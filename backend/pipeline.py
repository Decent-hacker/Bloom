"""Orchestrator: runs src/ stages in order and returns public results + intermediates."""
from __future__ import annotations

import time

import cv2
import numpy as np

from src import (
    enhancement,
    frequency_decomposition,
    imperfection_map,
    lesion_detection,
    postprocessing,
    preprocessing,
    skin_segmentation,
    skin_tone,
    visualization,
)


def _combine(masks: dict) -> np.ndarray:
    """Returns union of per-concern masks."""
    out = np.zeros_like(next(iter(masks.values())))
    for mask in masks.values():
        out = cv2.bitwise_or(out, mask)
    return out


def run_stages(image_bytes: bytes) -> dict:
    """Returns all pipeline stages, including arrays, for evaluation/experiments.

    Raises ``FaceLandmarkError`` when strict face landmark detection fails.
    """
    start = time.perf_counter()
    stage_times = {}
    checkpoint = start

    def mark(stage: str) -> None:
        nonlocal checkpoint
        now = time.perf_counter()
        stage_times[stage] = round(now - checkpoint, 4)
        checkpoint = now

    # 1. preprocess + brightness/sharpness gate (single BGR->RGB conversion)
    image, quality = preprocessing.run(image_bytes)
    mark("preprocessing")
    if not quality["passed"]:
        return {"quality": quality, "stage_times_sec": stage_times, "processing_time_sec": round(time.perf_counter() - start, 2)}

    # 2. enhancement (illumination normalisation + CLAHE + bilateral)
    enhanced = enhancement.run(image)
    mark("enhancement")
    # 3. strict FaceMesh ROI + skin segmentation
    segmentation = skin_segmentation.run(enhanced["enhanced"])
    mark("skin_segmentation")
    # 4. spatial frequency decomposition (LP / HP / DoG / LAB)
    frequency = frequency_decomposition.run(enhanced["enhanced"])
    mark("frequency_decomposition")
    # 5. raw concern masks
    raw = lesion_detection.run(frequency, segmentation)
    mark("lesion_detection")
    # 6. cleanup + region props + severity/extent
    concerns = postprocessing.run(raw, segmentation)
    mark("postprocessing")
    tone = skin_tone.compute_ita(frequency["lab"], segmentation["analysis_zone"])
    mark("skin_tone")
    surface = imperfection_map.compute_imperfection_map(enhanced["enhanced"], segmentation["analysis_zone"])
    mark("imperfection_map")

    # Naive baseline masks are for offline comparison/visualisation only.
    # Keep the baseline's thresholds intentionally simple, but never allow it
    # to mark protected facial features (eyes, lips, brows, or nose).
    baseline_masks = lesion_detection.naive_concern_masks(frequency, segmentation["analysis_zone"])
    baseline_mask = _combine(baseline_masks)
    improved_mask = _combine({name: c["mask"] for name, c in concerns.items()})
    mark("baseline")

    return {
        "quality": quality,
        "original": enhanced["original"],
        "enhanced": enhanced["enhanced"],
        "segmentation": segmentation,
        "frequency": frequency,
        "raw_masks": raw,
        "concerns": concerns,
        "skin_tone": tone,
        "imperfection_map": surface,
        "stage_times_sec": stage_times,
        "baseline_masks": baseline_masks,
        "baseline_mask": baseline_mask,
        "improved_mask": improved_mask,
        "detection": {
            "face_detected": True,
            "landmarks_available": bool(segmentation["landmarks_available"]),
            "fallback_used": False,
            "model": "mediapipe_face_mesh",
        },
        "mediapipe_mode": segmentation["mediapipe_mode"],
        "processing_time_sec": round(time.perf_counter() - start, 2),
    }


def run(image_bytes: bytes) -> dict:
    """Returns the public PipelineOutput used by the API (base64 images, no stage arrays)."""
    stages = run_stages(image_bytes)
    if not stages["quality"]["passed"]:
        return {"quality": stages["quality"], "stage_times_sec": stages["stage_times_sec"], "processing_time_sec": stages["processing_time_sec"]}

    images = visualization.generate_all(
        stages["original"],
        stages["enhanced"],
        stages["segmentation"]["skin_mask"],
        stages["concerns"],
        stages["baseline_mask"],
        stages["frequency"],
    )
    images["imperfection_heatmap"] = visualization.encode_image_to_base64(stages["imperfection_map"]["heatmap"])
    images["imperfection_overlay"] = visualization.encode_image_to_base64(stages["imperfection_map"]["overlay"])
    return {
        "quality": stages["quality"],
        "concerns": stages["concerns"],
        "skin_tone": stages["skin_tone"],
        "imperfection_map": {key: value for key, value in stages["imperfection_map"].items() if key not in ("heatmap", "overlay")},
        "stage_times_sec": stages["stage_times_sec"],
        "images": images,
        "detection": stages["detection"],
        "mediapipe_mode": stages["mediapipe_mode"],
        "processing_time_sec": stages["processing_time_sec"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m pipeline <image>")
    path = sys.argv[1]
    with open(path, "rb") as f:
        out = run(f.read())

    if not out["quality"]["passed"]:
        print("quality failed:", out["quality"]["issues"])
    else:
        print(f"time={out['processing_time_sec']}s mode={out['mediapipe_mode']}")
        for name, c in out["concerns"].items():
            print(f"  {name:8s} {c['severity']:<8} pct={c['area_pct']} extent={c['extent_score']}")
    print("pipeline.py: OK")
