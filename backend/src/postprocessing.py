"""Mask cleanup -> min-area filter -> region props -> severity + extent score.

``extent_score`` is a relative area measure in [0, 1], not detection certainty.
A successful run with no surviving regions is reported as ``Clear``.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from src.config import (
    LESION_MORPH_KERNEL,
    MIN_LESION_AREA_PX,
    SEVERITY,
)


def _severity(concern: str, area_px: int, area_pct: float) -> str:
    """Returns 'Clear' | 'Mild' | 'Moderate' | 'Severe' from unrounded extent."""
    if area_px == 0:
        return "Clear"
    band = SEVERITY[concern]
    if area_pct >= band["Severe"]:
        return "Severe"
    if area_pct >= band["Moderate"]:
        return "Moderate"
    if area_pct >= band["Mild"]:
        return "Mild"
    return "Clear"


def _extent_score(concern: str, area_pct: float) -> float:
    """Returns relative extent in [0, 1], normalised by the concern's Moderate threshold."""
    if area_pct <= 0:
        return 0.0
    scale = SEVERITY[concern]["Moderate"]
    return round(min(area_pct / scale, 1.0), 3)


def process_concern(raw_mask: np.ndarray, concern: str, skin_area: int) -> dict:
    """Returns ConcernResult (mask, count, area_px, area_pct, severity, extent_score, regions)."""
    # remove speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, LESION_MORPH_KERNEL)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # connected components + per-region props
    labels = sk_label(cleaned > 0, connectivity=2)
    kept = np.zeros_like(cleaned)
    regions: list[dict] = []
    for prop in regionprops(labels):
        if prop.area < MIN_LESION_AREA_PX:
            continue  # noise
        kept[labels == prop.label] = 255
        regions.append({
            "id": len(regions) + 1,
            "area_px": int(prop.area),
            "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
            "bbox": (int(prop.bbox[0]), int(prop.bbox[1]), int(prop.bbox[2]), int(prop.bbox[3])),
        })

    area_px = int((kept > 0).sum())
    area_pct_raw = 100.0 * area_px / skin_area if skin_area else 0.0
    # redness is diffuse; component count is not meaningful
    count = -1 if concern == "redness" else len(regions)

    return {
        "mask": kept,
        "count": count,
        "area_px": area_px,
        "area_pct": round(area_pct_raw, 2),
        "severity": _severity(concern, area_px, area_pct_raw),
        "extent_score": _extent_score(concern, area_pct_raw),
        "regions": regions,
    }


def run(raw_masks: dict, segmentation: dict) -> dict:
    """Returns {acne, redness, blemish} ConcernResult dicts."""
    skin_area = int((segmentation["analysis_zone"] > 0).sum())
    return {name: process_concern(mask, name, skin_area) for name, mask in raw_masks.items()}


if __name__ == "__main__":
    import sys

    from src.enhancement import run as enhance
    from src.frequency_decomposition import run as decompose
    from src.lesion_detection import run as detect
    from src.preprocessing import run as preprocess
    from src.skin_segmentation import run as segment

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.postprocessing <image>")
    path = sys.argv[1]
    with open(path, "rb") as f:
        image, _ = preprocess(f.read())

    enhanced = enhance(image)["enhanced"]
    segmentation = segment(enhanced)
    results = run(detect(decompose(enhanced), segmentation), segmentation)
    for name, res in results.items():
        print(f"{name:8s} count={res['count']:<3} area_px={res['area_px']:<5} "
              f"area_pct={res['area_pct']:<5} severity={res['severity']:<8} "
              f"extent={res['extent_score']} regions={len(res['regions'])}")
    print("postprocessing.py: OK")
