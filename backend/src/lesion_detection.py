"""Raw concern masks: redness (LAB-A), blemish (LAB-L), acne (DoG + watershed).

Production thresholds are ROI-adaptive. A deliberately naive fixed-threshold
baseline is kept for offline comparison only.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.config import (
    ACNE_DOG_THRESHOLD_RATIO,
    BLEMISH_L_MIN_DELTA,
    BLEMISH_L_STD_MULT,
    LESION_MORPH_KERNEL,
    NAIVE_ACNE_DOG_ABS_THRESHOLD,
    NAIVE_BLEMISH_L_THRESHOLD,
    NAIVE_REDNESS_A_THRESHOLD,
    REDNESS_A_MIN_DELTA,
    REDNESS_A_STD_MULT,
    WATERSHED_DIST_THRESHOLD,
    WATERSHED_MIN_DISTANCE,
)
from src.thresholding import mean_std_threshold, ratio_of_max_threshold


def _restrict(mask: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Returns mask limited to the analysis zone."""
    return cv2.bitwise_and(mask, zone)


def _redness_mask(lab: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Returns binary mask of pixels redder than the adaptive skin-A threshold."""
    a = lab[:, :, 1]
    _, mask = mean_std_threshold(
        a, zone, REDNESS_A_MIN_DELTA, REDNESS_A_STD_MULT, high_is_foreground=True
    )
    return _restrict(mask, zone)


def _blemish_mask(lab: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Returns binary mask of dark spots below the adaptive skin-L threshold."""
    lum = lab[:, :, 0]
    _, mask = mean_std_threshold(
        lum, zone, BLEMISH_L_MIN_DELTA, BLEMISH_L_STD_MULT, high_is_foreground=False
    )
    return _restrict(mask, zone)


def _separate_touching(binary: np.ndarray) -> np.ndarray:
    """Returns binary mask split at watershed lines so touching spots count separately."""
    if binary.max() == 0:
        return binary
    dist = ndi.distance_transform_edt(binary > 0)
    peaks = peak_local_max(
        dist,
        min_distance=WATERSHED_MIN_DISTANCE,
        labels=(binary > 0).astype(np.int32),
        exclude_border=False,
    )
    # drop weak peaks, but keep at least one seed
    strong = peaks[dist[tuple(peaks.T)] >= WATERSHED_DIST_THRESHOLD * dist.max()]
    seeds = strong if len(strong) else peaks
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    labels = watershed(-dist, markers, mask=(binary > 0), watershed_line=True)
    return np.where(labels > 0, 255, 0).astype(np.uint8)


def _acne_mask(dog: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Returns binary acne mask from |DoG|, then watershed-split."""
    magnitude = np.abs(dog)  # capture both dark and bright blobs
    _, mask = ratio_of_max_threshold(magnitude, zone, ACNE_DOG_THRESHOLD_RATIO)
    mask = _restrict(mask, zone)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, LESION_MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _separate_touching(mask)


def naive_concern_masks(frequency: dict, analysis_zone: np.ndarray) -> dict:
    """Returns per-concern masks for the fixed-threshold baseline.

    No adaptive thresholds or watershed. The supplied analysis zone still
    excludes eyes, lips, brows, and nose so the baseline cannot label them as
    visual concerns.
    """
    lab = frequency["lab"]
    lum, a = lab[:, :, 0], lab[:, :, 1]
    red = np.where(a > NAIVE_REDNESS_A_THRESHOLD, 255, 0).astype(np.uint8)
    dark = np.where(lum < NAIVE_BLEMISH_L_THRESHOLD, 255, 0).astype(np.uint8)
    acne = np.where(np.abs(frequency["dog"]) >= NAIVE_ACNE_DOG_ABS_THRESHOLD, 255, 0).astype(np.uint8)
    return {
        "acne": _restrict(acne, analysis_zone),
        "redness": _restrict(red, analysis_zone),
        "blemish": _restrict(dark, analysis_zone),
    }


def naive_combined_mask(image: np.ndarray, analysis_zone: np.ndarray) -> np.ndarray:
    """Returns union of the naive per-concern masks (visual baseline only)."""
    from src import frequency_decomposition

    frequency = frequency_decomposition.run(image)
    masks = naive_concern_masks(frequency, analysis_zone)
    out = np.zeros_like(analysis_zone)
    for mask in masks.values():
        out = cv2.bitwise_or(out, mask)
    return out


def run(frequency: dict, segmentation: dict) -> dict:
    """Returns RawConcernMasks dict: {acne, redness, blemish}."""
    lab = frequency["lab"]
    zone = segmentation["analysis_zone"]

    return {
        "surface_anomalies": _acne_mask(frequency["dog"], zone),
        "redness_contrast": _redness_mask(lab, zone),
        "dark_regions": _blemish_mask(lab, zone),
    }


if __name__ == "__main__":
    import sys

    from src.enhancement import run as enhance
    from src.frequency_decomposition import run as decompose
    from src.preprocessing import run as preprocess
    from src.skin_segmentation import run as segment

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.lesion_detection <image>")
    path = sys.argv[1]
    with open(path, "rb") as f:
        image, _ = preprocess(f.read())

    enhanced = enhance(image)["enhanced"]
    frequency = decompose(enhanced)
    segmentation = segment(enhanced)

    masks = run(frequency, segmentation)
    for key, m in masks.items():
        print(f"{key:8s} shape={m.shape} dtype={m.dtype} nonzero={int((m > 0).sum())}")
    print("lesion_detection.py: OK")
