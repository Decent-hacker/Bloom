import numpy as np

from src.lesion_detection import naive_concern_masks


def test_baseline_masks_respect_feature_exclusion_zone():
    """Eyes/lips removed from the zone must never appear in any baseline mask."""
    lab = np.zeros((20, 20, 3), dtype=np.uint8)
    lab[:, :, 0] = 20  # deliberately dark enough to trigger a naive dark mask
    lab[:, :, 1] = 255
    frequency = {"lab": lab, "dog": np.full((20, 20), 255, dtype=np.uint8)}
    zone = np.full((20, 20), 255, dtype=np.uint8)
    zone[5:15, 5:15] = 0  # simulated protected eye/lip region

    masks = naive_concern_masks(frequency, zone)

    for mask in masks.values():
        assert not mask[5:15, 5:15].any()
