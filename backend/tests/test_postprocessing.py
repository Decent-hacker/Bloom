import numpy as np

from src.postprocessing import _severity, process_concern


def test_zero_mask_is_clear_not_mild():
    mask = np.zeros((512, 512), np.uint8)
    result = process_concern(mask, "acne", skin_area=10000)
    assert result["severity"] == "Minimal variation"
    assert result["extent_score"] == 0.0
    assert result["count"] == 0


def test_redness_count_is_not_applicable():
    mask = np.zeros((512, 512), np.uint8)
    result = process_concern(mask, "redness", skin_area=10000)
    assert result["count"] == -1
    assert result["severity"] == "Minimal variation"


def test_small_regions_are_filtered_and_clear():
    mask = np.zeros((512, 512), np.uint8)
    mask[10:12, 10:12] = 255  # 4 px, below MIN_LESION_AREA_PX
    result = process_concern(mask, "acne", skin_area=10000)
    assert result["area_px"] == 0
    assert result["severity"] == "Minimal variation"


def test_severity_uses_unrounded_percentage():
    assert _severity("acne", 1, 0.999) == "Low variation"
    assert _severity("acne", 1, 1.0) == "Moderate variation"
    assert _severity("acne", 1, 3.0) == "High variation"
    assert _severity("acne", 0, 0.0) == "Minimal variation"


def test_extent_score_is_numeric_and_bounded():
    mask = np.zeros((512, 512), np.uint8)
    mask[0:20, 0:20] = 255  # 400 px
    result = process_concern(mask, "acne", skin_area=10000)
    assert 0.0 < result["extent_score"] <= 1.0
    assert isinstance(result["extent_score"], float)
    assert "confidence" not in result
