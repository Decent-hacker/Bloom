import numpy as np
import pytest

from src import skin_segmentation
from src.errors import FaceLandmarkError


def test_estimate_yaw_uses_nose_position_relative_to_eyes():
    landmarks = [(0.5, 0.5)] * 264
    landmarks[33] = (0.3, 0.4)
    landmarks[263] = (0.7, 0.4)
    landmarks[1] = (0.5, 0.5)
    assert skin_segmentation.estimate_yaw(landmarks) == 0.0

    landmarks[1] = (0.6, 0.5)
    assert skin_segmentation.estimate_yaw(landmarks) == 0.5


def test_missing_landmarks_raise_strict_error(monkeypatch):
    monkeypatch.setattr(skin_segmentation, "_run_face_mesh", lambda image: None)
    with pytest.raises(FaceLandmarkError):
        skin_segmentation.run(np.zeros((64, 64, 3), np.uint8))


def test_no_fallback_mask_is_returned(monkeypatch):
    monkeypatch.setattr(skin_segmentation, "_run_face_mesh", lambda image: None)
    with pytest.raises(FaceLandmarkError):
        skin_segmentation.get_face_mask(np.zeros((64, 64, 3), np.uint8))
