"""Augmentation pipeline for Bloom 2.0 training ablations (Task 4.4).

Modes (AGENT_PLAN.md §4.4):
  "none"        -> identity               (run a — baseline)
  "geometric"   -> geometric only         (run b)
  "photometric" -> geometric + photometric (run c)
  "full"        -> geometric + photometric + degradation (run d)

This is the DIP contribution: augmentation families are implemented with
Bloom's own histogram/restoration operators rather than stock transforms.
"""

from __future__ import annotations

import numpy as np

from .degradation import (
    gaussian_noise,
    jpeg_compression_artifact,
    motion_blur,
    salt_and_pepper_noise,
)
from .geometric import (
    random_flip_horizontal,
    random_rotate,
    random_scale_crop,
    random_translate,
)
from .photometric import (
    brightness_shift,
    clahe_jitter,
    gamma_jitter,
    white_balance_shift,
)

VALID_MODES = ("none", "geometric", "photometric", "full")

# Per-operation application probability
P_GEOMETRIC = 0.5
P_PHOTOMETRIC = 0.5
P_DEGRADATION = 0.3


class AugmentationPipeline:
    """Mode-gated augmentation used by train_yolo.py for the ablation matrix."""

    def __init__(self, mode: str):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode

    def __call__(self, image: np.ndarray, boxes: list) -> tuple[np.ndarray, list]:
        if self.mode == "none":
            return image, boxes

        # --- geometric family -------------------------------------------
        if np.random.rand() < P_GEOMETRIC:
            image, boxes = random_rotate(image, boxes)
        if np.random.rand() < P_GEOMETRIC:
            image, boxes = random_flip_horizontal(image, boxes)
        if np.random.rand() < P_GEOMETRIC:
            image, boxes = random_scale_crop(image, boxes)
        if np.random.rand() < P_GEOMETRIC:
            image, boxes = random_translate(image, boxes)

        # --- photometric family -----------------------------------------
        if self.mode in ("photometric", "full"):
            if np.random.rand() < P_PHOTOMETRIC:
                image = gamma_jitter(image)
            if np.random.rand() < P_PHOTOMETRIC:
                image = clahe_jitter(image)
            if np.random.rand() < P_PHOTOMETRIC:
                image = white_balance_shift(image)
            if np.random.rand() < P_PHOTOMETRIC:
                image = brightness_shift(image)

        # --- degradation family -----------------------------------------
        if self.mode == "full":
            if np.random.rand() < P_DEGRADATION:
                image = gaussian_noise(image)
            if np.random.rand() < P_DEGRADATION:
                image = salt_and_pepper_noise(image)
            if np.random.rand() < P_DEGRADATION:
                image = motion_blur(image)
            if np.random.rand() < P_DEGRADATION:
                image = jpeg_compression_artifact(image)

        return image, boxes


__all__ = [
    "AugmentationPipeline",
    "VALID_MODES",
    "gamma_jitter",
    "clahe_jitter",
    "white_balance_shift",
    "brightness_shift",
    "random_rotate",
    "random_flip_horizontal",
    "random_scale_crop",
    "random_translate",
    "gaussian_noise",
    "salt_and_pepper_noise",
    "motion_blur",
    "jpeg_compression_artifact",
]
