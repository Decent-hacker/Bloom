"""DIP-derived photometric augmentation.

Simulates real-world lighting variation and phone camera AWB errors on Indian
skin tones. Reuses the spirit of backend/src/histogram_processing.py (CLAHE,
gamma) — this IS the DIP contribution to the ML pipeline (AGENT_PLAN.md Task 4.1).

All functions: numpy BGR uint8 in -> numpy BGR uint8 out.
"""

from __future__ import annotations

import numpy as np


def gamma_jitter(image: np.ndarray, gamma_range: tuple[float, float] = (0.7, 1.5)) -> np.ndarray:
    """Apply a random gamma correction (DIP: power-law intensity transform)."""
    gamma = float(np.random.uniform(*gamma_range))
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return table[image]


def clahe_jitter(image: np.ndarray, clip_limit_range: tuple[float, float] = (1.0, 4.0)) -> np.ndarray:
    """Apply CLAHE on the LAB L channel with a randomized clip limit.

    Mirrors the contrast-limited histogram equalisation used in Bloom's
    enhancement stage (backend/src/histogram_processing.py).
    """
    import cv2

    clip = float(np.random.uniform(*clip_limit_range))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def white_balance_shift(image: np.ndarray, gain_range: tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
    """Random per-channel gain — simulates phone auto-white-balance errors."""
    gains = 1.0 + np.random.uniform(gain_range[0], gain_range[1], size=3)
    out = image.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_shift(image: np.ndarray, shift_range: tuple[float, float] = (-0.2, 0.2)) -> np.ndarray:
    """Additive brightness shift in normalized units (-0.2 = darken by 20%)."""
    shift = float(np.random.uniform(*shift_range)) * 255.0
    return np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
