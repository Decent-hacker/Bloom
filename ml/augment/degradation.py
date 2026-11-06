"""DIP-derived degradation augmentation (Task 4.3).

Reuses Bloom's restoration pipeline noise models (backend/src/restoration.py)
to simulate real-world upload quality degradation — phones, compression,
motion. Part of the DIP contribution.

All functions: numpy BGR uint8 in -> numpy BGR uint8 out.
"""

from __future__ import annotations

import numpy as np


def gaussian_noise(image: np.ndarray, sigma_range: tuple[int, int] = (5, 25)) -> np.ndarray:
    """Additive Gaussian sensor noise (mirrors restoration.py noise models)."""
    sigma = float(np.random.uniform(*sigma_range))
    noise = np.random.normal(0.0, sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper_noise(
    image: np.ndarray, amount_range: tuple[float, float] = (0.001, 0.01)
) -> np.ndarray:
    """Impulse noise — random salt (255) and pepper (0) pixels."""
    amount = float(np.random.uniform(*amount_range))
    out = image.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    mask = np.random.randint(0, 2, n)
    out[ys[mask == 0], xs[mask == 0]] = 0
    out[ys[mask == 1], xs[mask == 1]] = 255
    return out


def motion_blur(image: np.ndarray, kernel_size_range: tuple[int, int] = (3, 7)) -> np.ndarray:
    """Directional motion blur with a random angle (camera shake / subject motion)."""
    import cv2

    k = int(np.random.randint(kernel_size_range[0], kernel_size_range[1] + 1))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(np.random.uniform(0, 180))
    matrix = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(image, -1, kernel)


def jpeg_compression_artifact(
    image: np.ndarray, quality_range: tuple[int, int] = (40, 80)
) -> np.ndarray:
    """Round-trip through JPEG at low quality — matches real upload artifacts."""
    import cv2

    quality = int(np.random.randint(quality_range[0], quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
