"""Image appearance estimate from the analysable facial skin area."""
import numpy as np


def compute_ita(lab_image: np.ndarray, skin_mask: np.ndarray) -> dict:
    if lab_image.ndim != 3 or lab_image.shape[:2] != skin_mask.shape or lab_image.shape[2] != 3:
        raise ValueError("LAB image and skin mask dimensions do not match")

    pixels = lab_image[skin_mask > 0]
    if not len(pixels):
        return {"ita_angle": None, "appearance_band": "Unavailable"}

    # Convert OpenCV 8-bit LAB to CIE L* and b*
    # OpenCV: L in [0,255], b in [0,255] centered at 128
    l_star = pixels[:, 0].astype(np.float32) * (100.0 / 255.0)
    b_star = pixels[:, 2].astype(np.float32) - 128.0

    # ITA formula: atan((L* - 50) / b*) x 180/pi
    # arctan2 is WRONG here — use scalar atan with safe division
    b_safe = np.where(np.abs(b_star) < 0.001, 0.001, b_star)
    per_pixel_ita = np.degrees(np.arctan((l_star - 50.0) / b_safe))

    # Median of per-pixel angles (more robust than median L + median b separately)
    angle = float(np.median(per_pixel_ita))

    if angle > 55:
        band = "Very Light"
    elif angle >= 41:
        band = "Light"
    elif angle >= 28:
        band = "Intermediate"
    elif angle >= 10:
        band = "Tan"
    elif angle >= -30:
        band = "Brown"
    else:
        band = "Dark"
    return {"ita_angle": round(angle, 2), "appearance_band": band}
