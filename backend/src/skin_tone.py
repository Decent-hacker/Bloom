"""Image appearance estimate from the analysable facial skin area."""
import numpy as np


def compute_ita(lab_image: np.ndarray, skin_mask: np.ndarray) -> dict:
    if lab_image.ndim != 3 or lab_image.shape[:2] != skin_mask.shape or lab_image.shape[2] != 3:
        raise ValueError("LAB image and skin mask dimensions do not match")
    pixels = lab_image[skin_mask > 0]
    if not len(pixels):
        return {"ita_angle": None, "appearance_band": "Unavailable"}
    l_star = float(np.median(pixels[:, 0])) * 100.0 / 255.0
    b_star = float(np.median(pixels[:, 2])) - 128.0
    angle = float(np.degrees(np.arctan2(l_star - 50.0, b_star)))
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
