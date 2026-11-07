"""FFT surface detail visualization; regions are not medical findings."""
import cv2
import numpy as np

from src.frequency_domain import gaussian_bandpass


def compute_imperfection_map(image: np.ndarray, skin_mask: np.ndarray, sensitivity: float = 1.5) -> dict:
    if image.ndim != 3 or image.shape[:2] != skin_mask.shape or image.shape[2] != 3:
        raise ValueError("RGB image and skin mask dimensions do not match")
    if sensitivity < 0:
        raise ValueError("sensitivity must be nonnegative")
    response = np.abs(gaussian_bandpass(image)["filtered"])
    selected = skin_mask > 0
    overlay = image.copy()
    heatmap = np.zeros_like(image, dtype=np.uint8)
    if not selected.any():
        return {"heatmap": heatmap, "overlay": overlay, "region_count": 0,
                "coverage_percent": 0.0, "visual_label": "No analysable skin"}
    values = response[selected]
    threshold = float(values.mean() + sensitivity * values.std())
    binary = np.zeros(skin_mask.shape, dtype=np.uint8)
    binary[selected & (response > threshold)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel)
    binary[~selected] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 20]
    retained = np.zeros_like(binary)
    cv2.drawContours(retained, contours, -1, 255, cv2.FILLED)
    retained[~selected] = 0
    scale = float(np.percentile(values, 99))
    normalized = np.clip(response / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap[selected] = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)[selected]
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    coverage = round(100.0 * np.count_nonzero(retained) / np.count_nonzero(selected), 2)
    return {"heatmap": heatmap, "overlay": overlay, "region_count": len(contours),
            "coverage_percent": coverage, "visual_label": "Relative surface detail"}
