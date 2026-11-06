"""Geometric augmentation with YOLO-box coordinate updates (Task 4.2).

Boxes are YOLO-normalized lists: [class_id, cx, cy, w, h] with values in [0, 1].
Each function returns (image, boxes) with boxes transformed consistently with
the image. Boxes whose visible fraction falls below a threshold are dropped.
"""

from __future__ import annotations

import math

import numpy as np

Box = list  # [class_id, cx, cy, w, h] normalized
MIN_VISIBLE = 0.25  # drop boxes with less than this fraction inside the image


def _box_corners(b: Box) -> np.ndarray:
    _, cx, cy, w, h = b
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _apply_affine_to_boxes(
    boxes: list[Box], matrix: np.ndarray, out_w: int, out_h: int, src_w: int, src_h: int
) -> list[Box]:
    """Transform normalized boxes by a pixel-space 2x3 affine matrix.

    matrix maps source pixel coords -> destination pixel coords.
    """
    out: list[Box] = []
    for b in boxes:
        corners = _box_corners(b)
        # normalized -> source pixels
        pix = corners * np.array([src_w, src_h], dtype=np.float64)
        ones = np.ones((4, 1))
        transformed = matrix @ np.hstack([pix, ones]).T  # 2x4
        dst = transformed.T / np.array([out_w, out_h], dtype=np.float64)  # back to normalized

        x_min, y_min = dst.min(axis=0)
        x_max, y_max = dst.max(axis=0)
        # visible fraction of the transformed box
        vis_w = max(0.0, min(x_max, 1.0) - max(x_min, 0.0))
        vis_h = max(0.0, min(y_max, 1.0) - max(y_min, 0.0))
        full_w = max(x_max - x_min, 1e-9)
        full_h = max(y_max - y_min, 1e-9)
        if (vis_w * vis_h) / (full_w * full_h) < MIN_VISIBLE:
            continue

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = min(x_max - x_min, 1.0)
        h = min(y_max - y_min, 1.0)
        cx_c = min(max(cx, 0.0), 1.0)
        cy_c = min(max(cy, 0.0), 1.0)
        if w <= 0 or h <= 0:
            continue
        out.append([b[0], cx_c, cy_c, w, h])
    return out


def random_rotate(
    image: np.ndarray, boxes: list[Box], angle_range: tuple[float, float] = (-15, 15)
) -> tuple[np.ndarray, list[Box]]:
    """Rotate image about its center; transform boxes identically."""
    import cv2

    h, w = image.shape[:2]
    angle = float(np.random.uniform(*angle_range))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, _apply_affine_to_boxes(boxes, matrix, w, h, w, h)


def random_flip_horizontal(image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[Box]]:
    """Mirror horizontally — cosmetic lesions are not chirality-sensitive."""
    out = np.ascontiguousarray(image[:, ::-1])
    flipped = [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in boxes]
    return out, flipped


def random_scale_crop(
    image: np.ndarray, boxes: list[Box], scale_range: tuple[float, float] = (0.9, 1.1)
) -> tuple[np.ndarray, list[Box]]:
    """Random zoom about the center (border-replicated, no true crop)."""
    import cv2

    h, w = image.shape[:2]
    scale = float(np.random.uniform(*scale_range))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, scale)
    out = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, _apply_affine_to_boxes(boxes, matrix, w, h, w, h)


def random_translate(
    image: np.ndarray, boxes: list[Box], translate_fraction: float = 0.05
) -> tuple[np.ndarray, list[Box]]:
    """Shift image by up to translate_fraction of its dimensions."""
    import cv2

    h, w = image.shape[:2]
    dx = float(np.random.uniform(-translate_fraction, translate_fraction)) * w
    dy = float(np.random.uniform(-translate_fraction, translate_fraction)) * h
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    out = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, _apply_affine_to_boxes(boxes, matrix, w, h, w, h)


_ = math   # kept for potential future use; avoids unused-import lint churn
