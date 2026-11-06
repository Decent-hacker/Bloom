"""Inference for Bloom's evidence-based acne-only MVP.

Only ACNE04-trained lesion detection is supported.  Other visual concerns
remain in the classical pipeline until they have independent spatial labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(REPO_ROOT / "ml" / "runs" / "acne_mvp" / "best.pt")))
CLASS_NAMES = ["acne"]

# Lazy model loading — avoid import-time torch dependency
_model: Any | None = None
_model_loaded = False
_torch_available = False


def _ensure_torch():
    """Import torch if available; set flag for callers."""
    global _torch_available
    if _torch_available:
        return
    try:
        import torch  # noqa: F401  (import only, we use it via YOLO)
        _torch_available = True
    except ImportError:
        _torch_available = False


def _load_model() -> object | None:
    """Load a local, explicitly supplied acne-only YOLO checkpoint once."""
    global _model, _model_loaded

    if _model_loaded and _model is not None:
        return _model

    _ensure_torch()

    # If torch not available, we can't load YOLO
    if not _torch_available:
        import warnings
        warnings.warn("torch not available — ML model will not load")
        _model_loaded = True
        return None

    if not MODEL_PATH.is_file():
        import logging
        logging.getLogger(__name__).warning("Acne MVP checkpoint not found at %s", MODEL_PATH)
        return None

    # Load with YOLO — keep CPU threads limited for deployment
    import torch
    torch.set_num_threads(2)
    try:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
        _model_loaded = True
        return _model
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to load YOLO model: {e}")
        return None


def model_status() -> dict:
    """Expose capability without claiming that an untrained model is available."""
    return {
        "available": _model is not None,
        "scope": "acne_lesion_detection",
        "checkpoint_path": str(MODEL_PATH),
    }


def detect(image: np.ndarray) -> list[dict]:
    """Run YOLOv8 detection on a BGR numpy image.

    Returns list of detections:
    [
      {
        "class": "acne",
        "class_id": 0,
        "confidence": 0.82,
        "box": [x1, y1, x2, y2],   # pixel coords
        "severity": "Mild"          # from area-fraction postprocessing
      },
      ...
    ]
    Returns [] if model not loaded (graceful degradation).
    """
    global _model, _model_loaded

    if _model is None or not _model_loaded:
        return []

    # Ultralytics letterboxes internally and returns boxes in the original
    # image coordinate system.  Pre-resizing here would make UI boxes wrong.
    try:
        results = _model(image, imgsz=640, verbose=False)
    except Exception:
        return []

    detections = []
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id != 0:
                continue
            class_name = CLASS_NAMES[class_id]

            # Convert box from xyxy to [x1, y1, x2, y2] pixel coords
            x1, y1, x2, y2 = float(box.xyxy[0][0]), float(box.xyxy[0][1]), float(box.xyxy[0][2]), float(box.xyxy[0][3])

            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
            })

    return detections


# Auto-load on module import. Missing optional ML dependencies or a checkpoint
# leave the API in honest classical-only mode.
_ensure_torch()
_load_model()
