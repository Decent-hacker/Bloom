"""Inference for Bloom's evidence-based acne-only MVP.

Only ACNE04-trained lesion detection is supported.  Other visual concerns
remain in the classical pipeline until they have independent spatial labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as _torch_error:
    torch = None
    nn = None

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


SIGNALS_PATH = Path(__file__).resolve().parent / "models" / "skin_signals_best.pt"
ACNE_PATH = Path(__file__).resolve().parent / "models" / "acne_detector_best.pt"
SIGNAL_NAMES = ("structure", "hydration", "sun_damage", "elasticity")
DISCLAIMER = "ML outputs are research/educational estimates. Not a medical diagnosis."

_signals_model: Any | None = None
_acne_detector: Any | None = None
_signals_lock = Lock()
_acne_lock = Lock()


def analysis_model_status() -> dict:
    """Report installed checkpoints and whether lazy model loading has occurred."""
    signals_loaded = _signals_model is not None
    acne_loaded = _acne_detector is not None
    return {
        "available": signals_loaded and acne_loaded,
        "installed": SIGNALS_PATH.is_file() and ACNE_PATH.is_file(),
        "signals_loaded": signals_loaded,
        "acne_loaded": acne_loaded,
    }


class SkinSignalsModel(nn.Module if nn is not None else object):
    def __init__(self):
        if nn is None:
            raise ImportError("PyTorch is required for skin signal inference")
        super().__init__()
        import timm

        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.shared = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(4)
        ])

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def _get_signals_model():
    global _signals_model
    if _signals_model is None:
        with _signals_lock:
            if _signals_model is None:
                if not SIGNALS_PATH.is_file():
                    raise FileNotFoundError(
                        f"skin_signals_best.pt not found at {SIGNALS_PATH}. Run setup_ml.sh first."
                    )
                model = SkinSignalsModel()
                state = torch.load(str(SIGNALS_PATH), map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                compatibility = model.load_state_dict(state, strict=False)
                if compatibility.missing_keys or compatibility.unexpected_keys:
                    raise ValueError(
                        "Skin signal checkpoint does not match model architecture: "
                        f"missing={compatibility.missing_keys}, unexpected={compatibility.unexpected_keys}"
                    )
                model.eval()
                _signals_model = model
    return _signals_model


def _get_acne_detector():
    global _acne_detector
    if _acne_detector is None:
        with _acne_lock:
            if _acne_detector is None:
                if not ACNE_PATH.is_file():
                    raise FileNotFoundError(f"Missing model: {ACNE_PATH}")
                from ultralytics import YOLO
                _acne_detector = YOLO(str(ACNE_PATH))
    return _acne_detector


def run_skin_signals(img_pil) -> dict:
    """Return four normalized skin signal estimates from the PyTorch model."""
    try:
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = _get_signals_model()
        x = transform(img_pil.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            scores = model(x)[0].numpy()
        if scores.size != len(SIGNAL_NAMES) or not np.all(np.isfinite(scores)):
            raise ValueError("Skin signal model returned invalid scores")
        return {name: round(float(scores[index]), 3)
                for index, name in enumerate(SIGNAL_NAMES)}
    except Exception as exc:
        return {**dict.fromkeys(SIGNAL_NAMES), "error": str(exc)}


def run_acne_detection(img_pil) -> dict:
    """Return acne boxes in original image pixel coordinates."""
    try:
        model = _get_acne_detector()
        results = model.predict(np.asarray(img_pil.convert("RGB")), conf=0.35, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                names = result.names
                class_name = names[class_id]
                detections.append({
                    "class": str(class_name),
                    "confidence": float(box.conf[0]),
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                })
        return {
            "detections": detections,
            "count": len(detections),
            "classes": sorted({item["class"] for item in detections}),
        }
    except Exception as exc:
        return {"detections": [], "count": 0, "classes": [], "error": str(exc)}


def run_ml_analysis(img_pil) -> dict:
    """Run both optional models without interrupting the classical pipeline."""
    signals = run_skin_signals(img_pil)
    acne = run_acne_detection(img_pil)
    return {
        "signals": signals,
        "acne": acne,
        "available": "error" not in signals and "error" not in acne,
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    import sys

    from PIL import Image

    if len(sys.argv) != 2:
        raise SystemExit("usage: python ml_inference.py <image_path>")
    with Image.open(sys.argv[1]) as image:
        print(run_ml_analysis(image))
