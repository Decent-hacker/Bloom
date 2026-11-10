#!/usr/bin/env bash
# Install optional ML dependencies and download the two Bloom checkpoints.
set -eu

cd "$(dirname "$0")" || { echo "FAIL: locate project root"; exit 1; }

if [ -n "${BLOOM_PYTHON:-}" ]; then
    PYTHON="$BLOOM_PYTHON"
elif [ -x backend/.venv/bin/python ]; then
    PYTHON=backend/.venv/bin/python
elif [ -x backend/.venv/Scripts/python.exe ]; then
    PYTHON=backend/.venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "FAIL: Python is unavailable"
    exit 1
fi
echo "OK: using $PYTHON"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else "FAIL: Bloom production requires Python 3.11; select it with BLOOM_PYTHON")'

echo "Installing huggingface_hub, ultralytics, timm, torchvision, torch..."
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    if "$PYTHON" -m pip install -r backend/requirements.txt -r backend/requirements-ml.txt --quiet; then
        install_status=0
    else
        install_status=$?
    fi
elif command -v uv >/dev/null 2>&1; then
    if UV_CACHE_DIR="${UV_CACHE_DIR:-$(mktemp -d)}" uv pip install --python "$PYTHON" -r backend/requirements.txt -r backend/requirements-ml.txt --quiet; then
        install_status=0
    else
        install_status=$?
    fi
else
    echo "FAIL: neither pip nor uv is available for package installation"
    exit 1
fi
if [ "$install_status" -eq 0 ]; then
    echo "OK: dependencies installed"
else
    echo "FAIL: dependency installation"
    exit 1
fi

# Fail during the build if native libraries or the selected wheel set cannot load.
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip check
else
    uv pip check --python "$PYTHON"
fi
"$PYTHON" - <<'PY'
import cv2
import mediapipe
import torch
import torchvision
import timm
from ultralytics import YOLO
print("OK: image processing and CPU ML runtime imports")
PY

if mkdir -p backend/models; then
    echo "OK: backend/models ready"
else
    echo "FAIL: could not create backend/models"
    exit 1
fi

"$PYTHON" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "mufasabrownie/glowlytics-skin-models"
dest = Path("backend/models")
failed = False
for name in ("skin_signals_best.pt", "acne_detector_best.pt"):
    target = dest / name
    if target.is_file():
        print(f"SKIP: {name} already exists")
        continue
    try:
        hf_hub_download(repo_id=repo, filename=name, local_dir=str(dest))
        if not target.is_file():
            raise FileNotFoundError(target)
        print(f"OK: downloaded {name}")
    except Exception as exc:
        print(f"FAIL: {name}: {exc}")
        failed = True
raise SystemExit(1 if failed else 0)
PY

if rm -f backend/models/skin_signals.onnx; then
    echo "OK: removed obsolete skin_signals.onnx"
else
    echo "FAIL: could not remove obsolete skin_signals.onnx"
    exit 1
fi
