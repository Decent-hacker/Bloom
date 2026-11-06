# Bloom ML — acne-detection MVP

The deployable ML scope is deliberately one feature: **acne lesion detection**.
It uses ACNE04's real lesion-level boxes. Dark spots, diffuse pigmentation,
redness, scars, pores, and wrinkles are not ML classes until reviewed spatial
annotations exist for each one; image-folder labels and synthetic centre boxes
are never used for detector training.

## Layout

```
ml/
  datasets/    fetch, inspect, and MVP dataset builders
  augment/     DIP-derived augmentation library (photometric/geometric/degradation)
  train/       YOLOv8 training script, config, Colab notebook
  eval/        evaluation, classical comparison, report generation
  annotations/ future own-collection capture/annotation protocol
  runs/        checkpoints, logs, results (gitignored)
  data/        raw + harmonized datasets (gitignored)
```

## Quick start

```bash
# Fetch ACNE04 once (no torch needed)
python ml/datasets/fetch_acne04.py

# Build the one-class YOLO layout. It excludes every placeholder annotation.
python ml/datasets/build_acne_mvp_dataset.py

# Full training deps (prefer Colab for a full run)
pip install -r ml/requirements.txt
python ml/train/train_yolo.py --epochs 50
```

## Notes

- `ml/data/` and `ml/runs/` are gitignored — datasets and checkpoints never
  enter git.
- `*.pt` / `*.onnx` model weights are gitignored — upload to HuggingFace Hub
  via `ml/deploy/upload_model.py` instead.
- Own collected images live in `ml/annotations/images/` (gitignored, privacy).
- The MVP has one class: `acne`.
- Archive folders for dark spots, pigmentation, and redness are an annotation
  pool only. Deduplicate and manually annotate them before expanding scope.
