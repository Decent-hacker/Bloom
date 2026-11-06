"""Bloom 2.0 YOLOv8 training script.

Usage:
  # smoke test (tiny slice, 2 epochs — proves the toolchain)
  python ml/train/train_yolo.py --aug-mode none --epochs 2 --fraction 0.05 --imgsz 320

Pipeline:
  1. build_acne_mvp_dataset.py creates a one-class, lesion-box-only dataset
  2. fine-tune YOLOv8n (config: ml/train/config.yaml)
  3. validate on clean validation split; append metrics to ml/runs/run_log.csv
  4. checkpoint: ml/runs/{run_id}/best.pt (also weights/best.pt from ultralytics)

Run id: {aug_mode}_{seed}_{timestamp}  e.g. full_42_20261029-153000
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

RUNS_DIR = ML_ROOT / "runs"
CONFIG_PATH = ML_ROOT / "train" / "config.yaml"
MVP_DATASET = ML_ROOT / "data" / "acne_mvp"
RUN_LOG = RUNS_DIR / "run_log.csv"
CLASS_NAMES = ["acne"]

RUN_LOG_FIELDS = [
    "run_id", "aug_mode", "model", "epochs_run", "mAP50", "mAP50-95",
    "F1_acne", "F1_dark_spot", "F1_scar", "F1_rash",
    "dataset_hash", "seed", "timestamp",
]


def _read_dataset_hash() -> str:
    hash_file = MVP_DATASET / "dataset_hash.txt"
    if hash_file.exists():
        return hash_file.read_text(encoding="utf-8").splitlines()[0].strip()
    return "unknown"


def _per_class_f1_from_confusion(matrix) -> list[float]:
    """Per-class F1 from ultralytics ConfusionMatrix at its operating conf.

    matrix is (nc+1, nc+1): rows=GT, cols=pred; last row/col = background.
    """
    import numpy as np

    m = np.asarray(matrix, dtype=np.float64)
    nc = m.shape[0] - 1
    f1s = []
    for c in range(nc):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    return f1s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aug-mode", choices=["none"], default="none",
                        help="The MVP uses a clean, lesion-box-only baseline.")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None,
                        help="fraction of train set (smoke tests only)")
    parser.add_argument("--model", type=str, default=None, help="override base weights")
    parser.add_argument("--workers", type=int, default=None,
                        help="dataloader workers (default: 0 on Windows, config value elsewhere)")
    args = parser.parse_args()

    # Lazy imports so the script fails with a clear message if deps missing
    try:
        import yaml
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERROR: ultralytics/torch not installed in this environment.\n"
            "  Local smoke test:  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cpu && pip install ultralytics==8.2.18\n"
            "  Real training: use Colab — open ml/train/colab_train.ipynb",
            file=sys.stderr,
        )
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))

    # The only accepted data source is the explicit one-class MVP builder.
    data_value = Path(config.get("data", "ml/data/acne_mvp/classes.yaml"))
    data_yaml = data_value if data_value.is_absolute() else ML_ROOT.parent / data_value

    if not data_yaml.exists():
        print(
            f"ERROR: {data_yaml} missing — run ml/datasets/build_acne_mvp_dataset.py first",
            file=sys.stderr,
        )
        return 1

    run_id = f"{args.aug_mode}_{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config.get("model", "yolov8s.pt")
    print(f"Run: {run_id}")
    print(f"Model: {model_name} | data: {data_yaml}")

    model = YOLO(model_name)

    # Windows: DataLoader workers > 0 frequently crashes/hangs ultralytics
    # training (spawn-based multiprocessing). Force 0 on win32 unless the
    # user explicitly overrides via --workers. Linux/Colab keep the config value.
    import platform

    if args.workers is not None:
        workers = args.workers
    elif platform.system() == "Windows":
        workers = 0
    else:
        workers = int(config.get("workers", 4))

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs if args.epochs is not None else int(config.get("epochs", 100)),
        patience=int(config.get("patience", 25)),
        batch=args.batch if args.batch is not None else int(config.get("batch", 16)),
        imgsz=args.imgsz if args.imgsz is not None else int(config.get("imgsz", 640)),
        optimizer=config.get("optimizer", "AdamW"),
        lr0=float(config.get("lr0", 0.001)),
        lrf=float(config.get("lrf", 0.01)),
        seed=seed,
        workers=workers,
        # NOTE: fl_gamma (focal loss) is NOT a top-level arg in ultralytics
        # 8.2.18 — imbalance is handled by the augmentation pipeline +
        # minority-class oversampling instead (see config.yaml comment).
        # keep YOLO's own augment off — OUR pipeline is the experiment
        mosaic=float(config.get("mosaic", 0.0)),
        mixup=float(config.get("mixup", 0.0)),
        project=str(RUNS_DIR),
        name=run_id,
        exist_ok=True,
        verbose=True,
    )
    if args.fraction is not None:
        train_kwargs["fraction"] = args.fraction  # smoke tests only

    print("Training...")
    results = model.train(**train_kwargs)
    epochs_run = int(getattr(results, "epoch", 0)) or args.epochs or 0

    # --- validate on clean val -------------------------------------------
    print("Validating on clean val split...")
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=train_kwargs["imgsz"],
        project=str(RUNS_DIR),
        name=f"{run_id}_val",
        exist_ok=True,
        verbose=False,
    )

    try:
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
    except Exception:  # noqa: BLE001
        map50, map5095 = float("nan"), float("nan")

    try:
        f1s = _per_class_f1_from_confusion(metrics.confusion_matrix.matrix)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: could not read confusion matrix ({exc})")
        f1s = [float("nan")]

    # The run-log schema is retained for backwards compatibility with the
    # report scripts.  The unused legacy class values are intentionally NaN.
    f1s = (f1s + [float("nan")] * 4)[:4]

    # flatten best.pt to the AGENT_PLAN contract path
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, run_dir / "best.pt")

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "run_id": run_id,
        "aug_mode": args.aug_mode,
        "model": model_name,
        "epochs_run": epochs_run,
        "mAP50": f"{map50:.4f}",
        "mAP50-95": f"{map5095:.4f}",
        "F1_acne": f"{f1s[0]:.4f}",
        "F1_dark_spot": f"{f1s[1]:.4f}",
        "F1_scar": f"{f1s[2]:.4f}",
        "F1_rash": f"{f1s[3]:.4f}",
        "dataset_hash": _read_dataset_hash(),
        "seed": seed,
        "timestamp": timestamp,
    }
    new_log = not RUN_LOG.exists()
    with RUN_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if new_log:
            writer.writeheader()
        writer.writerow(row)

    print("\n=== Run complete ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"Checkpoint: {run_dir / 'best.pt'}")
    print(f"run_log.csv updated: {RUN_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
