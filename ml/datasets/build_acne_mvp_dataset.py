"""Build Bloom's evidence-based acne-only MVP dataset.

The previous harmonized dataset mixed ACNE04 lesion boxes with image-level
classification images converted into artificial centre boxes.  That makes an
object detector learn a location artefact instead of acne.  This builder uses
only the original ACNE04 lesion annotations and deliberately has one class:
``acne``.

Run after ``fetch_acne04.py``:

    python ml/datasets/build_acne_mvp_dataset.py

It creates a disposable, gitignored YOLO layout at ``ml/data/acne_mvp``.
The original ACNE04 test split remains the test set.  Its train split is
deterministically divided into train/validation; no image is in two splits.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ML_ROOT / "data" / "raw" / "acne04"
OUT_ROOT = ML_ROOT / "data" / "acne_mvp"
SEED = 42
VALIDATION_FRACTION = 0.15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy a source file, creating its parent without mutating the source."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_manifest() -> list[dict[str, str]]:
    manifest = RAW_ROOT / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} is missing. Run ml/datasets/fetch_acne04.py first."
        )
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_acne_label(label_path: Path) -> bool:
    """Return True only for non-empty, valid one-class YOLO annotations."""
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            return False
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            return False
        if not all(0.0 <= value <= 1.0 for value in values):
            return False
    return True


def main() -> int:
    rows = _load_manifest()
    images = RAW_ROOT / "images"
    labels = RAW_ROOT / "annotations"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("ACNE04 images/ or annotations/ directory is missing.")

    samples: list[tuple[str, Path, Path, str]] = []
    skipped = 0
    for row in rows:
        image_id = row["image_id"]
        candidates = [images / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES]
        image_path = next((path for path in candidates if path.exists()), None)
        label_path = labels / f"{image_id}.txt"
        if image_path is None or not _validate_acne_label(label_path):
            skipped += 1
            continue
        samples.append((image_id, image_path, label_path, row.get("split", "train").lower()))

    source_test = [sample for sample in samples if sample[3] == "test"]
    source_train = [sample for sample in samples if sample[3] != "test"]
    rng = random.Random(SEED)
    rng.shuffle(source_train)
    validation_count = max(1, round(len(source_train) * VALIDATION_FRACTION))
    splits = {
        "train": source_train[validation_count:],
        "val": source_train[:validation_count],
        "test": source_test,
    }
    if not all(splits.values()):
        raise RuntimeError("ACNE04 did not provide enough valid images for every split.")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split, split_samples in splits.items():
        for image_id, image_path, label_path, _ in split_samples:
            _safe_copy(image_path, OUT_ROOT / "images" / split / image_path.name)
            _safe_copy(label_path, OUT_ROOT / "labels" / split / f"{image_id}.txt")

    classes_yaml = (
        f"path: {OUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: acne\n"
    )
    (OUT_ROOT / "classes.yaml").write_text(classes_yaml, encoding="utf-8")
    print("Bloom acne MVP dataset created:")
    for split, split_samples in splits.items():
        print(f"  {split}: {len(split_samples)} images")
    print(f"  skipped invalid/missing: {skipped}")
    print("  class: acne (original ACNE04 lesion boxes only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
