"""Fetch Kaggle tiswan14 acne-dataset-image into the project's raw format.

Dataset: https://www.kaggle.com/datasets/tiswan14/acne-dataset-image
Structure: train/valid/test directories with class subfolders:
  Blackheads, Cyst, Papules, Pustules, Whiteheads

Maps 5 native classes -> project's 4-class taxonomy:
  Blackhead -> acne (0)
  Whitehead -> acne (0)
  Papule    -> acne (0)
  Pustule   -> acne (0)
  Cyst      -> acne (0)

Saves:
  ml/data/raw/kaggle_tiswan14/
    images/      JPG files (original filenames preserved)
    annotations/ YOLO txt: class cx cy w h (normalized)
    manifest.csv  image_id, source_id, original_label, split

Idempotent: skips if manifest.csv already exists.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SOURCE_ID = "kaggle_tiswan14"
ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ML_ROOT / "data" / "raw" / "kaggle_tiswan14"

# Task 3.1 — native label -> our class id
CLASS_MAP = {
    "blackhead": 0,
    "blackheads": 0,
    "whitehead": 0,
    "whiteheads": 0,
    "papule": 0,
    "papules": 0,
    "pustule": 0,
    "pustules": 0,
    "cyst": 0,
    "cysts": 0,
}


def _norm(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "-").replace("-", "")


def _map_class(native_label: str) -> int:
    """Map native label string to our class id, case-insensitive."""
    key = native_label.lower().replace(" ", "").replace("_", "-").replace("-", "")
    return CLASS_MAP.get(key)


def main() -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("AcneDataset"),
        help="Path to the downloaded Kaggle dataset root (relative to project root "
             "or absolute). Default: ./AcneDataset",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory for raw data",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_absolute():
        data_dir = ML_ROOT / data_dir

    out: Path = args.out
    manifest_path = out / "manifest.csv"

    if manifest_path.exists():
        print(f"Already fetched at {out} — skipping. Use --force to redo.")  # noqa: T201
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Expects: data_dir/train/{class}, data_dir/valid/{class}, data_dir/test/{class}
    # Classes: Blackheads, Cyst, Papules, Pustules, Whiteheads
    splits = ("train", "valid", "test")
    class_names = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]

    manifest_rows = []
    total_images = 0

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            print(f"  WARN: {split} directory not found in {data_dir}, skipping")
            continue

        for cls_name in class_names:
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                print(f"  WARN: {cls_name} directory not found in {split}, skipping")
                continue

            # Process each image in the class directory
            for img_path in sorted(cls_dir.rglob("*")):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                image_id = img_path.stem  # original stem without extension
                # Read image dimensions for box normalization
                from PIL import Image
                try:
                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                except Exception:
                    img_w, img_h = 640, 640  # fallback

                # Since this is an image-level classification dataset without
                # per-pixel bounding boxes, we create a YOLO label with a
                # central placeholder box so the format is compatible with
                # the training pipeline. The class is always 0 (acne).
                # Box: center at (0.5, 0.5), size 0.2 x 0.2 of image
                cx, cy = 0.5, 0.5
                bw, bh = 0.2, 0.2

                label_path = out / "annotations" / f"{image_id}.txt"
                label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                label_path.write_text(label_line, encoding="utf-8")

                # Copy image with ORIGINAL filename (no SOURCE_ID prefix),
                # so harmonize.py can match image_id from manifest to file stem
                dst_img = out / "images" / f"{image_id}{img_path.suffix.lower()}"
                if not dst_img.exists():
                    dst_img.write_bytes(img_path.read_bytes())

                manifest_rows.append(
                    {
                        "image_id": image_id,  # original stem, matches file stem
                        "source_id": SOURCE_ID,
                        "original_label": cls_name,
                        "split": split,
                    }
                )
                total_images += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_id", "original_label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: {total_images} images -> {out}")
    print("Class mapping used:")
    for native, our in CLASS_MAP.items():
        print(f"  {native:12s} -> our class {our} (acne)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())