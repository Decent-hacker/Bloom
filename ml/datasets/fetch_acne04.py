"""Fetch ACNE04 (ICCV 2019, Layered-Labs repack) from HuggingFace.

Saves raw images + YOLO-format bbox annotations to ml/data/raw/acne04/:
  images/          JPG files named <image_id>.jpg
  annotations/     YOLO txt: class cx cy w h (normalized); all boxes -> class 0 (acne)
  manifest.csv     image_id, source_id, original_label, split

Idempotent: skips download if ml/data/raw/acne04/manifest.csv already exists.

ACNE04 provides lesion bounding boxes for acne only. The Hayashi severity
grade (0-3) is preserved in manifest.csv as original_label for use in
severity-agreement evaluation (quadratic-weighted kappa) — it is NOT a
detection class.

License: academic/research use only (per original authors, xpwu95/LDL).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ID = "Layered-Labs/ACNE04"
SOURCE_ID = "acne04"
ACNE_CLASS_ID = 0  # all ACNE04 boxes are acne

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ML_ROOT / "data" / "raw" / "acne04"


def _write_yolo_label(path: Path, boxes: dict, img_w: int, img_h: int) -> int:
    """Write one YOLO txt file from parallel xmin/ymin/xmax/ymax lists.

    ACNE04 stores boxes as a struct of parallel pixel-coordinate lists.
    Returns the number of boxes written.
    """
    xmins = boxes["xmin"]
    ymins = boxes["ymin"]
    xmaxs = boxes["xmax"]
    ymaxs = boxes["ymax"]
    lines = []
    for x1, y1, x2, y2 in zip(xmins, ymins, xmaxs, ymaxs):
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        lines.append(
            f"{ACNE_CLASS_ID} {cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    out: Path = args.out
    manifest_path = out / "manifest.csv"
    if manifest_path.exists() and not args.force:
        n = sum(1 for _ in manifest_path.open(encoding="utf-8")) - 1
        print(f"Already fetched ({n} images) at {out} — skipping. Use --force to redo.")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        return 1

    from PIL import Image

    print(f"Loading {REPO_ID} from HuggingFace (first run downloads ~1.1 GB)...")
    ds = load_dataset(REPO_ID)

    # The packaged dataset has a single "train" split; the per-row `split`
    # column carries the original train/test membership (canonical fold 0).
    rows = ds["train"]

    images_dir = out / "images"
    labels_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    manifest_rows = []
    for i, row in enumerate(rows):
        image_id = Path(row["file_name"]).stem
        img: Image.Image = row["image"]
        img_w, img_h = img.size

        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=95)

        n_boxes = _write_yolo_label(labels_dir / f"{image_id}.txt", row["lesion_boxes"], img_w, img_h)
        total_boxes += n_boxes

        manifest_rows.append(
            {
                "image_id": image_id,
                "source_id": SOURCE_ID,
                "original_label": row["severity"],
                "split": row["split"],
            }
        )
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(rows)} images written...")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_id", "original_label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: {len(manifest_rows)} images, {total_boxes} lesion boxes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
