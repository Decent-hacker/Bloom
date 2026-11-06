"""Inspect a YOLO-format dataset folder (images/ + annotations/ or labels/).

Prints and optionally saves:
  - total images
  - total boxes per class
  - min/max/mean boxes per image
  - 5 sample images with box overlays written to an output dir

Usage:
  python ml/datasets/inspect_dataset.py --data-dir ml/data/raw/acne04
  python ml/datasets/inspect_dataset.py --data-dir ml/data/raw/acnedata3 \
      --out ml/runs/dataset_inspection.txt --samples-dir ml/data/inspect_samples

Class names default to the locked 4-class taxonomy (AGENT_PLAN.md §0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLASS_NAMES = ["acne", "dark_spot", "scar", "rash"]
CLASS_COLORS_BGR = [(0, 0, 255), (42, 42, 128), (128, 128, 128), (0, 140, 255)]  # red/brown/grey/orange


def _find_label_file(labels_dir: Path, image_stem: str) -> Path | None:
    for d in (labels_dir, labels_dir.parent / "annotations"):
        candidate = d / f"{image_stem}.txt"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="write report to this text file")
    parser.add_argument(
        "--samples-dir", type=Path, default=None, help="where to save overlay samples"
    )
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    root: Path = args.data_dir
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        labels_dir = root / "annotations"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"ERROR: expected {images_dir} and {labels_dir} to exist", file=sys.stderr)
        return 1

    images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        print("ERROR: no images found", file=sys.stderr)
        return 1

    class_box_counts = [0] * len(CLASS_NAMES)
    other_class_counts: dict[str, int] = {}
    boxes_per_image: list[int] = []

    sample_candidates = []
    for img_path in images:
        lbl_path = _find_label_file(labels_dir, img_path.stem)
        n_boxes = 0
        if lbl_path and lbl_path.exists():
            for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                n_boxes += 1
                if 0 <= cid < len(CLASS_NAMES):
                    class_box_counts[cid] += 1
                else:
                    other_class_counts[str(cid)] = other_class_counts.get(str(cid), 0) + 1
        boxes_per_image.append(n_boxes)
        if n_boxes > 0:
            sample_candidates.append(img_path)

    total_boxes = sum(boxes_per_image)
    lines = [
        f"Dataset: {root}",
        f"Total images: {len(images)}",
        f"Total boxes: {total_boxes}",
        "",
        "Boxes per class:",
    ]
    for cid, name in enumerate(CLASS_NAMES):
        pct = (100.0 * class_box_counts[cid] / total_boxes) if total_boxes else 0.0
        lines.append(f"  {name:10s} {class_box_counts[cid]:7d} ({pct:5.1f}%)")
    if other_class_counts:
        lines.append(f"  UNKNOWN native class ids (not in our taxonomy): {other_class_counts}")
    nonzero = [n for n in boxes_per_image if n > 0] or [0]
    lines += [
        "",
        f"Boxes per image (images with ≥1 box): min={min(nonzero)} max={max(nonzero)} "
        f"mean={sum(nonzero) / len(nonzero):.2f}",
        f"Images with zero boxes: {boxes_per_image.count(0)}",
    ]
    report = "\n".join(lines)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport written to {args.out}")

    # Overlay samples
    if sample_candidates:
        try:
            import cv2
        except ImportError:
            print("NOTE: opencv-python not installed — skipping overlay samples")
            return 0
        samples_dir = args.samples_dir or (root / "inspect_samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        step = max(len(sample_candidates) // args.num_samples, 1)
        for img_path in sample_candidates[:: step][: args.num_samples]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl_path = _find_label_file(labels_dir, img_path.stem)
            if lbl_path:
                for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    color = CLASS_COLORS_BGR[cid] if 0 <= cid < len(CLASS_COLORS_BGR) else (255, 255, 255)
                    label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"id{cid}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            out_path = samples_dir / f"sample_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), img)
        print(f"Overlay samples written to {samples_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
