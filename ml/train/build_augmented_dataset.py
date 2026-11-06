"""Build an augmented copy of the harmonized train split (offline pre-augmentation).

Why offline instead of in-loop: the augmentation ablation must be a clean,
reproducible comparison — each run trains on a frozen, inspectable dataset
folder. Val/test are NEVER augmented.

Output (per mode):
  ml/data/harmonized_aug/{mode}/
    images/{train,val,test}/
    labels/{train,val,test}/
    classes.yaml

Usage:
  python ml/train/build_augmented_dataset.py --mode full
  python ml/train/build_augmented_dataset.py --mode geometric --factor 2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

import cv2  # noqa: E402

from augment import AugmentationPipeline  # noqa: E402

SRC = ML_ROOT / "data" / "harmonized"
AUG_ROOT = ML_ROOT / "data" / "harmonized_aug"

CLASSES_YAML = """path: {path}
train: images/train
val: images/val
test: images/test
nc: 4
names:
  0: acne
  1: dark_spot
  2: scar
  3: rash
"""


def _load_boxes(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    boxes = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            boxes.append([float(x) for x in parts[:5]])
    return boxes


def _save_boxes(label_path: Path, boxes: list[list[float]]) -> None:
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["geometric", "photometric", "full"])
    parser.add_argument("--factor", type=int, default=1,
                        help="augmented copies per train image (frozen diversity knob)")
    parser.add_argument("--force", action="store_true", help="rebuild even if output exists")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (SRC / "images" / "train").is_dir():
        print(f"ERROR: {SRC} not found — run harmonize.py first", file=sys.stderr)
        return 1

    out = AUG_ROOT / args.mode
    if out.exists() and not args.force:
        print(f"{out} exists — skipping (use --force to rebuild)")
        return 0

    np.random.seed(args.seed)
    pipe = AugmentationPipeline(args.mode)

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # val + test: straight copies (clean eval is sacred)
    for split in ("val", "test"):
        for img in (SRC / "images" / split).iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = SRC / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)

    # train: augmented copies
    n_written = 0
    n_boxes_dropped = 0
    for img_path in sorted((SRC / "images" / "train").iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: unreadable {img_path.name}, skipping")
            continue
        boxes0 = _load_boxes(SRC / "labels" / "train" / f"{img_path.stem}.txt")

        copies = 1 if args.mode == "none" else args.factor
        for c in range(copies):
            aug_img, aug_boxes = pipe(img.copy(), [list(b) for b in boxes0])
            suffix = "" if c == 0 else f"_{c}"
            out_name = f"{img_path.stem}{suffix}{img_path.suffix.lower()}"
            cv2.imwrite(str(out / "images" / "train" / out_name), aug_img)
            _save_boxes(out / "labels" / "train" / f"{img_path.stem}{suffix}.txt", aug_boxes)
            n_written += 1
            n_boxes_dropped += len(boxes0) - len(aug_boxes)

    (out / "classes.yaml").write_text(CLASSES_YAML.format(path=out.as_posix()), encoding="utf-8")
    print(
        f"Built {out}\n"
        f"  train: {n_written} images (factor={args.factor}, mode={args.mode}), "
        f"boxes dropped by clipping: {n_boxes_dropped}\n"
        f"  val/test: copied clean"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
