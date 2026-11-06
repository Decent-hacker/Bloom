"""Process a Acnedata3 dataset from Ultralytics Universe.

[DECISION D1 — Mohit]: Acnedata3 lives on Ultralytics Universe and may require
a login/API key. Default mode is MANUAL: download the dataset from
https://platform.ultralytics.com/arjun-mukherjee/datasets/acnedata3, unzip it,
then run:

    python ml/datasets/fetch_acnedata3.py --data-dir /path/to/unzipped/acnedata3

The script auto-detects the format and maps native classes to our 4-class taxonomy:

Native class mapping (case-insensitive):
  blackhead, whitehead, papule, pustule, nodule -> acne (0)
  cyst -> dark_spot (1)   [extra class from some dataset versions]
  scar / scars -> scar (2)
Unknown classes are logged and skipped — never guessed.

Output: ml/data/raw/acnedata3/
  images/      JPG/PNG files
  annotations/ YOLO txt: class cx cy w h (normalized)
  manifest.csv   image_id, source_id, original_label, split
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

SOURCE_ID = "acnedata3"

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ML_ROOT / "data" / "raw" / "acnedata3"

# Native label (lowercased, spaces/underscores stripped) -> our class id
# Dataset versions may have different class sets; we map what we can
CLASS_MAP = {
    "blackhead": 0,
    "blackheads": 0,
    "cyst": 1,
    "darkspot": 1,
    "darkspots": 1,
    "papule": 0,
    "papules": 0,
    "pustule": 0,
    "pustules": 0,
    "nodule": 0,
    "nodules": 0,
    "whitehead": 0,
    "whiteheads": 0,
}


def _norm(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "-").replace("-", "")


def _detect_format(root: Path) -> str:
    """Detect the format of the Acnedata3 dataset."""
    # Check for folder with .ndjson files + images
    ndjson_files = list(root.rglob("*.ndjson"))
    has_images = (root / "images").is_dir() or any(root.rglob("*.jpg")) or any(root.rglob("*.png"))
    has_labels = (root / "labels").is_dir() or any(root.rglob("*.txt"))
    
    if ndjson_files and has_images:
        return "ultralytics_ndjson_folder"
    if ndjson_files:
        return "ultralytics_ndjson_standalone"
    if has_labels and not ndjson_files:
        return "yolo"
    # Check for COCO format: annotations*.json
    coco_candidates = list(root.rglob("annotations*.json")) + list(root.rglob("*coco*.json"))
    if coco_candidates:
        return "coco"
    # Check for Ultralytics data.yaml
    if list(root.rglob("data.yaml")) or list(root.rglob("data.yml")):
        return "yolo"
    return "unknown"


def _process_ultralytics_ndjson_folder(root: Path, out: Path) -> int:
    """Process Ultralytics Universe .ndjson format dataset FOLDER."""
    import shutil

    images_out = out / "images"
    labels_out = out / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    manifest_rows = []
    n_images = 0

    # Process each .ndjson file in the folder
    for ndjson_file in root.rglob("*.ndjson"):
        with open(ndjson_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # Get image info
                file_name = obj.get("file", "")
                split = obj.get("split", "train")

                if not file_name:
                    continue

                # Get boxes
                ann = obj.get("annotations", {})
                boxes = ann.get("boxes", []) if isinstance(ann, dict) else []

                # Determine image path
                # The .ndjson references images via CDN URLs or local paths
                # Try to find the image file
                img_stem = Path(file_name).stem
                img_ext = Path(file_name).suffix.lower()
                img_path = None
                
                # Try exact file name in the folder
                exact = root / file_name
                if exact.is_file():
                    img_path = exact
                else:
                    # Try stem + common extensions in the folder
                    for ext in [".jpg", ".jpeg", ".png"]:
                        cand = root / f"{img_stem}{ext}"
                        if cand.is_file():
                            img_path = cand
                            break
                
                # Also check if the file_name is just a stem without extension
                if img_path is None:
                    for ext in [".jpg", ".jpeg", ".png"]:
                        cand = root / f"{img_stem}{ext}"
                        if cand.is_file():
                            img_path = cand
                            break

                if img_path is None:
                    # Try to find any jpg/png with matching stem
                    for candidate in root.rglob(f"{img_stem}*"):
                        if candidate.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            img_path = candidate
                            break

                if img_path is None:
                    print(f"  WARN: could not find image: {file_name}")
                    continue

                # Copy image to output
                img_stem_out = f"{SOURCE_ID}_{Path(file_name).stem}{img_ext}"
                shutil.copy2(img_path, images_out / img_stem_out)

                # Process boxes
                mapped_lines = []
                for box in boxes:
                    if len(box) < 5:
                        continue
                    class_id = int(box[0])
                    native_label = CLASS_MAP.get(class_id)
                    if native_label is None:
                        print(f"  SKIP unknown class id {class_id} in {file_name}")
                        continue
                    cx, cy, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                    line_str = f"{native_label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    mapped_lines.append(line_str)
                    class_counts[native_label] = class_counts.get(native_label, 0) + 1

                # Write label file
                img_id = f"{SOURCE_ID}_{Path(file_name).stem}"
                label_path = labels_out / f"{img_id}.txt"
                label_content = "\n".join(mapped_lines) + ("\n" if mapped_lines else "")
                label_path.write_text(label_content, encoding="utf-8")

                manifest_rows.append(
                    {"image_id": img_id, "source_id": SOURCE_ID, "original_label": "", "split": split}
                )
                n_images += 1

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_id", "original_label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: {n_images} images -> {out}")
    print("Class distribution (our taxonomy):")
    for cid, name in enumerate(["acne", "dark_spot", "scar"]):
        print(f"  {name}: {class_counts.get(cid, 0)} boxes")
    return 0


def _process_ultralytics_ndjson_standalone(ndjson_path: Path, out: Path) -> int:
    """Process a standalone .ndjson file (images referenced via URLs or need downloading)."""
    import shutil
    import urllib.request
    
    images_out = out / "images"
    labels_out = out / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    manifest_rows = []
    n_images = 0

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Get image info
            file_name = obj.get("file", "")
            split = obj.get("split", "train")
            url = obj.get("url", "")
            
            if not file_name:
                continue

            # Get boxes
            ann = obj.get("annotations", {})
            boxes = ann.get("boxes", []) if isinstance(ann, dict) else []

            # Determine image path/download
            # If we have a URL, we'd need to download it - for now, skip if no local file
            img_stem = Path(file_name).stem
            
            # Try to find image by stem in current directory or subdirs
            img_path = None
            for candidate in [Path(f) for f in os.listdir('.') if os.path.isfile(f)]:
                if candidate.stem == img_stem and candidate.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img_path = candidate
                    break
            
            # Also check if file_name includes a path we can use
            if img_path is None and url:
                # Could implement URL download here, but for now require local images
                print(f"  SKIP: URL-based image {file_name} - download images manually from Ultralytics Universe")
                continue
            
            if img_path is None:
                print(f"  SKIP: could not find image file for {file_name}")
                continue

            # Copy image to output
            img_stem_out = f"{SOURCE_ID}_{Path(file_name).stem}{img_path.suffix.lower()}"
            shutil.copy2(img_path, images_out / img_stem_out)

            # Process boxes
            mapped_lines = []
            for box in boxes:
                if len(box) < 5:
                    continue
                class_id = int(box[0])
                native_label = CLASS_MAP.get(class_id)
                if native_label is None:
                    print(f"  SKIP unknown class id {class_id} in {file_name}")
                    continue
                cx, cy, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                line_str = f"{native_label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                mapped_lines.append(line_str)
                class_counts[native_label] = class_counts.get(native_label, 0) + 1

            # Write label file
            img_id = f"{SOURCE_ID}_{Path(file_name).stem}"
            label_path = labels_out / f"{img_id}.txt"
            label_content = "\n".join(mapped_lines) + ("\n" if mapped_lines else "")
            label_path.write_text(label_content, encoding="utf-8")

            manifest_rows.append(
                {"image_id": img_id, "source_id": SOURCE_ID, "original_label": "", "split": split}
            )
            n_images += 1

    print(f"\nDone: {n_images} images -> {out}")
    print("Class distribution (our taxonomy):")
    for cid, name in enumerate(["acne", "dark_spot", "scar"]):
        print(f"  {name}: {class_counts.get(cid, 0)} boxes")
    return 0


def _process_yolo(root: Path, out: Path) -> int:
    """Process YOLO format dataset (existing logic)."""
    images_dirs = [d for d in root.rglob("images") if d.is_dir()]
    labels_dirs = [d for d in root.rglob("labels") if d.is_dir()]
    if not images_dirs or not labels_dirs:
        print("ERROR: could not find images/ and labels/ directories", file=sys.stderr)
        return 1

    class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    images_out = out / "images"
    labels_out = out / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_dir, lbl_dir in zip(images_dirs, labels_dirs):
        for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            dst_img = images_out / f"{SOURCE_ID}_{img_path.stem}{img_path.suffix.lower()}"
            if not dst_img.exists():
                dst_img.write_bytes(img_path.read_bytes())

            mapped_lines = []
            for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    native_id = int(float(parts[0]))
                except ValueError:
                    continue
                ours = CLASS_MAP.get(native_id)
                if ours is None:
                    print(f"  SKIP unknown native class [{native_id}] in {lbl_path}")
                    continue
                mapped_lines.append(f"{ours} " + " ".join(parts[1:5]))
                class_counts[ours] += 1

            image_id = f"{SOURCE_ID}_{img_path.stem}"
            (labels_out / f"{image_id}.txt").write_text(
                "\n".join(mapped_lines) + ("\n" if mapped_lines else ""), encoding="utf-8"
            )
            manifest_rows.append(
                {"image_id": image_id, "source_id": SOURCE_ID, "original_label": "", "split": "train"}
            )

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_id", "original_label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: YOLO format")
    print("Class distribution (our taxonomy):")
    for cid, name in enumerate(["acne", "dark_spot", "scar"]):
        print(f"  {name}: {class_counts.get(cid, 0)} boxes")
    return 0


def _process_coco(root: Path, out: Path, coco_json: Path) -> int:
    """Process COCO format dataset (existing logic)."""
    import json
    import shutil

    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    print(f"COCO categories: {categories}")

    images_out = out / "images"
    labels_out = out / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    images_meta = {im["id"]: im for im in data.get("images", [])}
    boxes_per_image: dict[int, list] = {iid: [] for iid in images_meta}

    for ann in data.get("annotations", []):
        native_name = categories.get(ann.get("category_id", -1), "")
        ours = CLASS_MAP.get(_norm(native_name))
        if ours is None:
            continue
        x, y, w, h = ann["bbox"]
        boxes_per_image[ann["image_id"]].append((ours, x, y, w, h))

    class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    manifest_rows = []

    for iid, meta in images_meta.items():
        src = root / meta["file_name"]
        if not src.exists():
            candidates = list(root.rglob(Path(meta["file_name"]).name))
            if not candidates:
                continue
            src = candidates[0]
        image_id = f"{SOURCE_ID}_{Path(meta['file_name']).stem}"
        dst = images_out / f"{image_id}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)

        img_w, img_h = meta.get("width", 1), meta.get("height", 1)
        for ours, x, y, w, h in boxes_per_image.get(iid, []):
            cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
            lines.append(f"{ours} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
            class_counts[ours] += 1
        (labels_out / f"{image_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        manifest_rows.append(
            {"image_id": image_id, "source_id": SOURCE_ID, "original_label": "", "split": "train"}
        )

    print(f"\nDone: COCO format")
    print("Class distribution (our taxonomy):")
    for cid, name in enumerate(["acne", "dark_spot", "scar"]):
        print(f"  {name}: {class_counts.get(cid, 0)} boxes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="unzipped Acnedata3 folder (manual download from Ultralytics Universe)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    root: Path = args.data_dir
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    if (args.out / "manifest.csv").exists():
        print(f"Already processed at {args.out} — delete it to redo.")
        return 0

    fmt = _detect_format(root)
    print(f"Detected format: {fmt}")

    if fmt == "ultralytics_ndjson_folder":
        return _process_ultralytics_ndjson_folder(root, args.out)
    if fmt == "ultralytics_ndjson_standalone":
        print("WARNING: Standalone .ndjson file detected.")
        print("This format references images via CDN URLs from Ultralytics Universe.")
        print("Please download the images separately or use the folder format.")
        # Try to process anyway
        return _process_ultralytics_ndjson_standalone(Path(args.data_dir), args.out)
    if fmt == "yolo":
        return _process_yolo(root, args.out)
    if fmt == "coco":
        coco_json = next(iter(root.rglob("annotations*.json")), None) or next(
            iter(root.rglob("*coco*.json"))
        )
        return _process_coco(root, args.out, coco_json)

    print(
        "ERROR: could not detect label format (expected YOLO images/+labels/, "
        "COCO annotations.json, or Ultralytics .ndjson folder).\n"
        "Check the unzipped folder layout and re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())