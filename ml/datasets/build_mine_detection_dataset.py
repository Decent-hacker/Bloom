"""Build a leakage-aware YOLO dataset from the local ``mine`` annotations.

The source dataset uses one directory per condition, but its YOLO label files
all contain class id 0.  This script maps each directory to its documented
class id, preserves its boxes, groups byte-identical images before splitting,
and writes a deterministic 70/15/15 train/validation/test dataset.

Usage:
    .\\ml\\.venv\\Scripts\\python.exe ml\\datasets\\build_mine_detection_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mine"
OUT = ROOT / "ml" / "data" / "mine_detection"
CLASS_DISPLAY_NAMES = {
    "acne": "acne",
    "darkspots": "dark_spots",
    "pigmentation": "pigmentation",
    "redness": "redness",
}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SEED = 42


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(image: Path) -> Path:
    return SOURCE / "obj_train_data" / image.parent.name / f"{image.stem}.txt"


def remap_label(source: Path, class_id: int) -> str:
    lines = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{source}:{number}: expected 5 YOLO fields")
        values = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{source}:{number}: coordinates must be normalised")
        lines.append(" ".join([str(class_id), *fields[1:]]))
    return "\n".join(lines) + ("\n" if lines else "")


def select_splits(images: list[Path]) -> dict[Path, str]:
    """Split one class while ensuring byte-identical images stay together."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        grouped[image_hash(image)].append(image)
    groups = list(grouped.values())
    random.Random(SEED).shuffle(groups)

    targets = {name: round(len(images) * ratio) for name, ratio in SPLITS}
    targets["train"] += len(images) - sum(targets.values())
    counts = Counter()
    assigned: dict[Path, str] = {}
    for group in groups:
        split = min(
            (name for name, _ in SPLITS),
            key=lambda name: (counts[name] >= targets[name], counts[name] / max(targets[name], 1)),
        )
        for image in group:
            assigned[image] = split
            counts[split] += 1
    return assigned


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, _ in SPLITS:
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    # Empty files are not negative examples here: the source has a dedicated
    # condition folder for each image, and the annotation audit showed that
    # several folders are largely unannotated.  Using those as negatives would
    # teach the detector to suppress the very conditions it should learn.
    annotated: dict[str, list[tuple[Path, str]]] = {}
    excluded_empty = Counter()
    for folder in CLASS_DISPLAY_NAMES:
        items = []
        for image in sorted(path for path in (SOURCE / folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            source_label = label_for(image)
            if not source_label.is_file():
                raise FileNotFoundError(f"{image}: label file is missing")
            text = remap_label(source_label, 0)
            if text:
                items.append((image, text))
            else:
                excluded_empty[folder] += 1
        if items:
            annotated[folder] = items

    active_classes = {folder: index for index, folder in enumerate(annotated)}
    if not active_classes:
        raise ValueError("No non-empty annotation files were found")

    manifest: list[dict[str, str | int]] = []
    per_split = Counter()
    per_class = Counter()
    for folder, class_id in active_classes.items():
        images = [image for image, _ in annotated[folder]]
        label_texts = dict(annotated[folder])
        assignments = select_splits(images)
        for image in images:
            split = assignments[image]
            safe_stem = f"{folder}--{image.stem}"
            destination = OUT / "images" / split / f"{safe_stem}{image.suffix.lower()}"
            shutil.copy2(image, destination)
            (OUT / "labels" / split / f"{safe_stem}.txt").write_text(
                remap_label(label_for(image), class_id), encoding="utf-8"
            )
            manifest.append({"source": str(image.relative_to(ROOT)), "split": split, "class": folder, "class_id": class_id})
            per_split[split] += 1
            per_class[folder] += 1

    classes_yaml = "\n".join(
        [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
        + [f"  {index}: {CLASS_DISPLAY_NAMES[name]}" for name, index in active_classes.items()]
    ) + "\n"
    (OUT / "classes.yaml").write_text(classes_yaml, encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "total_annotated_images": len(manifest),
        "by_split": dict(per_split),
        "by_class": dict(per_class),
        "excluded_empty_annotations": dict(excluded_empty),
        "classes_trained": [CLASS_DISPLAY_NAMES[name] for name in active_classes],
        "seed": SEED,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {OUT / 'classes.yaml'}")


if __name__ == "__main__":
    main()
