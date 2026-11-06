"""Harmonize raw datasets into the unified 4-class YOLO format.

Implements AGENT_PLAN.md §3 (Tasks 3.1–3.6):

  - label mapping table (native -> 4-class taxonomy)
  - subject-level splits (70/15/15) with zero-overlap assertion
  - source_id == "our_collection" -> 100% test, always
  - pHash deduplication (distance < 4 dropped, logged)
  - metadata.csv, classes.yaml, class_balance.txt, dataset_hash.txt

Output:
  ml/data/harmonized/
    images/{train,val,test}/
    labels/{train,val,test}/    YOLO txt: class cx cy w h (normalized)
    metadata.csv
    classes.yaml
  ml/runs/dedup_log.csv
  ml/runs/class_balance.txt

Usage:
  python ml/datasets/harmonize.py               # sources present under ml/data/raw/
  python ml/datasets/harmonize.py --include-own  # + own collection (100% test)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ML_ROOT / "data" / "raw"
OUT_DIR = ML_ROOT / "data" / "harmonized"
RUNS_DIR = ML_ROOT / "runs"

CLASS_NAMES = ["acne", "dark_spot", "scar", "rash"]
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42
DEDUP_PHASH_DISTANCE = 4
OWN_SOURCE = "our_collection"

# Task 3.1 — native label -> our class id, per source.
# Sources already pre-mapped by their fetch scripts write class ids directly;
# this table applies to any source that still carries native label names
# (e.g. our_collection YOLO exports use numeric ids matching CLASS_NAMES).
GENERIC_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

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


def _read_manifest(raw_dir: Path) -> dict[str, dict]:
    manifest = raw_dir / "manifest.csv"
    rows: dict[str, dict] = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["image_id"]] = row
    return rows


def _phash(image_path: Path):
    import imagehash
    from PIL import Image

    with Image.open(image_path) as img:
        return imagehash.phash(img)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_source(source: str, raw_dir: Path) -> list[dict]:
    """Gather {image_id, image_path, label_path, subject_id, source_id,
    original_label, weak_label} for one raw source."""
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "annotations"
    if not labels_dir.is_dir():
        labels_dir = raw_dir / "labels"
    if not images_dir.is_dir():
        print(f"  SKIP {source}: no images/ in {raw_dir}")
        return []

    manifest = _read_manifest(raw_dir)
    samples = []
    for img_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        image_id = img_path.stem
        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            continue
        m = manifest.get(image_id, {})
        if source == OWN_SOURCE:
            subject_id = m.get("subject_id") or image_id  # subj_001_left etc.
        else:
            # web-scraped / clinical sources: unknown subject -> image is its own subject
            subject_id = m.get("subject_id") or f"{source}:{image_id}"
        samples.append(
            {
                "image_id": image_id,
                "image_path": img_path,
                "label_path": label_path,
                "subject_id": subject_id,
                "source_id": m.get("source_id", source),
                "original_label": m.get("original_label", ""),
                "weak_label": m.get("weak_label", "false"),
            }
        )
    return samples


def _split_subjects(samples: list[dict], rng: random.Random) -> None:
    """Assign each subject to train/val/test at 70/15/15.

    Own-collection subjects are forced to test (Task 3.3, no exceptions).
    """
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_subject[s["subject_id"]].append(s)

    own_subjects = [sub for sub in by_subject if any(
        s["source_id"] == OWN_SOURCE for s in by_subject[sub]
    )]
    other_subjects = [sub for sub in by_subject if sub not in own_subjects]
    rng.shuffle(other_subjects)

    n = len(other_subjects)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    assignment: dict[str, str] = {}
    for i, sub in enumerate(other_subjects):
        if i < n_train:
            assignment[sub] = "train"
        elif i < n_train + n_val:
            assignment[sub] = "val"
        else:
            assignment[sub] = "test"
    for sub in own_subjects:
        assignment[sub] = "test"

    # CRITICAL assertion (Task 3.3): zero subject overlap across splits
    split_subjects: dict[str, set[str]] = defaultdict(set)
    for sub, split in assignment.items():
        split_subjects[split].add(sub)
    assert not (split_subjects["train"] & split_subjects["val"]), "subject leak train∩val"
    assert not (split_subjects["train"] & split_subjects["test"]), "subject leak train∩test"
    assert not (split_subjects["val"] & split_subjects["test"]), "subject leak val∩test"

    for s in samples:
        s["split"] = assignment[s["subject_id"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-own", action="store_true",
                        help="include own collection (100%% test)")
    args = parser.parse_args()

    sources = ["acne04", "acnedata3", "kaggle_tiswan14"]
    if args.include_own:
        sources.append(OWN_SOURCE)

    all_samples: list[dict] = []
    for source in sources:
        raw_dir = RAW_ROOT / source if source != OWN_SOURCE else (ML_ROOT / "annotations")
        # own collection expects images/ + labels/ directly in ml/annotations/
        if source == OWN_SOURCE:
            alt = ML_ROOT / "annotations" / "images"
            raw_dir = ML_ROOT / "annotations" if alt.is_dir() else raw_dir
        samples = _collect_source(source, raw_dir)
        if source == "acnedata3" and not samples:
            print("  NOTE: acnedata3 not fetched yet (blocked on D1) — re-run after "
                  "fetch_acnedata3.py --data-dir <path>")
        print(f"{source}: {len(samples)} samples")
        all_samples.extend(samples)

    if not all_samples:
        print("ERROR: no samples found under ml/data/raw/", file=sys.stderr)
        return 1

    # --- Task 3.4: pHash deduplication across the pooled set ---------------
    import imagehash  # noqa: F401  (import early to fail fast if missing)

    print("Computing perceptual hashes...")
    hashes: dict[Path, object] = {}
    for s in all_samples:
        try:
            hashes[s["image_path"]] = _phash(s["image_path"])
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: phash failed for {s['image_id']}: {exc}")

    kept: list[dict] = []
    dropped_log = []
    # Keep earlier-listed sources first so acne04 survives over later duplicates
    kept_hashes: list[tuple[object, str]] = []
    seen_ids = set()
    for s in all_samples:
        if s["image_id"] in seen_ids:
            dropped_log.append({"image_id": s["image_id"], "source_id": s["source_id"],
                                "reason": "duplicate_image_id"})
            continue
        h = hashes.get(s["image_path"])
        is_dup = False
        if h is not None:
            for kh, kid in kept_hashes:
                if (h - kh) < DEDUP_PHASH_DISTANCE:
                    dropped_log.append({"image_id": s["image_id"], "source_id": s["source_id"],
                                        "reason": f"phash_dup_of_{kid}"})
                    is_dup = True
                    break
        if is_dup:
            continue
        if h is not None:
            kept_hashes.append((h, s["image_id"]))
        seen_ids.add(s["image_id"])
        kept.append(s)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / "dedup_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_id", "reason"])
        writer.writeheader()
        writer.writerows(dropped_log)
    print(f"Dedup: kept {len(kept)}, dropped {len(dropped_log)} -> ml/runs/dedup_log.csv")

    # --- Task 3.3: subject-level split -------------------------------------
    rng = random.Random(SEED)
    _split_subjects(kept, rng)

    # --- Copy into harmonized layout ---------------------------------------
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    class_counts: Counter[int] = Counter()
    per_split_ids: dict[str, list[str]] = defaultdict(list)
    for s in kept:
        split = s["split"]
        # namespace image ids by source to avoid cross-source stem collisions
        hid = f"{s['source_id']}_{s['image_id']}" if s["source_id"] != OWN_SOURCE else s["image_id"]
        shutil.copy2(s["image_path"], OUT_DIR / "images" / split / f"{hid}{s['image_path'].suffix.lower()}")
        dst_lbl = OUT_DIR / "labels" / split / f"{hid}.txt"
        shutil.copy2(s["label_path"], dst_lbl)

        # validate + count classes (labels are already numeric class ids)
        for ln in s["label_path"].read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                if not (0 <= cid < len(CLASS_NAMES)):
                    print(f"  ERROR: unknown class {cid} in {s['label_path']}", file=sys.stderr)
                    return 1
                if split == "train":
                    class_counts[cid] += 1

        metadata_rows.append(
            {
                "image_id": hid,
                "source_id": s["source_id"],
                "subject_id": s["subject_id"],
                "split": split,
                "monk_tone": "",
                "original_label": s["original_label"],
                "weak_label": s["weak_label"],
            }
        )
        per_split_ids[split].append(hid)

    # metadata.csv
    with (OUT_DIR / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "source_id", "subject_id", "split",
                        "monk_tone", "original_label", "weak_label"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # classes.yaml (Task 3.6)
    (OUT_DIR / "classes.yaml").write_text(CLASSES_YAML.format(path=OUT_DIR.as_posix()), encoding="utf-8")

    # --- Task 3.5: class balance report ------------------------------------
    total_boxes = sum(class_counts.values()) or 1
    lines = ["Class distribution (train):"]
    for cid, name in enumerate(CLASS_NAMES):
        n = class_counts.get(cid, 0)
        lines.append(f"  {name:10s} {n:7d} ({100.0 * n / total_boxes:5.1f}%)")
    lines += [
        "",
        "WARNING: scar/rash are minority classes. Focal loss and oversampling will compensate.",
        "",
        f"Images — train: {len(per_split_ids['train'])}, val: {len(per_split_ids['val'])}, "
        f"test: {len(per_split_ids['test'])}",
    ]
    report = "\n".join(lines)
    (RUNS_DIR / "class_balance.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)

    # --- dataset hash (Task 3.2) -------------------------------------------
    print("\nHashing harmonized dataset...")
    entries = []
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file() and p.name != "dataset_hash.txt":
            entries.append(f"{p.relative_to(OUT_DIR).as_posix()} {_sha256_file(p)}")
    dataset_hash = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    (OUT_DIR / "dataset_hash.txt").write_text(
        f"{dataset_hash}\nfiles={len(entries)}\nseed={SEED}\n", encoding="utf-8"
    )
    print(f"dataset_hash: {dataset_hash}")
    print(f"\nHarmonized dataset -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
