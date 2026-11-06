"""Run the existing Bloom classical pipeline on the held-out TEST set and score it.

Usage:
  python ml/eval/compare_classical.py

[Phase 3 — Task 6.2] Produces the "Arm A" (classical CV baseline) numbers for
the DIP+ML comparison table. Runs backend/src/lesion_detection.py +
postprocessing.py on every test image, converts concern masks to boxes, and
scores against ground-truth YOLO labels with IoU-based matching.

Output: ml/runs/classical_baseline/eval_results.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(ML_ROOT))

HARMONIZED = ML_ROOT / "data" / "harmonized"
OUT_DIR = ML_ROOT / "runs" / "classical_baseline"
CLASS_NAMES = ["acne", "dark_spot", "scar", "rash"]

# Classical pipeline produces concern masks keyed by {acne, redness, blemish}.
# Map to our 4-class taxonomy for box comparison.
#   redness  -> rash (both are diffuse erythematous regions)
#   blemish  -> dark_spot (post-inflammatory pigment)
CONCERN_TO_CLASS = {"acne": 0, "blemish": 1, "redness": 3}


def _mask_to_boxes(mask: np.ndarray, min_area: int = 50) -> list[list[float]]:
    """Convert a binary mask to a list of [x1, y1, x2, y2] pixel boxes via connected components."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append([x, y, x + w, y + h])
    return boxes


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(pred_boxes: list, gt_boxes: list, iou_thresh: float = 0.5) -> tuple[int, int, int]:
    """Greedy IoU matching. Returns (tp, fp, fn)."""
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> dict[int, list]:
    """Load YOLO labels -> {class_id: [[x1,y1,x2,y2], ...]} in pixel coords."""
    out: dict[int, list] = {c: [] for c in range(4)}
    if not label_path.exists():
        return out
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        if cid in out:
            out[cid].append([x1, y1, x2, y2])
    return out


def main() -> int:
    try:
        import cv2
        from pipeline import run_stages
    except ImportError as exc:
        print(f"ERROR: cannot import backend pipeline ({exc}).", file=sys.stderr)
        print("Run from repo root: python ml/eval/compare_classical.py", file=sys.stderr)
        return 1

    test_images_dir = HARMONIZED / "images" / "test"
    test_labels_dir = HARMONIZED / "labels" / "test"
    if not test_images_dir.is_dir():
        print(f"ERROR: {test_images_dir} missing — run harmonize.py first", file=sys.stderr)
        return 1

    img_paths = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_paths:
        print("No test images found.", file=sys.stderr)
        return 1

    print(f"Running classical pipeline on {len(img_paths)} test images...")

    # per-class accumulation
    tp = [0] * 4
    fp = [0] * 4
    fn = [0] * 4
    n_processed = 0
    n_failed = 0
    runtimes = []

    for img_path in img_paths:
        img_bytes = img_path.read_bytes()
        try:
            t0 = time.perf_counter()
            stages = run_stages(img_bytes)
            runtimes.append(time.perf_counter() - t0)
        except Exception:
            n_failed += 1
            continue

        if not stages.get("quality", {}).get("passed", False):
            n_failed += 1
            continue

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            n_failed += 1
            continue
        h, w = img.shape[:2]

        # collect classical boxes per our class id
        pred_by_class: dict[int, list] = {c: [] for c in range(4)}
        for concern_name, concern in stages.get("concerns", {}).items():
            cid = CONCERN_TO_CLASS.get(concern_name)
            if cid is None:
                continue
            mask = concern.get("mask")
            if mask is None:
                continue
            pred_by_class[cid].extend(_mask_to_boxes(mask))

        # ground truth
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_by_class = _load_gt_boxes(lbl_path, w, h)

        for cid in range(4):
            t, p, f = _match_boxes(pred_by_class[cid], gt_by_class[cid])
            tp[cid] += t
            fp[cid] += p
            fn[cid] += f

        n_processed += 1
        if n_processed % 20 == 0:
            print(f"  {n_processed}/{len(img_paths)}")

    # compute per-class P/R/F1 + IoU proxy
    per_class = {}
    f1s = []
    for cid, name in enumerate(CLASS_NAMES):
        precision = tp[cid] / (tp[cid] + fp[cid]) if (tp[cid] + fp[cid]) > 0 else float("nan")
        recall = tp[cid] / (tp[cid] + fn[cid]) if (tp[cid] + fn[cid]) > 0 else float("nan")
        denom = 2 * tp[cid] + fp[cid] + fn[cid]
        f1 = 2 * tp[cid] / denom if denom > 0 else float("nan")
        per_class[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": tp[cid],
            "fp": fp[cid],
            "fn": fn[cid],
        }
        if not np.isnan(f1):
            f1s.append(float(f1))

    mean_f1 = round(float(np.mean(f1s)), 4) if f1s else None
    mean_runtime_ms = round(1000.0 * float(np.mean(runtimes)), 1) if runtimes else None

    result = {
        "pipeline": "classical_cv",
        "split": "test",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_images_total": len(img_paths),
        "n_images_processed": n_processed,
        "n_images_failed_quality": n_failed,
        "per_class": per_class,
        "f1_avg": mean_f1,
        "runtime_ms_per_image": mean_runtime_ms,
        "note": (
            "Classical pipeline produces concern masks, not lesion boxes. "
            "Masks are converted to connected-component boxes for IoU matching "
            "against GT. Rash uses redness mask; dark_spot uses blemish mask. "
            "Scar has no classical analogue (expected F1=0)."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eval_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== Classical baseline complete ===")
    print(f"{'Class':<12} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 40)
    for name, vals in per_class.items():
        p = f"{vals['precision']:.4f}" if vals["precision"] == vals["precision"] else "  n/a "
        r = f"{vals['recall']:.4f}" if vals["recall"] == vals["recall"] else "  n/a "
        f = f"{vals['f1']:.4f}" if vals["f1"] == vals["f1"] else "  n/a "
        print(f"{name:<12} {p:>8} {r:>8} {f:>8}")
    print(f"\nF1 avg: {mean_f1}")
    print(f"Runtime: {mean_runtime_ms} ms/img")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
