# Bloom 2.0 — Agent Execution Plan

> **WHO READS THIS:** An AI coding agent executing tasks autonomously on behalf of a solo developer (Mohit).  
> **HOW TO USE:** Read the entire file before starting any task. Each section is a self-contained work order. Complete tasks in the order listed. When a task is done, mark it `[x]`. Never skip a task unless its SKIP condition is met. Ask Mohit only when a `[DECISION REQUIRED]` block appears.

---

## 0. Project Context

**What is Bloom?**  
A web app that analyses facial skin photos for cosmetic imperfections (acne, dark spots, scars, rash) and returns a skincare routine. It currently uses a classical computer vision pipeline. This upgrade adds a YOLOv8-based ML detector alongside the classical pipeline, with augmentation-driven training and honest accuracy metrics.

**Existing repo structure (assumed):**
```
bloom/
  backend/
    src/
      lesion_detection.py       ← classical pipeline
      skin_segmentation.py      ← MediaPipe skin mask
      histogram_processing.py   ← CLAHE, gamma
      restoration.py            ← noise/blur models
      postprocessing.py         ← severity bands (area-fraction logic)
    routine.py                  ← rule-based skincare routine
    main.py                     ← FastAPI app
  frontend/
    src/
      components/
        ResultsPanel.*          ← shows scan results
        RoutineCard.*           ← shows routine output
```

**Tech stack:**  
- Backend: Python 3.11, FastAPI, OpenCV, MediaPipe, numpy  
- Frontend: React (Vite)  
- Training: PyTorch, Ultralytics YOLOv8, Albumentations  
- Deployment: Render (backend), static host (frontend), HuggingFace Hub (model artifact)

**The 4 detection classes (locked — do not change):**
```
0: acne        → bounding boxes (papules, pustules, comedones, nodules)
1: dark_spot   → bounding boxes (hyperpigmented macules, PIH)
2: scar        → bounding boxes (atrophic/boxcar/rolling texture depressions)
3: rash        → segmentation mask (diffuse erythematous patches)
```

**Severity scale (per class):** `Clear / Mild / Moderate / Severe`  
Derived from detected area as fraction of total skin mask area. Reuse `postprocessing.py` thresholds — do not train a separate severity classifier.

---

## 1. Repository Scaffold

**Goal:** Create the `ml/` directory with all required files (empty stubs are fine — content filled in later tasks).

### Task 1.1 — Create `ml/` folder structure
```
ml/
  datasets/
    fetch_acne04.py
    fetch_acnedata3.py
    harmonize.py
    inspect_dataset.py
  augment/
    __init__.py
    photometric.py
    geometric.py
    degradation.py
  train/
    train_yolo.py
    config.yaml
  eval/
    evaluate.py
    compare_classical.py
    report.py
  annotations/
    protocol.md
    README.md
  runs/
    .gitkeep
  requirements.txt
  README.md
```

### Task 1.2 — Create `ml/requirements.txt`
```
ultralytics==8.2.18
torch==2.3.0
torchvision==0.18.0
albumentations==1.4.6
datasets==2.19.1
torchmetrics==1.4.0
scikit-learn==1.5.0
pandas==2.2.2
Pillow==10.3.0
imagehash==4.3.1
pyarrow==16.1.0
tqdm==4.66.4
pyyaml==6.0.1
```

> Pin every version. These are the last known stable combos as of mid-2025. Do not use `latest`.

### Task 1.3 — Create `ml/README.md`
Content: brief description of the ml/ module, how to run each script, and a note that `ml/data/` and `ml/runs/` are gitignored.

### Task 1.4 — Add to `.gitignore` (root)
```
ml/data/
ml/runs/
ml/annotations/images/
*.pt
*.onnx
```

**Done when:** `ml/` folder exists with all files. No errors importing from `ml/`.

---

## 2. Dataset Fetch Scripts

**Goal:** Scripts that download and cache the two primary training datasets.

### Task 2.1 — `ml/datasets/fetch_acne04.py`

Download ACNE04 from HuggingFace: `Layered-Labs/ACNE04`

Script requirements:
- Use `datasets` library: `load_dataset("Layered-Labs/ACNE04")`
- Save raw images + annotations to `ml/data/raw/acne04/`
- Print total image count and annotation count on completion
- Skip download if `ml/data/raw/acne04/` already exists (idempotent)
- Record `source_id = "acne04"` in a sidecar manifest CSV

Expected output structure:
```
ml/data/raw/acne04/
  images/          ← JPG/PNG files
  annotations/     ← YOLO bbox txt files or JSON
  manifest.csv     ← columns: image_id, source_id, original_label, split
```

### Task 2.2 — `ml/datasets/fetch_acnedata3.py`

Download Acnedata3 from Ultralytics Universe.

> [DECISION REQUIRED — Mohit] Acnedata3 is on Ultralytics Universe (platform.ultralytics.com). Check if it has a direct download URL or requires API key. If it requires login, agent should write a script that accepts a `--data-dir` argument pointing to a manually downloaded zip, then processes from there. Default to manual-download mode.

Script requirements (manual-download mode):
- Accept `--data-dir <path>` pointing to unzipped dataset folder
- Detect label format (YOLO or COCO) automatically
- Save to `ml/data/raw/acnedata3/` in same structure as Task 2.1
- Print class distribution on completion

### Task 2.3 — `ml/datasets/inspect_dataset.py`

A utility script that, given a dataset folder, prints:
- Total images
- Total boxes per class
- Min/max/mean boxes per image
- Sample 5 images with box overlays saved to `ml/data/inspect_samples/`

Run this on both datasets after fetching. Output goes in `ml/runs/dataset_inspection.txt`.

**Done when:** Both datasets downloaded, manifest CSVs exist, inspect script runs without error.

---

## 3. Harmonization Pipeline

**Goal:** Map both datasets into one unified 4-class YOLO format.

### Task 3.1 — Label mapping table (implement in `harmonize.py`)

| Source | Original label | → Our class | Notes |
|---|---|---|---|
| acne04 | all lesion boxes | `acne` (0) | severity column → keep in metadata, don't use as class |
| acnedata3 | blackhead | `acne` (0) | |
| acnedata3 | whitehead | `acne` (0) | |
| acnedata3 | papule | `acne` (0) | |
| acnedata3 | pustule | `acne` (0) | |
| acnedata3 | nodule | `acne` (0) | |
| acnedata3 | dark_spot / dark spot | `dark_spot` (1) | |
| acnedata3 | scar (if present) | `scar` (2) | only if label exists |
| our_collection | acne | `acne` (0) | identity |
| our_collection | dark_spot | `dark_spot` (1) | identity |
| our_collection | scar | `scar` (2) | identity |
| our_collection | rash | `rash` (3) | identity |

### Task 3.2 — Output schema

```
ml/data/harmonized/
  images/
    train/
    val/
    test/
  labels/
    train/        ← YOLO txt: one file per image, lines: class cx cy w h (normalized)
    val/
    test/
  metadata.csv    ← columns: image_id, source_id, subject_id, split, monk_tone (nullable), original_label, weak_label (bool)
  classes.yaml    ← 0: acne, 1: dark_spot, 2: scar, 3: rash
  dataset_hash.txt ← SHA256 of the entire harmonized/ folder (for reproducibility)
```

### Task 3.3 — Split logic (CRITICAL — do not skip)

- Split **by subject_id**, never by image
- If subject_id is unknown (web-scraped sources), treat each image as its own subject
- Ratio: 70% train / 15% val / 15% test
- Our own collected set (`source_id = "our_collection"`) → **100% test, always, no exceptions**
- Assert in code: `assert len(set(train_subjects) & set(test_subjects)) == 0` — if this fails, stop and fix

### Task 3.4 — Deduplication

- Compute pHash for every image using `imagehash` library
- Drop duplicates where hash distance < 4
- Log dropped images to `ml/runs/dedup_log.csv`

### Task 3.5 — Class balance report

After harmonization, print and save to `ml/runs/class_balance.txt`:
```
Class distribution (train):
  acne:       XXXX boxes (XX.X%)
  dark_spot:  XXXX boxes (XX.X%)
  scar:       XXXX boxes (XX.X%)
  rash:       XXXX boxes (XX.X%)

WARNING: scar/rash are minority classes. Focal loss and oversampling will compensate.
```

### Task 3.6 — `classes.yaml`
```yaml
path: ml/data/harmonized
train: images/train
val: images/val
test: images/test
nc: 4
names:
  0: acne
  1: dark_spot
  2: scar
  3: rash
```

**Done when:** `harmonize.py` runs end-to-end, split assertion passes, dataset_hash.txt exists.

---

## 4. Augmentation Library

**Goal:** A reusable augmentation library that wraps existing Bloom DIP code + Albumentations. This IS the DIP contribution — frame it as such in comments.

### Task 4.1 — `ml/augment/photometric.py`

Reuse logic from `backend/src/histogram_processing.py` where possible.

Implement:
```python
def gamma_jitter(image, gamma_range=(0.7, 1.5)) -> np.ndarray
def clahe_jitter(image, clip_limit_range=(1.0, 4.0)) -> np.ndarray
def white_balance_shift(image, gain_range=(-0.1, 0.1)) -> np.ndarray
def brightness_shift(image, shift_range=(-0.2, 0.2)) -> np.ndarray
```

Each function: takes numpy BGR image, returns augmented numpy BGR image.  
Add docstring: "DIP-derived photometric augmentation — simulates real-world lighting variation and phone camera AWB errors on Indian skin tones."

### Task 4.2 — `ml/augment/geometric.py`

```python
def random_rotate(image, boxes, angle_range=(-15, 15))
def random_flip_horizontal(image, boxes)
def random_scale_crop(image, boxes, scale_range=(0.9, 1.1))
def random_translate(image, boxes, translate_fraction=0.05)
```

Each function: takes image + YOLO-format boxes list, returns augmented image + updated boxes.

### Task 4.3 — `ml/augment/degradation.py`

Reuse logic from `backend/src/restoration.py` noise models.

```python
def gaussian_noise(image, sigma_range=(5, 25)) -> np.ndarray
def salt_and_pepper_noise(image, amount_range=(0.001, 0.01)) -> np.ndarray
def motion_blur(image, kernel_size_range=(3, 7)) -> np.ndarray
def jpeg_compression_artifact(image, quality_range=(40, 80)) -> np.ndarray
```

Add docstring: "DIP-derived degradation augmentation — reuses Bloom's restoration pipeline noise models to simulate real-world upload quality degradation."

### Task 4.4 — `ml/augment/__init__.py`

Export a single `AugmentationPipeline` class:
```python
class AugmentationPipeline:
    def __init__(self, mode: str):
        # mode: "none" | "geometric" | "photometric" | "full"
        ...
    
    def __call__(self, image, boxes) -> (image, boxes):
        ...
```

`mode` maps to ablation runs:
- `"none"` → identity (run a)
- `"geometric"` → geometric only (run b)
- `"photometric"` → geometric + photometric (run c)
- `"full"` → geometric + photometric + degradation (run d)

**Done when:** All 4 augmentation functions implemented and tested with a sample image. `AugmentationPipeline("full")` runs without error.

---

## 5. Training Script

**Goal:** A reproducible YOLOv8 fine-tuning script with 4 ablation modes.

### Task 5.1 — `ml/train/config.yaml`
```yaml
# Bloom 2.0 Training Config
# Do not modify seed or split_hash after baseline run is locked

model: yolov8s.pt          # start with small; switch to yolov8n for CPU deployment
data: ml/data/harmonized/classes.yaml
epochs: 100
patience: 25               # early stopping
batch: 16                  # reduce to 8 if OOM on T4
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01                  # final lr = lr0 * lrf (cosine decay)
seed: 42
workers: 4

# Loss
fl_gamma: 1.5              # focal loss — targets scar/rash minority classes
iou_type: ciou

# Class weights (inverse-sqrt frequency — update after harmonization)
# agent: compute these from class_balance.txt and fill in
cls_weights: null           # [w_acne, w_dark_spot, w_scar, w_rash]

# YOLO built-in augmentation
mosaic: 0.0                # OFF — mosaic hurts small lesion detection; ablate if time
mixup: 0.0

# Augmentation mode (controlled externally by train_yolo.py)
augmentation_mode: full     # none | geometric | photometric | full
```

### Task 5.2 — `ml/train/train_yolo.py`

```python
"""
Bloom 2.0 YOLOv8 training script.

Usage:
  python ml/train/train_yolo.py --aug-mode none     # run (a) — baseline
  python ml/train/train_yolo.py --aug-mode geometric  # run (b)
  python ml/train/train_yolo.py --aug-mode photometric # run (c)
  python ml/train/train_yolo.py --aug-mode full     # run (d) — full
"""
```

Script requirements:
- Accept `--aug-mode` argument
- Load config from `ml/train/config.yaml`
- Apply `AugmentationPipeline(mode)` from `ml/augment/`
- Train YOLOv8s using Ultralytics API
- Save checkpoint to `ml/runs/{run_id}/best.pt` where `run_id = f"{aug_mode}_{seed}_{timestamp}"`
- On completion, append one row to `ml/runs/run_log.csv`:
  ```
  run_id, aug_mode, model, epochs_run, mAP50, mAP50-95, F1_acne, F1_dark_spot, F1_scar, F1_rash, dataset_hash, seed, timestamp
  ```
- Print final metrics table to stdout

### Task 5.3 — Colab notebook `ml/train/colab_train.ipynb`

A notebook that:
1. `pip install -r ml/requirements.txt`
2. Mounts Google Drive for checkpointing
3. Calls `train_yolo.py --aug-mode full`
4. Downloads `best.pt` to Drive on completion
5. Has a resume cell: if interrupted, continues from last checkpoint

> This notebook is for Mohit to run — make it dead simple with one-cell-per-step.

**Done when:** `train_yolo.py --aug-mode none` runs for 2 epochs without error (smoke test). `run_log.csv` gets one row.

---

## 6. Evaluation Scripts

**Goal:** Run all 4 ablation experiments, compare with classical baseline, produce report tables.

### Task 6.1 — `ml/eval/evaluate.py`

```python
"""
Usage:
  python ml/eval/evaluate.py --run-id <run_id>
  python ml/eval/evaluate.py --all   # evaluates every run in run_log.csv
"""
```

Metrics to compute on the **test set only** (never val):
- Per-class: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
- Overall: weighted mAP@0.5
- Severity agreement: Cohen's κ (quadratic weighted) for acne severity vs ACNE04 ground truth grades
- Bootstrap 95% CIs on mAP@0.5 (1000 resamples)

Output: `ml/runs/{run_id}/eval_results.json` + printed table.

### Task 6.2 — `ml/eval/compare_classical.py`

Run the **existing Bloom classical pipeline** on the same held-out test images:
- Import `backend/src/lesion_detection.py` and `backend/src/postprocessing.py`
- For each test image: run classical pipeline → get detections + severity
- Compare against ground truth boxes using IoU-based matching
- Compute same metrics as Task 6.1
- Save to `ml/runs/classical_baseline/eval_results.json`

> This produces the "Arm A" results for the DIP+ML comparison table.

### Task 6.3 — `ml/eval/report.py`

Generate the final comparison table + figures:

**Table 1 — Augmentation Ablation:**
| Run | Aug Mode | mAP@0.5 | F1 (acne) | F1 (dark_spot) | F1 (scar) | F1 (rash) |
|---|---|---|---|---|---|---|
| a | none | ... | ... | ... | ... | ... |
| b | geometric | ... | ... | ... | ... | ... |
| c | photometric | ... | ... | ... | ... | ... |
| d | full | ... | ... | ... | ... | ... |

**Table 2 — Pipeline Comparison (on same test set):**
| Pipeline | mAP@0.5 | F1 avg | Runtime (ms/img) |
|---|---|---|---|
| Classical CV (Bloom v1) | ... | ... | ... |
| ML only (YOLOv8s, run d) | ... | ... | ... |
| Hybrid (ML + DIP skin mask) | ... | ... | ... |

Save tables as `ml/runs/final_report/table1_augmentation.csv` and `table2_comparison.csv`.  
Save confusion matrix figure as `ml/runs/final_report/confusion_matrix.png`.  
Save a "failure gallery" — 9 images where the model failed, annotated with predicted vs GT boxes — as `ml/runs/final_report/failure_gallery.png`.

**Done when:** `report.py` runs and produces both tables + failure gallery. All numbers traceable to `run_log.csv`.

---

## 7. Own Dataset — Annotation Protocol

**Goal:** Document the capture and annotation protocol so Mohit can execute it himself.

### Task 7.1 — `ml/annotations/protocol.md`

Write a clear, step-by-step capture and annotation guide for Mohit. Include:

**Capture protocol (Mohit executes):**
1. Device: any smartphone ≥12 MP, rear camera
2. Lighting: indirect daylight or consistent indoor white light; no flash; no mixed light sources
3. Subject prep: no makeup, no sunscreen, hair pulled back, no glasses; face washed 30 min before
4. Angle: **front only** (camera at eye level, subject facing forward)
5. Distance: 25–35 cm; fill frame with face + upper neck
6. Background: plain wall, non-reflective
7. Filename: `subj_001.jpg`, `subj_002.jpg`, etc. (never real names)
8. Target: **20–25 subjects minimum** (RGIPT hostel/campus)

**Consent (one per subject):**
- Verbal explanation: "This is for an academic project. Photos stored locally only, never uploaded publicly, deleted on request."
- Written consent: simple sign-off in a notebook with date. Not a formal form — this is campus academic research, not clinical.

**Annotation workflow (Mohit executes in Roboflow):**
1. Create Roboflow project: Object Detection, 4 classes matching our taxonomy
2. Upload subjects one by one
3. Draw bounding boxes only inside the face/neck skin region
4. For rash: draw the tightest box that contains the patch (Roboflow doesn't do masks easily — box is fine for this project)
5. Export: YOLOv8 format, normalized coordinates
6. Download zip → save to `ml/annotations/images/` (gitignored)

**Severity spot-check (after annotation):**
- Run `postprocessing.py` area-fraction severity on each annotated image
- Manually verify 5 random images per class — does the severity label match what you see?
- If >30% mismatch: recalibrate thresholds with Mohit

### Task 7.2 — `ml/annotations/README.md`

One-paragraph note: this folder holds Mohit's own collected Indian skin dataset. Images are gitignored for privacy. To add to harmonized dataset, run `python ml/datasets/harmonize.py --include-own`.

**Done when:** Protocol doc written. Mohit can follow it without asking any questions.

---

## 8. Backend Integration

**Goal:** Add ML inference to the existing FastAPI backend without breaking the classical pipeline.

### Task 8.1 — `backend/ml_inference.py`

```python
"""
Bloom 2.0 ML inference module.
Loads YOLOv8 model once at startup, exposes detect().
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

MODEL_URL = os.getenv("MODEL_URL", "")       # HuggingFace Hub URL
MODEL_SHA256 = os.getenv("MODEL_SHA256", "") # expected checksum
MODEL_CACHE = Path("ml/runs/best.pt")

CLASS_NAMES = ["acne", "dark_spot", "scar", "rash"]

def load_model():
    """
    Download model from MODEL_URL if not cached.
    Verify SHA256. Load into memory.
    Returns YOLO model or None if load fails (triggers classical fallback).
    """
    ...

def detect(image: np.ndarray) -> List[Dict]:
    """
    Run YOLOv8 detection on a BGR numpy image.
    Returns list of detections:
    [
      {
        "class": "acne",
        "class_id": 0,
        "confidence": 0.82,
        "box": [x1, y1, x2, y2],   # pixel coords
        "severity": "Mild"          # from area-fraction postprocessing
      },
      ...
    ]
    Returns [] if model not loaded (graceful degradation).
    """
    ...
```

Requirements:
- Model loads once at startup, not per-request
- If `MODEL_URL` is empty or download fails: log warning, return empty list (do NOT raise)
- Use `torch.set_num_threads(2)` for CPU deployment
- Input size: resize to 512×512 before inference (CPU latency budget)
- After YOLO detection: run skin mask from `skin_segmentation.py` + `postprocessing.py` to derive severity per detection

### Task 8.2 — Extend `/api/analyse` in `backend/main.py`

Add `pipeline` query parameter:
```python
@app.post("/api/analyse")
async def analyse(
    file: UploadFile,
    pipeline: str = Query("hybrid", enum=["classical", "ml", "hybrid"])
):
```

Response: extend existing response schema with an `ml` block:
```json
{
  "concerns": { ... },           // existing classical output — always present
  "ml": {                        // new block — present if pipeline != "classical"
    "detections": [ ... ],       // list of detect() output
    "pipeline_used": "ml",
    "model_loaded": true
  },
  "metrics": null                // keep existing null convention
}
```

If `model_loaded` is false: return `ml.pipeline_used = "classical_fallback"`. Never return a 500.

### Task 8.3 — Environment variables

Add to `backend/.env.example`:
```
MODEL_URL=https://huggingface.co/your-repo/bloom-ml/resolve/main/best.pt
MODEL_SHA256=
PIPELINE_DEFAULT=hybrid
```

**Done when:** `POST /api/analyse?pipeline=hybrid` returns valid JSON with both `concerns` and `ml` blocks. Classical pipeline still works when model is not loaded.

---

## 9. Frontend Integration

**Goal:** Add a pipeline toggle and detection box overlay to ResultsPanel. Keep it minimal.

### Task 9.1 — Pipeline toggle in `ResultsPanel`

Add a toggle above the results:
```
[ Classical | ML | Hybrid ]   ← three-way toggle
```

- Default: `Hybrid`
- On toggle: re-call `/api/analyse?pipeline={selected}` with the same uploaded image
- Show a loading spinner during re-analysis

### Task 9.2 — Detection box overlay

When pipeline is `ml` or `hybrid` and `ml.detections` is non-empty:
- Overlay bounding boxes on the uploaded image preview
- Color per class: acne=red, dark_spot=brown, scar=grey, rash=orange
- Show class label + confidence % above each box
- Toggle: "Show boxes" checkbox (default: on)

### Task 9.3 — `RoutineCard` — no change needed for now

The routine card is fed by `concerns` from the classical pipeline. After routine engine upgrade (§10), it will consume ML concerns. For now: no change.

**Done when:** Toggle works, boxes render on the image, re-analysis on toggle works.

---

## 10. Routine Engine Upgrade (`backend/routine.py`)

**Goal:** Align routine engine with ML taxonomy. Add scar rules. Add conflict matrix. Keep it minimal — no tracker, no PDF, no step-up warnings.

### Task 10.1 — Taxonomy mapping layer

At the top of `routine.py`, add a mapping from ML classes → routine concern keys:
```python
ML_TO_ROUTINE = {
    "acne": "acne",
    "dark_spot": "blemish",
    "rash": "redness",
    "scar": "scar",      # new key — see Task 10.2
}
```

When `/api/analyse` calls the routine engine, map ML detections through this dict before rule lookup.

### Task 10.2 — New `scar` rules

Add to the routine rule engine:

```python
"scar": {
    "Mild": {
        "am": [
            "Niacinamide 10% serum (fades PIH, strengthens barrier)",
            "SPF 50+ sunscreen — non-negotiable, UV worsens scar pigmentation"
        ],
        "pm": [
            "Adapalene 0.1% gel (OTC in India — Differin) — start 2x/week, build up",
            "Silicone-based scar gel if atrophic"
        ],
        "avoid": ["Harsh physical scrubs on active scar tissue"],
        "lifestyle": ["Avoid picking or touching scars"]
    },
    "Moderate": {
        "am": [
            "Niacinamide 10% serum",
            "SPF 50+ — critical"
        ],
        "pm": [
            "Adapalene 0.1% (as tolerated)",
            "Azelaic acid 10% on alternate nights (OTC in India)"
        ],
        "avoid": ["Retinoids + AHAs on same night"],
        "lifestyle": [
            "Consult a dermatologist for persistent moderate scarring.",
            "Clinical options (microneedling, chemical peels) require professional assessment."
        ]
    },
    "Severe": {
        "am": ["Niacinamide 10% serum", "SPF 50+"],
        "pm": ["Gentle moisturiser only — let skin barrier recover"],
        "avoid": ["All active ingredients until dermatologist reviewed"],
        "lifestyle": [
            "See a dermatologist. Severe scarring warrants professional assessment.",
            "Do not self-treat with OTC actives at this severity."
        ],
        "prescription_flag": true
    }
}
```

### Task 10.3 — Conflict matrix

Add a `resolve_conflicts(concerns: List[str], rules: Dict) -> Dict` function:

| Conflict | Resolution |
|---|---|
| `acne` + `rash` (any severity) | Suppress BPO; use only niacinamide + gentle cleanser in AM; barrier-repair focus in PM |
| `acne` (any) + benzoyl peroxide in PM + retinoid in PM | Stagger: BPO on even nights, retinoid on odd nights |
| `rash` (Severe) | Remove ALL active ingredients (AHA, BHA, retinoids, vitamin C) regardless of other concerns |
| `dark_spot` + vitamin C (AM) + `scar` + azelaic acid (PM) | Keep split — no conflict |

### Task 10.4 — Sunscreen rule (non-negotiable)

Every routine response where severity is not `Clear` for ALL concerns must end with:
```
PM final step: SPF 50+ sunscreen (next morning) — UV worsens pigmentation on Indian skin tones (Fitzpatrick III–V).
```

### Task 10.5 — Disclaimer on every response

Every `/api/routine` response must include:
```json
{
  "disclaimer": "Cosmetic guidance only — not medical advice. Consult a dermatologist for persistent, painful, or worsening conditions.",
  ...
}
```

**Done when:** `POST /api/routine` with `concerns: ["acne", "scar"]` returns valid rules with scar block + conflict resolution + disclaimer.

---

## 11. Deployment

**Goal:** App runs on Render (classical always, ML when model loaded). Model artifact on HuggingFace Hub.

### Task 11.1 — HuggingFace Hub model repo setup

Create `ml/deploy/upload_model.py`:
```
Upload best.pt to HuggingFace Hub after training.
Usage: python ml/deploy/upload_model.py --run-id <run_id> --repo <hf_username/bloom-ml>

Requires: huggingface_hub, HF_TOKEN env var set.
```

Script: uploads `ml/runs/{run_id}/best.pt` + a `model_card.md` to the specified HF repo.

`model_card.md` template (fill actuals after training):
```markdown
# Bloom 2.0 — Skin Lesion Detector

**Model:** YOLOv8s fine-tuned on fused acne/skin dataset  
**Classes:** acne, dark_spot, scar, rash  
**mAP@0.5:** [fill after training]  
**Dataset:** ACNE04 + Acnedata3 + own Indian collection (20–25 subjects)  
**Intended use:** Academic demo only. Not a medical device.  
**Limitations:** Low F1 on scar/rash classes (data-limited). Tested on Indian skin tones (Fitzpatrick III–V).  
```

### Task 11.2 — Model fetch on boot in backend

In `backend/ml_inference.py` `load_model()` (Task 8.1):
1. Check if `MODEL_CACHE` exists and SHA256 matches
2. If not: download from `MODEL_URL` with progress bar
3. Verify SHA256 after download — if mismatch, delete and log error
4. Load with `YOLO(MODEL_CACHE)`
5. Wrap entire function in try/except — any failure → return None

### Task 11.3 — Render config

Update `render.yaml` (or create if missing):
```yaml
services:
  - type: web
    name: bloom-api
    env: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: MODEL_URL
        sync: false      # set in Render dashboard
      - key: MODEL_SHA256
        sync: false
      - key: PIPELINE_DEFAULT
        value: hybrid
    healthCheckPath: /api/health
```

Add `/api/health` endpoint to `backend/main.py` if not present:
```python
@app.get("/api/health")
def health():
    from backend.ml_inference import model_loaded
    return {"status": "ok", "ml_model_loaded": model_loaded()}
```

### Task 11.4 — HuggingFace Gradio Space (demo mirror)

Create `ml/deploy/gradio_app.py`:

A stripped Gradio app that:
- Takes an uploaded face photo
- Runs `detect()` from `ml_inference.py`
- Draws bounding boxes on the image
- Shows detected classes + confidence scores
- Shows a severity summary

Keep it < 100 lines. This is a backup demo link, not the main product.

### Task 11.5 — Demo-day checklist (for Mohit)

Create `ml/deploy/DEMO_DAY.md`:
```markdown
# Demo Day Checklist

48 hours before:
- [ ] Hit https://bloom-api.onrender.com/api/health — if model_loaded: false, check MODEL_URL
- [ ] Verify HF Gradio Space link works
- [ ] Run full demo flow locally (uvicorn + npm run dev)
- [ ] Confirm 3–4 example images return results (cache these in frontend)

1 hour before:
- [ ] Wake up Render: hit /api/health to prevent cold start sleep
- [ ] Confirm local machine has the app running as offline fallback
- [ ] Slides have both the Render link AND the Gradio Space link

During demo:
- [ ] Upload a real photo first (dramatic effect)
- [ ] Show classical vs ML side-by-side
- [ ] Show routine card
- [ ] Quote one honest accuracy number (mAP@0.5 on held-out test set)
```

**Done when:** `/api/health` returns `model_loaded: true` on Render. Gradio Space accessible.

---

## 12. Report Draft

**Goal:** Agent drafts the technical report structure with placeholder sections Mohit fills in with actual numbers.

### Task 12.1 — `ml/report/REPORT_DRAFT.md`

Structure:
```
1. Introduction
   - Problem: skin imperfection detection on Indian skin
   - Motivation: existing tools not calibrated for Fitzpatrick III–V
   - Contribution: DIP+ML hybrid pipeline + own Indian test set

2. Related Work
   - ACNE04 / LDL (ICCV 2019): [cite]
   - DermaCon-IN (NeurIPS 2025): [cite]
   - YOLOv8 (Ultralytics 2023): [cite]

3. Dataset
   3.1 Sources (ACNE04, Acnedata3, own collection)
   3.2 Label taxonomy and harmonization
   3.3 Split strategy (subject-level — explain why)
   3.4 Class distribution table [FILL FROM class_balance.txt]
   3.5 Own collection: N subjects, capture protocol, annotator agreement

4. Methodology
   4.1 Classical CV baseline (existing Bloom pipeline — describe 7 stages)
   4.2 YOLOv8 detector (architecture, loss, hyperparams)
   4.3 Severity derivation (area-fraction method — this IS the DIP+ML fusion)
   4.4 Augmentation library (DIP contribution — describe 3 families + ablation design)
   4.5 Routine engine v2

5. Experiments
   5.1 Augmentation ablation (Table 1) [FILL FROM run_log.csv]
   5.2 Pipeline comparison: Classical vs ML vs Hybrid (Table 2) [FILL]
   5.3 Failure analysis [FILL — use failure_gallery.png]

6. Results
   [FILL all numbers with CIs from evaluate.py output]
   Honest note: scar/rash F1 expected to trail acne — explain why

7. Discussion
   - What worked: augmentation gains, Indian skin focus
   - What didn't: rash data scarcity, scar texture difficulty
   - Limitations: dataset size, single annotator, no clinical validation

8. Conclusion + Future Work
   - Multi-angle fusion (not implemented — future work)
   - Larger own dataset collection
   - Severity classifier head

9. Ethics Statement
   - Dataset licenses (ACNE04: academic, Acnedata3: check)
   - Own collection: consent obtained, no public upload
   - Not a medical device disclaimer
   - Skin-tone fairness: metrics reported per-tone if DermaCon-IN integrated

References: [FILL]
```

**Done when:** Report draft exists with all section headers and placeholder markers. Mohit fills `[FILL]` items after training.

---

## 13. Execution Order for Agent

Execute tasks in this order. Do not skip ahead.

```
Phase 0 (Start here):
  [x] 1.1 → 1.4   (scaffold)
  [x] 2.1          (fetch ACNE04)
  [x] 2.3          (inspect script)

Phase 1 (Data):
  [ ] 2.2          (fetch Acnedata3 — wait for Mohit on download method)
  [x] 3.1 → 3.6   (harmonize — ran on ACNE04; re-run after Acnedata3 arrives)
  [x] 4.1 → 4.4   (augmentation library)

Phase 2 (Train):
  [ ] [x] 5.1 → 5.3 (smoke test passed — 2 epochs, ACNE04 only)
  ↳ SMOKE TEST PASSED — proceeding to Phase 3
.

Phase 3 (Eval):
  [ ] 6.1 → 6.3   (evaluation + report scripts)
  ↳ PAUSE: Mohit runs full training (4 augmentation modes). Wait for run_log.csv.

Phase 4 (Integration):
  [ ] 7.1 → 7.2   (annotation protocol — Mohit collects own set in parallel)
  [ ] 8.1 → 8.3   (backend integration)
  [ ] 9.1 → 9.3   (frontend integration)
  [ ] 10.1 → 10.5 (routine engine v2)

Phase 5 (Deploy + Report):
  [ ] 11.1 → 11.5 (deployment)
  [ ] 12.1         (report draft)
```

---

## 14. Decisions Required from Mohit

These are the only times the agent should stop and ask:

| # | Decision | Blocking? | When |
|---|---|---|---|
| D1 | Acnedata3 download method (API key or manual zip?) | Yes — blocks Task 2.2 | Phase 1 |
| D2 | Professor sign-off on 4-class taxonomy confirmed? | Yes — blocks training | Before Phase 2 |
| D3 | Own collection done? How many subjects? | Yes — blocks test set eval | Before Phase 3 eval |
| D4 | HuggingFace repo name for model upload? | Yes — blocks deploy | Phase 5 |
| D5 | Render account — free tier or paid ($7/mo)? | No — can default to free | Phase 5 |

---

## 15. What Agent Should Never Do

- Never train on `our_collection` data (it is test-only, always)
- Never commit model `.pt` files to git
- Never hardcode API keys or HF tokens in code
- Never change the 4-class taxonomy without Mohit's approval
- Never claim severity from ML confidence scores — severity comes from area-fraction only
- Never add a fifth dataset without checking its license
- Never suggest medical diagnoses in routine output — only "detected X consistent with Y"
- Never remove the disclaimer from `/api/routine` responses

---

## 16. Key File Locations (Quick Reference)

| File | Purpose |
|---|---|
| `ml/runs/run_log.csv` | Single source of truth for all training results |
| `ml/data/harmonized/dataset_hash.txt` | Dataset version pin |
| `ml/data/harmonized/metadata.csv` | Per-image provenance |
| `ml/runs/final_report/` | Tables, figures, failure gallery |
| `backend/ml_inference.py` | ML model loading + inference |
| `backend/routine.py` | Skincare routine engine |
| `ml/train/config.yaml` | All training hyperparameters |
| `ml/deploy/DEMO_DAY.md` | Mohit's demo day checklist |

---

*Plan version: 1.0 — Solo execution (Mohit + AI agent)*  
*Scope: Academic project, non-commercial, not a medical device*  
*Last updated: October 2026*
