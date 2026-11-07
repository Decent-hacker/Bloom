# Bloom 2.0 — DIP + ML Upgrade Plan

Turning Bloom from a classical-CV-only pipeline into a complete **Digital Image
Processing + Machine Learning** project: hybrid detection of cosmetic
imperfections (acne, rash, dark spots, scars) on **Indian skin**, with real
accuracy metrics, augmentation-driven data efficiency, and multi-angle capture.

> Status: PLAN — nothing here is implemented yet. This is the shared reference
> for the team and the professor.

---

## 1. Goals and scope (locked)

| Requirement (from prof) | How this plan satisfies it |
|---|---|
| Use augmentation + ML | YOLOv8 detector + severity classifier, DIP-derived augmentation library |
| Dataset not readily available → build own | Fuse 3–4 public datasets + collect a small Indian multi-angle test set |
| Limited features: acne, rash, dark spots, scars | 4-class taxonomy (§2), no more, no less |
| Must provide accuracy | Held-out evaluation: mAP, per-class F1, κ, skin-tone-sliced metrics (§9) |
| Images from multiple angles | 3-angle capture protocol + multi-angle fusion (§5, §8) |
| Indian skin only | Indian-sourced training data + tone-sliced fairness reporting |
| Complete DIP and ML project | Classical CV baseline retained as comparison arm (§10) |
| Deploy somewhere | Two-tier deployment: free-tier Render (classical) + ML inference service (§13) |
| Provide treatment / skincare routine | Severity-aware rule engine "Bloom Care" fed by detection output, with derm-escalation rules (§12) |

**Non-goals:** no medical diagnosis claims, no commercial deployment, no
account system, no cloud training dependency. Treatment output is
**cosmetic skincare guidance, not prescriptions** — the escalation ladder in
§12 keeps clinical advice behind a "see a dermatologist" wall.

---

## 2. Label taxonomy (decide first — everything depends on this)

Four target classes. Sub-types recorded as metadata but not trained as separate
classes (data too small):

| Class | Definition (annotation guideline) | Annotation type | Difficulty |
|---|---|---|---|
| `acne` | Open/closed comedones, papules, pustules, nodules | Bounding box per lesion | Medium — most data available |
| `dark_spot` | Flat hyperpigmented macules: post-inflammatory pigment, melasma patches | Box (or mask if time permits) | Medium |
| `scar` | Atrophic/boxcar/rolling texture depressions; exclude active acne | Box | Hard — texture, not colour |
| `rash` | Diffuse erythematous patches (eczema/allergy-like), no discrete lesion border | Segmentation mask (classical redness mask fused) | Hard — diffuse, boxes are wrong shape |

Severity grading (Clear / Mild / Moderate / Severe) derived **per class** from
detected area fraction of the skin mask, reusing the existing
`postprocessing.py` severity-band logic — thresholds re-calibrated against the
new dataset, not the old prototype guesses.

**Design decision to present to the prof:** two-head model — YOLO boxes for
acne/dark_spot/scar + segmentation-style detection for rash. This is a genuine
DIP+ML fusion (the classical redness mask from `lesion_detection.py` becomes
the rash head's prior), and it is the honest way to treat diffuse erythema.

---

## 3. Datasets — what to get, what each is for

### 3.1 Primary sources

| # | Dataset | Size | Labels relevant to us | India? | License | Role |
|---|---|---|---|---|---|---|
| D1 | **DermaCon-IN** (NeurIPS 2025, Harvard Dataverse, ~4 GB) | 5,450 imgs / 3,002 patients | Acne (appendageal), pigmentary disorders, inflammatory/eczema (rash-adjacent), scars as descriptors; Fitzpatrick + Monk tone per image | **Yes — South India** | CC BY-NC-SA 4.0 (research only) | Tone diversity, Indian skin prior, fairness slicing, rash/pigment examples |
| D2 | **ACNE04** (ICCV 2019; HuggingFace `Layered-Labs/ACNE04`, 1.13 GB) | 1,457 face photos, **18,983 lesion boxes**, 4-level Hayashi severity | Acne boxes + severity grades; 5-fold CV splits preserved | No (China) | Academic use only | **Core acne detection + grading data**; severity-grade benchmark |
| D3 | **Acnedata3** (Ultralytics Universe) | 11,400 imgs, 174k boxes, 6 classes | acne lesion types + dark spots | No | Check on download | Volume for detection pre-training |
| D4 | **Face-derm VOC set** (Baidu, 8,093 imgs, 11 classes) | 65k boxes | Acne, Dark-Spots, Atrophic Scars, redness | Unknown | Unclear — **verify before use** | Only source with all 4 of our classes boxed |
| D5 | **TrainingDataPro dermatology** (HF, CC BY-NC-ND) | 30 subjects × **3 angles** (front/left/right) | acne / redness labels per subject | Diverse, unclear | Non-commercial, no derivatives | Multi-angle protocol template + fusion validation |

### 3.2 Supporting / analysis-only

- **Fitzpatrick17k** — tone-fairness cross-check only (not training).
- **DDI (Stanford)** — fairness benchmark reference in the write-up only.
- **SkinDis-India** (Mendeley) — Indian psoriasis/vitiligo; useful only if we
  expand beyond the 4 classes later. Skip for now.

### 3.3 Critical caveat about DermaCon-IN (read before promising anything)

DermaCon-IN **masks eyes and crops facial regions for patient privacy** (paper
§5, Limitations). It is excellent for Indian skin tones and body-site
conditions, but **weak for facial cosmetic imperfections specifically**. Plan
accordingly:

- Use DermaCon-IN for: skin-tone metadata, pigmentary/rash examples on body
  sites, fairness slicing, and as evidence of "Indian-skin-centric" design.
- Use ACNE04 + Acnedata3 + D4 + **our own collection** for the actual facial
  training signal.

Tell the prof this upfront — it is the kind of honest limitation that scores
well, and it is exactly why our own collected set (§5) matters.

### 3.4 Expected combined training pool (after harmonization, rough)

| Class | Est. boxes/images | Source |
|---|---|---|
| acne | ~150k+ boxes | D2 + D3 + D4 |
| dark_spot | ~10–20k boxes | D3 + D4 |
| scar | ~1–3k boxes | D4 (+ our collection) |
| rash | ~500–2k masks | D1 subset + our collection |

Scar and rash are the bottleneck classes — augmentation (§7) and focal loss
target them specifically. Expect the report to be honest that rash/scar F1
will trail acne by a wide margin.

---

## 4. Hardware and software requirements

### 4.1 Compute

| Need | Minimum | Recommended |
|---|---|---|
| Training YOLOv8s | Google Colab free tier (T4 16 GB) — works, slow | Colab Pro / Kaggle 2×T4 / college lab GPU |
| Full ablation matrix (~8–12 runs) | ~15–25 GPU-hours | Budget 1 weekend of continuous training |
| Inference in Bloom API | **CPU-only** — YOLOv8n on 512×512 ≈ 100–300 ms | fine for the existing FastAPI service |
| Dataset storage | ~10 GB locally | SSD recommended |

No need to buy anything. Colab/Kaggle free tiers are sufficient if runs are
batched overnight.

### 4.2 Software (new additions to the repo)

```
ml/                          # new top-level folder (keeps backend/ clean)
  datasets/                  # download + harmonization scripts (one per source)
    fetch_dermacon_in.py
    fetch_acne04.py          # pulls Layered-Labs/ACNE04 via `datasets`
    fetch_acnedata3.py
    harmonize.py             # maps every source → our 4-class YOLO format
  augment/                   # DIP-derived augmentation library
    photometric.py           # gamma, CLAHE variants, tone shifts (reuse backend/src)
    geometric.py             # rotations, flips, elastic
    degradation.py           # blur/noise from backend/src/restoration.py
  train/
    train_yolo.py            # YOLOv8s/n fine-tune, k-fold support
    train_severity.py        # EfficientNet-B0 severity head
    config.yaml              # hyperparameters, paths, seeds
  eval/
    evaluate.py              # mAP, F1, κ, tone-sliced metrics (wraps ultralytics val)
    compare_classical.py     # runs existing Bloom pipeline on same test set
    report.py                # tables + confusion matrices + figures
  annotations/               # our own collected set lives here
    protocol.md
    consent_form_template.pdf
```

Python additions: `ultralytics`, `torch`/`torchvision`, `albumentations`,
`datasets` (HF), `torchmetrics`, `scikit-learn` (κ, PR curves). Everything
else (OpenCV, MediaPipe, numpy) already exists in `backend/`.

Python 3.11, CUDA 12.x runtime on Colab. Pin versions in
`ml/requirements.txt` separate from backend.

---

## 5. Preparing our own dataset (the Indian multi-angle set)

This is the **held-out evaluation set** and the project's ethical core. Never
train on it.

### 5.1 Target

- **60–100 subjects**, each contributing **3 angles** (front, left 45°,
  right 45°) → **180–300 images**.
- Subject pool: Fitzpatrick III–V (typical Indian range), both genders, ages
  ~18–35 (campus-feasible), varied regional backgrounds.
- Deliberately oversample subjects with visible acne (the common case) and
  recruit ~10 subjects with known dark spots/scars/rash through dermatology
  department contacts if available.

### 5.2 Capture protocol (write this down and follow it exactly)

1. **Device:** any smartphone, ≥12 MP, rear camera. Record device model per
   subject (metadata column).
2. **Lighting:** indirect daylight or consistent indoor white light; no mixed
   lighting; no direct flash.
3. **Subject prep:** no makeup, no sunscreen residue, hair pulled back, no
   glasses; face washed 30 min before (avoid post-wash redness).
4. **Angles:** front (camera at eye level), left 45°, right 45° — subject
   turns head, camera fixed. Fill frame with face + upper neck.
5. **Distance:** ~25–35 cm; consistent across the 3 shots.
6. **Background:** plain, non-reflective.
7. **Consent:** signed consent form per subject (template in
   `ml/annotations/consent_form_template.pdf`) stating: educational research
   use only, no storage on servers, right to withdraw and demand deletion.
8. **Privacy:** store locally only; never upload our own set to public
   datasets or demos; anonymize filenames (`subj_007_left.jpg`).

### 5.3 Annotation workflow

1. **Tool:** Roboflow (free plan) or labelImg — YOLO bbox format at native
   resolution, normalized on export.
2. **Guideline document:** a 2-page illustrated annotation manual with
   positive/negative examples per class ("this is acne, this is a scar, this
   is not a rash") — write it before annotating anything.
3. **Double annotation:** every image labelled independently by 2 annotators;
   disagreements adjudicated by a third (or by discussion). Report **Cohen's
   κ between annotators** in the report — DermaCon-IN achieved κ=0.84, which
   is the bar to mention.
4. **Order of operations per image:** (a) run MediaPipe face detect, (b) draw
   skin mask with existing `skin_segmentation.py`, (c) annotate only inside
   skin region, (d) record Monk tone estimate per subject.
5. **Severity labels:** after box annotation, derive area-fraction severity
   with the existing `postprocessing.py` code, then manually spot-check 20%
   against a 4-level scale (Clear/Mild/Moderate/Severe).

### 5.4 Realistic effort

Annotation is the long pole: ~15–25 min per 3-angle subject for a trained
annotator → **3–5 full days** for 60–100 subjects with double annotation.
Start this in week 1, in parallel with everything else — do not leave it to
the end.

---

## 6. Dataset preparation pipeline (harmonization)

One script per source, one common output schema. Output everything in
**YOLO detection format** + a `metadata.parquet` sidecar.

### 6.1 Common schema

```
ml/data/
  raw/                 # immutable downloads, one folder per source
  harmonized/
    images/{split}/    # train/ val/ test/
    labels/{split}/    # YOLO txt: class cx cy w h (normalized)
    metadata.csv       # image_id, source, subject_id, angle, monk_tone, fitzpatrick
    classes.yaml       # 0: acne, 1: dark_spot, 2: scar, 3: rash
```

### 6.2 Label mapping (the harmonization table — the core intellectual work)

| Source | Native labels | → our classes |
|---|---|---|
| ACNE04 | lesion boxes (acne, 4 severity levels) | all boxes → `acne`; severity column kept in metadata |
| Acnedata3 | blackhead, whitehead, papule, pustule, nodule, dark spot | first 5 → `acne`; dark spot → `dark_spot` |
| D4 VOC | Acne, Blackheads, Whiteheads, pustules, Dark-Spots, Atrophic Scars, redness | Acne/Blackheads/Whiteheads/pustules → `acne`; Dark-Spots → `dark_spot`; Atrophic Scars → `scar`; redness → **review manually** (discrete vs diffuse; diffuse → rash masks) |
| DermaCon-IN | hierarchical disease + descriptors, no boxes | classification-level weak labels only; body-site images with pigmentary/inflammatory labels → candidate `rash`/`dark_spot` crops, **manually reviewed** before inclusion |
| Ours | our 4 classes | identity |

Rules:
- **Never invent a box DermaCon-IN doesn't have.** DermaCon-IN contributes
  image-level labels and metadata, not detection boxes. If we crop lesions
  from it, those crops become classification samples, flagged as
  `weak_label=true`.
- Every harmonized image keeps a `source` column — all metrics reported
  overall **and** per-source, so we can see if the model only works on one
  dataset.

### 6.3 Splits

- Split **by subject**, never by image (prevent same-face leakage across
  train/test — the most common error in face datasets).
- 70 / 15 / 15 train/val/test on the fused pool; **our own collection is
  100% test, always**.
- ACNE04's published 5-fold splits: preserve fold 0 as an additional
  reproducible benchmark row.

### 6.4 Deduplication

Near-duplicate images across web-scraped sources are likely (D3/D4 overlap
with public acne galleries). Run perceptual hashing (pHash) across the fused
pool and drop hash-distance < 4 collisions.

---

## 7. Augmentation strategy (the DIP contribution — make it visible)

Frame augmentation as a DIP experiment, not a checkbox. Three families, each
ablated separately:

### 7.1 Photometric (skin-tone aware)
- Gamma in [0.7, 1.5], brightness ±20%, simulated diffuse lighting
- CLAHE clip-limit jitter (reuse `histogram_processing.py` logic)
- White-balance shift (R/G/B channel gain ±10%) — models real phone AWB errors
- **Tone-preserving rule:** never push augmented pixel statistics outside the
  Monk-tone range observed in Indian subjects (check with the skin-segmentation
  mask mean)

### 7.2 Geometric
- Rotation ±15°, horizontal flip (face asymmetry is acceptable for lesions)
- Small random scale/crop (0.9–1.1), translation ±5%
- Elastic deformation: subtle only (α small) — skin doesn't stretch

### 7.3 Degradation (our existing DIP code as an augmentation generator)
- Gaussian + salt-and-pepper noise (reuse `restoration.py` noise models)
- Motion/defocus blur kernels
- JPEG compression artifacts (quality 40–80) — matches real uploads

### 7.4 Imbalance handling
- Per-class oversampling of `scar` and `rash` (×3–5 effective epochs)
- Focal loss (γ=1.5) for the detector head
- Copy-paste augmentation for rare lesions (paste scar/rash crops onto
  healthy skin crops with Poisson blending) — optional stretch goal

### 7.5 The ablation that sells the project
Train 4 identical YOLO runs: (a) no augmentation, (b) +geometric, (c)
+photometric, (d) +degradation (= full). Report mAP per run. **If (d) − (a)
is positive, augmentation is proven, not claimed.**

---

## 8. Models and training recipes

### 8.1 Stage 1 — lesion detector (the main model)

- **Architecture:** YOLOv8s (start), YOLOv8n for the deployable CPU variant
- **Input:** 640×640 (letterbox); keep original aspect ratio
- **Init:** COCO pretrained weights, fine-tune
- **Schedule:** 100–150 epochs, image-size 640, batch 16 (T4), AdamW,
  lr0=0.001, cosine decay, early-stop patience 25
- **Loss:** focal + CIoU; class weights inverse-sqrt frequency
- **Seed policy:** fix seed=42 for headline runs; repeat best config over 3
  seeds and report mean±std (DermaCon-IN paper averaged 5 seeds — cite that
  as precedent)
- **Augmentation inside YOLO** (mosaic/mixup) — evaluate with and without;
  mosaic can hurt small-lesion detection, that's a finding either way

### 8.2 Stage 2 — severity head (classification)

- Crop each detected lesion (box + 10% context padding) → classify
  Clear/Mild/Moderate/Severe with EfficientNet-B0 (ImageNet init)
- Alternative simpler route (if time is short): **skip a learned severity
  head** and reuse area-fraction severity from `postprocessing.py` on the
  ML-detected mask regions. This is legitimate DIP+ML fusion and much less
  work. Decide after Stage 1 metrics are in.

### 8.3 Stage 3 — multi-angle fusion

- Run detector independently on front/left/right
- Merge detections: map boxes to a canonical face coordinate frame using
  MediaPipe landmarks (you already have FaceMesh — use it for alignment)
- Deduplicate across angles by landmark-relative position overlap (IoU in
  face-frame, not image-frame)
- Per-subject severity = max across angles (conservative) + per-angle
  breakdown shown in UI
- Evaluation compares per-angle and fused results — fusion should win; if it
  doesn't, that's an honest negative result worth reporting

### 8.4 Training-time tracking

- Log every run (hyperparams, seed, dataset hash, metrics) to
  `ml/runs/run_log.csv` — no wandb needed, a CSV is enough and reproducible
- Save best checkpoint per run as `ml/runs/{run_id}/best.pt`

---

## 9. What to expect — realistic accuracy targets

Set expectations now so results don't look like failures later. These are
**plausible ranges** based on published acne-detection work, not promises:

| Task | Metric | Expected range | Notes |
|---|---|---|---|
| Acne detection (boxes) | mAP@0.5 | **0.55 – 0.75** | ACNE04-level data; small lesions are hard |
| Acne detection | F1 @ best conf | 0.60 – 0.80 | Conf threshold tuned on val only |
| Acne severity (4-level) | QWK | 0.55 – 0.75 | ACNE04 published LDL baselines ≈ this range |
| Dark spot detection | mAP@0.5 | 0.40 – 0.60 | Larger regions, fewer data |
| Scar detection | mAP@0.5 | **0.20 – 0.45** | Texture-based; genuinely hard; say so in report |
| Rash (diffuse) | Dice / IoU (mask) | 0.45 – 0.65 | Diffuse boundary ambiguity |
| **Fused (our 3-angle test set)** | overall mAP@0.5 | should beat single-angle by 3–8% | the multi-angle claim |
| Classical CV baseline (current Bloom) | IoU vs same GT | **0.15 – 0.35** | Expected to lose to ML — that's the point |
| Tone-sliced | Δ mAP deep vs light | < 0.10 desired | Report whatever it is, honestly |

**How to read these:** the project succeeds if (1) ML beats the classical
baseline on the same held-out set, (2) fusion beats single-angle, (3)
augmentation ablation shows gains, and (4) all numbers are honest with CIs.
It does **not** require beating published SOTA.

Statistical hygiene:
- Bootstrap 95% CIs on every headline metric (1,000 resamples)
- Report per-class metrics, never only mAP-avg
- Any config choice made on val set is frozen before the test set is touched
  once

---

## 10. The DIP+ML comparison experiment (the project's centerpiece)

Run **both pipelines on the exact same held-out images**:

| Arm | Pipeline |
|---|---|
| A — Classical CV (current Bloom) | existing 7-stage pipeline → masks → severity |
| B — ML only | YOLOv8s + severity head → severity |
| C — Fused (DIP+ML) | ML detection inside DIP skin mask; rash head uses classical redness prior; severity from area fraction |

Metrics table: per-class precision/recall/F1/IoU + runtime + failure-case
gallery. Arm C winning (or trading off honestly) is the "complete DIP and
ML project" story the prof asked for.

---

## 11. Integration into Bloom (after models are validated)

- New backend module `backend/ml_inference.py`: loads `best.pt` once at
  startup, exposes `detect(image) -> List[Detection]`
- `/api/analyse` extended: `pipeline: "classical" | "ml" | "hybrid"` request
  param; response adds `ml` block alongside existing `concerns`
- Keep `metrics: null` honesty convention for unlabelled uploads — ML
  confidence scores are **not** accuracy; the report already establishes this
  principle, don't undo it
- Frontend: add a Classical vs ML toggle in `ResultsPanel` and a detection
  box overlay view; keep the existing disclaimer
- CPU latency budget: < 1 s total added; YOLOv8n if YOLOv8s is too slow
- Model artifact (`best.pt`, ~20–40 MB) served from object storage / HF Hub
  at startup — never baked into the git history (see §13)
- Routine engine (`routine.py`) upgraded to consume ML concerns — see §12

---

## 12. Treatment & skincare routine ("Bloom Care")

The repo already ships a rule-based routine engine
(`backend/routine.py`, `POST /api/routine`) with AM/PM steps for
acne/redness/blemish. Upgrade it from "demo extra" to a first-class feature
that the prof can demo after every scan.

### 12.1 What stays

- Pure rule-based, ingredient-level (not brand-level) recommendations —
  avoids commercial bias and licensing issues
- Structure: `am` steps, `pm` steps, `avoid` list, `lifestyle` tips
- Dedupe logic and the existing API shape (frontend `RoutineCard` already
  renders it)

### 12.2 What changes for the ML era

1. **Concern vocabulary alignment.** ML taxonomy is
   `{acne, dark_spot, scar, rash}`; routine keys today are
   `{acne, redness, blemish}`. Add a mapping layer:
   - `acne → acne`
   - `dark_spot → blemish` (PIH-focused rules already exist)
   - `rash → redness` (calming/barrier rules already exist)
   - `scar → new "scar" rule block` (must be written — see below)
2. **New `scar` rules** (cosmetic, not medical):
   - Mild: gentle retinoid introduction (adapalene 0.1% OTC in India),
     silicone-based primer/scar gel, strict SPF (UV worsens scar
     pigmentation)
   - Moderate/Severe: cosmetic options only (retinoid + niacinamide +
     sunscreen) + **explicit dermatologist referral** — microneedling,
     lasers, and chemical peels are clinical procedures, the app must not
     recommend them beyond "consult a dermatologist"
3. **Severity-driven intensity.** Rules already branch on
   Clear/Mild/Moderate/Severe — feed ML severity directly. Add a
   **"step-up" warning**: if severity increased since the user's previous
   scan (localStorage history already stores past concerns), append a
   gentle "consider seeing a dermatologist" banner rather than silently
   escalating actives.
4. **Conflict resolution.** When multiple concerns co-fire (e.g. acne +
   rash), naive rule-merging can suggest contradicting actives (BPO +
   strong exfoliants on inflamed skin). Add a small **conflict matrix**:
   - benzoyl peroxide × retinoid → stagger nights, never same PM
   - AHA/BHA × irritated rash skin → pause exfoliation until rash clears
   - vitamin C (AM) × azelaic acid (PM) → fine, keep split
   - any "active" × Severe rash → suppress actives, barrier-repair only
5. **Indian-market grounding.** Keep ingredient-level, but note OTC
   availability in India (salicylic acid, niacinamide, azelaic acid 10–20%,
   adapalene 0.1% are all OTC; tretinoin/clindamycin are prescription —
   the rule engine must mark those as `prescription: true` so the UI can
   style them differently and never present them as buy-now advice)
6. **Sunscreen non-negotiable.** Every non-Clear path ends with SPF 50+,
   with a note that UV worsens PIH on Indian skin tones (Fitzpatrick
   III–V) — this is the single highest-value recommendation the app can
   make and ties directly to the project's Indian-skin thesis.

### 12.3 Guardrails (non-negotiable)

- Every routine response ships with
  `"disclaimer": "Cosmetic guidance only, not medical advice. See a
  dermatologist for persistent, painful, or worsening conditions."`
- No dosage beyond OTC-typical concentrations; no oral medication ever
  suggested; no diagnosis language ("you have eczema") — only "detected
  redness consistent with irritation"
- Rules reviewed once by a dermatology-aware person (ideally a med student
  co-author or the prof) before demo day; cite source guidelines (e.g.
  Indian Dermatology Society patient-education materials, AAD patient
  pages) in the report

### 12.4 Nice-to-have (stretch, only if time remains)

- 4-week routine tracker in localStorage: user checks off AM/PM adherence;
  next scan compares severity → simple "routine adherence vs outcome" chart
- Export routine as a printable PDF card

---

## 13. Deployment

### 13.1 Constraint analysis (why the current setup can't just absorb ML)

| Component | Current (Render free) | With ML |
|---|---|---|
| Backend RAM | 512 MB free tier | torch CPU + YOLOv8 ≈ 1.2–2 GB → **needs paid instance or split service** |
| Cold starts | acceptable | model load (3–10 s) worsens them |
| Model artifact | n/a | `best.pt` 20–40 MB — too big for git, needs external fetch |
| Upload processing | in-memory, fine | unchanged — ML is CPU-fast enough |

### 13.2 Recommended two-tier architecture (free/cheap, course-appropriate)

```
                    ┌─────────────────────────────┐
  User ── HTTPS ──▶ │  Render: bloom (static SPA)  │   FREE static site
                    └──────────────┬──────────────┘
                                   │ VITE_API_URL
                    ┌──────────────▼──────────────┐
                    │  Render: bloom-api (FastAPI) │   Starter $7/mo OR
                    │  ├─ classical pipeline       │   free tier if ML
                    │  ├─ routine engine           │   stays optional
                    │  └─ /api/analyse (hybrid)    │
                    └──────────────┬──────────────┘
                                   │ (startup) fetch
                    ┌──────────────▼──────────────┐
                    │  HuggingFace Hub / Releases │   best.pt artifact
                    └─────────────────────────────┘
```

- **Model artifact:** upload `best.pt` to a HuggingFace model repo (private)
  or a GitHub Release; backend downloads it on first boot to a cached
  volume, verifies SHA256, loads into memory. Keeps the git repo clean and
  lets you update the model without redeploying code.
- **Backend sizing:** try free tier first with YOLOv8n + `torch.set_num_threads(2)`
  — if RSS exceeds limits, upgrade Render Starter ($7/mo, 512 MB→ more RAM)
  or reduce input to 512×512. For demo day, a paid week is ~$2.
- **Fallback path:** `/api/analyse?pipeline=classical` always works even if
  the model fails to load — deploy resilience is part of the grade.

### 13.3 Alternative / supplementary hosts (pick one as plan B)

| Host | Pros | Cons | Verdict |
|---|---|---|---|
| **HuggingFace Spaces (Gradio)** | Free GPU/CPU demos, ML-native, instant share link | Replaces (not extends) your React UI; Gradio look | **Plan B demo link** — ship a stripped Gradio demo of just the ML detector as a backup demo path |
| Streamlit Cloud | Fast to build | Python-only UI, duplicates React work | Skip |
| Vercel + serverless fn | Great frontend hosting | Cold-start + 50 MB fn limit kills torch | Frontend-only hosting option |
| College server / local demo | Zero cost, offline-proof | Not a URL | **Mandatory offline backup** — always have the app running locally for demo day regardless of cloud status |

### 13.4 Demo-day resilience checklist

- [ ] App runs fully local (`uvicorn` + `npm run dev`) as offline fallback
- [ ] Cloud deployment verified 48 h before demo (Render free tier sleeps —
      hit `/api/health` 5 min before to wake it)
- [ ] Gradio HF Space as second link in the demo slides
- [ ] Pre-computed results for 3–4 example images in the UI (cached
      responses) in case inference hiccups live
- [ ] Model checksum verified at startup; clear error → automatic classical
      fallback, never a 500 to the user

### 13.5 Deployment work items (add to timeline §15)

1. Verify the local `start.ps1` launcher with the optional model installed.
2. HF Hub private model repo + fetch-on-boot with checksum
3. Render blueprint update: health checks, env vars
   (`MODEL_URL`, `MODEL_SHA256`, `PIPELINE_DEFAULT=hybrid`)
4. Rate limits reconsidered: ML inference is heavier — keep 10/min for
   classical, consider 5/min for hybrid
5. HF Gradio Space mirroring the detector (half-day of work, high demo value)

---

## 14. Ethics, licensing, and honesty checklist

- [ ] DermaCon-IN cited as CC BY-NC-SA 4.0; research-only use stated in README
- [ ] ACNE04 academic-use-only terms respected (contact author if unsure)
- [ ] D4 (Baidu VOC set) license verified before any public demo uses it — if
      unverifiable, train on it privately, don't redistribute
- [ ] Signed consent forms for every own-collection subject; deletion-on-
      request honored
- [ ] Own-collection images never uploaded to public platforms
- [ ] "Not a medical device / not medical advice" banner retained everywhere
- [ ] Routine engine: no prescriptions suggested as OTC, escalation rules in
      §12.2 followed, disclaimer shipped with every `/api/routine` response
- [ ] Every reported number traces to a run_log.csv row + a frozen test split
- [ ] Failure cases shown, not hidden — include a "when Bloom gets it wrong"
      section in the report
- [ ] Skin-tone-sliced metrics reported even if they look bad

---

## 15. Timeline (≈7 weeks, parallelizable)

| Week | Dataset track | Model track | Own-set track | Product track (routine + deploy) |
|---|---|---|---|---|
| 1 | Download D1–D5; inspect labels; write harmonize.py for D2 | Set up ml/ scaffold, Colab notebooks, YOLO smoke test | Write consent form + annotation guideline; first 10 pilot subjects | Map routine keys to new taxonomy; write `scar` rules |
| 2 | Harmonize D3, D4; dedupe; build classes.yaml; fix splits | Baseline YOLOv8s train on fused pool (run a) | Continue collection (target 40 subjects) | Conflict matrix in routine engine + unit tests |
| 3 | DermaCon-IN weak-label crop pipeline; manual review | Augmentation ablation runs (b, c, d) | Annotation in earnest; κ measurement | Local launcher + model-fetch-on-boot test |
| 4 | Freeze v1 dataset (hash it) | Severity head or area-fraction severity; 3-seed repeat | Finish collection (60–100 subjects) | Deploy classical path to Render; verify CI |
| 5 | — | Multi-angle fusion; per-angle vs fused eval | Adjudication pass | Deploy ML path; Gradio HF Space; load test |
| 6 | — | Full evaluation: classical vs ML vs hybrid; tone slicing; CIs | — | Routine demo polish; step-up warnings; disclaimer audit |
| 7 | — | Backend integration + frontend toggle | — | Demo-day rehearsal; offline fallback verified |
| buffer | Report writing, figure polishing, viva demo prep (1 week slack) | | | |

---

## 16. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| D4 license unverifiable | Medium | Use for training only, never redistribute; prefer D2/D3+own if it falls through |
| Scar/rash classes too rare | High | Honest low-F1 reporting; consider merging scar into "texture anomaly" or scoping it as future work |
| Colab session limits break long runs | Medium | Checkpoint every epoch; resume-capable scripts; run ablations serially overnight |
| Own-set annotation slower than planned | High | Start week 1; pilot 10 subjects to time yourself; accept 60 not 100 |
| Same-face leakage across splits | Medium | Subject-level split enforced in harmonize.py + unit test asserting zero subject overlap |
| Fusion doesn't beat single angle | Low–Med | Honest negative result; diagnose which angle contributes errors |
| DermaCon-IN face-crop limitation | Certain | Already accounted (§3.3) — it's metadata/tone/fairness data, not facial detection data |
| Render free tier OOMs with torch | Medium | YOLOv8n + thread cap first; $7/mo Starter for demo week; classical fallback always live (§13.2) |
| Routine engine suggests conflicting actives | Medium | Conflict matrix (§12.2.4) + unit tests for every pair of co-firing concerns |
| Cloud asleep / network down at demo | Medium | Demo-day checklist §13.4: local run, Gradio mirror, cached example results |

---

## 17. Deliverables checklist

- [ ] `ml/` codebase: fetch, harmonize, augment, train, evaluate — all runnable
      with one command each
- [ ] Frozen v1 fused dataset + hash + datasheet (like a mini model card)
- [ ] Trained `best.pt` (YOLOv8s) + `best_n.pt` (CPU variant)
- [ ] Our own 3-angle Indian test set + annotation guideline + κ score
- [ ] Experiment matrix results: augmentation ablation, 3 seeds, classical vs
      ML vs hybrid, per-angle vs fused, tone-sliced
- [ ] Report: methodology, dataset card, ethics, results with CIs, failure
      gallery, limitations
- [ ] Bloom app with ML/hybrid mode + comparison UI
- [ ] **Routine engine v2:** taxonomy-aligned rules incl. `scar`, conflict
      matrix, prescription flags, step-up dermatologist warnings, disclaimer
      on every response
- [ ] **Live deployment:** Render (static + API) with model fetched from HF Hub
      at boot; Gradio Space mirror; local offline fallback rehearsed
- [ ] 5-minute demo script: live upload → boxes drawn → classical vs ML
      side-by-side → routine card shown → deployment link in slides

---

## 18. First five actions (start here)

1. **Freeze the taxonomy** (§2) — get prof's sign-off on the 4 classes and the
   two-head (rash = mask) decision.
2. **Download ACNE04** from HuggingFace (`Layered-Labs/ACNE04`) and inspect 50
   images + boxes — this calibrates annotation quality expectations.
3. **Request DermaCon-IN access** on Harvard Dataverse (check institutional
   email acceptance of CC BY-NC-SA).
4. **Write the consent form + annotation guideline** before touching a camera.
5. **Colab smoke test:** `pip install ultralytics`, train YOLOv8n for 5 epochs
   on ACNE04 — proves the toolchain end-to-end in one evening.

(Sixth, when you catch a breath: upgrade `routine.py` keys to the new
taxonomy — it's a half-day change unblocks the whole product track.)

---

## Appendix A — key links

- DermaCon-IN paper: arXiv:2506.06099 · dataset on Harvard Dataverse ·
  code: github.com/shan4035/DermaCon-IN-code
- ACNE04: huggingface.co/datasets/Layered-Labs/ACNE04 · original:
  github.com/xpwu95/LDL (ICCV 2019)
- Acnedata3: platform.ultralytics.com/arjun-mukherjee/datasets/acnedata3
- TrainingDataPro (3-angle): huggingface.co/datasets/TrainingDataPro/
  dermatology-dataset-acne-redness-and-bags-under-the-eyes
- Fitzpatrick17k: github.com/mattgroh/fitzpatrick17k
- DDI: ddi-dataset.github.io

## Appendix B — glossary

- **mAP@0.5**: mean average precision at IoU threshold 0.5 — standard detection
  accuracy; combines precision and recall across confidence thresholds.
- **QWK**: quadratic weighted Cohen's κ — agreement with ordinal severity
  grades, chance-corrected; the standard for acne grading papers.
- **Monk skin tone / Fitzpatrick**: 10-point and 6-point skin tone scales;
  DermaCon-IN provides both per image.
- **Harmonization**: mapping heterogeneous dataset label spaces into one
  consistent taxonomy.
- **Weak label**: image-level label with no precise box/mask annotation.
