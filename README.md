# Bloom — Classical CV Skin Image Analysis

Upload a facial skin photo and get a **rule-based classical computer-vision**
analysis of **acne**, **redness** and **blemishes**, with a side-by-side view of a
naive baseline and an improved detection pipeline.

> **Bloom is for educational purposes only and does not constitute medical advice.
> Consult a dermatologist for skin concerns.**
>
> This is not an AI/ML model and not a diagnostic device. Detection is a set of
> hand-tuned image-processing rules; accuracy is not established.

---

## What it does

1. Upload a JPEG/PNG (max 10 MB) from files, drag-drop, or the device camera.
2. Validate the image safely (format, dimensions, pixel budget) and check
   brightness/sharpness.
3. Detect one face with MediaPipe FaceMesh; strict mode rejects images without
   usable landmarks.
4. Segment skin (YCrCb + HSV), exclude eyes/lips/brows/nose, and run the
   detection pipeline.
5. Return annotated images, `Clear`/`Mild`/`Moderate`/`Severe` bands, component
   counts, and a relative extent score.
6. Show a visual baseline-vs-improved comparison. **No accuracy metrics are
   shown for unlabelled uploads**; independent metrics live in the offline
   evaluation workflow.

Accounts keep their scan measurements separate. Original and annotated images
are processed in memory and discarded; they are never saved in scan history.

---

## Architecture

```
Browser (React 18 + Vite + Tailwind)
      │  POST /api/analyse  (multipart/form-data)
      ▼
FastAPI (Python 3.11, synchronous)
      ▼
Classical CV pipeline (OpenCV, scikit-image, scipy, MediaPipe, numpy)
      ▼
In-memory result → base64 images + JSON
```

### Pipeline steps

```
1. preprocessing        dimension/pixel validation, decode, BGR→RGB, 512×512, brightness/blur gate
2. enhancement          LAB L illumination normalisation + CLAHE + bilateral
3. skin_segmentation    FaceMesh oval (strict), YCrCb+HSV skin mask, ROI erosion, exclusions
4. frequency_decomp     spatial LP / HP / DoG bands + LAB
5. lesion_detection     redness (LAB-A), blemish (LAB-L), acne (DoG) + watershed
6. postprocessing       morphology, min-area filter, region props, Clear/severity, extent score
7. visualization        annotated overlays → base64 JPEG
```

Offline DIP experiments (not in the API request path):

- true 2D FFT Gaussian band-pass (`src/frequency_domain.py`)
- Sobel/Scharr, Canny, Laplacian (`src/edge_detection.py`)
- mean±kσ, Otsu, adaptive thresholding (`src/thresholding.py`)
- median/Gaussian/Wiener restoration (`src/restoration.py`)
- global equalisation/gamma vs CLAHE (`src/histogram_processing.py`)
- GLCM/LBP/shape features (`src/features.py`)
- PSNR/SSIM/entropy/contrast (`src/quality_metrics.py`)
- independent evaluation (`evaluation/`)

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18, Vite 5, Tailwind CSS 3 |
| Backend | FastAPI 0.111, Uvicorn, SlowAPI (rate limiting) |
| CV | OpenCV (contrib), scikit-image, scipy, MediaPipe, numpy, Pillow |
| Deploy | Render (web service + static site) |

All numeric parameters live in `backend/src/config.py`. They are prototype
thresholds, not medically validated cut-offs.

---

## Project structure

```
backend/
  main.py                 FastAPI routes + upload validation
  pipeline.py             orchestrator (public result + stage intermediates)
  src/
    config.py             all numeric constants
    preprocessing.py  enhancement.py  skin_segmentation.py
    frequency_decomposition.py  frequency_domain.py
    edge_detection.py  thresholding.py  restoration.py
    histogram_processing.py  lesion_detection.py
    postprocessing.py  features.py  quality_metrics.py
    evaluation.py  visualization.py  image_validation.py  errors.py
  evaluation/             independent GT workflow, manifest, template script
  experiments/            DIP experiment runner (synthetic + optional image)
  tests/                  pytest unit + API integration tests
  requirements.txt  requirements-dev.txt  pyproject.toml
  .env.example  Dockerfile
frontend/
  src/App.jsx  src/api.js
  src/components/  (UploadZone, ProcessingView, ResultsPanel,
                    ComparisonGrid, ConcernBadges, ErrorView, ResetButton)
docs/
  implementation-audit.md  dataset-and-annotation.md
  dip-methodology.md  evaluation-report.md
.github/workflows/ci.yml
render.yaml
```

---

## Local development

**Backend** (Python 3.11):

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements-dev.txt
copy .env.example .env                             # then edit if needed
uvicorn main:app --reload --port 8000
```

**Frontend** (Node 18+):

```bash
cd frontend
npm install
copy .env.example .env                             # VITE_API_URL=http://localhost:8000
npm run dev                                        # http://localhost:5173
```

## Run with Docker

With Docker Desktop running, start the complete application from the project
root:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080). The browser talks to the
same-origin frontend container, which proxies `/api` requests to the backend;
there is no separate API URL to configure. Stop it with `Ctrl+C`, or run
`docker compose down` in another terminal.

The Docker scan is classical-only by default. An optional YOLO checkpoint is
not baked into the image, so the health response will report its ML capability
as unavailable unless you deliberately add a model volume/configuration later.

In Docker, account records and the generated local signing key are kept in the
named volume, so sessions survive container restarts. For hosted deployment,
set a long random `AUTH_SECRET` in the service environment. Source photos remain
in memory only.

Module CLIs require an explicit image path:

```bash
cd backend
python -m src.lesion_detection path/to/test.jpg
python -m pipeline path/to/test.jpg
```

---

## Tests, lint, and CI

```bash
cd backend
ruff check .
pytest
```

Current local result: `ruff check .` passes; `pytest` passes 57 tests.
CI (`.github/workflows/ci.yml`) runs backend lint/tests and the frontend build.

---

## Independent evaluation

Ground truth must be **manual or licensed**, never derived from predictions.

```bash
cd backend
python -m evaluation.annotate_template --image-id img_001
# paint evaluation/ground_truth/img_001_{acne,redness,blemish}.png at 512x512
# set annotation_status=annotated in evaluation/manifest.csv
python -m evaluation.run_evaluation --dataset evaluation --out evaluation/results/run_001
```

Outputs: `metrics_per_image.csv`, `summary.json` (macro mean/median, support,
excluded undefined values), `status.json`, predictions. If no annotated images
exist, the runner writes a `pending` status and **does not fabricate metrics**.
See `backend/evaluation/README.md` and `docs/dataset-and-annotation.md`.

---

## DIP experiments

```bash
cd backend
python -m experiments.dip_experiments --out experiments/output
# or with an image:
python -m experiments.dip_experiments --image path/to/image.jpg
```

Synthetic restoration uses the clean scene as reference, so PSNR/SSIM are valid
there. Real-image enhancement reports reference-free entropy/contrast only. No
detection accuracy is claimed without independent masks. See
`docs/evaluation-report.md`.

---

## API

```
GET  /              → {"name":"Bloom API","version":"1.0"}
GET  /api/health    → {"status":"ok"}
POST /api/analyse   → multipart/form-data, field "file" (JPEG/PNG, ≤10MB)
```

**200**

```json
{
  "status": "success",
  "processing_time_sec": 1.23,
  "quality": {"passed": true, "issues": []},
  "detection": {"face_detected": true, "landmarks_available": true,
                "fallback_used": false, "model": "mediapipe_face_mesh"},
  "images": {"original": "...", "enhanced": "...", "skin_mask": "...",
             "baseline_annotated": "...", "improved_annotated": "..."},
  "concerns": {
    "acne": {"severity": "Clear", "count": 0, "area_pct": 0.0, "extent_score": 0.0},
    "redness": {"severity": "Clear", "area_pct": 0.0, "extent_score": 0.0},
    "blemishes": {"severity": "Clear", "count": 0, "area_pct": 0.0, "extent_score": 0.0}
  },
  "metrics": null,
  "metrics_note": "Per-image accuracy is not computed for unlabelled uploads...",
  "mediapipe_mode": "mediapipe"
}
```

- `severity`: `Clear` (no regions detected by the algorithm), `Mild`, `Moderate`,
  `Severe`.
- `extent_score`: relative area in `[0, 1]`, **not** a confidence value.
- `metrics` is `null` on uploads. Accuracy is computed only offline against
  independent ground truth.

**Errors**

- `422 invalid_image` — unsupported/corrupt/undersized/oversized upload.
- `422 quality_error` — brightness/sharpness failure; `quality.reason_codes` may
  contain `too_dark`, `overexposed`, `too_blurry`.
- `422 face_error` — strict-mode FaceMesh failure; `reason_code` and
  `detection.fallback_used=false`.
- `429` — rate limit (10 requests/min/IP).
- `500` — generic processing failure.

---

## Security and reliability

- Content-type → compressed size → magic bytes → Pillow dimension/pixel budget →
  OpenCV decode.
- Dimension limits: min side 200 px, max 6000×6000, max 24 MP.
- CORS uses an explicit, trimmed origin list (never `*`).
- Rate limiting is in-process (per worker, resets on restart), scoped to a
  single-instance deployment.
- Errors are generic to the client and detailed only in logs. Nothing is written
  to disk for uploads.
- Frontend requests have a 60 s timeout, AbortController cancellation, stale
  response protection, and object-URL cleanup.

---

## Privacy

Bloom does not store, log, or transmit photos to a third party. Images are
processed in memory and discarded after the response. Accounts retain only
their email, salted credential/recovery hashes, and compact scan measurements;
users can delete each saved measurement. There are no tracking cookies.

---

## Deployment

`render.yaml` defines both services (backend web service + static frontend). Set
the repository in Render, update the `onrender.com` origins/URLs to your actual
service names, and deploy. In-memory rate limiting assumes a single backend
process; use a shared store if scaling to multiple workers/instances.

---

## Known limitations

- Rule-based classical CV, not AI/ML; no diagnosis.
- No labelled dataset is shipped; detection accuracy is **unverified**.
- One face only; no alignment/registration, multi-face handling, or occlusion
  modelling.
- Prototype thresholds may shift with skin tone, lighting, makeup, hair, and
  glasses.
- Acne `|DoG|` detection can flag highlights and can suppress weaker regions via
  a global-maximum-relative threshold.
- Runtime API deliberately withholds accuracy metrics for unlabelled uploads.
- A small annotated set would be a feasibility check, not proof of generalization.
