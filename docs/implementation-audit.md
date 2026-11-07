# Implementation Audit

> Historical snapshot from 2026-10-21. See the root README and `docs/project-summary.md` for current behavior, names, limits, and file locations.

Date: 2026-10-21  
Repository: `C:\D-Drive\Projects\DIP`  
Audited commit: `cb623f3 stage-10`  
Scope: backend CV pipeline, FastAPI contract, frontend flow, evaluation, engineering.

## 1. Request flow (verified from source)

1. `frontend/src/App.jsx` validates MIME type and 10 MB compressed size, creates an object URL preview, and calls `analyseImage()` in `frontend/src/api.js`.
2. `frontend/src/api.js` POSTs `multipart/form-data` field `file` to `VITE_API_URL/api/analyse`.
3. `backend/main.py` validates declared content type, byte size (`MAX_IMAGE_SIZE_MB`), and magic bytes, then calls `pipeline.run(content)`.
4. `backend/pipeline.py` runs, in order:
   - `preprocessing.run` — decode, single BGR→RGB conversion, resize to 512×512, brightness/Laplacian-variance/FaceDetection quality gate.
   - `enhancement.run` — LAB L mean shift, CLAHE, bilateral filter.
   - `skin_segmentation.run` — MediaPipe FaceMesh oval, YCrCb+HSV skin ranges, morphology, ROI erosion, eye/lip/brow/nose exclusion.
   - `frequency_decomposition.run` — spatial Gaussian low-pass, high-pass, DoG, LAB.
   - `lesion_detection.run` — redness (LAB A), blemish (LAB L), acne (|DoG| + watershed).
   - `postprocessing.run` — morphology, connected components, min-area filter, severity/confidence.
   - `evaluation.run` — baseline vs improved against a pseudo-GT built from both predictions.
   - `visualization.generate_all` — five base64 JPEG images.
5. `main.build_success` maps internal `blemish` to API `blemishes` and drops `regions`.
6. Frontend results view renders `ComparisonGrid`, `ConcernBadges`, `MetricsTable`, `ResetButton`.

## 2. Confirmed API contract (before changes)

`POST /api/analyse` success (`200`):

```json
{
  "status": "success",
  "processing_time_sec": 0.0,
  "quality": {"passed": true, "issues": []},
  "images": {
    "original": "base64",
    "enhanced": "base64",
    "skin_mask": "base64",
    "baseline_annotated": "base64",
    "improved_annotated": "base64"
  },
  "concerns": {
    "acne": {"severity": "Mild|Moderate|Severe", "count": 0, "area_pct": 0.0, "confidence": "High|Medium|Low"},
    "redness": {"severity": "...", "area_pct": 0.0, "confidence": "..."},
    "blemishes": {"severity": "...", "count": 0, "area_pct": 0.0, "confidence": "..."}
  },
  "metrics": {"baseline": {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, "improved": {"...": 0.0}},
  "mediapipe_mode": "mediapipe|fallback"
}
```

Errors: `422` `quality_error` with `quality.issues`, `429` rate limit, `500` generic error.

## 3. Confirmed issues

### Correctness and evaluation
- **Circular evaluation.** `evaluation.consensus_ground_truth` derives GT from the baseline and improved predictions (`backend/src/evaluation.py:25-33`). Recall is structurally 1.0 for both pipelines; the comparison cannot demonstrate an improvement.
- **Zero detections labelled Mild.** `postprocessing._severity` returns `Mild` when `area_pct == 0` (`backend/src/postprocessing.py:28-35`). Severity is also computed from a rounded percentage.
- **Misleading confidence.** `postprocessing._confidence` and `_region_confidence` use area and count only (`backend/src/postprocessing.py:19-45`), not detection certainty.
- **Fallback/quality-gate contradiction.** The quality gate rejects no-face images (`backend/src/preprocessing.py:70-74`), while the UI has a fallback warning and segmentation has fallback code (`backend/src/skin_segmentation.py:102-108`, `frontend/src/App.jsx:80-84`). The fallback warning is effectively unreachable.
- **Duplicate MediaPipe work.** FaceDetection runs in preprocessing, FaceMesh runs again in segmentation. FaceMesh alone can answer the face-presence question.
- **Runtime metrics are not evidence.** `pipeline.py` computes and returns baseline/improved metrics for unlabelled uploads; the frontend displays them as a pipeline comparison.
- **Detector failure modes.** Acne uses `abs(DoG)` with a threshold tied to the global maximum (`backend/src/lesion_detection.py:75-85`). Bright highlights, hair, pores, and one strong edge can create false positives or suppress weaker regions.
- **Unused frequency outputs.** `low_pass` and `high_pass` are computed but never consumed; only `dog` and `lab` feed detection (`backend/src/lesion_detection.py:98-107`). "Frequency decomposition" is spatial filtering, not FFT.

### Robustness and security
- No decompression-bomb guard: only compressed bytes are limited (`backend/main.py:53-58`); decode happens before any dimension/pixel check (`backend/src/preprocessing.py:32-44`).
- No minimum/maximum resolution checks; multiple faces are not handled explicitly (`FACE_MESH_MAX_FACES = 1`).
- CORS origins are split without trimming (`backend/main.py:27`).
- Rate limiting is in-process and resets on restart (`backend/main.py:41-43`).
- Frontend fetch has no timeout or abort (`frontend/src/api.js:11-31`); previous preview object URLs are not revoked when a new file is selected (`frontend/src/App.jsx:27-38`).

### Engineering
- No tests, CI, linter, type checker, or Python lockfile.
- `__main__` blocks default to a nonexistent `test_image.jpg`.
- No sample images, dataset, ground truth, or annotation workflow.
- No LICENSE, screenshots, or DIP experiment report.
- `HP_KERNEL_SIZE` is configured but unused; several MediaPipe/FaceDetection constants become dead if face detection is removed.

## 4. Baseline checks (before changes)

| Check | Command | Result |
|---|---|---|
| Backend imports | `.venv\Scripts\python.exe -c "import main, pipeline"` | PASS |
| Frontend production build | `npm run build` | PASS (41 modules, 5.09 s) |
| Python tests | none configured | NOT RUN — no test suite |
| Lint/type check | none configured | NOT RUN — not configured |
| Reproduction with real face image | no fixture in repository | NOT RUN — no licensed fixture |

## 5. Planned changes

### P0 — correctness
- Replace the circular pseudo-GT with an independent, manifest-driven evaluation workflow. Remove runtime metrics from the API; return `metrics: null` and an explicit note until independent GT exists.
- Add `Clear` for zero detected regions, use full-precision area for banding, keep thresholds in `config.py`.
- Rename `confidence` to `extent_score` (numeric, relative extent, not certainty) in backend, API, and frontend.
- Adopt strict face mode: FaceMesh is the single face/landmark source; no face or no landmarks returns a structured `face_error`; remove fallback code and UI warning.
- Add `detection` status (`face_detected`, `landmarks_available`, `fallback_used`) and quality `reason_codes`.

### P1 — DIP depth
- Add `frequency_domain.py` (2D FFT Gaussian band-pass, spectrum).
- Add `edge_detection.py` (Sobel/Scharr, Canny, Laplacian).
- Add `thresholding.py` (mean±kσ, Otsu, adaptive) and use the mean±std path in production.
- Add `restoration.py` (median, Gaussian, Wiener) with synthetic noise experiments.
- Add `histogram_processing.py` (global equalisation, gamma) for comparison with CLAHE.
- Add `features.py` (GLCM, LBP, shape/colour features, CSV export).
- Add `quality_metrics.py` (PSNR, SSIM, entropy, RMS contrast).
- Add `experiments/dip_experiments.py` with synthetic and optional real-image runs; no performance claims without independent GT.

### P2 — reliability
- Validate image dimensions and pixel count with Pillow before OpenCV decode.
- Remove FaceDetection from preprocessing; reuse FaceMesh as the only face pass.
- Add timeout/abort, stale-response protection, and object URL cleanup to the frontend.
- Trim CORS origins; document in-memory rate limiting.
- Add pytest unit/integration tests, ruff config, CI, and dev requirements.
- Update README, add `docs/dataset-and-annotation.md`, `docs/dip-methodology.md`, `docs/evaluation-report.md`, and evaluation CLI/README.

## 6. Non-goals / limitations
- No medical diagnosis is claimed.
- No labelled dataset is supplied with this change; comparative metrics remain **pending** until independent masks are added.
- No deep learning is introduced.
- Thresholds remain prototype values in `config.py`.
