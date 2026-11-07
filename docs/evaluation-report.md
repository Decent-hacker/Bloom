# Evaluation Report

> Historical report. The offline `backend/evaluation/` and `backend/experiments/` tools described below were removed during production cleanup. No classical detection accuracy is currently established.

Date: 2026-10-21  
Status: **detection evaluation pending — no independent annotated images exist in
this repository.**

## 1. Dataset

| item | value |
|---|---|
| Manifest rows | 0 |
| Annotated images | 0 |
| Held-out test images | 0 |
| Ground-truth source | none shipped |
| Dataset version | hash of empty manifest |
| Image licenses | n/a |

The repository contains the workflow, empty directories, and a header-only
manifest. It does **not** contain real images or masks. No detection precision,
recall, IoU, Dice, or F1 is reported.

## 2. Detection evaluation

`backend/evaluation/run_evaluation.py`:

- runs baseline and improved methods on identical images;
- scores both against the same independent masks;
- writes per-image metrics, supports, errors, config, and software versions.

Current result:

```text
PENDING: no annotated images. Add manual masks and mark manifest rows annotated.
```

No baseline-vs-improved claim is made. The previous runtime pseudo-GT workflow was
removed because it derived ground truth from the predictions and forced recall to
1.0.

## 3. Synthetic DIP experiments (not detection performance)

Run:

```bash
cd backend
python -m experiments.dip_experiments --out experiments/output
```

This uses a deterministic synthetic scene with a known clean reference. Values
below are from that run and are reproducible; they do **not** measure lesion
detection accuracy.

### Restoration (clean reference)

| noise | filter | PSNR (dB) | SSIM |
|---|---|---:|---:|
| salt & pepper | none | 20.98 | 0.361 |
| salt & pepper | median | 41.34 | 0.995 |
| salt & pepper | Gaussian | 30.66 | 0.725 |
| salt & pepper | Wiener | 21.01 | 0.361 |
| Gaussian | none | 24.61 | 0.290 |
| Gaussian | median | 34.67 | 0.871 |
| Gaussian | Gaussian | 33.25 | 0.830 |
| Gaussian | Wiener | 24.81 | 0.299 |

Interpretation: median filtering is appropriate for impulse noise; Gaussian
smoothing/median help Gaussian corruption; the simple Wiener configuration does
not help on this synthetic scene. These are synthetic observations only.

### Threshold alternatives (descriptive, no GT)

| method | threshold | foreground |
|---|---:|---:|
| mean ± 1.4σ | 148.45 | 8.18% |
| Otsu | 104.85 | 57.73% |
| adaptive mean | C = 5 | 87.62% |

No claim is made that one method is better; selection requires independent masks.

### FFT / edges / features

- FFT: constant image is suppressed; impulse and in-band sinusoid produce
  expected non-zero responses.
- Edges: constant field gives zero response; a vertical step produces strong
  Scharr/Canny response.
- Features: synthetic circles produce a CSV with shape, colour, GLCM, LBP, and
  edge-density columns.

Artifacts are written under `backend/experiments/output/` (gitignored):
`fft_*.png`, `edge_*.png`, `threshold_*.png`, `restore_*.png`, `hist_*.png`,
and comparison CSVs.

## 4. Unit and integration validation

`pytest` currently passes 57 tests. Covered:

- metric conventions and aggregation;
- `Clear` severity and unrounded banding;
- extent score is numeric and not called confidence;
- thresholding, Otsu, adaptive, ROI restriction;
- FFT constant/impulse/sinusoid behaviour;
- Sobel/Canny/Laplacian and edge density;
- median/Gaussian/Wiener restoration on synthetic noise;
- histogram equalisation/gamma;
- GLCM/LBP/shape features and CSV export;
- PSNR/SSIM/entropy/contrast;
- upload format/dimension/pixel guards;
- strict face-landmark failure;
- API success, quality error, face error, invalid image, size limit, CORS;
- pending evaluation workflow and mask validation.

Lint: `ruff check .` passes.

## 5. Failure cases and limitations

- No labelled real-world data; no generalization evidence.
- Acne detection uses `|DoG|` with a global-max-relative threshold; highlights,
  hair, pores, and strong edges can produce false positives or suppress weaker
  regions.
- Fixed skin-colour ranges may under/over-segment across skin tones and lighting.
- One face only; no alignment, registration, or multi-face handling.
- Quality gate and severity bands are prototype heuristics, not clinical cut-offs.
- PSNR/SSIM are only valid for synthetic reference-backed comparisons here.
- A 20–30 image set, once annotated, would still be a feasibility set, not proof
  of broad generalization.

## 6. Next steps

1. Add 20–30 licensed images with varied skin tones/lighting and manual 512x512
   masks; record them in `manifest.csv`.
2. Split calibration/validation and held-out test data; tune only on the former.
3. Run the evaluation CLI and report macro metrics with support counts.
4. Compare threshold alternatives and DoG variants against the same GT.
5. Keep or revert experimental detection changes based on held-out results only.
