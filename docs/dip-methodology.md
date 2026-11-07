# DIP Methodology

> Historical methodology note. The current production outputs are surface anomalies, redness contrast, dark regions, ITA appearance index, and an FFT surface detail map. The offline experiment runner referenced below was removed; use the root README for the current pipeline.

This document records what each stage does, why it exists, its parameters, and
what it does **not** prove. All numeric parameters live in `backend/src/config.py`.

## Pipeline overview

```text
upload validation
  -> decode + RGB conversion + 512x512 resize
  -> brightness/sharpness quality gate
  -> illumination normalisation + CLAHE + bilateral
  -> strict FaceMesh ROI + YCrCb/HSV skin mask + exclusions
  -> spatial band decomposition + LAB
  -> redness / blemish / acne detection
  -> morphology + connected components + severity/extent
  -> overlays -> base64 JPEG
```

Offline only:

```text
independent masks -> baseline vs improved -> IoU/Dice/precision/recall/F1
FFT / edges / threshold alternatives / restoration / histogram / features / quality metrics
```

## 1. Acquisition, colour, and resampling

- `preprocessing.load_image_from_bytes` decodes with OpenCV and performs the only
  BGR→RGB conversion. All later modules receive RGB.
- Images are resized to `RESIZE_TARGET = (512, 512)` with `INTER_AREA`, which
  averages pixels when downscaling and reduces aliasing.
- Colour spaces:
  - **LAB** for luminance (L) and redness (A).
  - **YCrCb** and **HSV** for skin-range segmentation.
- Uploads are inspected with Pillow before full decode: format, minimum side,
  maximum width/height, and total pixel budget (`MAX_IMAGE_PIXELS`).

Limitation: a 512x512 resize changes the effective lesion scale; parameters are
calibrated for this resolution.

## 2. Quality gate

- Brightness = mean grayscale intensity; reject outside
  `[MIN_BRIGHTNESS, MAX_BRIGHTNESS]`.
- Sharpness = variance of the Laplacian; reject below `MIN_BLUR_SCORE`.
- Face presence is **not** checked here. FaceMesh is the single face/landmark
  source; strict mode returns `face_error` when landmarks are missing.
- Reason codes: `too_dark`, `overexposed`, `too_blurry`, `no_face_landmarks`.

These are heuristic gate values, not calibrated quality predictors.

## 3. Enhancement

### Illumination normalisation

Shift the LAB L channel so its mean equals `TARGET_L_MEAN = 130`, clipping to
`[0, 255]`. Formula: `L' = clip(L + (130 - mean(L)), 0, 255)`.

### CLAHE

Contrast-limited adaptive histogram equalisation on L only:
`clipLimit=2.0`, `tileGridSize=(8, 8)`. A and B channels are untouched, so hue is
preserved. CLAHE is a local histogram method; it can amplify noise and does not
guarantee better downstream detection.

### Bilateral filter

`cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)` smooths skin texture
while preserving strong edges. It is enhancement, not restoration.

### Alternatives (experiments only)

`histogram_processing.py` adds global LAB-L equalisation and gamma correction
(`GAMMA_CORRECTION=0.8`) for comparison. They are not in the production path.

## 4. Face ROI and skin segmentation

- MediaPipe FaceMesh returns normalised landmarks for one face
  (`FACE_MESH_MAX_FACES=1`). The face oval is filled as a polygon.
- The oval is eroded by `ROI_EROSION_PX=15` to avoid hair/background.
- Skin-colour mask = `inRange` in YCrCb **AND** HSV, followed by morphological
  open/close with an elliptical `(5, 5)` kernel.
- Eyes, brows, lips, and nose are filled from landmark groups and dilated by
  `EXCLUSION_DILATE_PX=5`; excluded from the analysis zone.
- `analysis_zone = skin_mask AND NOT exclusion_mask`.

Limitation: one face only; glasses, facial hair, makeup, occlusion, and extreme
angles are not modelled. Skin ranges are broad and not calibrated per skin tone.

## 5. Spatial band decomposition

On the enhanced grayscale image:

- Low-pass: Gaussian blur, `LP_KERNEL_SIZE=(21, 21)`.
- High-pass: `gray - low_pass + 128` (diagnostic; not used by detection).
- Difference of Gaussians: `G(sigma=1) - G(sigma=3)`, a band-pass blob response.
- LAB is computed for colour thresholds.

These are **spatial** filters. They are not described as FFT frequency-domain
processing.

## 6. True 2D FFT filtering

`frequency_domain.py` implements a Gaussian band-pass in the Fourier domain:

1. Convert to luminance `f(x, y)`.
2. `F = fftshift(fft2(f))`, with DC at the centre.
3. Build radial distance `D(u, v)` and mask
   `H = (1 - exp(-D² / (2·D_low²))) · exp(-D² / (2·D_high²))`.
4. `filtered = real(ifft2(ifftshift(F · H)))`.
5. Expose `mask` and a log-magnitude `spectrum` (display only).

Cut-offs are fractions of `min(rows, cols)`:
`FFT_LOW_CUTOFF=0.02`, `FFT_HIGH_CUTOFF=0.25`.

Validation: constant image → near-zero response; impulse → non-zero; in-band
sinusoid retains most RMS energy. This is a diagnostic/experimental path, not
currently part of lesion detection; no accuracy gain is claimed.

## 7. Gradient and edge detection

- `sobel_gradient`: Scharr 3x3 derivatives (`ksize=3`), magnitude
  `sqrt(Gx² + Gy²)`; other kernel sizes use Sobel.
- `canny_edges`: `CANNY_LOW_THRESHOLD=50`, `CANNY_HIGH_THRESHOLD=150`.
- `laplacian_map`: absolute Laplacian response for diagnostics.
- `edge_density(edge_map, region)` is used as a per-region feature.

Edge maps are not lesion masks. Synthetic step-edge tests verify response and
zero response on constant fields. Real-image sensitivity to hair, wrinkles, and
pores is documented as a limitation.

## 8. Thresholding

All thresholds are computed inside the analysis ROI.

- **Mean ± k·σ** (production):
  - Redness: `threshold = mean(A) + max(10, 1.4·std(A))`.
  - Blemish: `threshold = mean(L) - max(25, 1.5·std(L))`.
  - Rationale: adapts to per-image contrast; the floor prevents over-detection on
    uniform skin.
- **Otsu** (experiment): `skimage.filters.threshold_otsu` on ROI values only.
- **Adaptive mean** (experiment): OpenCV local mean, block size 31, C=5, ROI
  restricted.
- **Ratio of ROI maximum**: used by acne (`0.30 · max|DoG|`) and kept as the
  current production behaviour. This is sensitive to one strong edge/highlight.

Otsu/adaptive are not assumed to be better; they are compared in the experiment
runner and must be evaluated against independent masks before adoption.

## 9. Lesion detection

- **Redness:** LAB A above the adaptive threshold.
- **Blemish:** LAB L below the adaptive threshold.
- **Acne:** `|DoG|` above `0.30 · max(|DoG|)`, restricted to the zone, closed
  with a `(3, 3)` ellipse, then watershed-split.
- **Watershed:** distance transform → `peak_local_max`
  (`min_distance=10`) → keep peaks ≥ `0.4 · max_distance` → marker-controlled
  watershed with `watershed_line=True`.
- **Postprocessing:** morphological open, connected components
  (`connectivity=2`), drop regions below `MIN_LESION_AREA_PX=50`.
- **Severity:** `Clear` when area is zero; otherwise bands from unrounded
  `area_pct` (`acne 1/3`, `redness 4/12`, `blemish 2/6`).
- **Extent score:** `min(area_pct / Moderate_threshold, 1)` — relative extent,
  explicitly **not** detection confidence.

Known failure modes: bright highlights count as acne due to `|DoG|`; the global
max threshold can suppress weaker spots; hair and face boundaries can create
edges; parameter values are prototype settings.

## 10. Features

`features.py` computes per connected region:

- Shape: area, perimeter, circularity `4πA/P²`, aspect ratio, extent, solidity.
- Colour: mean L, A, B.
- Texture: GLCM contrast, homogeneity, energy, correlation (32 levels, distance 1,
  four angles); uniform LBP histogram entropy (P=8, R=1).
- Edge density from Canny.

Degenerate regions return zero or `None` rather than dividing by zero. Features
describe regions; they do not identify a medical condition.

## 11. Restoration experiments

- Corruptions: salt-and-pepper (`amount=0.03`) and Gaussian (`sigma=15`) on a
  synthetic scene with a clean reference.
- Filters: median `(5, 5)`, Gaussian `sigma=1`, Wiener `(5, 5)`.
- Metrics: PSNR and SSIM against the clean reference; entropy and contrast are
  also reported.

PSNR/SSIM are only valid here because the clean reference is retained. No
real-photo restoration quality is claimed.

## 12. Histogram comparison

Global LAB-L equalisation vs CLAHE vs gamma vs original. Reference-free entropy
and contrast are reported. Higher entropy/contrast is not automatically better.

## 13. Independent evaluation

- Ground truth comes from manual masks only; no prediction-derived GT exists.
- Baseline (fixed absolute thresholds on broad skin) and improved methods run on
  the same images and are scored against the same GT.
- Metrics: IoU, Dice, precision, recall, F1, with pixel supports.
- Aggregation: macro mean/median across images/classes; undefined cases excluded
  and counted.
- Results include dataset hash, config snapshot, software versions, and errors.

No baseline-vs-improved claim is made until a labelled set is available.

## 14. Limitations

- Rule-based classical CV; no learned classifier, no clinical diagnosis.
- One face, frontal-ish assumption; no alignment/registration.
- Prototype thresholds calibrated on the 512x512 enhanced representation, not on
  a representative labelled population.
- Skin tone, lighting, makeup, hair, glasses, and occlusion can change results.
- Runtime API returns no accuracy metrics for unlabelled uploads.
