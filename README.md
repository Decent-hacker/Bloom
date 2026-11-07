# Bloom — Facial Skin Image Analysis

Educational skin image analysis using classical computer vision. Bloom reviews a facial photo and displays visible image patterns. Its scores are visual estimates, not medical findings.

## What it does

1. Accepts JPEG or PNG images up to 5 MB and checks format, dimensions, exposure, and sharpness before analysis.
2. Detects a face and identifies an analysable skin region, excluding eyes, lips, brows, and nose.
3. Normalises illumination and enhances local contrast for inspection.
4. Maps surface detail, relative redness contrast, and dark regions in the facial image.
5. Displays an FFT surface irregularity heatmap and relative region coverage.
6. Reports an ITA appearance index from the image's LAB colour values. This describes image appearance, not skin type, race, or ethnicity.
7. Presents original and processed views, per-concern variation bands, and optional guided multi-angle review.

Uploaded image bytes are processed in memory. Account history stores measurements, not the raw photos. The classical concern masks and variation bands have not been clinically validated; no accuracy is assigned to an unlabelled upload.

## Tech stack

| Layer | Technology |
|---|---|
| Interface | React 18, Vite, Tailwind CSS |
| API | FastAPI, Python 3.11, Uvicorn |
| Image processing | OpenCV, NumPy, SciPy, scikit-image, Pillow |
| Face landmarks | MediaPipe FaceMesh |
| Deployment | Docker Compose |

MediaPipe FaceMesh uses a pretrained model for landmark location. The skin measurements and visual maps use classical image processing; the default scan does not use a custom trained disease classifier.

## DIP techniques used

- Grayscale mean and Laplacian variance for exposure and focus checks.
- LAB luminance normalization, CLAHE, and bilateral filtering for enhancement.
- YCrCb and HSV thresholding, erosion, opening, and closing for the skin region.
- Gaussian low/high-pass decomposition and Difference of Gaussians for fine detail.
- ROI-adaptive LAB thresholding for relative colour and brightness contrast.
- Distance transform, watershed, and connected components for candidate regions.
- 2D FFT Gaussian band-pass filtering for the surface detail map.
- Contours and overlays for visual inspection.

These methods measure image properties. They cannot by themselves identify a skin disease.

## Setup

Install Docker Desktop, then from the project root run:

```bash
docker compose up --build
```

Open `http://localhost:8080`. The frontend proxies API requests to the backend. To stop, use `Ctrl+C`, or run `docker compose down` in another terminal. Account data is stored in a named Docker volume.

For local development, install `backend/requirements-dev.txt`, run `uvicorn main:app --reload` inside `backend/`, then run `npm install` and `npm run dev` inside `frontend/`.

Run backend tests from `backend/` with `python -m pytest tests -q`. Production Python dependencies are pinned in `requirements.txt`; the development file adds pytest, HTTPX, and Ruff. A model checkpoint is optional for the default scan.

## API and limitations

`POST /api/analyse` accepts a multipart `file` field. The response contains processed image views, visual concern measurements, the ITA appearance index, and per-stage timing. The optional routine endpoint uses rule-based suggestions; its output is educational guidance rather than a treatment plan.

Variation bands and coverage values depend on photo lighting, pose, image quality, and the skin mask. They are relative image measurements, not detection confidence or clinical severity. No independent annotated test set currently establishes diagnostic accuracy.

## Project structure

```text
backend/   FastAPI routes, classical pipeline, active image modules and tests
frontend/  React interface
ml/        ML research code and training notebook; not in the default production scan
docs/      Background and project notes
```

## Disclaimer

Bloom provides educational visual analysis only. This is not a medical diagnosis or medical advice. Consult a dermatologist for skin health concerns.

## Academic context

Bloom is a Digital Image Processing course project at RGIPT. It demonstrates acquisition checks, enhancement, segmentation, spatial and frequency filtering, thresholding, morphology, and visualization. The visible maps are presented with their limitations rather than as validated diagnostic accuracy.

## ML research (separate)

ML research code — not part of production pipeline. See `ml/train/kaggle_train_acne.ipynb` for the training notebook. The optional `hybrid` API mode can use an acne-only YOLO checkpoint when one is installed; the default classical scan does not require it. Datasets, model weights, and training runs under `ml/` are ignored by Git.
