# Bloom — Project Summary and Analysis Brief

## 1. Project overview

**Bloom** is an educational skin-image-analysis web application. A user uploads a facial photo and receives a visual analysis of surface detail, redness contrast, and dark regions. The application must not be presented as a medical diagnostic tool.

The project combines:

- **Classical digital image processing (DIP):** the current production analysis path.
- **Deep learning:** a planned and partially prepared acne-lesion-detection module.
- **Web application engineering:** a React frontend, FastAPI backend, authentication, safe upload handling, and a local PowerShell launcher.

Suggested academic title:

> **Bloom: Multi-Modal Skin Image Analysis Using Classical Computer Vision and Deep Learning**

## 2. Current user flow

```text
User uploads photo
        ↓
Image validation and quality checks
        ↓
Face/skin region extraction
        ↓
Classical CV analysis
        ↓
Surface / redness contrast / dark-region overlays
        ↓
Relative extent and variation bands
```

The API analyses photos in memory. The README states that source photos and generated annotations are not retained in scan history.

## 3. Current technical architecture

| Layer | Current implementation |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.11 |
| Classical CV | OpenCV, scikit-image, SciPy, NumPy, MediaPipe FaceMesh |
| ML experiments | PyTorch, Ultralytics YOLOv8 |
| Storage/auth | Backend account/auth store with per-account measurements |
| Local run / hosting | `start.ps1` for local use; Render configuration for optional hosting |

### Classical CV pipeline

1. Validate image format, dimensions, file size, brightness, and blur.
2. Resize and enhance illumination using LAB/CLAHE and bilateral filtering.
3. Detect face landmarks and generate a skin-region mask.
4. Exclude non-skin regions such as eyes, lips, brows, and nose.
5. Extract relative redness, dark regions, and surface detail using colour-space, frequency, and morphological operations.
6. Filter components and calculate relative extent and nonclinical variation bands.
7. Calculate the ITA appearance index and FFT surface detail map; return overlays and measurements to the frontend.

## 4. Existing ML direction

The intended deployable ML scope is **acne lesion detection**, rather than a broad medical-diagnosis model.

The repository already contains:

- A local ACNE04-based MVP dataset in `ml/data/acne_mvp/` (ignored by Git; upload it separately for training).
- A YOLOv8 training script in `ml/train/train_yolo.py`.
- Dataset build, evaluation, augmentation, and reporting utilities under `ml/`.
- A Kaggle GPU notebook in `ml/train/kaggle_train_acne.ipynb`.

### Why acne is the correct first ML target

ACNE04 provides lesion-level annotations. That means the model can be evaluated honestly using bounding-box detection metrics. The dataset supports both:

- **Acne lesion detection:** localisation/count of lesions.
- **Acne severity classification:** Mild, Moderate, Severe, and Very Severe.

For object detection, the main metrics should be precision, recall, F1, mAP@0.5, and mAP@0.5:0.95—not ordinary classification accuracy.

## 5. Audit of the local `mine/` dataset

The `mine/` directory was inspected as a possible four-class object-detection dataset.

| Intended folder class | Images | Images with non-empty bounding-box labels | Annotated boxes |
|---|---:|---:|---:|
| Acne | 50 | 45 | 321 |
| Dark spots | 50 | 4 | 20 |
| Pigmentation | 30 | 14 | 45 |
| Redness | 30 | 0 | 0 |
| **Total** | **160** | **63** | **386** |

Important findings:

- Every source label file used YOLO class ID `0`; intended condition classes were inferred from parent folders.
- 97 images have empty label files. Since they belong to condition folders, treating them as negative examples would be incorrect.
- Redness has no object annotations at all, so it cannot be trained or evaluated as a detection class.
- Dark spots and pigmentation have too few positive images for credible held-out evaluation.

Conclusion: `mine/` is useful as a future **annotation pool**, but it is not a production-quality multi-class detection dataset.

## 6. Local baseline experiment on `mine/`

A leakage-aware experimental YOLO dataset was constructed from only the 63 non-empty annotations:

- Training images: 43
- Validation images: 10
- Held-out test images: 10
- Classes: acne, dark spots, pigmentation
- Model: YOLOv8s, pretrained, 320px, CPU, 5 epochs

Held-out test result:

| Metric | Score |
|---|---:|
| Precision | 70.85% |
| Recall | 10.05% |
| mAP@0.5 | 4.22% |
| mAP@0.5:0.95 | 1.25% |

These results are intentionally retained as evidence that the dataset is inadequate. High precision with very low recall means the model detects only a small fraction of actual labelled lesions. This baseline must not be deployed.

Artifacts:

- `ml/runs/mine_detection_cpu_5e_detached/weights/best.pt`
- `ml/runs/mine_detection_cpu_5e_detached/test_metrics.json`
- `ml/datasets/build_mine_detection_dataset.py`

## 7. Recommended final scope

Do not change the entire project. Change the ML scope from a broad, unsupported “all skin conditions detector” to a dataset-aligned modular system.

### Module A — Facial acne analysis (priority)

- Input: ordinary facial photograph.
- Dataset: ACNE04.
- ML task: acne lesion detection and severity estimation.
- Output: lesion overlay, lesion count/extent, and research-only severity estimate.
- Evaluation: held-out test mAP, precision, recall, F1, and severity confusion matrix.

### Module B — Classical visual indicators

- Input: facial photograph.
- Method: existing OpenCV/DIP pipeline.
- Output: redness/pigmentation/blemish visual indicators and photo-quality feedback.
- UI wording: clearly mark these as **experimental visual estimates**, not trained medical classifications.

### Module C — Future common-condition explorer

- Input: general skin-concern photo.
- Candidate dataset: SCIN, after licence review and class-distribution analysis.
- Output: top educational image categories and guidance to seek professional care when appropriate.
- Constraint: this must not claim a diagnosis.

### Module D — Future dermoscopy mode

- Input: dermoscope or standardised close-up lesion image only.
- Candidate dataset: ISIC/HAM10000.
- Task: lesion segmentation and research classification.
- Constraint: keep this separate from ordinary selfie analysis because the image modalities are different.

## 8. Key risks and controls

| Risk | Control |
|---|---|
| Incorrect medical claim | State educational/research-only purpose and retain dermatologist disclaimer. |
| Dataset-label mismatch | Train only tasks that match the type of available labels. |
| Biased or unrepresentative results | Record dataset limitations and report performance by available demographic/skin-tone strata where valid. |
| Data leakage | Split by source/patient where metadata permits; deduplicate before splitting. |
| Inflated results | Report metrics only on an untouched test set. |
| Personal-photo privacy | Keep uploads in memory; avoid saving raw images without explicit consent. |

## 9. Immediate implementation plan

1. Upload `ml/data/acne_mvp/` to a private Kaggle Dataset.
2. Import `ml/train/kaggle_train_acne.ipynb` into a Kaggle Notebook.
3. Enable GPU and run the 50-epoch baseline.
4. Download `best.pt`, `test_metrics.json`, `results.png`, and `results.csv` from Kaggle Output.
5. Add the validated acne model to the FastAPI inference path only if held-out metrics are acceptable.
6. Update the frontend so that only acne is labelled as an ML prediction; keep other concerns explicitly experimental.
7. Treat `mine/` as an annotation backlog. Expand a condition only after collecting and reviewing enough bounding-box labels.

## 10. Success criteria for the first ML release

- Reproducible Kaggle training notebook runs with GPU.
- Train/validation/test splits remain fixed and documented.
- Test mAP, precision, recall, and F1 are saved with the model version.
- A clear model card documents dataset, intended use, limitations, and disclaimer.
- UI never implies clinical diagnosis or treatment recommendation.

## 11. One-sentence conclusion

Bloom should be presented as an educational multi-module image-analysis project: a properly evaluated acne ML module supported by ACNE04, complemented by transparent classical CV visual indicators and future dataset-specific modules.
