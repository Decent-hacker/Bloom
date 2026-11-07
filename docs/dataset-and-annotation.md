# Dataset and Annotation

> Historical research note. The `backend/evaluation/` runner and annotation templates were removed from the production tree. The paths and commands below describe the former workflow and do not run in the current checkout.

## Status

No labelled dataset is shipped with this repository. The evaluation workflow is
implemented and validated on empty/pending inputs, but **no detection performance
is claimed** until independent masks are added.

## What counts as ground truth

Ground truth must be created independently of both the baseline and improved
predictions. Never union, intersect, or otherwise derive masks from pipeline
outputs. Masks are manual annotations or come from a properly licensed dataset
with a documented protocol.

## Layout

```text
backend/evaluation/
  images/                       # source images (JPEG/PNG)
  ground_truth/
    img_001_acne.png
    img_001_redness.png
    img_001_blemish.png
  predictions/                  # generated predictions (gitignored)
  results/                      # generated CSV/JSON (gitignored)
  manifest.csv
  annotate_template.py
  run_evaluation.py
```

## Mask format

| property | requirement |
|---|---|
| Classes | `acne`, `redness`, `blemish` (API: `blemishes`) |
| Resolution | `512x512`, matching `src.config.RESIZE_TARGET` |
| Type | 8-bit grayscale |
| Foreground | 1 or 255 |
| Background | 0 |
| Alignment | Same coordinate frame as the resized pipeline input |
| Resizing | Never silently resized; the runner fails on shape mismatch |

If the source image is not 512x512, annotate on the resized 512x512 image (or
resize once, manually, and document it). The runner cannot tell whether a
resizing operation was applied correctly, so keep the raw annotation and document
the transformation in the manifest notes.

## Annotation protocol

1. Use a consistent zoom level and a white brush on a black canvas.
2. Include visible lesions of the target class only.
3. Exclude hair, eyes, lips, nostrils, eyebrows, shadows, specular highlights, and
   the face boundary unless the study protocol explicitly includes them.
4. Mark ambiguous regions in the manifest instead of guessing silently.
5. Review every mask at full resolution; masks are binary, not anti-aliased.
6. Do not include private user images without explicit permission.

### Class rules (prototype)

- **Acne:** discrete raised/inflamed spots or comedones visible as local blobs.
- **Redness:** diffuse or localised erythema within the facial skin area.
- **Blemish:** dark spots/marks distinct from shading and shadows.

These definitions are intentionally narrow and are not clinical criteria.

## Manifest

`manifest.csv` columns:

| column | meaning |
|---|---|
| `image_id` | Stable ID used in mask filenames |
| `image_file` | Filename inside `images/` |
| `source` | Source or `self-captured` |
| `license` | License/permission; use `private` when not redistributable |
| `width`, `height` | Original dimensions |
| `annotation_status` | `pending`, `annotated`, or `excluded` |
| `concerns` | Comma-separated labels |
| `notes` | Ambiguity, lighting, skin tone, exclusion reason |

Only `annotation_status=annotated` rows are evaluated. Rows with missing or
malformed masks produce an error entry in `summary.json`, not silent exclusion.

## Splits and tuning

- Use separate calibration/validation and held-out test subsets.
- Never tune thresholds on images reported as independent test data.
- If the dataset is too small to split, disclose that limitation explicitly.
- 20–30 images is an initial feasibility set, not proof of generalization.
- Aim for varied skin tones, lighting, face sizes, and difficult cases when
  ethically and practically available.

## Licensing and privacy

- Record the source and license for every image.
- Do not commit private photos, identifiable faces without consent, or images
  whose license forbids redistribution.
- The repository ships only empty `.gitkeep` placeholders and a header-only
  manifest.
- Predictions and results are gitignored because they may reveal image content.

## Creating the first set

```bash
cd backend
python -m evaluation.annotate_template --image-id img_001
# paint ground_truth/img_001_{acne,redness,blemish}.png
# set annotation_status=annotated in evaluation/manifest.csv
python -m evaluation.run_evaluation --dataset evaluation --out evaluation/results/run_001
```

If no annotated rows exist, the runner writes a `pending` status file and
produces no performance numbers.

## Empty-mask conventions

Documented in `src/evaluation.py`:

- Both prediction and GT empty: IoU = Dice = 1; precision/recall/F1 undefined.
- GT empty, prediction non-empty: IoU = Dice = 0; precision = 0;
  recall/F1 undefined.
- Prediction empty, GT non-empty: IoU = Dice = 0; recall = 0;
  precision/F1 undefined.
- Undefined values are excluded from macro averages and counted per metric.
