# Capture and Annotation Protocol for Mohit

## Capture protocol (Mohit executes):

1. **Device:** any smartphone ≥12 MP, rear camera
2. **Lighting:** indirect daylight or consistent indoor white light; no flash; no mixed light sources
3. **Subject prep:** no makeup, no sunscreen, hair pulled back, no glasses; face washed 30 min before
4. **Angle:** **front only** (camera at eye level, subject facing forward)
5. **Distance:** 25–35 cm; fill frame with face + upper neck
6. **Background:** plain wall, non-reflective
7. **Filename:** `subj_001.jpg`, `subj_002.jpg`, etc. (never real names)
8. **Target:** **20–25 subjects minimum** (RGIPT hostel/campus)

## Consent (one per subject):

- **Verbal explanation:** "This is for an academic project. Photos stored locally only, never uploaded publicly, deleted on request."
- **Written consent:** simple sign-off in a notebook with date. Not a formal form — this is campus academic research, not clinical.

## Annotation workflow (Mohit executes in Roboflow):

1. **Create Roboflow project:** Object Detection, 4 classes matching our taxonomy
2. **Upload subjects one by one**
3. **Draw bounding boxes only inside the face/neck skin region**
4. **For rash:** draw the tightest box that contains the patch (Roboflow doesn't do masks easily — box is fine for this project)
5. **Export:** YOLOv8 format, normalized coordinates
6. **Download zip → save to `ml/annotations/images/`** (gitignored)

## Severity spot-check (after annotation):

- Run `postprocessing.py` area-fraction severity on each annotated image
- Manually verify 5 random images per class — does the severity label match what you see?
- If >30% mismatch: recalibrate thresholds with Mohit