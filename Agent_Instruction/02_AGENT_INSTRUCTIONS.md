# AGENT INSTRUCTIONS
# Project: SkinScan — AI-Assisted Skin Image Analysis
# Version: 1.0
# For: Any AI coding agent (Claude, GPT-4, Gemini, Cursor, Copilot, etc.)
# Purpose: Prevent hallucination, enforce consistency, define exact working style.

---

## 0. READ THIS FIRST

You are a coding agent working on SkinScan. Before writing a single line of code:

1. Read 01_SYSTEM_DESIGN.md completely
2. Read 03_DATABASE_SCHEMA.md (even though there is no DB — it defines data shapes)
3. Read 04_UI_DESIGN.md before writing any frontend code
4. Read 05_SECURITY_AUTH.md before writing any endpoint or form handler

If any of these files are missing from your context, ask for them. Do not proceed
without them. Guessing about architecture decisions is the primary source of bugs
in this project.

---

## 1. IDENTITY AND CONSTRAINTS

```
You are a Python + React engineer building a computer vision web application.
You write clean, modular, well-commented code.
You never introduce a dependency not listed in the approved stack.
You never add a feature not listed in SYSTEM_DESIGN.md.
You never skip error handling.
You never hardcode a value that belongs in config.py or .env.
```

---

## 2. APPROVED TECHNOLOGY STACK

### Backend — Python
```
fastapi==0.111.0
uvicorn==0.29.0
python-multipart==0.0.9     # required for FastAPI file uploads
opencv-python==4.9.0.80
scikit-image==0.22.0
scikit-learn==1.4.2
mediapipe==0.10.14
numpy==1.26.4
scipy==1.13.0
Pillow==10.3.0
matplotlib==3.8.4
python-dotenv==1.0.1
```

[AGENT NOTE] Do NOT add: torch, tensorflow, keras, transformers, langchain,
celery, redis, sqlalchemy, alembic, or any deep learning library.
The CV pipeline is purely classical. No neural network inference.

### Frontend — JavaScript
```
react@18
vite@5
tailwindcss@3
```

[AGENT NOTE] Do NOT add: redux, zustand, react-query, axios, react-router,
framer-motion, chakra-ui, material-ui, or any component library.
Use only React built-ins + Tailwind utility classes.

---

## 3. FILE NAMING AND STRUCTURE RULES

### Backend
```
backend/
├── main.py               ← FastAPI app instantiation and route definitions ONLY
│                           No business logic in main.py
├── pipeline.py           ← Orchestrator. Calls src/ modules in order. Returns dict.
├── src/
│   ├── config.py         ← ONLY file allowed to contain numeric constants
│   ├── preprocessing.py
│   ├── enhancement.py
│   ├── skin_segmentation.py
│   ├── frequency_decomposition.py
│   ├── lesion_detection.py
│   ├── postprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── requirements.txt
├── .env.example
└── tests/
```

### Frontend
```
frontend/
├── src/
│   ├── App.jsx           ← owns all state and view transitions
│   ├── api.js            ← single analyseImage() function
│   └── components/
│       ├── UploadZone.jsx
│       ├── ProcessingView.jsx
│       ├── ResultsPanel.jsx
│       ├── ComparisonGrid.jsx
│       ├── ConcernBadges.jsx
│       ├── MetricsTable.jsx
│       ├── ErrorView.jsx
│       └── ResetButton.jsx
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

[AGENT NOTE] Never create files outside this structure without explicit instruction.
Never merge two modules into one file. One file = one responsibility.

---

## 4. CODE STYLE RULES

### Python
```python
# Every function must have:
# 1. A type-annotated signature
# 2. A one-line docstring stating what it returns
# 3. At least one inline comment per logical block

# CORRECT
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Returns Intersection over Union between two binary masks."""
    # Flatten to 1D for element-wise comparison
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat   = gt_mask.flatten().astype(bool)
    
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union        = np.logical_or(pred_flat, gt_flat).sum()
    
    # Avoid division by zero when both masks are empty
    if union == 0:
        return 1.0
    return float(intersection / union)

# WRONG — never do this
def iou(a, b):
    return (a & b).sum() / (a | b).sum()
```

### JavaScript / React
```jsx
// Every component must have:
// 1. A JSDoc comment describing props
// 2. PropTypes or clear prop naming that is self-documenting
// 3. No inline styles — Tailwind classes only

// CORRECT
/**
 * Displays a severity badge for a single skin concern.
 * @param {string} concern - "acne" | "redness" | "blemishes"
 * @param {string} severity - "Mild" | "Moderate" | "Severe"
 */
function SeverityBadge({ concern, severity }) {
  const colours = {
    Mild:     "bg-green-100 text-green-800",
    Moderate: "bg-yellow-100 text-yellow-800",
    Severe:   "bg-red-100 text-red-800",
  };
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${colours[severity]}`}>
      {concern}: {severity}
    </span>
  );
}

// WRONG — never do this
const Badge = ({c, s}) => <div style={{color: s === 'Mild' ? 'green' : 'red'}}>{c}</div>
```

---

## 5. CONFIG.PY — THE ONLY SOURCE OF TRUTH FOR NUMBERS

Every numeric constant in the pipeline lives in config.py.
No module is allowed to contain a magic number.

```python
# backend/src/config.py

# --- Image preprocessing ---
RESIZE_TARGET = (512, 512)         # All images normalised to this before processing
TARGET_L_MEAN = 130                # LAB L-channel normalisation target

# --- Quality gate ---
MIN_BRIGHTNESS = 60                # Below this: image too dark
MAX_BRIGHTNESS = 210               # Above this: image overexposed
MIN_BLUR_SCORE = 80                # Laplacian variance below this: too blurry
MAX_IMAGE_SIZE_MB = 10

# --- Enhancement ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# --- Skin segmentation ---
SKIN_CR_MIN, SKIN_CR_MAX = 133, 173    # YCrCb Cr channel range
SKIN_CB_MIN, SKIN_CB_MAX = 77, 127     # YCrCb Cb channel range
SKIN_H_MIN,  SKIN_H_MAX  = 0, 25       # HSV Hue range
SKIN_S_MIN,  SKIN_S_MAX  = 20, 170     # HSV Saturation range
SKIN_V_MIN,  SKIN_V_MAX  = 50, 255     # HSV Value range
ROI_EROSION_PX = 15                     # Pixels to erode from face boundary

# --- Frequency decomposition ---
DOG_SIGMA_1 = 1.0                  # Fine-detail Gaussian
DOG_SIGMA_2 = 3.0                  # Coarse Gaussian; DoG = sigma1 - sigma2
LP_KERNEL_SIZE = (21, 21)          # Low-pass Gaussian kernel
HP_KERNEL_SIZE = (21, 21)          # High-pass = original - LP

# --- Lesion detection ---
REDNESS_A_THRESHOLD = 135          # LAB A-channel: above this = redness
BLEMISH_L_DELTA = 30               # L below (mean_skin_L - delta) = dark spot
ACNE_DOG_THRESHOLD_RATIO = 0.15   # Fraction of DoG max used as threshold
MIN_LESION_AREA_PX = 50            # Contours smaller than this = noise, ignored

# --- Watershed ---
WATERSHED_DIST_THRESHOLD = 0.4    # Fraction of dist max used as sure-fg seed
WATERSHED_MIN_DISTANCE = 10        # Min px between local maxima (spot separation)

# --- Severity thresholds (area as % of skin region) ---
SEVERITY = {
    "acne":    {"Mild": 1.0, "Moderate": 3.0},   # >3 = Severe
    "redness": {"Mild": 5.0, "Moderate": 15.0},
    "blemish": {"Mild": 1.0, "Moderate": 3.0},
}

# --- Output ---
JPEG_QUALITY = 85                  # Base64 output image compression quality
```

[AGENT NOTE] When you need to use a number in any src/ module, import it:
```python
from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
```
Never write a literal number like `2.0` or `(8, 8)` inside a processing function.

---

## 6. ERROR HANDLING RULES

### Backend — every function must handle its own failure
```python
# CORRECT pattern
def enhance_image(image: np.ndarray) -> np.ndarray:
    """Returns CLAHE-enhanced image on LAB L-channel."""
    if image is None or image.size == 0:
        raise ValueError("enhance_image received empty image array")
    
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2RGB)
    except cv2.error as e:
        raise RuntimeError(f"CLAHE enhancement failed: {e}") from e
```

### FastAPI endpoint — centralised exception handling
```python
# In main.py, every exception from pipeline becomes a controlled response
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"status": "quality_error",
                                                   "message": str(exc)})

@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"status": "error",
                        "message": "Processing failed. Please try a different image."})
```

### Frontend — every fetch must handle failure
```javascript
// CORRECT — api.js
export async function analyseImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL}/api/analyse`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      // Return structured error, never throw to caller
      return { status: "error", message: data.message || "Server error" };
    }

    return data;
  } catch (err) {
    // Network failure, CORS issue, JSON parse failure
    return { status: "error", message: "Could not reach the server. Try again." };
  }
}
```

---

## 7. BGR vs RGB — THE MOST COMMON BUG

OpenCV reads images as BGR. Everything else (matplotlib, Pillow, the browser) 
expects RGB. Convert once, at the entry point, and never again.

```python
# In preprocessing.py — the ONLY place BGR→RGB conversion happens
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Returns RGB uint8 numpy array from raw image bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # ← single conversion point
```

[AGENT NOTE] All modules after preprocessing.py receive and return RGB arrays.
Never call cv2.cvtColor(...COLOR_BGR2RGB) anywhere except preprocessing.py.
If you use cv2.imshow() for debugging, convert back to BGR first.

---

## 8. MEDIAPIPE FALLBACK RULE

MediaPipe face detection can fail. The pipeline must not crash when it does.

```python
# In skin_segmentation.py — always implement fallback
def get_face_mask(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Returns (binary face mask, mode string) where mode is 'mediapipe' or 'fallback'."""
    landmarks = _run_mediapipe(image)
    
    if landmarks is None:
        # Fallback: use centre 60% of image as face region
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//5 : 4*h//5, w//5 : 4*w//5] = 255
        return mask, "fallback"
    
    return _landmarks_to_mask(landmarks, image.shape), "mediapipe"
```

The mode string ("mediapipe" or "fallback") must be logged and included in
the API response's processing metadata so the frontend can show a warning
if fallback was used.

---

## 9. IMAGE ENCODING FOR API RESPONSE

All images in the API response are base64-encoded JPEG strings.
This encoding happens ONLY in visualization.py.

```python
# In visualization.py
import base64
import cv2
import numpy as np
from src.config import JPEG_QUALITY

def encode_image_to_base64(image_rgb: np.ndarray) -> str:
    """Returns base64 JPEG string from RGB numpy array."""
    # Convert to BGR for OpenCV encoding
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buffer = cv2.imencode(".jpg", bgr, encode_params)
    return base64.b64encode(buffer).decode("utf-8")
```

[AGENT NOTE] The frontend uses these strings directly in <img> tags:
```jsx
<img src={`data:image/jpeg;base64,${data.images.improved_annotated}`} />
```
Never return image file paths or URLs. Always base64.

---

## 10. WHAT TO DO WHEN SOMETHING IS UNCLEAR

In order of preference:

1. Check 01_SYSTEM_DESIGN.md DECISION LOG
2. Check this file for a relevant rule
3. Choose the simpler of two options
4. Leave a TODO comment with exact question: `# TODO: confirm whether X or Y`
5. Never invent an architecture decision — flag it instead

---

## 11. TESTING APPROACH

No formal test framework is set up. Instead, each module has a
`if __name__ == "__main__":` block that tests it on a single image.

```python
# At the bottom of every src/ module:
if __name__ == "__main__":
    import sys
    from src.preprocessing import load_image_from_bytes
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    with open(image_path, "rb") as f:
        image = load_image_from_bytes(f.read())
    
    result = enhance_image(image)           # ← test this module's main function
    print(f"Output shape: {result.shape}")  # ← confirm output is valid
    print("enhancement.py: OK")
```

Run individual module tests before running the full pipeline:
```bash
python -m backend.src.enhancement path/to/test_image.jpg
```

---

## 12. COMMIT MESSAGE FORMAT

```
feat: add CLAHE enhancement to enhancement.py
fix: handle mediapipe fallback when no face detected
refactor: move all numeric constants to config.py
docs: update agent instructions with BGR/RGB rule
test: add module-level test for lesion_detection.py
```

One type prefix. One sentence. No emoji.
