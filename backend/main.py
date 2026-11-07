"""Bloom API: FastAPI routes + upload validation. Business logic lives in pipeline.py."""
import logging
import os

# Keep MediaPipe/TFLite C++ logs quiet (must be set before importing MediaPipe).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import auth_store
import pipeline
import routine
from ml_inference import detect, model_status
from src import enhancement, preprocessing, skin_segmentation
from src.config import MAX_IMAGE_SIZE_MB, RATE_LIMIT
from src.errors import FaceLandmarkError, InvalidImageError
from src.image_validation import inspect_image

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("bloom")

MAX_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
# Trim whitespace and ignore empty entries (e.g. "https://a, https://b,").
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8080"
).split(",") if origin.strip()]
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAGIC_BYTES = {"image/jpeg": b"\xff\xd8\xff", "image/png": b"\x89PNG\r\n\x1a\n"}
REQUIRED_ANGLES = {"front", "left", "right"}
VALID_ANGLES = REQUIRED_ANGLES
SEVERITY_RANK = {"Clear": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
                 "Minimal variation": 0, "Low variation": 1, "Moderate variation": 2, "High variation": 3}
FRONT_YAW_MAX = 0.18
SIDE_YAW_MIN = 0.14

app = FastAPI(title="Bloom API", version="1.0")
auth_store.initialise()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # explicit list, never "*" in production
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# In-process rate limiting: per worker, reset on restart. This deployment is scoped
# to a single Render web service process; use a shared store for multi-instance use.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class RegisterPayload(BaseModel):
    email: str
    password: str
    security_question: str
    security_answer: str


class LoginPayload(BaseModel):
    email: str
    password: str


class ResetPasswordPayload(BaseModel):
    email: str
    security_answer: str
    new_password: str


class ScanRecordPayload(BaseModel):
    scan_type: str = "single_photo"
    concerns: dict
    processing_time_sec: float | None = None


def current_user(authorization: str | None = Header(None)) -> dict:
    """Require the signed bearer token for account-private routes."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to access your private scan history.")
    user = auth_store.user_from_token(authorization.removeprefix("Bearer ").strip())
    if not user:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.")
    return user


# --- Upload validation (content-type -> size -> magic bytes -> dimensions -> decode) ---
def validate_content_type(file: UploadFile) -> None:
    """Raises ValueError if the declared content-type is not JPEG/PNG."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(f"Unsupported file type: {file.content_type}. Upload JPEG or PNG only.")


async def read_image_safely(file: UploadFile) -> bytes:
    """Returns file bytes, raising ValueError if over the size limit."""
    content = await file.read(MAX_SIZE_BYTES + 1)
    if len(content) > MAX_SIZE_BYTES:
        raise InvalidImageError(f"File too large. Maximum size is {MAX_IMAGE_SIZE_MB}MB.")
    return content


def verify_file_signature(content: bytes, content_type: str | None = None) -> None:
    """Raises ValueError if the bytes don't match a JPEG/PNG signature."""
    if not any(content.startswith(signature) for signature in MAGIC_BYTES.values()) or (
        content_type is not None and not content.startswith(MAGIC_BYTES.get(content_type, b"\x00"))
    ):
        raise InvalidImageError("File content does not match an allowed image format.")


def make_concern_json(concern: dict) -> dict:
    """Returns the API concern shape (count omitted for redness)."""
    out = {
        "severity": concern["severity"],
        "area_pct": round(float(concern["area_pct"]), 2),
        "extent_score": float(concern["extent_score"]),
        "visual_estimate_only": True,
    }
    if concern["count"] != -1:
        out["count"] = int(concern["count"])
    return out


def build_success(result: dict) -> dict:
    """Returns the HTTP 200 response body from a PipelineOutput."""
    response = {
        "status": "success",
        "processing_time_sec": result["processing_time_sec"],
        "quality": {"passed": True, "issues": result["quality"]["issues"]},
        "detection": result["detection"],
        "images": result["images"],
        "concerns": {
            "surface_anomalies": make_concern_json(result["concerns"].get("surface_anomalies", result["concerns"].get("acne"))),
            "redness_contrast": make_concern_json(result["concerns"].get("redness_contrast", result["concerns"].get("redness"))),
            "dark_regions": make_concern_json(result["concerns"].get("dark_regions", result["concerns"].get("blemish"))),
        },
        "metrics": None,
        "metrics_note": (
            "Per-image accuracy is not computed for unlabelled uploads. "
            "See the offline evaluation workflow for independent ground-truth metrics."
        ),
        "mediapipe_mode": result["mediapipe_mode"],
        "skin_tone": result.get("skin_tone"),
        "imperfection_map": result.get("imperfection_map"),
        "stage_times_sec": result.get("stage_times_sec", {}),
    }
    if "ml" in result:
        response["ml"] = result["ml"]
    return response


def _best_angle_concerns(results: list[dict]) -> dict:
    """Use the strongest finding seen from one angle, never sum overlap.

    Adjacent face views often contain the same skin area.  Choosing the most
    affected angle per concern avoids presenting duplicate regions as a larger
    total.  Individual angle images remain available for review.
    """
    combined = {}
    for concern in results[0]["concerns"]:
        combined[concern] = max(
            (result["concerns"][concern] for result in results),
            key=lambda item: (SEVERITY_RANK[item["severity"]], item["area_pct"]),
        )
    return combined


def _angle_result(angle: str, result: dict) -> dict:
    """Return the useful, reviewable part of one successful angle scan."""
    response = build_success(result)
    return {
        "angle": angle,
        "concerns": response["concerns"],
        "images": response["images"],
        "imperfection_map": response["imperfection_map"],
        "stage_times_sec": response["stage_times_sec"],
        "processing_time_sec": response["processing_time_sec"],
    }


# --- Routes ---

@app.get("/")
async def root() -> dict:
    """Returns API name and version."""
    return {"name": "Bloom API", "version": "1.0"}


@app.get("/api/health")
async def health() -> dict:
    """Health check used by Render. Returns whether ML model is loaded."""
    return {"status": "ok", "ml": model_status()}


@app.post("/api/auth/register")
@limiter.limit("5/minute")
async def register(request: Request, payload: RegisterPayload) -> dict:
    user = auth_store.create_user(
        payload.email, payload.password, payload.security_question, payload.security_answer,
    )
    return {"user": user, "token": auth_store.issue_token(user)}


@app.post("/api/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, payload: LoginPayload) -> dict:
    user = auth_store.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return {"user": user, "token": auth_store.issue_token(user)}


@app.get("/api/auth/recovery-question")
@limiter.limit("5/minute")
async def recovery_question(request: Request, email: str = Query(...)) -> dict:
    question = auth_store.security_question(email)
    # Keep response shape and status the same for known and unknown emails.
    return {"security_question": question or "If an account exists, its recovery question will be shown here."}


@app.post("/api/auth/reset-password")
@limiter.limit("5/minute")
async def reset_password(request: Request, payload: ResetPasswordPayload) -> dict:
    if not auth_store.reset_password(payload.email, payload.security_answer, payload.new_password):
        raise HTTPException(status_code=400, detail="The account details could not be verified.")
    return {"status": "success"}


@app.get("/api/auth/me")
async def me(user: dict = Depends(current_user)) -> dict:
    return {"user": user}


@app.get("/api/history")
async def history(user: dict = Depends(current_user)) -> dict:
    return {"records": auth_store.get_history(user["id"])}


@app.post("/api/history")
async def create_history(payload: ScanRecordPayload, user: dict = Depends(current_user)) -> dict:
    auth_store.save_history(user["id"], payload.model_dump())
    return {"records": auth_store.get_history(user["id"])}


@app.delete("/api/history/{record_id}")
async def delete_history(record_id: int, user: dict = Depends(current_user)) -> dict:
    if not auth_store.delete_history_entry(user["id"], record_id):
        raise HTTPException(status_code=404, detail="That scan record was not found.")
    return {"records": auth_store.get_history(user["id"])}


@app.post("/api/analyse")
@limiter.limit(RATE_LIMIT)
async def analyse(
    request: Request,
    file: UploadFile = File(...),
    pipeline_mode: str = Query("classical", alias="pipeline"),
) -> JSONResponse:
    """Accepts an image and returns the full analysis as JSON.

    ``classical`` is the default. ``hybrid`` adds the optional, acne-only ML
    detector when a trained checkpoint is installed. ``ml`` is not exposed as
    a standalone mode because it would discard the validated image/face gates.
    """
    if pipeline_mode not in {"classical", "hybrid"}:
        return JSONResponse(status_code=422, content={
            "status": "invalid_pipeline",
            "message": "pipeline must be classical or hybrid.",
        })
    validate_content_type(file)
    content = await read_image_safely(file)
    verify_file_signature(content, file.content_type)
    inspect_image(content)

    # --- Classical pipeline ---
    stages = pipeline.run(content)

    if not stages.get("quality", {}).get("passed", False):
        return JSONResponse(status_code=422, content={
            "status": "quality_error",
            "quality": {
                "passed": False,
                "issues": stages["quality"]["issues"],
                "reason_codes": stages["quality"].get("reason_codes", []),
            },
        })

    logger.info(
        "Analysed image in %ss (model=%s)",
        stages.get("processing_time_sec", 0),
        stages.get("detection", {}).get("model", "mediapipe"),
    )

    if pipeline_mode == "classical":
        return JSONResponse(status_code=200, content=build_success(stages))

    # Hybrid: run the optional, single-class acne detector on the same image.
    ml = model_status()
    if ml["available"]:
        import cv2
        import numpy as np
        nparr = np.frombuffer(content, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ml_detections = detect(bgr_img) if bgr_img is not None else []
    else:
        ml_detections = []
    return JSONResponse(status_code=200, content=build_success({
        **stages,
        "ml": {
            "detections": ml_detections,
            "available": ml["available"],
            "scope": ml["scope"],
        },
    }))


@app.post("/api/analyse/multi")
@limiter.limit(RATE_LIMIT)
async def analyse_multi(
    request: Request,
    files: list[UploadFile] = File(...),
    angles: list[str] = Form(...),
) -> JSONResponse:
    """Analyse guided face angles and return a combined, reviewable scan.

    A front, left, and right image are required.
    Each image goes through the full pipeline independently so its annotated
    output can be inspected; the summary uses the strongest single angle for
    each concern rather than adding potentially overlapping regions.
    """
    if len(files) != len(angles) or not files:
        return JSONResponse(status_code=422, content={
            "status": "invalid_request",
            "message": "Send one angle label for every image.",
        })
    if len(files) > len(VALID_ANGLES) or len(set(angles)) != len(angles):
        return JSONResponse(status_code=422, content={
            "status": "invalid_request",
            "message": "Each guided angle can be uploaded once.",
        })
    if any(angle not in VALID_ANGLES for angle in angles) or not REQUIRED_ANGLES.issubset(angles):
        return JSONResponse(status_code=422, content={
            "status": "invalid_request",
            "message": "Front, left, and right photos are required.",
        })

    processed: list[tuple[str, dict]] = []
    for angle, file in zip(angles, files):
        validate_content_type(file)
        content = await read_image_safely(file)
        verify_file_signature(content, file.content_type)
        inspect_image(content)
        result = pipeline.run(content)
        if not result.get("quality", {}).get("passed", False):
            return JSONResponse(status_code=422, content={
                "status": "quality_error",
                "angle": angle,
                "quality": {
                    "passed": False,
                    "issues": result["quality"]["issues"],
                    "reason_codes": result["quality"].get("reason_codes", []),
                },
                "message": f"The {angle} photo needs to be retaken before the multi-angle scan can continue.",
            })
        processed.append((angle, result))

    by_angle = dict(processed)
    combined_concerns = _best_angle_concerns([result for _, result in processed])
    front_result = by_angle["front"]
    combined = build_success({
        **front_result,
        "concerns": combined_concerns,
        "processing_time_sec": round(sum(result["processing_time_sec"] for _, result in processed), 2),
    })
    combined["scan_type"] = "multi_angle"
    combined["angles_analyzed"] = angles
    combined["aggregation_note"] = (
        "Each concern reflects the strongest single angle to avoid double-counting overlapping skin areas."
    )
    combined["angle_results"] = [_angle_result(angle, result) for angle, result in processed]
    return JSONResponse(status_code=200, content=combined)


@app.post("/api/capture-check")
async def capture_check(
    file: UploadFile = File(...),
    angle: str = Form(...),
    reference_yaw: float | None = Form(None),
) -> JSONResponse:
    """Gate one guided camera frame before it is kept for multi-angle analysis."""
    if angle not in VALID_ANGLES:
        return JSONResponse(status_code=422, content={
            "accepted": False,
            "message": "Choose a valid guided camera angle.",
        })

    validate_content_type(file)
    content = await read_image_safely(file)
    verify_file_signature(content, file.content_type)
    inspect_image(content)
    image, quality = preprocessing.run(content)
    if not quality["passed"]:
        return JSONResponse(status_code=200, content={
            "accepted": False,
            "message": quality["issues"][0] if quality["issues"] else "Hold the camera steady and improve the lighting.",
        })

    try:
        enhanced = enhancement.run(image)["enhanced"]
        landmarks = skin_segmentation._run_face_mesh(enhanced)
    except Exception:
        logger.exception("Guided camera face check failed")
        return JSONResponse(status_code=200, content={
            "accepted": False,
            "message": "We could not verify the face. Centre your face and try again.",
        })
    if landmarks is None:
        return JSONResponse(status_code=200, content={
            "accepted": False,
            "message": "Keep one unobstructed face centred in the frame.",
        })

    yaw = skin_segmentation.estimate_yaw(landmarks)
    if angle == "front" and abs(yaw) > FRONT_YAW_MAX:
        return JSONResponse(status_code=200, content={
            "accepted": False,
            "message": "Face the camera more directly before we capture.",
            "yaw": yaw,
        })
    if angle == "left" and abs(yaw) < SIDE_YAW_MIN:
        return JSONResponse(status_code=200, content={
            "accepted": False,
            "message": "Turn a little more to either side of the preview.",
            "yaw": yaw,
        })
    if angle == "right":
        if abs(yaw) < SIDE_YAW_MIN:
            return JSONResponse(status_code=200, content={
                "accepted": False,
                "message": "Turn a little more to the other side of the preview.",
                "yaw": yaw,
            })
        if reference_yaw is not None and yaw * reference_yaw >= 0:
            return JSONResponse(status_code=200, content={
                "accepted": False,
                "message": "Turn to the opposite side from the previous side-view capture.",
                "yaw": yaw,
            })

    return JSONResponse(status_code=200, content={
        "accepted": True,
        "message": "Angle accepted.",
        "yaw": yaw,
    })


@app.post("/api/routine")
@limiter.limit(RATE_LIMIT)
async def get_routine(request: Request, concerns: dict) -> dict:
    """Returns a personalized skincare routine based on detected concerns."""
    return routine.build_routine(concerns)


# --- Centralised error handling (generic to client, detail to logs) ---
@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError) -> JSONResponse:
    """Maps decode/format/size validation failures to HTTP 422."""
    return JSONResponse(status_code=422, content={
        "status": "invalid_image",
        "message": str(exc),
    })


@app.exception_handler(FaceLandmarkError)
async def landmark_error_handler(request: Request, exc: FaceLandmarkError) -> JSONResponse:
    """Maps strict-mode face/landmark failures to HTTP 422."""
    return JSONResponse(status_code=422, content={
        "status": "face_error",
        "message": str(exc),
        "reason_code": exc.reason_code,
        "detection": {
            "face_detected": False,
            "landmarks_available": False,
            "fallback_used": False,
        },
    })


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Maps validation/quality ValueErrors to HTTP 422."""
    return JSONResponse(status_code=422, content={
        "status": "quality_error",
        "quality": {"passed": False, "issues": [str(exc)], "reason_codes": []},
        "message": str(exc),
    })


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Maps malformed requests (e.g. missing file) to HTTP 422."""
    return JSONResponse(status_code=422, content={
        "status": "error",
        "message": "Invalid request. Send an image file as multipart/form-data.",
    })


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Logs full detail, returns a generic message to the client."""
    logger.error("Unhandled error: %s: %s", type(exc).__name__, exc, exc_info=True)
    return JSONResponse(status_code=500, content={
        "status": "error",
        "message": "Processing failed. Please try a different image.",
    })
