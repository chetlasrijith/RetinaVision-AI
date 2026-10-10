"""Retinal fundus image classification API.

Serves the existing local Keras student model (``server/stundent_model.keras``).
No retraining, no model edits. Preprocessing (RGB, 224x224, /255, batch dim)
matches the original demo exactly.

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "stundent_model.keras"),
)
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
MAX_FILE_MB = 15
MIN_SIDE_PX = 64

CLASS_NAMES = [
    "AMD",                      # 0
    "Cataract",                 # 1
    "DR",                       # 2
    "Glaucoma",                 # 3
    "Hypertensive_Retinopathy",  # 4
    "Normal_Fundus",            # 5
    "Pathological_Myopia",      # 6
]

DISPLAY_NAMES = {
    "AMD": "Age-related macular degeneration (AMD)",
    "Cataract": "Cataract",
    "DR": "Diabetic retinopathy (DR)",
    "Glaucoma": "Glaucoma",
    "Hypertensive_Retinopathy": "Hypertensive retinopathy",
    "Normal_Fundus": "Normal fundus (no disease detected)",
    "Pathological_Myopia": "Pathological myopia",
}

EXPECTED_CLASSES = len(CLASS_NAMES)


class ClassScore(BaseModel):
    key: str
    label: str
    score: float


class PredictionResponse(BaseModel):
    predicted_class: str
    display_label: str
    confidence: float
    probs_verified: bool
    scores: list[ClassScore]
    margin_pp: float
    image_warnings: list[str] = []


app = FastAPI(title="Eye Disease Detection API", version="1.0.0")

origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Model (loaded once, lazily; tensorflow import is lazy for testability)
# ----------------------------------------------------------------------------

_MODEL = None


def get_model():
    """Load the Keras model on first use and verify its 7-class output."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install server dependencies with "
            "`pip install -r server/requirements.txt`."
        ) from exc
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    try:
        n_out = int(model.output_shape[-1])
    except Exception:
        n_out = -1
    if n_out != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Unexpected model output shape {getattr(model, 'output_shape', '?')}: "
            f"expected {EXPECTED_CLASSES} class scores."
        )
    _MODEL = model
    return _MODEL


def preprocess_image(image: Image.Image) -> np.ndarray:
    """RGB -> 224x224 -> float32 in the training range (0-255) -> batch dim."""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def is_probability_vector(vec: np.ndarray) -> bool:
    total = float(np.sum(vec))
    return bool(
        np.all(vec >= -1e-6) and np.all(vec <= 1.0 + 1e-6) and abs(total - 1.0) < 1e-3
    )


def validate_image(filename: str, data: bytes) -> tuple[Image.Image, list[str]]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type “.{ext or '?'}”. Please upload a JPG, JPEG, or PNG image.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File is {len(data) / 1024 / 1024:.1f} MB — please upload an image under {MAX_FILE_MB} MB.",
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read as an image. It may be corrupt or not a real "
            "JPG/PNG — please try a different file.",
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="This file could not be decoded. Please try a different JPG or PNG image.",
        )
    warnings: list[str] = []
    w, h = image.size
    if min(w, h) < MIN_SIDE_PX:
        warnings.append(
            f"Image is very small ({w}×{h}px). Results may be unreliable — "
            "a larger, well-lit fundus photo works best."
        )
    return image, warnings


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded, "classes": EXPECTED_CLASSES}


@app.get("/classes")
def classes() -> dict:
    return {
        "classes": [
            {"key": k, "label": DISPLAY_NAMES[k]} for k in CLASS_NAMES
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    data = file.file.read()
    image, warnings = validate_image(file.filename or "upload", data)
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        raw = model.predict(preprocess_image(image), verbose=0)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}. Please try a different image.",
        )
    vec = np.asarray(raw[0], dtype=np.float64).flatten()
    if vec.shape[0] != EXPECTED_CLASSES:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned {vec.shape[0]} scores; expected {EXPECTED_CLASSES}.",
        )
    verified = is_probability_vector(vec)
    total = float(np.sum(vec))
    shares = vec / total if total > 0 else vec
    order = np.argsort(shares)[::-1]
    scores = [
        ClassScore(
            key=CLASS_NAMES[i],
            label=DISPLAY_NAMES[CLASS_NAMES[i]],
            score=round(float(shares[i]), 4),
        )
        for i in order
    ]
    margin_pp = round(float((shares[order[0]] - shares[order[1]]) * 100), 2)
    top = CLASS_NAMES[int(order[0])]
    return PredictionResponse(
        predicted_class=top,
        display_label=DISPLAY_NAMES[top],
        confidence=round(float(shares[order[0]]), 4),
        probs_verified=verified,
        scores=scores,
        margin_pp=margin_pp,
        image_warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
