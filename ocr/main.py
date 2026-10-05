import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "best_crnn_model.keras"
)


# ============================================================
# VOCABULARY
# ============================================================

DIGITS = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

LETTERS = [
    "D",
    "S",
    "alef",
    "beh",
    "dal",
    "ein",
    "feh",
    "gaf",
    "ghaf",
    "he",
    "jim",
    "kaf",
    "lam",
    "mim",
    "noon",
    "peh",
    "sad",
    "seh",
    "shin",
    "sin",
    "ta",
    "teh",
    "vav",
    "ye",
    "zeh",
    "zhe",
]

VOCABULARY = DIGITS + LETTERS

TOKEN_TO_ID = {
    token: index + 1
    for index, token in enumerate(VOCABULARY)
}

ID_TO_TOKEN = {
    index + 1: token
    for index, token in enumerate(VOCABULARY)
}


# ============================================================
# SPECIAL TOKENS
# ============================================================

BLANK_ID = 0
UNKNOWN_ID = len(VOCABULARY) + 1

NUM_CLASSES = len(VOCABULARY) + 2

INPUT_HEIGHT = 32
INPUT_WIDTH = 160


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Persian Plate OCR",
    version="1.0.0",
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading OCR model...")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"OCR model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print(f"Model path : {MODEL_PATH}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Vocabulary: {len(VOCABULARY)}")
print(f"Blank ID: {BLANK_ID}")
print(f"Unknown ID: {UNKNOWN_ID}")
print(f"Total classes: {NUM_CLASSES}")

print("=" * 60)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocessing must match training:

        1. Read image
        2. Convert to grayscale
        3. Resize to 32x160
        4. Convert to float32
        5. Normalize to [0, 1]
        6. Add channel dimension
        7. Add batch dimension
    """

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Grayscale
        image = image.convert("L")

        # Training used:
        # tf.image.resize(image, [32, 160])
        image = image.resize(
            (INPUT_WIDTH, INPUT_HEIGHT),
            Image.Resampling.BILINEAR,
        )

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        # Same normalization as:
        # tf.image.convert_image_dtype(...)
        image /= 255.0

        # (32, 160) -> (32, 160, 1)
        image = np.expand_dims(
            image,
            axis=-1,
        )

        # (32, 160, 1) -> (1, 32, 160, 1)
        image = np.expand_dims(
            image,
            axis=0,
        )

        return image

    except Exception as exc:
        raise ValueError(
            f"Image preprocessing failed: {exc}"
        ) from exc


# ============================================================
# CTC DECODER
# ============================================================

def decode_prediction(
    prediction: np.ndarray,
) -> str:
    """
    Greedy CTC decoding.

    Expected prediction shape:

        (batch, time_steps, num_classes)

    For this model:

        (1, 40, 37)
    """

    input_length = np.array(
        [prediction.shape[1]],
        dtype=np.int32,
    )

    decoded, _ = tf.keras.backend.ctc_decode(
        prediction,
        input_length=input_length,
        greedy=True,
    )

    decoded = decoded[0].numpy()[0]

    tokens = []

    for class_id in decoded:

        class_id = int(class_id)

        # CTC blank
        if class_id == BLANK_ID:
            continue

        # Unknown character
        if class_id == UNKNOWN_ID:
            tokens.append("?")
            continue

        # Normal vocabulary token
        token = ID_TO_TOKEN.get(class_id)

        if token is not None:
            tokens.append(token)

    return "".join(tokens)


# ============================================================
# OCR INFERENCE
# ============================================================

def recognize_plate(
    image_bytes: bytes,
) -> str:

    image = preprocess_image(
        image_bytes
    )

    prediction = model.predict(
        image,
        verbose=0,
    )

    text = decode_prediction(
        prediction
    )

    return text


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "best_crnn_model.keras",
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Persian Plate OCR",
        "status": "running",
    }


# ============================================================
# RECOGNIZE
# ============================================================

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
):
    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image.",
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        text = recognize_plate(
            image_bytes
        )

        return {
            "text": text,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OCR inference failed: {exc}",
        ) from exc
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002
    )