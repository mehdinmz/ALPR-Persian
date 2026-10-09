
import io
import os
import logging
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import uvicorn


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_crnn_model_2.keras")

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50

# Must match the vocabulary used during training.
VOCABULARY = list("0123456789") + [
    "A", "B", "C", "D", "H", "J", "K", "L",
    "M", "N", "S", "T", "V", "X", "Y", "Z",
]

# StringLookup assigned IDs 1..26 to characters.
# ID 0 is reserved for [UNK].
ID_TO_TOKEN = {
    index + 1: token
    for index, token in enumerate(VOCABULARY)
}

NUM_CLASSES = len(VOCABULARY) + 2  # 28, including UNK and CTC blank
BLANK_ID = NUM_CLASSES - 1         # 27

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-service")


# ============================================================
# MODEL LOADING
# ============================================================


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # During inference, return the model's predictions.
        return y_pred


prediction_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_model

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"OCR model not found: {MODEL_PATH}")

    logger.info("Loading OCR model...")

    try:
        training_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"CTCLayer": CTCLayer},
        )

        # The training model includes a CTC loss layer.
        # For inference, use the character probabilities from dense2.
        prediction_model = tf.keras.Model(
            inputs=training_model.inputs[0],
            outputs=training_model.get_layer("dense2").output,
        )

        if prediction_model.input_shape[1:] != (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            1,
        ):
            raise RuntimeError(
                f"Unexpected input shape: {prediction_model.input_shape}"
            )

        if prediction_model.output_shape[-1] != NUM_CLASSES:
            raise RuntimeError(
                f"Unexpected output classes: "
                f"{prediction_model.output_shape[-1]} "
                f"(expected {NUM_CLASSES})"
            )

        logger.info("OCR model loaded successfully.")
        logger.info("Input shape: %s", prediction_model.input_shape)
        logger.info("Output shape: %s", prediction_model.output_shape)

    except Exception as exc:
        logger.exception("Failed to load OCR model")
        raise RuntimeError(f"Failed to load OCR model: {exc}") from exc

    yield

    prediction_model = None


app = FastAPI(
    title="Plate OCR Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class OCRResponse(BaseModel):
    text: str


# ============================================================
# IMAGE PREPROCESSING
# Must match the notebook's preprocessing pipeline.
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=1,
            expand_animations=False,
        )

        # Same normalization used during training.
        image = tf.image.convert_image_dtype(image, tf.float32)

        # TensorFlow resize: [height, width].
        image = tf.image.resize(
            image,
            [IMAGE_HEIGHT, IMAGE_WIDTH],
        )

        # Training transposed the image after resizing.
        image = tf.transpose(image, perm=[1, 0, 2])

        # (200, 50, 1) -> (1, 200, 50, 1)
        image = tf.expand_dims(image, axis=0)

        return image.numpy()

    except Exception as exc:
        raise ValueError(f"Could not preprocess image: {exc}") from exc


# ============================================================
# CTC DECODING
# ============================================================

def decode_prediction(prediction: np.ndarray) -> str:
    """
    prediction shape: (1, time_steps, num_classes)
    """

    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(
            f"Unexpected prediction shape: {prediction.shape}"
        )

    input_length = np.full(
        shape=(prediction.shape[0],),
        fill_value=prediction.shape[1],
        dtype=np.int32,
    )

    decoded, _ = tf.keras.backend.ctc_decode(
        prediction,
        input_length=input_length,
        greedy=True,
        # The model's CTC blank is the final class, ID 27.
        # Depending on the TensorFlow/Keras version, ctc_decode
        # uses its default blank index, which is normally the last class.
    )

    class_ids = decoded[0].numpy()[0]

    tokens = []
    for class_id in class_ids:
        class_id = int(class_id)

        # Ignore padding/unknown IDs and any non-character classes.
        token = ID_TO_TOKEN.get(class_id)
        if token is not None:
            tokens.append(token)

    return "".join(tokens)


# ============================================================
# OCR INFERENCE
# ============================================================

def recognize_plate(image_bytes: bytes) -> str:
    if prediction_model is None:
        raise RuntimeError("OCR model is not loaded.")

    image = preprocess_image(image_bytes)

    prediction = prediction_model.predict(
        image,
        verbose=0,
    )

    return decode_prediction(prediction)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if prediction_model is not None else "loading",
        "service": "ocr",
    }


@app.post("/predict", response_model=OCRResponse)
async def predict(file: UploadFile = File(...)):
    if prediction_model is None:
        raise HTTPException(
            status_code=503,
            detail="OCR model is not loaded.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    # Validate that the uploaded content is an actual image.
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image.",
        )

    try:
        text = recognize_plate(image_bytes)
        return OCRResponse(text=text)

    except Exception as exc:
        logger.exception("OCR inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"OCR inference failed: {exc}",
        ) from exc
if __name__ == "__main__":
 
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
    )