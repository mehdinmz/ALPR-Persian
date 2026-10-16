
import io
import os
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import uvicorn

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from hezar.models import Model
from hezar.preprocessors import preprocessor


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "crnn-fa-license-plate-recognition-v2",
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-service")


# ============================================================
# MODEL LOADING
# ============================================================

ocr_model = None
ocr_preprocessor = None
ocr_device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_model, ocr_preprocessor, ocr_device

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"OCR model directory not found: {MODEL_PATH}"
        )

    logger.info("Loading Hezar OCR model from: %s", MODEL_PATH)

    try:
        ocr_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("OCR device: %s", ocr_device)

        ocr_model = Model.load(
            MODEL_PATH,
            load_locally=True,
        )

        ocr_preprocessor = preprocessor.Preprocessor.load(
            subfolder="preprocessor",
            hub_or_local_path=MODEL_PATH,
        )

        logger.info("Hezar OCR model loaded successfully.")

    except Exception as exc:
        logger.exception("Failed to load OCR model")
        raise RuntimeError(
            f"Failed to load OCR model: {exc}"
        ) from exc

    yield

    ocr_model = None
    ocr_preprocessor = None

    logger.info("OCR model unloaded.")


app = FastAPI(
    title="Plate OCR Service",
    version="2.1.0",
    lifespan=lifespan,
)


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class OCRResponse(BaseModel):
    text: str


# ============================================================
# OCR INFERENCE
# Matches the working test pipeline's Hezar invocation.
# The detection service is responsible for cropping the plate.
# ============================================================

def recognize_plate(image_bytes: bytes) -> str:
    if ocr_model is None or ocr_preprocessor is None:
        raise RuntimeError("OCR model or preprocessor is not loaded.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image_array = np.array(image)

        # The test pipeline passes OpenCV BGR images to Hezar.
        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR,
        )

        results = ocr_model.predict(
            preprocess=ocr_preprocessor,
            device=ocr_device,
            inputs=image_array,
        )

        if not results:
            raise RuntimeError(
                "OCR model returned no predictions."
            )

        text = results[0].text

        if text is None:
            raise RuntimeError(
                "OCR model returned empty text."
            )

        return str(text).strip()

    except Exception as exc:
        logger.exception("OCR inference failed")
        raise RuntimeError(
            f"OCR inference failed: {exc}"
        ) from exc


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": (
            "ok"
            if ocr_model is not None
            and ocr_preprocessor is not None
            else "loading"
        ),
        "service": "ocr",
        "device": ocr_device,
    }


@app.post("/predict", response_model=OCRResponse)
async def predict(file: UploadFile = File(...)):
    if ocr_model is None or ocr_preprocessor is None:
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

    # Validate that the uploaded file is a readable image.
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
        logger.error("OCR prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="OCR inference failed.",
        ) from exc


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
    )
