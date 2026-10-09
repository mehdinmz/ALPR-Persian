import asyncio
import json
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import cv2
import httpx
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# ============================================================
# CONFIG
# ============================================================

DETECTION_URL = "http://127.0.0.1:8001"
OCR_URL = "http://127.0.0.1:8002"

DETECTION_TIMEOUT = 30.0
OCR_TIMEOUT = 10.0

MAX_STREAM_RESULTS = 100
DEFAULT_PROCESS_FPS = 5.0

RTSP_RECONNECT_DELAY = 3.0
MAX_RTSP_RECONNECT_ATTEMPTS = 10

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
}

CONFIDENCE_THRESHOLD = 0.60  # حداقل آستانه پذیرش (۶۰٪)


# ============================================================
# GLOBAL HTTP CLIENT
# ============================================================

http_client: Optional[httpx.AsyncClient] = None


# ============================================================
# STREAM STATE
# ============================================================

class StreamState:
    def __init__(
        self,
        stream_id: str,
        source: str,
        process_fps: float,
    ):
        self.stream_id = stream_id
        self.source = source
        self.process_fps = process_fps

        self.running = True
        self.status = "starting"

        self.task: Optional[asyncio.Task] = None

        self.results = deque(maxlen=MAX_STREAM_RESULTS)

        self.frames_read = 0
        self.frames_processed = 0

        self.started_at = None
        self.last_frame_at = None

        self.error = None

        self.latest_annotated_frame: Optional[bytes] = None

        # حافظه نگهداری اطلاعات اختصاصی هر پلاک براساس track_id
        # { track_id: {"confidence": float, "text": str, "bbox": list} }
        self.tracked_plates: Dict[int, dict] = {}


streams: dict[str, StreamState] = {}
stream_lock = asyncio.Lock()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client

    streams.clear()

    http_client = httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(
            connect=10.0,
            read=DETECTION_TIMEOUT,
            write=DETECTION_TIMEOUT,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
    )

    yield

    for stream in list(streams.values()):
        stream.running = False

    tasks = [stream.task for stream in streams.values() if stream.task]
    if tasks:
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    if http_client:
        await http_client.aclose()

    http_client = None


app = FastAPI(
    title="ALPR Gateway",
    description="Production Gateway / Orchestrator for Persian ALPR",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# MODELS
# ============================================================

class StreamRequest(BaseModel):
    url: str
    process_fps: float = DEFAULT_PROCESS_FPS


# ============================================================
# HELPERS
# ============================================================

def get_http_client() -> httpx.AsyncClient:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")
    return http_client


def validate_process_fps(process_fps: float) -> float:
    if process_fps <= 0:
        return DEFAULT_PROCESS_FPS
    return min(float(process_fps), 30.0)


def extract_boundary(content_type: str) -> Optional[str]:
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            return part.split("=", 1)[1].strip('"')
    return None


def draw_annotations(frame_np: np.ndarray, detections: list) -> bytes:
    """
    رسم Bounding Box و متن OCR (بدون نمایش plate_id)
    """
    annotated = frame_np.copy()

    for det in detections:
        bbox = det.get("bbox")
        text = det.get("text") or ""
        conf = det.get("confidence", 0.0)

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)

            # رسم مستطیل سبز دور پلاک
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # برچسب شامل متن OCR و میزان کانفیدنس
            label = f"{text} ({conf:.2f})" if text else f"({conf:.2f})"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # پس‌زمینه برچسب
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 5, y1),
                (0, 255, 0),
                -1,
            )

            # متن اصلی پلاک
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    success, encoded_image = cv2.imencode(
        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if success:
        return encoded_image.tobytes()
    return None

# ============================================================
# DETECTION RESPONSE PARSER
# ============================================================

def parse_detection_multipart(response: httpx.Response):
    content_type = response.headers.get("content-type", "")

    if "multipart/mixed" not in content_type:
        raise HTTPException(
            status_code=502,
            detail=f"Detection returned an invalid response format: {content_type}",
        )

    boundary = extract_boundary(content_type)
    if not boundary:
        raise HTTPException(
            status_code=502,
            detail="Detection response has no multipart boundary.",
        )

    boundary_bytes = f"--{boundary}".encode("utf-8")
    parts = response.content.split(boundary_bytes)

    metadata = None
    crops = {}

    for part in parts:
        part = part.strip()
        if not part or part in (b"--", b"---"):
            continue

        if b"\r\n\r\n" not in part:
            continue

        headers_raw, body = part.split(b"\r\n\r\n", 1)
        headers = headers_raw.decode("utf-8", errors="ignore").lower()
        body = body.rstrip(b"\r\n")

        if "application/json" in headers:
            try:
                metadata = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Detection returned invalid JSON.",
                ) from exc

        elif "image/jpeg" in headers:
            filename = None
            for line in headers_raw.decode("utf-8", errors="ignore").split("\r\n"):
                if "filename=" in line.lower():
                    filename = line.split("=", 1)[1].strip('"\r\n ')
                    break

            if not filename:
                filename = f"{uuid.uuid4().hex}.jpg"

            crops[filename] = body

    if metadata is None:
        raise HTTPException(
            status_code=502,
            detail="Detection did not return metadata.",
        )

    return metadata, crops


# ============================================================
# DETECTION CLIENT
# ============================================================

async def detect_frame(frame_bytes: bytes):
    client = get_http_client()

    try:
        response = await client.post(
            f"{DETECTION_URL}/detect",
            files={
                "file": (
                    "frame.jpg",
                    frame_bytes,
                    "image/jpeg",
                )
            },
            timeout=DETECTION_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(f"Detection service unavailable: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Detection error: {response.status_code} {response.text}"
        )

    return parse_detection_multipart(response)


# ============================================================
# OCR CLIENT
# ============================================================

async def recognize_crop(crop_bytes: bytes, filename: str):
    client = get_http_client()

    try:
        response = await client.post(
            f"{OCR_URL}/predict",
            files={
                "file": (
                    filename,
                    crop_bytes,
                    "image/jpeg",
                )
            },
            timeout=OCR_TIMEOUT,
        )
    except httpx.RequestError as exc:
        return {
            "text": None,
            "error": f"OCR service unavailable: {exc}",
        }

    if response.status_code != 200:
        return {
            "text": None,
            "error": {
                "status_code": response.status_code,
                "response": response.text,
            },
        }

    try:
        result = response.json()
    except ValueError:
        return {
            "text": None,
            "error": "OCR returned invalid JSON.",
        }

    return {
        "text": result.get("text"),
    }


# ============================================================
# FRAME PIPELINE (SINGLE FRAME)
# ============================================================

async def process_frame(frame_bytes: bytes):
    metadata, crops = await detect_frame(frame_bytes)
    raw_detections = metadata.get("detections", [])

    # فیلتر فریم‌های زیر ۶۰٪ کانفیدنس
    valid_detections = [
        d for d in raw_detections if d.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD
    ]

    if not valid_detections:
        return {"detections": [], "crops": {}}

    return {"detections": valid_detections, "crops": crops}


# ============================================================
# IMAGE API ENDPOINT
# ============================================================

@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()

    if not image_data:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        res = await process_frame(image_data)
        detections = res["detections"]
        crops = res["crops"]

        tasks = []
        for det in detections:
            crop_name = det.get("crop")
            if crop_name in crops:
                tasks.append(recognize_crop(crops[crop_name], crop_name))

        ocr_results = await asyncio.gather(*tasks) if tasks else []

        final_detections = []
        for det, ocr in zip(detections, ocr_results):
            det_copy = dict(det)
            det_copy["plate_id"] = det.get("track_id")
            det_copy["text"] = ocr.get("text")
            final_detections.append(det_copy)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "success": True,
        "source": "image",
        "detections": final_detections,
    }


# ============================================================
# RTSP / CCTV WORKER (UNIQUE PLATE ID + HIGHEST CONFIDENCE LOCK)
# ============================================================

async def open_rtsp(url: str):
    capture = await asyncio.to_thread(cv2.VideoCapture, url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


async def stream_worker(stream: StreamState):
    stream.status = "connecting"
    stream.started_at = asyncio.get_running_loop().time()
    reconnect_attempts = 0
    capture = None

    try:
        while stream.running:
            if capture is None:
                if reconnect_attempts >= MAX_RTSP_RECONNECT_ATTEMPTS:
                    stream.status = "failed"
                    stream.error = "Maximum reconnect attempts exceeded."
                    break

                capture = await open_rtsp(stream.source)
                if capture is None:
                    reconnect_attempts += 1
                    stream.status = "reconnecting"
                    await asyncio.sleep(RTSP_RECONNECT_DELAY)
                    continue

                reconnect_attempts = 0
                stream.status = "running"

            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_delay = 1.0 / fps

            while stream.running:
                start_time = asyncio.get_running_loop().time()

                success, frame = await asyncio.to_thread(capture.read)
                if not success:
                    break

                stream.frames_read += 1
                stream.last_frame_at = asyncio.get_running_loop().time()

                encoded, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not encoded:
                    continue

                frame_bytes = buffer.tobytes()

                # اجرای دیتکشن ریل‌تایم روی فریم
                try:
                    async with stream_lock:
                        frame_res = await process_frame(frame_bytes)
                except Exception as exc:
                    stream.error = str(exc)
                    await asyncio.sleep(0.01)
                    continue

                stream.frames_processed += 1
                detections = frame_res.get("detections", [])
                crops = frame_res.get("crops", {})

                active_frame_detections = []

                # پردازش و به‌روزرسانی وضعیت هر پلاک براساس track_id (plate_id)
                for det in detections:
                    track_id = det.get("track_id")
                    confidence = det.get("confidence", 0.0)
                    crop_name = det.get("crop")

                    if track_id is None:
                        continue

                    # اگر پلاک جدید باشد یا دقت جدید بالاتر از دقت قبلی باشد -> فراخوانی OCR
                    if track_id not in stream.tracked_plates:
                        stream.tracked_plates[track_id] = {
                            "confidence": 0.0,
                            "text": None,
                            "bbox": det.get("bbox"),
                        }

                    cached_plate = stream.tracked_plates[track_id]

                    if confidence > cached_plate["confidence"] and crop_name in crops:
                        # ارسال به OCR تنها در صورت ثبت بالاترین Confidence
                        ocr_res = await recognize_crop(crops[crop_name], crop_name)
                        new_text = ocr_res.get("text")

                        if new_text:
                            cached_plate["text"] = new_text
                            cached_plate["confidence"] = confidence

                    # به‌روزرسانی موقعیت BBox ریل‌تایم در فریم جاری
                    cached_plate["bbox"] = det.get("bbox")

                    active_frame_detections.append(
                        {
                            "plate_id": track_id,
                            "bbox": cached_plate["bbox"],
                            "confidence": cached_plate["confidence"],
                            "text": cached_plate["text"],
                        }
                    )

                if active_frame_detections:
                    stream.results.append(
                        {
                            "timestamp": asyncio.get_running_loop().time(),
                            "detections": active_frame_detections,
                        }
                    )

                # رسم تمام پلاک‌های فعال موجود در فریم جاری با مقادیر یونیک
                annotated_bytes = draw_annotations(frame, active_frame_detections)
                if annotated_bytes:
                    stream.latest_annotated_frame = annotated_bytes

                # کنترل سرعت پخش واقعی
                elapsed = asyncio.get_running_loop().time() - start_time
                sleep_time = max(0.001, frame_delay - elapsed)
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        stream.status = "stopped"
        raise
    except Exception as exc:
        stream.status = "failed"
        stream.error = str(exc)
    finally:
        if capture is not None:
            capture.release()
        if stream.status != "failed":
            stream.status = "stopped"


# ============================================================
# START / STOP / LIVE ENDPOINTS
# ============================================================

@app.post("/streams/start")
async def start_stream(request: StreamRequest):
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="RTSP URL is required.")

    process_fps = validate_process_fps(request.process_fps)

    running_streams = [s for s in streams.values() if s.running]
    if running_streams:
        raise HTTPException(
            status_code=409,
            detail="Another stream is already running. The tracker supports one active stream at a time.",
        )

    stream_id = uuid.uuid4().hex
    stream = StreamState(
        stream_id=stream_id,
        source=request.url.strip(),
        process_fps=process_fps,
    )

    streams[stream_id] = stream
    stream.task = asyncio.create_task(stream_worker(stream))

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
        "process_fps": process_fps,
    }


@app.get("/streams/{stream_id}")
async def get_stream(stream_id: str):
    stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    return {
        "stream_id": stream.stream_id,
        "source": stream.source,
        "status": stream.status,
        "process_fps": stream.process_fps,
        "frames_read": stream.frames_read,
        "frames_processed": stream.frames_processed,
        "results_available": len(stream.results),
        "error": stream.error,
    }


@app.get("/streams/{stream_id}/results")
async def get_stream_results(stream_id: str, clear: bool = False):
    stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    results = list(stream.results)
    if clear:
        stream.results.clear()

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
        "results": results,
    }


@app.post("/streams/{stream_id}/stop")
async def stop_stream(stream_id: str):
    stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    stream.running = False
    if stream.task:
        try:
            await asyncio.wait_for(stream.task, timeout=5.0)
        except asyncio.TimeoutError:
            stream.task.cancel()
            await asyncio.gather(stream.task, return_exceptions=True)

    stream.status = "stopped"
    return {
        "success": True,
        "stream_id": stream_id,
        "status": "stopped",
    }


async def mjpeg_generator(stream_id: str):
    stream = streams.get(stream_id)
    if not stream:
        return

    while stream.running:
        if stream.latest_annotated_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + stream.latest_annotated_frame
                + b"\r\n"
            )
        await asyncio.sleep(0.04)


@app.get("/streams/{stream_id}/live")
async def live_stream(stream_id: str):
    stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    return StreamingResponse(
        mjpeg_generator(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
async def health():
    client = get_http_client()
    detection_status = "unavailable"
    ocr_status = "unavailable"

    try:
        response = await client.get(f"{DETECTION_URL}/health", timeout=5.0)
        detection_status = "ok" if response.status_code == 200 else "error"
    except httpx.RequestError:
        detection_status = "unavailable"

    try:
        response = await client.get(f"{OCR_URL}/health", timeout=5.0)
        ocr_status = "ok" if response.status_code == 200 else "error"
    except httpx.RequestError:
        ocr_status = "unavailable"

    active_streams = sum(1 for s in streams.values() if s.running)
    status = "ok" if (detection_status == "ok" and ocr_status == "ok") else "degraded"

    return {
        "status": status,
        "services": {
            "detection": detection_status,
            "ocr": ocr_status,
        },
        "active_streams": active_streams,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)