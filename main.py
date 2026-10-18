import asyncio
import difflib
import json
import os
import tempfile
import time
import uuid

from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import arabic_reshaper
import cv2
import httpx
import numpy as np
import uvicorn

from bidi.algorithm import get_display
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DETECTION_URL = os.getenv(
    "DETECTION_URL",
    "http://127.0.0.1:8001",
)
OCR_URL = os.getenv(
    "OCR_URL",
    "http://127.0.0.1:8002",
)

DETECTION_TIMEOUT = 30.0
OCR_TIMEOUT = 10.0

MAX_STREAM_RESULTS = 500
DEFAULT_PROCESS_FPS = 10.0

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

MAX_VIDEO_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB
VIDEO_UPLOAD_DIR = Path(tempfile.gettempdir()) / "alpr_uploads"
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.40
MIN_PLATE_CONFIDENCE = 0.50

# بافر پخش
PRE_BUFFER_FRAMES = 150
PLAYBACK_QUEUE_SIZE = 900

# ============================================================
# GLOBAL HTTP CLIENT
# ============================================================

http_client: Optional[httpx.AsyncClient] = None


# ============================================================
# PLATE TRACKER
# ============================================================

class PersistentPlateTracker:
    def __init__(self):
        self.next_plate_id = 1
        self.tracks: Dict[int, dict] = {}

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = max(0, box1[2] - box1[0]) * max(
            0, box1[3] - box1[1]
        )
        area2 = max(0, box2[2] - box2[0]) * max(
            0, box2[3] - box2[1]
        )
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def compute_score(text: Optional[str], confidence: float) -> float:
        if not text:
            return 0.0

        clean_text = text.strip()

        if len(clean_text) != 8 or confidence < MIN_PLATE_CONFIDENCE:
            return 0.0

        return float(confidence)

    @staticmethod
    def text_similarity(t1: str, t2: str) -> float:
        if not t1 or not t2:
            return 0.0

        return difflib.SequenceMatcher(
            None,
            t1.strip(),
            t2.strip(),
        ).ratio()

    def update(self, detections: list):
        updated_detections = []
        matched_track_ids = set()

        for det in detections:
            bbox = det.get("bbox")
            conf = float(det.get("confidence", 0.0))
            det_text = det.get("text")

            if not bbox or len(bbox) != 4:
                continue

            best_match_id = None
            best_score = 0.0

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                iou = self.calculate_iou(bbox, track["bbox"])
                text_sim = (
                    self.text_similarity(det_text, track["text"])
                    if det_text and track["text"]
                    else 0.0
                )

                if iou > 0.40 or text_sim > 0.50:
                    score = iou + text_sim
                    if score > best_score:
                        best_score = score
                        best_match_id = track_id

            candidate_score = self.compute_score(det_text, conf)

            if best_match_id is not None:
                track = self.tracks[best_match_id]
                track["bbox"] = bbox
                track["missed_frames"] = 0

                if (
                    candidate_score > 0
                    and candidate_score > track["confidence"]
                ):
                    track["confidence"] = conf
                    track["text"] = det_text.strip()

                matched_track_ids.add(best_match_id)

                det_copy = dict(det)
                det_copy["plate_id"] = best_match_id
                det_copy["text"] = track["text"]
                updated_detections.append(det_copy)

            else:
                new_id = self.next_plate_id
                self.next_plate_id += 1

                valid_text = (
                    det_text.strip()
                    if det_text
                    and len(det_text.strip()) == 8
                    and candidate_score > 0
                    else None
                )
                valid_conf = conf if valid_text else 0.0

                self.tracks[new_id] = {
                    "bbox": bbox,
                    "confidence": valid_conf,
                    "text": valid_text,
                    "missed_frames": 0,
                }

                matched_track_ids.add(new_id)

                det_copy = dict(det)
                det_copy["plate_id"] = new_id
                det_copy["text"] = valid_text
                updated_detections.append(det_copy)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["missed_frames"] += 1

                if self.tracks[track_id]["missed_frames"] > 50:
                    del self.tracks[track_id]

        return updated_detections


# ============================================================
# STREAM STATE
# ============================================================

class StreamState:
    def __init__(
        self,
        stream_id: str,
        source: str,
        process_fps: float,
        source_type: str = "rtsp",
        temporary_file: Optional[str] = None,
    ):
        self.stream_id = stream_id
        self.source = source
        self.process_fps = process_fps
        self.source_type = source_type
        self.temporary_file = temporary_file

        self.running = True
        self.status = "starting"
        self.task: Optional[asyncio.Task] = None

        self.results = deque(maxlen=MAX_STREAM_RESULTS)
        self.unique_saved_plates = {}

        self.frames_read = 0
        self.frames_processed = 0
        self.native_fps = 25.0
        self.started_at = None
        self.last_frame_at = None
        self.error = None

        self.playback_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=PLAYBACK_QUEUE_SIZE
        )
        self.tracker = PersistentPlateTracker()


streams: Dict[str, StreamState] = {}
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

    tasks = [
        stream.task
        for stream in streams.values()
        if stream.task
    ]

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    if http_client:
        await http_client.aclose()

    http_client = None


app = FastAPI(
    title="ALPR Gateway",
    description="Production Gateway / Orchestrator for Persian ALPR",
    version="1.1.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODELS
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

    return min(float(process_fps), 60.0)


def extract_boundary(content_type: str) -> Optional[str]:
    for part in content_type.split(";"):
        part = part.strip()

        if part.startswith("boundary="):
            return part.split("=", 1)[1].strip('"')

    return None


def safe_source_display(stream: StreamState) -> str:
    if stream.source_type != "rtsp":
        return Path(stream.source).name

    try:
        parsed = urlsplit(stream.source)

        if parsed.password is None and parsed.username is None:
            return stream.source

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"

        return urlunsplit((
            parsed.scheme,
            f"***:***@{host}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        ))

    except Exception:
        return "RTSP source"


def public_stream_data(stream: StreamState) -> dict:
    return {
        "stream_id": stream.stream_id,
        "source": safe_source_display(stream),
        "source_type": stream.source_type,
        "status": stream.status,
        "running": stream.running,
        "process_fps": stream.process_fps,
        "native_fps": stream.native_fps,
        "frames_read": stream.frames_read,
        "frames_processed": stream.frames_processed,
        "unique_plates": len(stream.unique_saved_plates),
        "started_at": stream.started_at,
        "last_frame_at": stream.last_frame_at,
        "error": stream.error,
    }


@lru_cache(maxsize=8)
def load_annotation_font(size: int):
    font_candidates = [
        os.getenv("ALPR_FONT_PATH"),
        str(BASE_DIR / "fonts" / "Vazirmatn-Regular.ttf"),
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    ]

    for font_path in font_candidates:
        if font_path and Path(font_path).is_file():
            return ImageFont.truetype(font_path, size)

    raise RuntimeError(
        "No Unicode font found. Set ALPR_FONT_PATH to a Persian-capable TTF font."
    )


def draw_annotations(frame_np: np.ndarray, detections: list) -> Optional[bytes]:
    annotated = frame_np.copy()

    for det in detections:
        bbox = det.get("bbox")

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

    pil_image = Image.fromarray(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    )
    draw = ImageDraw.Draw(pil_image)
    font = load_annotation_font(20)

    for det in detections:
        bbox = det.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, _, _ = map(int, bbox)
        text = str(det.get("text") or "Reading...")
        confidence = float(det.get("confidence", 0.0))
        label = f"{text} ({confidence:.2f})"

        visual_label = arabic_reshaper.reshape(label)
        visual_label = get_display(visual_label)

        text_bbox = draw.textbbox((0, 0), visual_label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_top = max(0, y1 - text_h - 10)

        draw.rectangle(
            (x1, label_top, x1 + text_w + 8, y1),
            fill=(0, 255, 0),
        )

        draw.text(
            (
                x1 + 4,
                label_top + max(0, (y1 - label_top - text_h) // 2),
            ),
            visual_label,
            font=font,
            fill=(0, 0, 0),
        )

    annotated_bgr = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )

    if success:
        return encoded_image.tobytes()

    return None


# ============================================================
# DETECTION & OCR CLIENTS
# ============================================================

def parse_detection_multipart(response: httpx.Response):
    content_type = response.headers.get("content-type", "")

    if "multipart/mixed" not in content_type:
        raise RuntimeError(f"Invalid detection response format: {content_type}")

    boundary = extract_boundary(content_type)

    if not boundary:
        raise RuntimeError("Detection response boundary is missing.")

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
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        elif "image/jpeg" in headers:
            filename = None

            for line in headers_raw.decode(
                "utf-8", errors="ignore"
            ).split("\r\n"):
                if "filename=" in line.lower():
                    filename = line.split("=", 1)[1].strip('" \r\n')
                    break

            if not filename:
                filename = f"{uuid.uuid4().hex}.jpg"

            crops[filename] = body

    return metadata or {"detections": []}, crops


async def detect_frame(frame_bytes: bytes):
    client = get_http_client()

    response = await client.post(
        f"{DETECTION_URL}/detect",
        files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
        timeout=DETECTION_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Detection service returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    return parse_detection_multipart(response)


async def recognize_crop(crop_bytes: bytes, filename: str):
    client = get_http_client()

    try:
        response = await client.post(
            f"{OCR_URL}/predict",
            files={"file": (filename, crop_bytes, "image/jpeg")},
            timeout=OCR_TIMEOUT,
        )

        if response.status_code == 200:
            return response.json().get("text")

    except Exception:
        pass

    return None


async def process_frame(frame_bytes: bytes):
    metadata, crops = await detect_frame(frame_bytes)

    raw_detections = metadata.get("detections", [])
    valid_detections = [
        det for det in raw_detections
        if float(det.get("confidence", 0.0)) >= CONFIDENCE_THRESHOLD
    ]

    return {
        "detections": valid_detections,
        "crops": crops,
    }


# ============================================================
# SINGLE IMAGE API
# ============================================================

@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    await file.close()

    if not image_data:
        raise HTTPException(status_code=400, detail="Empty image.")

    image_array = np.frombuffer(image_data, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    ok, encoded = cv2.imencode(".jpg", decoded)

    if not ok:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    res = await process_frame(encoded.tobytes())
    detections = res["detections"]
    crops = res["crops"]

    final_detections = []

    async def recognize_detection(det):
        crop_name = det.get("crop")

        if not crop_name or crop_name not in crops:
            return None

        text = await recognize_crop(crops[crop_name], crop_name)

        if not text:
            return None

        clean_text = text.strip()
        confidence = float(det.get("confidence", 0.0))

        if len(clean_text) != 8 or confidence < MIN_PLATE_CONFIDENCE:
            return None

        det_copy = dict(det)
        det_copy["text"] = clean_text
        return det_copy

    results = await asyncio.gather(
        *(recognize_detection(det) for det in detections)
    )

    for result in results:
        if result is not None:
            final_detections.append(result)

    return {
        "success": True,
        "source": "image",
        "detections": final_detections,
    }


# ============================================================
# STREAM CAPTURE
# ============================================================

async def open_rtsp(url: str):
    capture = await asyncio.to_thread(
        cv2.VideoCapture,
        url,
        cv2.CAP_FFMPEG,
    )

    if not capture.isOpened():
        capture.release()
        return None

    return capture


async def stream_worker(stream: StreamState):
    stream.status = "connecting"
    stream.started_at = time.time()

    reconnect_attempts = 0
    capture = None
    completed = False

    try:
        while stream.running:
            if capture is None:
                if (
                    stream.source_type == "rtsp"
                    and reconnect_attempts >= MAX_RTSP_RECONNECT_ATTEMPTS
                ):
                    stream.status = "failed"
                    stream.error = "Maximum RTSP reconnect attempts exceeded."
                    break

                if stream.source_type == "video":
                    capture = await asyncio.to_thread(
                        cv2.VideoCapture,
                        stream.source,
                    )
                else:
                    capture = await open_rtsp(stream.source)

                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()

                    capture = None

                    if stream.source_type == "video":
                        stream.status = "failed"
                        stream.error = "Could not open uploaded video."
                        break

                    reconnect_attempts += 1
                    stream.status = "reconnecting"
                    await asyncio.sleep(RTSP_RECONNECT_DELAY)
                    continue

                reconnect_attempts = 0
                stream.status = "running"

            native_fps = capture.get(cv2.CAP_PROP_FPS)

            if not native_fps or native_fps <= 0:
                native_fps = 25.0

            stream.native_fps = float(native_fps)

            inference_interval = max(
                1,
                int(round(stream.native_fps / stream.process_fps)),
            )

            last_valid_dets = []

            while stream.running:
                success, frame = await asyncio.to_thread(capture.read)

                if not success:
                    capture.release()
                    capture = None

                    if stream.source_type == "video":
                        completed = True
                        break

                    reconnect_attempts += 1
                    stream.status = "reconnecting"
                    break

                stream.frames_read += 1
                stream.last_frame_at = time.time()

                if stream.frames_read % inference_interval == 0:
                    encoded, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 80],
                    )

                    if encoded:
                        try:
                            # حفظ رفتار فعلی Gateway؛ درخواست‌های مدل سری می‌شوند.
                            async with stream_lock:
                                frame_res = await process_frame(buffer.tobytes())

                            stream.frames_processed += 1
                            raw_dets = frame_res.get("detections", [])
                            crops = frame_res.get("crops", {})

                            for det in raw_dets:
                                crop_name = det.get("crop")
                                confidence = float(
                                    det.get("confidence", 0.0)
                                )

                                if crop_name and crop_name in crops:
                                    ocr_text = await recognize_crop(
                                        crops[crop_name],
                                        crop_name,
                                    )

                                    if (
                                        ocr_text
                                        and len(ocr_text.strip()) == 8
                                        and confidence >= MIN_PLATE_CONFIDENCE
                                    ):
                                        det["text"] = ocr_text.strip()
                                    else:
                                        det["text"] = None

                            tracked_dets = stream.tracker.update(raw_dets)
                            active_dets = []

                            for det in tracked_dets:
                                plate_id = det.get("plate_id")
                                track_info = stream.tracker.tracks.get(plate_id)

                                if track_info and track_info.get("text"):
                                    det["text"] = track_info["text"]
                                    active_dets.append(det)

                                    stream.unique_saved_plates[plate_id] = {
                                        "plate_id": plate_id,
                                        "text": track_info["text"],
                                        "confidence": float(
                                            track_info["confidence"]
                                        ),
                                        "timestamp": time.time(),
                                    }

                            last_valid_dets = active_dets

                            if active_dets:
                                stream.results.append({
                                    "timestamp": time.time(),
                                    "detections": active_dets,
                                })

                        except Exception as exc:
                            stream.error = str(exc)

                annotated_bytes = draw_annotations(frame, last_valid_dets)

                if annotated_bytes:
                    if stream.playback_queue.full():
                        try:
                            stream.playback_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    try:
                        stream.playback_queue.put_nowait(annotated_bytes)
                    except asyncio.QueueFull:
                        pass

            if completed or not stream.running:
                break

            if stream.source_type == "rtsp":
                await asyncio.sleep(RTSP_RECONNECT_DELAY)

    except asyncio.CancelledError:
        stream.status = "stopped"
        raise

    except Exception as exc:
        stream.status = "failed"
        stream.error = str(exc)

    finally:
        if capture is not None:
            capture.release()

        stream.running = False

        if stream.status != "failed":
            if completed:
                stream.status = "completed"
            elif stream.status != "stopped":
                stream.status = "stopped"

        if stream.temporary_file:
            try:
                Path(stream.temporary_file).unlink(missing_ok=True)
            except OSError as exc:
                stream.error = (
                    f"{stream.error or ''} "
                    f"Temporary-file cleanup failed: {exc}"
                ).strip()


# ============================================================
# STREAM API
# ============================================================

@app.get("/streams")
async def list_streams():
    return {
        "streams": [
            public_stream_data(stream)
            for stream in reversed(list(streams.values()))
        ]
    }


@app.post("/streams/start")
async def start_stream(request: StreamRequest):
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    if not url.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only RTSP/RTSPS URLs are accepted here. "
                "Upload video files using /streams/upload-video."
            ),
        )

    stream_id = uuid.uuid4().hex

    stream = StreamState(
        stream_id=stream_id,
        source=url,
        process_fps=validate_process_fps(request.process_fps),
        source_type="rtsp",
    )

    streams[stream_id] = stream
    stream.task = asyncio.create_task(stream_worker(stream))

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
    }


@app.post("/streams/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    process_fps: float = Form(DEFAULT_PROCESS_FPS),
):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension: {suffix or '(none)'}",
        )

    stream_id = uuid.uuid4().hex
    destination = VIDEO_UPLOAD_DIR / f"{stream_id}{suffix}"
    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 1 GiB upload limit.",
                    )

                output.write(chunk)

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    stream = StreamState(
        stream_id=stream_id,
        source=str(destination),
        process_fps=validate_process_fps(process_fps),
        source_type="video",
        temporary_file=str(destination),
    )

    streams[stream_id] = stream
    stream.task = asyncio.create_task(stream_worker(stream))

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
        "filename": Path(file.filename or "video").name,
        "size_bytes": total_bytes,
    }


@app.get("/streams/{stream_id}/save-json")
async def save_stream_json(stream_id: str):
    stream = streams.get(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    output_dir = Path(tempfile.gettempdir()) / "alpr_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"plates_{stream_id}.json"

    data = {
        "stream_id": stream_id,
        "source": safe_source_display(stream),
        "status": stream.status,
        "total_unique_plates": len(stream.unique_saved_plates),
        "frames_read": stream.frames_read,
        "frames_processed": stream.frames_processed,
        "error": stream.error,
        "plates": list(stream.unique_saved_plates.values()),
    }

    with file_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=4)

    return FileResponse(
        path=file_path,
        filename=f"plates_{stream_id}.json",
        media_type="application/json",
    )


@app.post("/streams/{stream_id}/stop")
async def stop_stream(stream_id: str):
    stream = streams.get(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    if stream.running:
        stream.running = False

    if stream.task:
        try:
            await asyncio.wait_for(stream.task, timeout=10.0)

        except asyncio.TimeoutError:
            stream.task.cancel()
            await asyncio.gather(stream.task, return_exceptions=True)

    if stream.status not in ("failed", "completed"):
        stream.status = "stopped"

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
    }


# ============================================================
# LIVE MJPEG
# ============================================================

async def mjpeg_generator(stream_id: str):
    stream = streams.get(stream_id)

    if stream is None:
        return

    # منتظر پرشدن بافر می‌مانیم، اما ویدیوی کوتاه را معطل نمی‌کنیم.
    while (
        stream.running
        and stream.playback_queue.qsize() < PRE_BUFFER_FRAMES
    ):
        await asyncio.sleep(0.2)

    while stream.running or not stream.playback_queue.empty():
        try:
            frame_bytes = await asyncio.wait_for(
                stream.playback_queue.get(),
                timeout=2.0,
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            stream.playback_queue.task_done()

            frame_delay = 1.0 / max(stream.native_fps, 1.0)
            await asyncio.sleep(frame_delay)

        except asyncio.TimeoutError:
            if not stream.running:
                break

        except asyncio.CancelledError:
            raise

        except Exception:
            break


@app.get("/streams/{stream_id}/live")
async def live_stream(stream_id: str):
    stream = streams.get(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    return StreamingResponse(
        mjpeg_generator(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# HEALTH & DASHBOARD
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_streams": sum(
            1 for stream in streams.values() if stream.running
        ),
        "total_streams": len(streams),
    }


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_path = BASE_DIR / "template" / "index.html"

    if html_path.is_file():
        return html_path.read_text(encoding="utf-8")

    return "<h1>فایل template/index.html پیدا نشد.</h1>"


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )