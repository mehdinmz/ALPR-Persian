import asyncio
import json
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse


# ============================================================
# CONFIG
# ============================================================

DETECTION_URL = "http://127.0.0.1:8001"
OCR_URL = "http://127.0.0.1:8002"

DETECTION_TIMEOUT = 60.0
OCR_TIMEOUT = 20.0

# Maximum number of results kept for each active stream.
MAX_STREAM_RESULTS = 100

# Number of frames per second sent from CCTV/video to Detection.
DEFAULT_PROCESS_FPS = 5.0

# RTSP reconnect settings.
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

        self.results = deque(
            maxlen=MAX_STREAM_RESULTS
        )

        self.frames_read = 0
        self.frames_processed = 0

        self.started_at = None
        self.last_frame_at = None

        self.error = None


streams: dict[str, StreamState] = {}

# Current architecture has a global tracker in Detection.
# Therefore only one stream may actively use Detection.
stream_lock = asyncio.Lock()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global http_client

    http_client = httpx.AsyncClient(
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

    # Stop all streams.
    for stream in list(streams.values()):
        stream.running = False

    tasks = []

    for stream in list(streams.values()):
        if stream.task:
            tasks.append(stream.task)

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
        raise RuntimeError(
            "HTTP client is not initialized."
        )

    return http_client


def validate_process_fps(
    process_fps: float,
) -> float:

    if process_fps <= 0:
        return DEFAULT_PROCESS_FPS

    return min(
        float(process_fps),
        30.0,
    )


def extract_boundary(
    content_type: str,
) -> Optional[str]:

    for part in content_type.split(";"):

        part = part.strip()

        if part.startswith("boundary="):

            return part.split(
                "=",
                1,
            )[1].strip('"')

    return None


# ============================================================
# DETECTION RESPONSE PARSER
# ============================================================

def parse_detection_multipart(
    response: httpx.Response,
):
    content_type = response.headers.get("content-type", "")

    if "multipart/mixed" not in content_type:
        raise HTTPException(
            status_code=502,
            detail=f"Detection invalid format: {content_type}",
        )

    boundary = extract_boundary(content_type)
    if not boundary:
        raise HTTPException(
            status_code=502,
            detail="Detection response missing boundary.",
        )

    # جداسازی بر اساس boundary استاندارد
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

        # تفکیک دقیق هدر و بدنه باینری
        headers_raw, body = part.split(b"\r\n\r\n", 1)
        headers = headers_raw.decode("utf-8", errors="ignore").lower()
        body = body.rstrip(b"\r\n")

        # ----------------------------------------------------
        # JSON Metadata
        # ----------------------------------------------------
        if "application/json" in headers:
            try:
                metadata = json.loads(body.decode("utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Invalid JSON from Detection: {exc}",
                ) from exc

        # ----------------------------------------------------
        # JPEG Crop Image
        # ----------------------------------------------------
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
            detail="Detection did not return metadata JSON.",
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
            files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
            timeout=DETECTION_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(f"Detection service unavailable: {exc}") from exc

    if response.status_code != 200:
        # چاپ هدرها و بدنه کامل پاسخ جهت عیب‌یابی دقیق
        raise RuntimeError(
            f"Detection HTTP {response.status_code} | "
            f"Headers: {dict(response.headers)} | "
            f"Body: {response.text[:300]}"
        )

    return parse_detection_multipart(response)


# ============================================================
# OCR CLIENT
# ============================================================

# ============================================================
# OCR CLIENT
# ============================================================

async def recognize_crop(
    crop_bytes: bytes,
    filename: str,
):
    client = get_http_client()

    try:
        response = await client.post(
            f"{OCR_URL}/predict",  # تغییر مسیر از /recognize به /predict
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
            "error": (
                f"OCR service unavailable: {exc}"
            ),
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
            "error": (
                "OCR returned invalid JSON."
            ),
        }

    return {
        "text": result.get("text"),
    }


# ============================================================
# FRAME PIPELINE
# ============================================================

async def process_frame(
    frame_bytes: bytes,
):
    """
    Complete ALPR pipeline:

        Frame
          ↓
        Detection
          ↓
        Plate crops
          ↓
        OCR
          ↓
        Result
    """

    metadata, crops = await detect_frame(
        frame_bytes
    )

    detections = metadata.get(
        "detections",
        [],
    )

    if not detections:

        return {
            "detections": [],
        }

    # --------------------------------------------------------
    # OCR all plates concurrently
    # --------------------------------------------------------

    tasks = []

    for detection in detections:

        crop_filename = detection.get(
            "crop"
        )

        if not crop_filename:

            tasks.append(
                asyncio.sleep(
                    0,
                    result={
                        "text": None,
                        "error": (
                            "Missing crop filename."
                        ),
                    },
                )
            )

            continue

        crop_bytes = crops.get(
            crop_filename
        )

        if crop_bytes is None:

            tasks.append(
                asyncio.sleep(
                    0,
                    result={
                        "text": None,
                        "error": (
                            "Crop not found."
                        ),
                    },
                )
            )

            continue

        tasks.append(
            recognize_crop(
                crop_bytes,
                crop_filename,
            )
        )

    ocr_results = await asyncio.gather(
        *tasks
    )

    # --------------------------------------------------------
    # Merge Detection + OCR
    # --------------------------------------------------------

    results = []

    for detection, ocr_result in zip(
        detections,
        ocr_results,
    ):

        result = {
            "track_id": detection.get(
                "track_id"
            ),
            "bbox": detection.get(
                "bbox"
            ),
            "confidence": detection.get(
                "confidence"
            ),
            "text": ocr_result.get(
                "text"
            ),
        }

        if ocr_result.get("error"):

            result["ocr_error"] = (
                ocr_result["error"]
            )

        results.append(result)

    return {
        "detections": results,
    }


# ============================================================
# IMAGE
# ============================================================

@app.post("/process/image")
async def process_image(
    file: UploadFile = File(...),
):
    """
    Process one image.
    """

    image_data = await file.read()

    if not image_data:

        raise HTTPException(
            status_code=400,
            detail="Empty image file.",
        )

    # Validate image.
    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    try:

        result = await process_frame(
            image_data
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "success": True,
        "source": "image",
        **result,
    }


# ============================================================
# VIDEO
# ============================================================

async def process_video_file(
    video_path: Path,
    process_fps: float,
):
    """
    Process video sequentially.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    process_fps = min(
        process_fps,
        fps,
    )

    frame_interval = max(
        1,
        int(
            round(
                fps / process_fps
            )
        ),
    )

    frame_index = 0
    processed_frames = 0

    results = []

    try:

        while True:

            success, frame = (
                await asyncio.to_thread(
                    capture.read
                )
            )

            if not success:
                break

            if (
                frame_index
                % frame_interval
                != 0
            ):

                frame_index += 1
                continue

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )

            if not encoded:

                frame_index += 1
                continue

            frame_bytes = buffer.tobytes()

            try:

                frame_result = (
                    await process_frame(
                        frame_bytes
                    )
                )

            except RuntimeError as exc:

                raise RuntimeError(
                    f"Frame {frame_index}: {exc}"
                ) from exc

            processed_frames += 1

            if frame_result["detections"]:

                results.append(
                    {
                        "frame": frame_index,
                        "timestamp": (
                            frame_index / fps
                        ),
                        "detections": (
                            frame_result[
                                "detections"
                            ]
                        ),
                    }
                )

            frame_index += 1

    finally:

        capture.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "results": results,
    }


@app.post("/process/video")
async def process_video(
    file: UploadFile = File(...),
    process_fps: float = DEFAULT_PROCESS_FPS,
):
    """
    Process uploaded video.
    """

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format."
            ),
        )

    process_fps = validate_process_fps(
        process_fps
    )

    temp_dir = Path("/tmp/alpr")
    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        temp_dir
        / f"{uuid.uuid4().hex}{suffix}"
    )

    try:

        with video_path.open("wb") as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                buffer.write(chunk)

        try:

            result = await process_video_file(
                video_path,
                process_fps,
            )

        except RuntimeError as exc:

            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

        return {
            "success": True,
            "source": "video",
            **result,
        }

    finally:

        try:
            video_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


# ============================================================
# RTSP / CCTV
# ============================================================

async def open_rtsp(
    url: str,
):
    """
    Open RTSP stream using OpenCV.
    """

    capture = await asyncio.to_thread(
        cv2.VideoCapture,
        url,
        cv2.CAP_FFMPEG,
    )

    if not capture.isOpened():

        capture.release()

        return None

    return capture


async def stream_worker(
    stream: StreamState,
):
    """
    Long-running CCTV worker.

        RTSP
          ↓
        Frame
          ↓
        Detection
          ↓
        OCR
          ↓
        Stored result
    """

    stream.status = "connecting"
    stream.started_at = (
        asyncio.get_running_loop().time()
    )

    reconnect_attempts = 0

    capture = None

    try:

        while stream.running:

            # ------------------------------------------------
            # Connect / reconnect
            # ------------------------------------------------

            if capture is None:

                if (
                    reconnect_attempts
                    >= MAX_RTSP_RECONNECT_ATTEMPTS
                ):

                    stream.status = "failed"

                    stream.error = (
                        "Maximum RTSP reconnect "
                        "attempts exceeded."
                    )

                    break

                capture = await open_rtsp(
                    stream.source
                )

                if capture is None:

                    reconnect_attempts += 1
                    stream.status = "reconnecting"

                    await asyncio.sleep(
                        RTSP_RECONNECT_DELAY
                    )

                    continue

                reconnect_attempts = 0
                stream.status = "running"

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            success, frame = (
                await asyncio.to_thread(
                    capture.read
                )
            )

            if not success:

                capture.release()
                capture = None

                stream.status = "reconnecting"

                await asyncio.sleep(
                    RTSP_RECONNECT_DELAY
                )

                continue

            stream.frames_read += 1
            stream.last_frame_at = (
                asyncio.get_running_loop().time()
            )

            # ------------------------------------------------
            # FPS control
            # ------------------------------------------------

            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            if not fps or fps <= 0:
                fps = 25.0

            frame_interval = max(
                1,
                int(
                    round(
                        fps
                        / stream.process_fps
                    )
                ),
            )

            if (
                stream.frames_read
                % frame_interval
                != 0
            ):
                continue

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )

            if not encoded:
                continue

            frame_bytes = buffer.tobytes()

            # ------------------------------------------------
            # Detection + OCR
            # ------------------------------------------------

            try:

                async with stream_lock:

                    result = await process_frame(
                        frame_bytes
                    )

            except Exception as exc:

                stream.error = str(exc)

                # Do not immediately kill CCTV
                # because one frame failed.
                await asyncio.sleep(0.1)

                continue

            stream.frames_processed += 1

            # ------------------------------------------------
            # Store only frames with detections
            # ------------------------------------------------

            if result["detections"]:

                stream.results.append(
                    {
                        "timestamp": (
                            asyncio.get_running_loop()
                            .time()
                        ),
                        "detections": (
                            result["detections"]
                        ),
                    }
                )

    except asyncio.CancelledError:

        stream.status = "stopped"

        raise

    except Exception as exc:

        stream.status = "failed"
        stream.error = str(exc)

    finally:

        if capture is not None:

            capture.release()

        if stream.status not in (
            "failed",
        ):

            stream.status = "stopped"


# ============================================================
# START STREAM
# ============================================================

@app.post("/streams/start")
async def start_stream(
    request: StreamRequest,
):
    """
    Start a CCTV / RTSP stream.

    Example:

        {
            "url": "rtsp://user:password@camera/stream",
            "process_fps": 5
        }
    """

    if not request.url.strip():

        raise HTTPException(
            status_code=400,
            detail="RTSP URL is required.",
        )

    process_fps = validate_process_fps(
        request.process_fps
    )

    # Current Detection tracker is global.
    # Only one stream can use it at a time.
    running_streams = [
        stream
        for stream in streams.values()
        if stream.running
    ]

    if running_streams:

        raise HTTPException(
            status_code=409,
            detail=(
                "Another stream is already running. "
                "The current Detection tracker supports "
                "one active stream at a time."
            ),
        )

    stream_id = uuid.uuid4().hex

    stream = StreamState(
        stream_id=stream_id,
        source=request.url.strip(),
        process_fps=process_fps,
    )

    streams[stream_id] = stream

    stream.task = asyncio.create_task(
        stream_worker(stream)
    )

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
        "process_fps": process_fps,
    }


# ============================================================
# STREAM STATUS
# ============================================================

@app.get("/streams/{stream_id}")
async def get_stream(
    stream_id: str,
):
    """
    Get stream status.
    """

    stream = streams.get(
        stream_id
    )

    if stream is None:

        raise HTTPException(
            status_code=404,
            detail="Stream not found.",
        )

    return {
        "stream_id": stream.stream_id,
        "source": stream.source,
        "status": stream.status,
        "process_fps": stream.process_fps,
        "frames_read": stream.frames_read,
        "frames_processed": (
            stream.frames_processed
        ),
        "results_available": len(
            stream.results
        ),
        "error": stream.error,
    }


# ============================================================
# STREAM RESULTS
# ============================================================

@app.get("/streams/{stream_id}/results")
async def get_stream_results(
    stream_id: str,
    clear: bool = False,
):
    """
    Get detected plates from CCTV.

    clear=true removes returned results
    from the in-memory queue.
    """

    stream = streams.get(
        stream_id
    )

    if stream is None:

        raise HTTPException(
            status_code=404,
            detail="Stream not found.",
        )

    results = list(
        stream.results
    )

    if clear:
        stream.results.clear()

    return {
        "success": True,
        "stream_id": stream_id,
        "status": stream.status,
        "results": results,
    }


# ============================================================
# STOP STREAM
# ============================================================

@app.post("/streams/{stream_id}/stop")
async def stop_stream(
    stream_id: str,
):
    """
    Stop a running CCTV stream.
    """

    stream = streams.get(
        stream_id
    )

    if stream is None:

        raise HTTPException(
            status_code=404,
            detail="Stream not found.",
        )

    stream.running = False

    if stream.task:

        try:

            await asyncio.wait_for(
                stream.task,
                timeout=5.0,
            )

        except asyncio.TimeoutError:

            stream.task.cancel()

            await asyncio.gather(
                stream.task,
                return_exceptions=True,
            )

    stream.status = "stopped"

    return {
        "success": True,
        "stream_id": stream_id,
        "status": "stopped",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    client = get_http_client()

    detection_status = "unavailable"
    ocr_status = "unavailable"

    try:

        response = await client.get(
            f"{DETECTION_URL}/health",
            timeout=5.0,
        )

        if response.status_code == 200:
            detection_status = "ok"
        else:
            detection_status = "error"

    except httpx.RequestError:

        detection_status = "unavailable"

    try:

        response = await client.get(
            f"{OCR_URL}/health",
            timeout=5.0,
        )

        if response.status_code == 200:
            ocr_status = "ok"
        else:
            ocr_status = "error"

    except httpx.RequestError:

        ocr_status = "unavailable"

    active_streams = sum(
        1
        for stream in streams.values()
        if stream.running
    )

    if (
        detection_status == "ok"
        and ocr_status == "ok"
    ):

        status = "ok"

    else:

        status = "degraded"

    return {
        "status": status,
        "services": {
            "detection": detection_status,
            "ocr": ocr_status,
        },
        "active_streams": active_streams,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "ALPR Gateway is running",
        "services": {
            "detection": DETECTION_URL,
            "ocr": OCR_URL,
        },
        "endpoints": {
            "image": "POST /process/image",
            "video": "POST /process/video",
            "stream_start": "POST /streams/start",
            "stream_status": (
                "GET /streams/{stream_id}"
            ),
            "stream_results": (
                "GET /streams/{stream_id}/results"
            ),
            "stream_stop": (
                "POST /streams/{stream_id}/stop"
            ),
            "health": "GET /health",
        },
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
