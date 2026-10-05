import cv2
import torch
import numpy as np
import json
import uuid
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO


# ============================================================
# ANPR
# ============================================================

class ANPR:

    def __init__(self, model_path="Models/license_plate_detector_s.engine"):

        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Device:", self.device)
        print("Loading model:", model_path)

        self.model = YOLO(
            model_path,
            task="detect"
        )

        # ----------------------------------------------------
        # Tracker
        # ----------------------------------------------------

        self.tracks = {}
        self.next_track_id = 0

        self.max_missing_frames = 5
        self.iou_threshold = 0.10

    # ========================================================
    # Detection
    # ========================================================

    def detect_plates(self, im0):

        results = self.model.predict(
            im0,
            verbose=False,
            conf=0.37,
            device=0 if torch.cuda.is_available() else "cpu"
        )

        boxes = []

        if not results:
            return boxes

        result = results[0]

        if result.boxes is None:
            return boxes

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = xyxy

            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

        return boxes

    # ========================================================
    # IoU
    # ========================================================

    def calculate_iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_width *
            intersection_height
        )

        area1 = (
            max(0, box1[2] - box1[0]) *
            max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0]) *
            max(0, box2[3] - box2[1])
        )

        union = (
            area1 +
            area2 -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ========================================================
    # Tracker
    # ========================================================

    def update_tracks(self, detections):

        updated_tracks = {}

        used_track_ids = set()

        # ----------------------------------------------------
        # Match detections with existing tracks
        # ----------------------------------------------------

        for detection in detections:

            bbox = detection["bbox"]

            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():

                if track_id in used_track_ids:
                    continue

                iou = self.calculate_iou(
                    bbox,
                    track["bbox"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_track_id = track_id

            # ------------------------------------------------
            # Existing track
            # ------------------------------------------------

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):

                track = self.tracks[
                    best_track_id
                ]

                track["bbox"] = bbox
                track["confidence"] = (
                    detection["confidence"]
                )
                track["missing"] = 0

                updated_tracks[
                    best_track_id
                ] = track

                used_track_ids.add(
                    best_track_id
                )

            # ------------------------------------------------
            # New track
            # ------------------------------------------------

            else:

                track_id = self.next_track_id

                self.next_track_id += 1

                updated_tracks[
                    track_id
                ] = {
                    "bbox": bbox,
                    "confidence": (
                        detection["confidence"]
                    ),
                    "missing": 0
                }

                used_track_ids.add(
                    track_id
                )

        # ----------------------------------------------------
        # Keep missing tracks
        # ----------------------------------------------------

        for track_id, track in self.tracks.items():

            if track_id in updated_tracks:
                continue

            track["missing"] += 1

            if (
                track["missing"]
                <= self.max_missing_frames
            ):

                updated_tracks[
                    track_id
                ] = track

        self.tracks = updated_tracks

        return self.tracks

    # ========================================================
    # Crop plate
    # ========================================================

    def crop_plate(self, image, bbox):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(
            0,
            min(x1, w - 1)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        x2 = max(
            0,
            min(x2, w)
        )

        y2 = max(
            0,
            min(y2, h)
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None

        return crop


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="ALPR Detection API",
    version="1.0.0"
)


# ============================================================
# Load model ONCE
# ============================================================

anpr = ANPR(
    model_path="Models/license_plate_detector_s.engine"
)


# ============================================================
# Tracker lock
# ============================================================

tracker_lock = threading.Lock()


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "detection"
    }


# ============================================================
# Multipart response helper
# ============================================================

def create_multipart_response(
    metadata,
    images
):
    """
    Creates a multipart/mixed HTTP response.

    metadata:
        JSON serializable dictionary

    images:
        list of tuples:
        (filename, jpeg_bytes)
    """

    boundary = (
        "ALPRBoundary"
        + uuid.uuid4().hex
    )

    body = bytearray()

    # --------------------------------------------------------
    # JSON metadata
    # --------------------------------------------------------

    body.extend(
        (
            f"--{boundary}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Disposition: form-data; "
            'name="metadata"\r\n'
            "\r\n"
        ).encode("utf-8")
    )

    body.extend(
        json.dumps(
            metadata,
            ensure_ascii=False
        ).encode("utf-8")
    )

    body.extend(b"\r\n")

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    for filename, image_bytes in images:

        body.extend(
            (
                f"--{boundary}\r\n"
                "Content-Type: image/jpeg\r\n"
                f'Content-Disposition: attachment; '
                f'filename="{filename}"\r\n'
                "\r\n"
            ).encode("utf-8")
        )

        body.extend(image_bytes)
        body.extend(b"\r\n")

    # --------------------------------------------------------
    # End boundary
    # --------------------------------------------------------

    body.extend(
        (
            f"--{boundary}--\r\n"
        ).encode("utf-8")
    )

    return Response(
        content=bytes(body),
        media_type=(
            f"multipart/mixed; "
            f"boundary={boundary}"
        )
    )


# ============================================================
# Detection API
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read binary image
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty image"
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    np_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    detections = anpr.detect_plates(
        image
    )

    # --------------------------------------------------------
    # Tracking
    #
    # Same ANPR instance is kept alive between requests.
    # --------------------------------------------------------

    with tracker_lock:

        tracks = anpr.update_tracks(
            detections
        )

    # --------------------------------------------------------
    # Prepare response
    # --------------------------------------------------------

    results = []
    images = []

    for track_id, track in tracks.items():

        bbox = track["bbox"]

        crop = anpr.crop_plate(
            image,
            bbox
        )

        if crop is None:
            continue

        # ----------------------------------------------------
        # Encode crop directly to JPEG binary
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            crop
        )

        if not success:
            continue

        crop_bytes = buffer.tobytes()

        filename = (
            f"plate_{track_id}.jpg"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        results.append({
            "track_id": int(track_id),

            "bbox": [
                int(v)
                for v in bbox
            ],

            "confidence": float(
                track["confidence"]
            ),

            "crop": filename
        })

        # ----------------------------------------------------
        # Binary image
        # ----------------------------------------------------

        images.append(
            (
                filename,
                crop_bytes
            )
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "detections": results
    }

    # --------------------------------------------------------
    # Return multipart/mixed
    # --------------------------------------------------------

    return create_multipart_response(
        metadata,
        images
    )


# ============================================================
# Local execution
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )