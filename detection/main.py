
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

    def __init__(self, model_path="Models/license_plate_detector_m.engine"):

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

    # ========================================================
    # Detection
    # ========================================================

    def detect_plates(self, im0):

        results = self.model.predict(
            source=im0,
            verbose=False,
            conf=0.37,
            device=0 if torch.cuda.is_available() else "cpu",
            task="detect"
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = xyxy.tolist()

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

        return detections

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

    def update_tracks(
        self,
        boxes,
        iou_threshold=0.10,
        max_missing_frames=5
    ):

        # Detection dictionaries -> bounding boxes
        current_boxes = [
            np.asarray(d["bbox"], dtype=np.float32)
            for d in boxes
        ]

        predicted_boxes = {}

        for track_id, track in self.tracks.items():

            predicted_boxes[track_id] = (
                track["bbox"] + track["velocity"]
            )

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_box_indices = set(range(len(current_boxes)))
        matches = []

        while unmatched_track_ids and unmatched_box_indices:

            best_iou = 0.0
            best_track_id = None
            best_box_index = None

            for track_id in unmatched_track_ids:

                predicted_box = predicted_boxes[track_id]

                for box_index in unmatched_box_indices:

                    iou = self.calculate_iou(
                        predicted_box,
                        current_boxes[box_index]
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                        best_box_index = box_index

            if (
                best_track_id is None
                or best_iou < iou_threshold
            ):
                break

            matches.append((best_track_id, best_box_index))

            unmatched_track_ids.remove(best_track_id)
            unmatched_box_indices.remove(best_box_index)

        # ----------------------------------------------------
        # Update matched tracks
        # ----------------------------------------------------

        for track_id, box_index in matches:

            track = self.tracks[track_id]

            previous_bbox = track["bbox"].copy()
            current_bbox = current_boxes[box_index]

            movement = current_bbox - previous_bbox

            track["velocity"] = (
                0.7 * track["velocity"]
                + 0.3 * movement
            )

            track["bbox"] = current_bbox
            track["confidence"] = boxes[box_index]["confidence"]
            track["missed"] = 0
            track["predicted"] = False

        # ----------------------------------------------------
        # Update unmatched tracks
        # ----------------------------------------------------

        for track_id in list(unmatched_track_ids):

            track = self.tracks[track_id]

            track["bbox"] = predicted_boxes[track_id]
            track["missed"] += 1
            track["predicted"] = True

            if track["missed"] > max_missing_frames:
                del self.tracks[track_id]

        # ----------------------------------------------------
        # Create new tracks
        # ----------------------------------------------------

        for box_index in unmatched_box_indices:

            self.tracks[self.next_track_id] = {
                "bbox": current_boxes[box_index],
                "velocity": np.zeros(4, dtype=np.float32),
                "confidence": boxes[box_index]["confidence"],
                "missed": 0,
                "predicted": False
            }

            self.next_track_id += 1

        return self.tracks

    # ========================================================
    # Crop plate
    # ========================================================

    def crop_plate(self, image, bbox):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = [
            int(v) for v in bbox
        ]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))

        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

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

def create_multipart_response(metadata, images):

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
            'Content-Disposition: form-data; name="metadata"\r\n'
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
                f'Content-Disposition: attachment; filename="{filename}"\r\n'
                "\r\n"
            ).encode("utf-8")
        )

        body.extend(image_bytes)
        body.extend(b"\r\n")

    # --------------------------------------------------------
    # End boundary
    # --------------------------------------------------------

    body.extend(
        f"--{boundary}--\r\n".encode("utf-8")
    )

    return Response(
        content=bytes(body),
        media_type=f"multipart/mixed; boundary={boundary}"
    )


# ============================================================
# Detection API
# ============================================================

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

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

    detections = anpr.detect_plates(image)

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    with tracker_lock:

        tracks = anpr.update_tracks(
            detections
        )

        # Copy current tracks to avoid processing tracks
        # that were removed during the update.
        tracks = {
            track_id: {
                **track,
                "bbox": track["bbox"].copy()
            }
            for track_id, track in tracks.items()
        }

    # --------------------------------------------------------
    # Prepare response
    # --------------------------------------------------------

    results = []
    images = []

    for track_id, track in tracks.items():

        # Do not return predicted boxes without a current
        # detection, because they may not match this image.
        if track["predicted"]:
            continue

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

        filename = f"plate_{track_id}.jpg"

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        results.append({
            "track_id": int(track_id),

            "bbox": [
                int(v) for v in bbox
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
    # Response metadata
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
