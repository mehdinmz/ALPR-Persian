# ==================================

# Ultralytics YOLO26 + EasyOCR

# Automatic Number Plate Recognition

# ==================================

import cv2

import torch

import time  # ADDED: For FPS calculation

# import easyocr

import numpy as np

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors



class ANPR:

    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR.

    This class handles license plate detection using a YOLO model and text extraction

    using EasyOCR. It supports both image and video streams for real-time inference.

    Attributes:

        model (YOLO): The YOLO model for license plate detection.

        reader (easyocr.Reader): The OCR reader instance for text recognition.

        device (torch.device): Computation device (CPU or CUDA).

    """

    def __init__(self, model_path: str = "yolo26m.engine"):

        """Initializes the ANPR system."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLO(model_path, task="detect")  # Load YOLO model for license plate detection

        # self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

        # ADDED: Lightweight tracking variables

        self.tracks = {}

        self.next_track_id = 0

    def detect_plates(self, im0: np.ndarray):

        """Detects license plates in a image."""

        results = self.model.predict(im0, verbose=False, conf=0.37)

        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

        return boxes

    # ADDED: Calculate IoU between two bounding boxes

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray):

        x1 = max(box1[0], box2[0])

        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])

        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)

        intersection_height = max(0, y2 - y1)

        intersection_area = intersection_width * intersection_height

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])

        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        union_area = area1 + area2 - intersection_area

        if union_area <= 0:

            return 0.0

        return intersection_area / union_area

    # ADDED: Lightweight tracker using IoU and previous movement

    def update_tracks(
        self,
        boxes: np.ndarray,
        iou_threshold: float = 0.10,
        max_missing_frames: int = 5
    ):

        current_boxes = [
            np.asarray(box, dtype=np.float32)
            for box in boxes
        ]

        # ADDED: Predict the current position of existing tracks

        predicted_boxes = {}

        for track_id, track in self.tracks.items():

            predicted_boxes[track_id] = (
                track["bbox"] + track["velocity"]
            )

        unmatched_track_ids = set(self.tracks.keys())

        unmatched_box_indices = set(
            range(len(current_boxes))
        )

        matches = []

        # ADDED: Greedy IoU matching

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

            matches.append(
                (best_track_id, best_box_index)
            )

            unmatched_track_ids.remove(best_track_id)

            unmatched_box_indices.remove(best_box_index)

        # ADDED: Update matched tracks

        for track_id, box_index in matches:

            track = self.tracks[track_id]

            previous_bbox = track["bbox"].copy()

            current_bbox = current_boxes[box_index]

            movement = current_bbox - previous_bbox

            # ADDED: Smooth the estimated movement

            track["velocity"] = (
                0.7 * track["velocity"]
                + 0.3 * movement
            )

            track["bbox"] = current_bbox

            track["missed"] = 0

            track["predicted"] = False

        # ADDED: Predict boxes for missed tracks

        for track_id in list(unmatched_track_ids):

            track = self.tracks[track_id]

            track["bbox"] = predicted_boxes[track_id]

            track["missed"] += 1

            track["predicted"] = True

            # ADDED: Remove tracks that have been missing too long

            if track["missed"] > max_missing_frames:

                del self.tracks[track_id]

        # ADDED: Create new tracks for new detections

        for box_index in unmatched_box_indices:

            self.tracks[self.next_track_id] = {

                "bbox": current_boxes[box_index],

                "velocity": np.zeros(4, dtype=np.float32),

                "missed": 0,

                "predicted": False

            }

            self.next_track_id += 1

        # ADDED: Return all active tracked boxes

        tracked_boxes = []

        for track in self.tracks.values():

            tracked_boxes.append(
                track["bbox"].copy()
            )

        return tracked_boxes

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):

        """Performs OCR on the cropped license plate region."""

        x1, y1, x2, y2 = map(int, bbox)

        roi = im0[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # text = self.reader.readtext(gray, detail=0, paragraph=True)
        # return " ".join(text).strip() if text else ""


    # text = self.reader.readtext(gray, detail=0, paragraph=True)
    # return " ".join(text).strip() if text else ""
    def infer_video(self, source: str = 0, output_path: str = None, display: bool = True):

        """Performs real-time ANPR on a video stream."""

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():

            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None

        if output_path:

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("🚀 Starting ANPR video inference... Press 'q' to quit.")

        # ADDED: Variables for OpenCV display FPS

        fps_start_time = time.perf_counter()

        fps_frame_count = 0

        display_fps = 0.0

        # ADDED: Reset tracker for this video

        self.tracks = {}

        self.next_track_id = 0

        while True:

            ret, im0 = cap.read()

            if not ret:

                break

            # ADDED: Count frames displayed by OpenCV

            fps_frame_count += 1

            elapsed = time.perf_counter() - fps_start_time

            if elapsed >= 1.0:

                display_fps = fps_frame_count / elapsed

                fps_frame_count = 0

                fps_start_time = time.perf_counter()

            boxes = self.detect_plates(im0)

            # ADDED: Update lightweight tracker

            tracked_boxes = self.update_tracks(
                boxes,
                iou_threshold=0.10,
                max_missing_frames=5
            )

            ann = Annotator(im0, line_width=4)

            # MODIFIED: Use tracked boxes instead of raw detection boxes

            for bbox in tracked_boxes:

                text = self.extract_text(im0, bbox)

                ann.box_label(bbox, label=text, color=colors(17, True))

            # ADDED: Display OpenCV FPS on the frame

            cv2.putText(

                im0,

                f"FPS: {display_fps:.1f}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2,

                cv2.LINE_AA

            )

            if display:

                cv2.imshow("ANPR (Press 'q' to exit)", im0)

            if writer:

                writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

        cap.release()

        if writer:

            writer.release()

        cv2.destroyAllWindows()



if __name__ == "__main__":

    anpr = ANPR(model_path="Models\\license_plate_detector_s.engine")  # Use trained YOLO license plate model

    anpr.infer_video(source="sample.mp4", output_path="anpr_output.mp4", display=True)