# ==================================
# Ultralytics YOLO26 + EasyOCR
# Automatic Number Plate Recognition
# ==================================

import cv2
import torch
import time
import easyocr
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from ultralytics import YOLO
from ultralytics.utils.plotting import colors


class ANPR:
    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR."""

    def __init__(self, model_path: str = "yolo26m.engine"):
        """Initializes the ANPR system."""

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load YOLO TensorRT model
        self.model = YOLO(model_path, task="detect")

        # EasyOCR
        self.reader = easyocr.Reader(
            ["fa"],
            gpu=torch.cuda.is_available()
        )

        # Lightweight tracking variables
        self.tracks = {}
        self.next_track_id = 0

        # Persian font for OpenCV visualization
        # Tahoma supports Persian/Arabic characters on Windows.
        self.font_path = r"C:\Windows\Fonts\tahoma.ttf"

    def detect_plates(self, im0: np.ndarray):
        """Detects license plates in an image."""

        results = self.model.predict(
            im0,
            verbose=False,
            conf=0.37
        )

        boxes = (
            results[0].boxes.xyxy.cpu().numpy()
            if results and results[0].boxes is not None
            else []
        )

        return boxes

    # ==================================
    # IoU
    # ==================================

    def calculate_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection_area = (
            intersection_width * intersection_height
        )

        area1 = (
            max(0, box1[2] - box1[0])
            * max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0])
            * max(0, box2[3] - box2[1])
        )

        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # ==================================
    # Lightweight Tracker
    # ==================================

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

        # Predict current position
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

        # Greedy IoU matching
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

        # Update matched tracks
        for track_id, box_index in matches:

            track = self.tracks[track_id]

            previous_bbox = track["bbox"].copy()
            current_bbox = current_boxes[box_index]

            movement = current_bbox - previous_bbox

            # Smooth movement
            track["velocity"] = (
                0.7 * track["velocity"]
                + 0.3 * movement
            )

            track["bbox"] = current_bbox
            track["missed"] = 0
            track["predicted"] = False

        # Predict missed tracks
        for track_id in list(unmatched_track_ids):

            track = self.tracks[track_id]

            track["bbox"] = predicted_boxes[track_id]
            track["missed"] += 1
            track["predicted"] = True

            # Remove old tracks
            if track["missed"] > max_missing_frames:
                del self.tracks[track_id]

        # Create new tracks
        for box_index in unmatched_box_indices:

            self.tracks[self.next_track_id] = {
                "bbox": current_boxes[box_index],
                "velocity": np.zeros(
                    4,
                    dtype=np.float32
                ),
                "missed": 0,
                "predicted": False
            }

            self.next_track_id += 1

        # Return active tracks
        tracked_boxes = []

        for track in self.tracks.values():
            tracked_boxes.append(
                track["bbox"].copy()
            )

        return tracked_boxes

    # ==================================
    # OCR
    # ==================================

    def extract_text(self, im0, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        h, w = im0.shape[:2]

        # کمی padding دور پلاک
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.15)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        roi = im0[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        # بزرگ کردن تصویر برای OCR
        scale = 3

        roi = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # کمی افزایش کنتراست
        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        results = self.reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            mag_ratio=1.0
        )

        # -------------------------------------------------
        # Filter results
        # -------------------------------------------------

        CONF_THRESHOLD = 0.50
        MIN_TEXT_LENGTH = 6

        candidates = []

        for result in results:
            text = result[1].strip()
            conf = result[2]

            clean_text = text.replace(" ", "")

            print(
                "OCR:",
                repr(text),
                "CONF:",
                round(conf, 3)
            )

            if conf < CONF_THRESHOLD:
                continue

            if len(clean_text) < MIN_TEXT_LENGTH:
                continue

            candidates.append((text, conf))

        if not candidates:
            return ""

        # فقط بهترین prediction
        best_text, best_conf = max(
            candidates,
            key=lambda x: x[1]
        )

        print(
            f"SELECTED: {repr(best_text)} "
            f"| CONF: {best_conf:.3f}"
        )

        return best_text
        # # -------------------------------------------------
        # # Filter OCR results
        # # -------------------------------------------------

        # CONF_THRESHOLD = 0.50
        # MIN_TEXT_LENGTH = 6

        # candidates = []

        # for result in results:
        #     text = result[1].strip()
        #     conf = result[2]

        #     # Remove spaces for length check
        #     clean_text = text.replace(" ", "")

        #     if conf < CONF_THRESHOLD:
        #         continue

        #     if len(clean_text) < MIN_TEXT_LENGTH:
        #         continue

        #     candidates.append((text, conf))

        # # Nothing reliable found
        # if not candidates:
        #     return ""

        # # -------------------------------------------------
        # # Select only ONE result
        # # Highest confidence
        # # -------------------------------------------------

        # best_text, best_conf = max(
        #     candidates,
        #     key=lambda x: x[1]
        # )

        # print(
        #     f"SELECTED OCR: {repr(best_text)} "
        #     f"| CONF: {best_conf:.3f}"
        # )

        # return best_text

    # ==================================
    # Persian Text Drawing
    # ==================================

    def draw_persian_text(
        self,
        image: np.ndarray,
        text: str,
        bbox: np.ndarray
    ):
        """
        Draws Persian/Arabic text correctly using
        Pillow instead of cv2.putText.
        """

        if not text:
            return image

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        # Convert OpenCV BGR -> PIL RGB
        pil_image = Image.fromarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        draw = ImageDraw.Draw(
            pil_image
        )

        # Load Persian-compatible font
        try:
            font = ImageFont.truetype(
                self.font_path,
                28
            )
        except Exception:
            # Fallback
            font = ImageFont.load_default()

        # Persian/Arabic reshaping
        reshaped_text = arabic_reshaper.reshape(
            text
        )

        # Correct RTL display order
        display_text = get_display(
            reshaped_text
        )

        # Measure text
        bbox_text = draw.textbbox(
            (0, 0),
            display_text,
            font=font
        )

        text_width = (
            bbox_text[2] - bbox_text[0]
        )

        text_height = (
            bbox_text[3] - bbox_text[1]
        )

        # Put label above bbox
        text_x = x1
        text_y = y1 - text_height - 10

        # If there isn't enough space above,
        # put it inside/under the bbox.
        if text_y < 0:
            text_y = y1 + 5

        # Background rectangle
        padding = 5

        background_x1 = text_x - padding
        background_y1 = text_y - padding
        background_x2 = (
            text_x
            + text_width
            + padding
        )
        background_y2 = (
            text_y
            + text_height
            + padding
        )

        draw.rectangle(
            [
                background_x1,
                background_y1,
                background_x2,
                background_y2
            ],
            fill=(0, 0, 0)
        )

        # Draw Persian text
        draw.text(
            (text_x, text_y),
            display_text,
            font=font,
            fill=(255, 255, 255)
        )

        # Convert PIL RGB -> OpenCV BGR
        image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

        return image

    # ==================================
    # Video Inference
    # ==================================

    def infer_video(
        self,
        source: str = 0,
        output_path: str = None,
        display: bool = True
    ):
        """Performs real-time ANPR on a video stream."""

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video source: {source}"
            )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or 30
        )

        writer = None

        if output_path:

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

        print(
            "🚀 Starting ANPR video inference... "
            "Press 'q' to quit."
        )

        # FPS variables
        fps_start_time = time.perf_counter()
        fps_frame_count = 0
        display_fps = 0.0

        # Reset tracker
        self.tracks = {}
        self.next_track_id = 0

        while True:

            ret, im0 = cap.read()

            if not ret:
                break

            # FPS
            fps_frame_count += 1

            elapsed = (
                time.perf_counter()
                - fps_start_time
            )

            if elapsed >= 1.0:

                display_fps = (
                    fps_frame_count
                    / elapsed
                )

                fps_frame_count = 0
                fps_start_time = (
                    time.perf_counter()
                )

            # Detection
            boxes = self.detect_plates(im0)

            # Tracking
            tracked_boxes = self.update_tracks(
                boxes,
                iou_threshold=0.10,
                max_missing_frames=5
            )

            # Process each tracked plate
            for bbox in tracked_boxes:

                # Draw bbox with OpenCV
                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                cv2.rectangle(
                    im0,
                    (x1, y1),
                    (x2, y2),
                    colors(17, True),
                    4
                )

                # OCR
                text = self.extract_text(
                    im0,
                    bbox
                )

                # Draw Persian OCR text
                im0 = self.draw_persian_text(
                    im0,
                    text,
                    bbox
                )

            # FPS
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
                cv2.imshow(
                    "ANPR (Press 'q' to exit)",
                    im0
                )

            if writer:
                writer.write(im0)

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

        cap.release()

        if writer:
            writer.release()

        cv2.destroyAllWindows()


# ==================================
# Main
# ==================================

if __name__ == "__main__":

    anpr = ANPR(
        model_path=
        "Models\\license_plate_detector_s.engine"
    )

    anpr.infer_video(
        source="1271645.jpg",
        output_path="anpr_output.mp4",
        display=True
    )