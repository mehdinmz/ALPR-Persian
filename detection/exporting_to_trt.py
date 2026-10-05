from pathlib import Path

from ultralytics import YOLO
import ultralytics
import torch
import tensorrt as trt

import torch

_original_torch_load = torch.load

def trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = trusted_torch_load

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "license_plate_detector_s.pt"


class ModelExporter:

    def __init__(self, model_path: Path):
        self.model_path = model_path

        print("=" * 60)
        print("TensorRT Export")
        print("=" * 60)

        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("Ultralytics:", ultralytics.__version__)
        print("TensorRT:", trt.__version__)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required during Docker build for TensorRT export."
            )

        print("GPU:", torch.cuda.get_device_name(0))
        print("=" * 60)

    def export(self):

        print(f"Loading model: {self.model_path}")

        model = YOLO(str(self.model_path))

        print("Exporting model to TensorRT...")

        engine_path = model.export(
            format="engine",
            device=0
        )

        print(f"TensorRT engine created: {engine_path}")

        return engine_path


if __name__ == "__main__":

    exporter = ModelExporter(MODEL_PATH)

    exporter.export()