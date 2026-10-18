
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app


# System dependencies
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/cuda*.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN python3 -m pip install --upgrade pip

# PyTorch with CUDA 12.8
RUN python3 -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

# TensorRT for CUDA 12
RUN python3 -m pip install \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt-cu12

# API and detection dependencies
RUN python3 -m pip install \
    ultralytics==8.4.173 \
    fastapi==0.142.2 \
    "uvicorn[standard]==0.54.0" \
    python-multipart==0.0.32 \
    opencv-python-headless==4.11.0.86 \
    numpy==1.24.3

# Pinned dependencies for ONNX / TensorRT export
RUN python3 -m pip install \
    protobuf==4.25.8 \
    onnx==1.17.0 \
    onnxruntime-gpu==1.20.2 \
    onnxslim==0.1.82

# Copy application
COPY main.py ./main.py
COPY Scripts/exporting_to_trt.py ./Scripts/exporting_to_trt.py
COPY Models/license_plate_detector_s.pt ./Models/license_plate_detector_s.pt

# Detection API
EXPOSE 8001

CMD ["python3", "main.py"]
