FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# PyTorch CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# TensorRT
RUN pip install --no-cache-dir \
    tensorrt==11.3.0.99

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY exporting_to_trt.py .
COPY util.py .
COPY sort ./sort
COPY Models ./Models

# Export PyTorch model to TensorRT during BUILD
RUN python3 exporting_to_trt.py

# Detection API
EXPOSE 8001

# Run API only
CMD ["python3", "main.py"]