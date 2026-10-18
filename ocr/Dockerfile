FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch
RUN python -m pip install \
    torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY main.py .
COPY Models/crnn-fa-license-plate-recognition-v2 \
     ./Models/crnn-fa-license-plate-recognition-v2

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
