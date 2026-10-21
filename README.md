# ALPR-Persian

### Dockerized Iranian License Plate Detection & Recognition Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker)](https://www.docker.com/)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-black)](https://www.ultralytics.com/)
[![OCR](https://img.shields.io/badge/OCR-CRNN%20%2B%20Hezar-6A5ACD)](https://github.com/mehdinmz/ALPR-Persian)
[![License](https://img.shields.io/badge/License-Proprietary-red)](https://github.com/mehdinmz/ALPR-Persian)

[English](#english) | [فارسی](#فارسی)

---

<a name="english"></a>
## English Documentation

### Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Services and Ports](#services-and-ports)
5. [Repository Structure](#repository-structure)
6. [Requirements and Hardware Compatibility](#requirements-and-hardware-compatibility)
7. [Installation and Docker Setup](#installation-and-docker-setup)
8. [API Reference and Usage](#api-reference-and-usage)
9. [Configuration Reference](#configuration-reference)
10. [Detection and OCR Model Information](#detection-and-ocr-model-information)
11. [Dataset Provenance and Licensing Notes](#dataset-provenance-and-licensing-notes)
12. [Limitations and Troubleshooting](#limitations-and-troubleshooting)
13. [Development Roadmap](#development-roadmap)
14. [Proprietary License](#proprietary-license)
15. [Contact Information](#contact-information)

---

### Overview

**ALPR-Persian** is a microservice-based pipeline engineered for automatic license plate detection and optical character recognition (OCR) on Iranian vehicles. Built using FastAPI, Docker, YOLO, and the Hezar deep learning framework, the system provides a unified entry point to process static images, uploaded video files, and real-time RTSP/RTSPS camera streams.

> **Reliability Notice:** The system enforces strict gateway-level filtering (minimum detection confidence and an exact 8-character string length requirement). These filters remove invalid candidates but **do not guarantee** that every accepted plate prediction is semantically correct.

---

### Features

- **Persian Plate Recognition:** CRNN-based model fine-tuned for Iranian plate characters via the Hezar framework.
- **YOLO Plate Detection:** Fine-tuned YOLO detection model with TensorRT optimization support.
- **Multi-Input Ingestion:** Supports single image processing, uploaded recorded videos, and live RTSP/RTSPS network video streams.
- **Microservice Design:** Decoupled Gateway, Detection, and OCR services operating over a private Docker network.
- **Temporal Tracking & Filtering:** IoU and string-similarity tracking to correlate detections across video frames.
- **Live Stream Preview:** Real-time MJPEG preview stream for monitoring active stream tracking.
- **JSON Exporting:** Structured JSON metadata export for audit logs and downstream database integration.
- **Persian Text Rendering:** Built-in support for Vazirmatn font, `arabic_reshaper`, and `python-bidi` for correct text layout in visual overlays.

---

### Architecture

```mermaid
flowchart TD
    Client["Client / User"] -->|HTTP / REST| Gateway["FastAPI Gateway (:8000)"]

    subgraph Docker Network [Private Container Network]
        Gateway -->|POST image crop| OCR["OCR Service (:8002)<br/>Hezar CRNN (CPU)"]
        Gateway -->|POST frame| Detect["Detection Service (:8001)<br/>YOLO + TensorRT (GPU)"]
    end

    Detect -->|Multipart Crops & Metadata| Gateway
    OCR -->|Predicted Text| Gateway

    Gateway -->|MJPEG Stream| Live["Live Dashboard Preview"]
    Gateway -->|JSON Payload| Export["Stream Summary / JSON Export"]
```

---

### Services and Ports

| Service | Internal Port | Host Port | Runtime | Primary Responsibility |
|---|---|---|---|---|
| Gateway | 8000 | 8000 | Python 3.12 (CPU) | Orchestration, stream management, temporal tracking, API routes, JSON export, UI. |
| Detection | 8001 | Unexposed | Python 3.11 / CUDA 12.8 (GPU) | License plate localization, bounding box tracking, frame cropping, multipart response. |
| OCR | 8002 | Unexposed | Python 3.11 (CPU) | Text recognition on cropped plate images using Hezar CRNN. |

---

### Repository Structure

```text
ALPR-Persian/
├── main.py                     # Gateway API entrypoint and stream orchestrator
├── Dockerfile                  # Gateway container manifest
├── requirements.txt            # Gateway dependencies
├── docker-compose.yml          # Multi-container service definition
├── template/
│   └── index.html              # Web dashboard template
├── detection/
│   ├── main.py                 # Detection FastAPI service
│   ├── Dockerfile              # GPU-enabled CUDA/TensorRT container manifest
│   ├── Models/
│   │   └── license_plate_detector_s.pt # Fine-tuned YOLO weights
│   └── Scripts/
│       └── exporting_to_trt.py # TensorRT engine generation script
└── ocr/
    ├── main.py                 # OCR FastAPI service
    ├── Dockerfile              # CPU container manifest
    ├── requirements.txt        # OCR dependencies
    └── Models/
        └── crnn-fa-license-plate-recognition-v2/ # Local Hezar CRNN model
```

---

### Requirements and Hardware Compatibility

**Tested Hardware**

- **GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM)
- **Host OS:** Linux / Windows with Docker Desktop & NVIDIA Container Toolkit

**Software Dependencies**

- Docker Engine 20.10+ and Docker Compose v2+
- NVIDIA Driver compatible with CUDA 12.8
- NVIDIA Container Toolkit (`nvidia-ctk`) for GPU passthrough to the detection service

---

### Installation and Docker Setup

**1. Clone the Repository**

```bash
git clone https://github.com/mehdinmz/ALPR-Persian.git
cd ALPR-Persian
```

**2. Build Docker Images**

Build the three service containers explicitly:

```bash
docker build -t alpr-detection:latest ./detection
docker build -t alpr-ocr:latest ./ocr
docker build -t alpr-gateway:latest .
```

**3. Start Services with Docker Compose**

```bash
docker compose up -d
```

> **Note on TensorRT Engine Export:** On first launch, the detection service checks for `detection/Models/license_plate_detector_s.engine`. If missing, it automatically compiles the `.pt` weights into a TensorRT engine. This process may take several minutes depending on your GPU.

**4. Verify Container Status**

```bash
docker compose ps
docker compose logs -f detection
```

---

### API Reference and Usage

All requests are directed to the Gateway service running at `http://localhost:8000`.

**1. System Health Check**

```bash
curl -X GET "http://localhost:8000/health"
```

Response:

```json
{
  "status": "healthy",
  "active_streams": 0,
  "total_streams": 0
}
```

**2. Process Single Image**

```bash
curl -X POST "http://localhost:8000/process/image" \
  -F "file=@/path/to/vehicle.jpg"
```

Response:

```json
{
  "success": true,
  "source": "image",
  "detections": [
    {
      "track_id": 1,
      "bbox": [120, 340, 280, 390],
      "confidence": 0.92,
      "crop": "plate_1.jpg",
      "text": "۱۲ ب ۳۴۵ ایران ۶۷"
    }
  ]
}
```

**3. Upload and Process Video File**

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.mpeg`, `.mpg` (Max: 1 GiB).

```bash
curl -X POST "http://localhost:8000/streams/upload-video" \
  -F "file=@/path/to/traffic.mp4" \
  -F "process_fps=10"
```

Response:

```json
{
  "success": true,
  "stream_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "starting",
  "filename": "traffic.mp4",
  "size_bytes": 15420100
}
```

**4. Start RTSP Stream Processing**

```bash
curl -X POST "http://localhost:8000/streams/start" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "process_fps": 10
  }'
```

**5. View Live MJPEG Stream**

Open in any standard browser:

```text
http://localhost:8000/streams/{stream_id}/live
```

**6. Export Stream Results as JSON**

```bash
curl -X GET "http://localhost:8000/streams/{stream_id}/save-json" \
  --output stream_results.json
```

**7. Stop and Delete Stream**

```bash
# Stop stream
curl -X POST "http://localhost:8000/streams/{stream_id}/stop"

# Delete stream records
curl -X DELETE "http://localhost:8000/streams/{stream_id}"
```

---

### Configuration Reference

Gateway settings configured in `main.py` and overridable via environment variables:

| Parameter | Default Value | Description |
|---|---|---|
| `DETECTION_URL` | `http://detection:8001` | Detection service endpoint within Docker network. |
| `OCR_URL` | `http://ocr:8002` | OCR service endpoint within Docker network. |
| `CONFIDENCE_THRESHOLD` | `0.40` | Min confidence passed to YOLO detection pipeline. |
| `MIN_PLATE_CONFIDENCE` | `0.50` | Min confidence required to accept plate candidate. |
| `DEFAULT_PROCESS_FPS` | `10.0` | Frame rate target for video/stream sampling. |
| `MAX_VIDEO_UPLOAD_BYTES` | `1,073,741,824` | Maximum video file upload size (1 GiB). |
| `DETECTION_TIMEOUT` | `30.0s` | HTTP timeout for detection requests. |
| `OCR_TIMEOUT` | `10.0s` | HTTP timeout for OCR inference requests. |
| `RTSP_RECONNECT_DELAY` | `3.0s` | Delay between RTSP connection retry attempts. |
| `MAX_RTSP_RECONNECT_ATTEMPTS` | `10` | Maximum consecutive RTSP reconnection attempts. |

---

### Detection and OCR Model Information

- **Detection Model:** YOLO26-based fine-tuned license plate detector.
  - Filepath: `detection/Models/license_plate_detector_s.pt`
  - Engine: Auto-generated `license_plate_detector_s.engine` via TensorRT.
- **OCR Model:** Hezar Persian License Plate Recognition CRNN v2.
  - Filepath: `ocr/Models/crnn-fa-license-plate-recognition-v2/`
  - Runtime: PyTorch (CPU mode).

---

### Dataset Provenance and Licensing Notes

- **Detection Dataset:** ANPR-2 on Ultralytics Platform
  - Description: Street-level surveillance footage of vehicle license plates in parking environments, providing annotated carplate data for automated number plate recognition.
- **OCR Dataset:** `persian-license-plate-v1` on Hugging Face
  - Description: Open-source dataset hosted by HezarAI for Persian vehicle license plate character recognition.

Third-party weights and datasets remain subject to their respective creators' licensing terms.

---

### Limitations and Troubleshooting

- **Character Filtering:** An 8-character length check prevents short/long invalid reads but does not verify if the recognized characters are accurate.
- **GPU Requirement:** TensorRT engine generation requires a compatible NVIDIA GPU and NVIDIA Container Toolkit installed on the host.
- **Ports:** Detection (8001) and OCR (8002) services are isolated within the internal Docker network and cannot be accessed directly from the host unless explicitly exposed in `docker-compose.yml`.

---

### Development Roadmap

- [ ] Optimize OCR recognition stability and accuracy on low-resolution crops.
- [ ] Implement end-to-end latency benchmarks on target edge hardware.
- [ ] Add native RTSP hardware decoding support using NVDEC.
- [ ] Implement standardized unit and integration tests across microservices.
- [ ] Package multi-arch Docker images for deployment on ARM/Jetson platforms.

---

### Proprietary License

Copyright (c) 2026 Mohammad Mehdi Namazian. All Rights Reserved.

This repository and its source code are proprietary. Standard open-source rights (e.g., MIT, Apache, GPL) are **NOT** granted. Unauthorized copying, distribution, modification, public display, or commercial exploitation of this software without explicit prior written authorization from the copyright holder is strictly prohibited. Refer to the `LICENSE` file in the repository root for formal terms.

---

### Contact Information

- **Author:** Mohammad Mehdi Namazian
- **Email:** mmahdinamazian@gmail.com
- **LinkedIn:** [Mohammad Mehdi Namazian](https://www.linkedin.com/in/mohammad-mehdi-namazian/)
- **Repository:** https://github.com/mehdinmz/ALPR-Persian

---

<a name="فارسی"></a>
## مستندات فارسی

### فهرست مطالب
۱. [درباره پروژه](#درباره-پروژه)
۲. [قابلیت‌ها](#قابلیت‌ها)
۳. [معماری سیستم](#معماری-سیستم)
۴. [سرویس‌ها و پورت‌ها](#سرویس‌ها-و-پورت‌ها)
۵. [ساختار مخزن](#ساختار-مخزن)
۶. [پیش‌نیازها و سخت‌افزار](#پیش‌نیازها-و-سخت‌افزار)
۷. [راه‌اندازی با داکر](#راه‌اندازی-با-داکر)
۸. [راهنمای API](#راهنمای-api)
۹. [تنظیمات و متغیرها](#تنظیمات-و-متغیرها)
۱۰. [اطلاعات مدل‌ها](#اطلاعات-مدل‌ها)
۱۱. [مجموعه داده و مجوزها](#مجموعه-داده-و-مجوزها)
۱۲. [محدودیت‌ها و عیب‌یابی](#محدودیت‌ها-و-عیب‌یابی)
۱۳. [نقشه راه توسعه](#نقشه-راه-توسعه)
۱۴. [مجوز مالکیت فکری](#مجوز-مالکیت-فکری)
۱۵. [اطلاعات تماس](#اطلاعات-تماس)

---

### درباره پروژه

سامانه **ALPR-Persian** یک خط پردازشی مبتنی بر میکروپایپ‌لاین برای تشخیص و بازخوانی خودکار پلاک خودروهای ایرانی است. این سیستم با استفاده از FastAPI، Docker، YOLO و فریم‌ورک هِزار (Hezar) توسعه یافته و امکان پردازش تصویر، ویدیوهای ضبط‌شده و استریم‌های زنده شبکه (RTSP/RTSPS) را فراهم می‌سازد.

> **توجه ایمنی:** سیستم در لایه Gateway از فیلترهای حد آستانه اطمینان و شرط طول دقیق ۸ کاراکتر استفاده می‌کند. این فیلترها ورودی‌های نامعتبر را کاهش می‌دهند اما صحت ۱۰۰٪ کاراکترهای خوانده‌شده را تضمین نمی‌کنند.

---

### قابلیت‌ها

- **بازخوانی پلاک ایرانی:** استفاده از مدل CRNN آموزش‌دیده برای کاراکترهای پلاک ایرانی در فریم‌ورک Hezar.
- **تشخیص پلاک با YOLO:** مدل اختصاصی YOLO به همراه پشتیبانی از موتور TensorRT جهت تسریع پردازش.
- **ورودی‌های متنوع:** پشتیبانی از عکس، فایل ویدیو و استریم‌های دوربین مداربسته (RTSP/RTSPS).
- **معماری میکروخدمات:** تفکیک کامل سرویس‌های Gateway، Detection و OCR در کانتینرهای مجزا.
- **ردیابی زمانی (Temporal Tracking):** استفاده از IoU و شباهت متنی برای تجمیع نتایج در فریم‌های متوالی.
- **پیش‌نمایش زنده:** ارائه استریم MJPEG برای مشاهده لحظه‌ای باکس‌ها و متن پلاک‌ها.
- **خروجی JSON:** قابلیت استخراج گزارش‌های ساختاریافته JSON از پلاک‌های ثبت‌شده.
- **پشتیبانی از فونت و متن فارسی:** بازنویسی صحیح کاراکترهای فارسی با فونت وزیرمتن و کتابخانه‌های `arabic_reshaper` و `python-bidi`.

---

### معماری سیستم

```mermaid
flowchart TD
    Client["کاربر / کلاینت"] -->|HTTP / REST| Gateway["درگاه ارتباطی Gateway (:8000)"]

    subgraph Docker Network [شبکه داخلی داکر]
        Gateway -->|ارسال برش پلاک| OCR["سرویس OCR (:8002)<br/>Hezar CRNN (پردازنده)"]
        Gateway -->|ارسال فریم| Detect["سرویس تشخیص (:8001)<br/>YOLO + TensorRT (کارت گرافیک)"]
    end

    Detect -->|خروجی برش‌ها و دیتای مکانی| Gateway
    OCR -->|متن پیش‌بینی‌شده| Gateway

    Gateway -->|استریم MJPEG| Live["داشبورد پیش‌نمایش زنده"]
    Gateway -->|فایل JSON| Export["گزارش خروجی JSON"]
```

---

### سرویس‌ها و پورت‌ها

| سرویس | پورت داخلی | پورت اصلی (Host) | محیط اجرا | وظیفه اصلی |
|---|---|---|---|---|
| Gateway | 8000 | 8000 | Python 3.12 (CPU) | مدیریت استریم‌ها، ردیابی، ارائه API، تولید JSON و داشبورد. |
| Detection | 8001 | بسته | Python 3.11 / CUDA 12.8 (GPU) | تشخیص پلاک، برش تصویر، ردیابی باکس‌ها و ارسال Multipart. |
| OCR | 8002 | بسته | Python 3.11 (CPU) | خواندن متن پلاک از روی تصویر برش‌خورده با مدل Hezar. |

---

### ساختار مخزن

```text
ALPR-Persian/
├── main.py                     # نقطه ورود API سرویس Gateway
├── Dockerfile                  # فایل ساخت کانتینر Gateway
├── requirements.txt            # پیش‌نیازهای Gateway
├── docker-compose.yml          # فایل تعریف سرویس‌های داکر
├── template/
│   └── index.html              # قالب داشبورد وب
├── detection/
│   ├── main.py                 # API سرویس تشخیص (Detection)
│   ├── Dockerfile              # فایل ساخت کانتینر CUDA/TensorRT
│   ├── Models/
│   │   └── license_plate_detector_s.pt # وزنده‌شده‌های YOLO
│   └── Scripts/
│       └── exporting_to_trt.py # اسکریپت تبدیل به TensorRT
└── ocr/
    ├── main.py                 # API سرویس پردازش متن (OCR)
    ├── Dockerfile              # فایل ساخت کانتینر OCR
    ├── requirements.txt        # پیش‌نیازهای OCR
    └── Models/
        └── crnn-fa-license-plate-recognition-v2/ # مدل Hezar
```

---

### پیش‌نیازها و سخت‌افزار

**سخت‌افزار تست‌شده**

- **کارت گرافیک:** NVIDIA GeForce GTX 1650 (حافظه ۴ گیگابایت)
- **سیستم‌عامل:** لینوکس / ویندوز به همراه Docker Desktop و NVIDIA Container Toolkit

**نرم‌افزارهای مورد نیاز**

- Docker Engine نسخه 20.10+ و Docker Compose نسخه v2+
- درایور NVIDIA سازگار با CUDA 12.8
- ابزار NVIDIA Container Toolkit جهت اشتراک‌گذاری کارت گرافیک با داکر

---

### راه‌اندازی با داکر

**۱. دریافت مخزن**

```bash
git clone https://github.com/mehdinmz/ALPR-Persian.git
cd ALPR-Persian
```

**۲. ساخت ایمیج‌های داکر**

```bash
docker build -t alpr-detection:latest ./detection
docker build -t alpr-ocr:latest ./ocr
docker build -t alpr-gateway:latest .
```

**۳. اجرا با Docker Compose**

```bash
docker compose up -d
```

> **نکته درباره TensorRT:** در اولین اجرا، سرویس Detection عدم وجود فایل `license_plate_detector_s.engine` را بررسی کرده و فایل `.pt` را به موتور TensorRT تبدیل می‌کند. این فرایند ممکن است چند دقیقه طول بکشد.

**۴. بررسی وضعیت کانتینرها**

```bash
docker compose ps
docker compose logs -f detection
```

---

### راهنمای API

تمام درخواست‌ها به سرویس Gateway روی پورت 8000 ارسال می‌شوند.

**۱. بررسی سلامت سیستم**

```bash
curl -X GET "http://localhost:8000/health"
```

**۲. پردازش تک تصویر**

```bash
curl -X POST "http://localhost:8000/process/image" \
  -F "file=@/path/to/vehicle.jpg"
```

**۳. آپلود و پردازش ویدیو**

فرمت‌های پشتیبانی‌شده: `.mp4`، `.avi`، `.mov`، `.mkv`، `.webm`، `.mpeg`، `.mpg` (حداکثر ۱ گیگابایت).

```bash
curl -X POST "http://localhost:8000/streams/upload-video" \
  -F "file=@/path/to/traffic.mp4" \
  -F "process_fps=10"
```

**۴. شروع استریم زنده RTSP**

```bash
curl -X POST "http://localhost:8000/streams/start" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "process_fps": 10
  }'
```

**۵. مشاهده پیش‌نمایش زنده MJPEG**

در مرورگر وارد آدرس زیر شوید:

```text
http://localhost:8000/streams/{stream_id}/live
```

**۶. دریافت خروجی JSON**

```bash
curl -X GET "http://localhost:8000/streams/{stream_id}/save-json" \
  --output stream_results.json
```

**۷. توقف و حذف استریم**

```bash
# توقف استریم
curl -X POST "http://localhost:8000/streams/{stream_id}/stop"

# حذف اطلاعات استریم
curl -X DELETE "http://localhost:8000/streams/{stream_id}"
```

---

### تنظیمات و متغیرها

مقادیر پیش‌فرض تنظیم‌شده در `main.py`:

| پارامتر | مقدار پیش‌فرض | توضیح |
|---|---|---|
| `DETECTION_URL` | `http://detection:8001` | آدرس سرویس تشخیص در شبکه داکر. |
| `OCR_URL` | `http://ocr:8002` | آدرس سرویس OCR در شبکه داکر. |
| `CONFIDENCE_THRESHOLD` | `0.40` | حد آستانه اطمینان مدل YOLO. |
| `MIN_PLATE_CONFIDENCE` | `0.50` | حداقل اطمینان پذیرش پلاک در درگاه. |
| `DEFAULT_PROCESS_FPS` | `10.0` | نرخ فریم پردازش ویدیو و استریم. |
| `MAX_VIDEO_UPLOAD_BYTES` | `1,073,741,824` | حداکثر حجم ویدیو (۱ گیگابایت). |
| `DETECTION_TIMEOUT` | `30.0s` | تایم‌اوت درخواست‌های تشخیص. |
| `OCR_TIMEOUT` | `10.0s` | تایم‌اوت درخواست‌های OCR. |

---

### اطلاعات مدل‌ها

- **مدل تشخیص (Detection):** مدل YOLO26 آموزش‌دیده روی پلاک‌های ایرانی.
  - مسیر: `detection/Models/license_plate_detector_s.pt`
  - موتور شتاب‌دهنده: `license_plate_detector_s.engine`
- **مدل بازخوانی متن (OCR):** مدل Hezar CRNN نسخه ۲.
  - مسیر: `ocr/Models/crnn-fa-license-plate-recognition-v2/`

---

### مجموعه داده و مجوزها

- **داده‌های Detection:** مجموعه داده ANPR-2 در پلتفرم Ultralytics
  - توضیحات: تصاویری از دوربین‌های نظارتی سطح خیابان و محیط‌های پارکینگی شامل دیتای نشانه‌گذاری‌شده پلاک خودرو برای الگوریتم‌های خوانش خودکار.
- **داده‌های OCR:** مجموعه داده `persian-license-plate-v1` در Hugging Face
  - توضیحات: مجموعه داده متن‌باز منتشرشده توسط هِزار جهت بازخوانی کاراکترهای پلاک خودروهای ایرانی.

---

### محدودیت‌ها و عیب‌یابی

۱. شرط ۸ کاراکتر تنها از ورود پلاک‌های ناقص جلوگیری می‌کند و صحت حروف خوانده‌شده را تضمین نمی‌کند.
۲. تبدیل به TensorRT نیازمند کارت گرافیک NVIDIA و نصب NVIDIA Container Toolkit است.
۳. پورت‌های Detection (8001) و OCR (8002) در شبکه داخلی داکر ایزوله هستند و از محیط Host قابل دسترسی مستقیم نیستند.

---

### نقشه راه توسعه

- [ ] بهبود دقت خوانش OCR در تصاویر کم‌کیفیت و تار.
- [ ] اندازه‌گیری و ثبت بنچمارک دقیق زمان تأخیر (Latency) روی سخت‌افزار هدف.
- [ ] افزودن قابلیت رمزکشی سخت‌افزاری استریم RTSP با NVDEC.
- [ ] نگارش تست‌های خودکار برای APIها و کانتینرها.
- [ ] بسته‌بندی ایمیج‌های چندمعماری داکر برای استقرار روی پلتفرم‌های ARM/Jetson.

---

### مجوز مالکیت فکری

حقوق نشر (c) ۲۰۲۶ محمد مهدی نمازیان. تمامی حقوق محفوظ است.

این پروژه اختصاصی (Proprietary) است. هیچ‌گونه مجوز متن‌باز (مانند MIT، Apache یا GPL) برای این مخزن صادر نشده است. هرگونه کپی‌برداری، تغییر، توزیع یا استفاده تجاری بدون دریافت مجوز مکتوب از مالک اثر ممنوع می‌باشد. جهت مشاهده جزئیات قانونی به فایل `LICENSE` موجود در ریشه مخزن مراجعه کنید.

---

### اطلاعات تماس

- **توسعه‌دهنده:** محمد مهدی نمازیان
- **ایمیل:** mmahdinamazian@gmail.com
- **لینکدین:** [Mohammad Mehdi Namazian](https://www.linkedin.com/in/mohammad-mehdi-namazian/)
- **مخزن گیت‌هاب:** https://github.com/mehdinmz/ALPR-Persian