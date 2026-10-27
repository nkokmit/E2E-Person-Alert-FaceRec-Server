# E2E-Person-Alert-FaceRec-Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23%20(CUDA%20%2B%20CPU)-green.svg)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-throughput, production-grade end-to-end edge-cloud server for **Real-Time Person Detection, 5-Point Landmark Face Alignment, Deep Metric Face Recognition, and Live Event Notification**. Built with **FastAPI**, **ONNX Runtime (CUDA / CPU)**, **YOLOv8**, and **ArcFace / MobileFaceNet**.

Designed to handle continuous RTSP / IP camera / ESP32 snapshot streams with guaranteed **sub-150ms real-time latency**, decoupled asynchronous deep learning execution, per-person alert debounce cooldown, and live WebSocket broadcast with automatic dead-connection reclamation.

---

## Table of Contents
- [1. System Architecture & Data Flow](#1-system-architecture--data-flow)
- [2. Key Features](#2-key-features)
- [3. Performance Benchmarks](#3-performance-benchmarks)
- [4. Quickstart Guide](#4-quickstart-guide)
  - [Native Setup (Virtualenv)](#native-setup-virtualenv)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [5. Face Enrollment & CLI Testing](#5-face-enrollment--cli-testing)
- [6. REST & WebSocket API Reference](#6-rest--websocket-api-reference)
- [7. Project Directory Structure](#7-project-directory-structure)
- [8. Configuration Reference (`configs.yaml`)](#8-configuration-reference-configsyaml)

---

## 1. System Architecture & Data Flow

The server employs a **Producer-Consumer architecture** separating network I/O from compute-intensive neural network inference. The FastAPI async event loop ingests frames and returns immediate HTTP acknowledgments, while a dedicated background daemon worker executes batched GPU/CPU inference without inducing GIL stalls or event-loop starvation.

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Layer (I/O Bound)"]
        Cam["IP Cameras / ESP32-CAM / RTSP Stream"]
        FastAPI["FastAPI App (app/api_min.py)<br/>POST /snapshot & POST /burst"]
        Cam -->|HTTP Multipart / JPEG| FastAPI
    end

    subgraph QueueLayer["Decoupled Buffer & Backpressure Guard"]
        SFQ["SmartFrameQueue (app/snapshot_worker.py)<br/>• Capacity: 15 frames<br/>• Drop-Oldest on Buffer Overflow<br/>• Latency Guard: Age > 150ms Drop"]
        FastAPI -->|Non-blocking put_nowait| SFQ
    end

    subgraph DLWorker["Deep Learning Inference Engine (CPU / GPU Bound)"]
        WorkerThread["Background Inference Worker Thread"]
        SFQ -->|Dequeue < 150ms fresh frame| WorkerThread
        
        YOLO["YOLOv8n Person Detector<br/>Bounding Box [x1, y1, x2, y2]"]
        FaceDet["SCRFD / YuNet Face Detector<br/>Bbox + 5 Keypoints (Eyes, Nose, Mouth)"]
        Align["Umeyama 5-Point Similarity Transform<br/>Canonical 112x112 ArcFace Template"]
        Sharp["Laplacian Variance Sharpness Gate<br/>Filter blurry faces (Var < 25)"]
        Embed["MobileFaceNet / ArcFace ONNX (512-D)<br/>CUDA / CPU Intra-Op Parallelism"]
        L2Norm["Vectorized L2 Normalization"]
        Match["FaceGallery Cosine Matcher (BLAS GEMM)<br/>Top-2 Similarity + Margin Check"]
        
        WorkerThread --> YOLO
        WorkerThread --> FaceDet
        FaceDet --> Align --> Sharp --> Embed --> L2Norm --> Match
    end

    subgraph AlertSystem["Alert Debounce & Annotation"]
        Debounce["AlertDebounceManager<br/>Cooldown: 30s per Person per Camera"]
        Annotator["draw_detection_overlay<br/>Annotate Person & Face Bboxes, Badges, OSD"]
        Match --> Debounce
        YOLO --> Annotator
        Match --> Annotator
    end

    subgraph Persistence["Storage & Database (Thread-Safe)"]
        Disk["data/snapshots/events/{evt_id}.jpg"]
        SQLite[("SQLite (data/app.db)<br/>WAL Mode + Busy Timeout 5000ms")]
        Annotator -->|cv2.imwrite| Disk
        Debounce -->|Commit Event Metadata| SQLite
    end

    subgraph Notification["Real-Time Dispatch"]
        WSMgr["ConnectionManager (app/ws_manager.py)<br/>• Concurrent Non-blocking Broadcast<br/>• Heartbeat (Ping/Pong: 15s/45s)<br/>• Dead Client Reaper & Camera Filter"]
        Debounce -->|asyncio.run_coroutine_threadsafe| WSMgr
        Client["Web / Mobile Dashboards / iOS Clients<br/>WebSocket /ws/alerts & /stream"]
        WSMgr -->|Live JSON Alert Event| Client
    end
```

---

## 2. Key Features

###  Real-Time Camera Stream Ingestion
- Non-blocking ingest endpoints (`/snapshot`, `/burst/frame`) supporting raw image bytes, multipart files, and in-memory frames.
- Fast I/O dispatch returns HTTP 200 within $< 1\text{ ms}$, preventing edge camera HTTP client timeouts.

###  Face Alignment & Deep Metric Embeddings
- **5-Point Facial Landmark Alignment**: Batched Umeyama similarity transform maps detected landmarks (left eye, right eye, nose, left mouth, right mouth) onto the standard canonical ArcFace $112 \times 112$ template.
- **Laplacian Variance Quality Gate**: Vectorized batch sharpness calculation filters out severely blurred or partially occluded faces ($\text{Var}(\Delta) < 25.0$) to eliminate false alerts.
- **Deep Feature Representation**: Exports MobileFaceNet to ONNX with dynamic batch axes, generating unit-hypersphere $512\text{-D}$ embeddings normalized via $L_2$ norm ($\|v\|_2 = 1.0$).
- **Vectorized Top-2 Cosine Classifier**:
  $$\text{Classification} = \begin{cases} \text{Person Name}, & \text{if } s_{\text{best}} \ge \tau \;\wedge\; (s_{\text{best}} - s_{\text{2nd}}) \ge \text{margin} \\ \text{Unknown}, & \text{otherwise} \end{cases}$$
  Matching $20$ candidate faces against a gallery of $10,000$ enrolled embeddings takes only **$\approx 4.0\text{ ms}$ on CPU** using single-instruction matrix GEMM.

###  Dual-Stage Real-Time Frame Drop Policy
- **Drop-Oldest on Enqueue**: If the camera framerate exceeds inference capacity, the oldest buffered frame is discarded, ensuring the queue depth never induces compounding latency.
- **Latency Guard on Dequeue**: Any frame whose waiting age $(t_{\text{now}} - t_{\text{frame}}) > 150\text{ ms}$ is immediately dropped before running inference, strictly guaranteeing real-time responsiveness.

###  Intelligent 30-Second Alert Debounce Engine
- Prevents notification storms when an authorized individual or intruder remains within camera view.
- Tracks state per `(camera_id, person_id)`. Subsequent detections within $30.0\text{s}$ are suppressed from triggering alarm webhooks and WebSocket alerts while maintaining live session telemetry.

###  Production WebSocket ConnectionManager
- **Concurrent Non-Blocking Broadcast**: Employs `asyncio.gather` with per-client timeouts ($3.0\text{s}$) so slow or stalled network clients cannot bottleneck the server.
- **Bidirectional Heartbeat**: Automated background task transmits periodic pings ($15\text{s}$) and prunes unresponsive connections ($45\text{s}$ timeout) to prevent memory leaks.
- **Dynamic Camera Subscriptions**: Clients can subscribe to specific camera feeds via `{"type": "subscribe", "cameras": ["front_gate"]}`.

###  Thread-Safe SQLite Event Store & REST API
- SQLite database configured with **WAL (Write-Ahead Logging)** mode, `PRAGMA synchronous = NORMAL`, and a $5000\text{ms}$ busy timeout.
- Full-featured REST API (`GET /events`) supporting cursor/offset pagination, timestamp range filtering (`start_ts`, `end_ts`), person identifier filtering, and camera filtering.

---

## 3. Performance Benchmarks

All benchmarks were measured on a physical testbed using the built-in benchmark harness (`python -m recognizer.export_onnx --runs 100`) with **$20$ warmup iterations** and **$100$ timed runs**.

**Hardware Specifications:**
- **CPU:** 12th Gen Intel Core (12 logical cores)
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, CUDA 12.1, cuDNN 9.x)
- **Software:** Python 3.12, PyTorch 2.5.1+cu121, ONNX Runtime 1.23.0

### Latency (ms) & Throughput (FPS) Comparison

| Model / Task | Backend | Device | Input Shape | Mean Latency | P50 (Median) | P95 Latency | P99 Latency | Throughput (FPS) | Speedup |
|:---|:---|:---|:---:|---:|---:|---:|---:|---:|---:|
| **Face Detector (SCRFD-Lite)** | PyTorch Eager (Baseline) | CPU | `[1, 3, 640, 640]` | 89.36 ms | 85.10 ms | 126.10 ms | 159.97 ms | 11.2 FPS | 1.00x |
| **Face Detector (SCRFD-Lite)** | ONNX Runtime (`threads=4`) | CPU | `[1, 3, 640, 640]` | 78.66 ms | 69.85 ms | 134.82 ms | 155.57 ms | 12.7 FPS | 1.14x |
| **Face Detector (SCRFD-Lite)** | ONNX Runtime (`threads=6`) | CPU | `[1, 3, 640, 640]` | **52.18 ms** | 50.81 ms | 65.42 ms | 73.28 ms | **19.2 FPS** | **1.71x** |
| **Face Detector (SCRFD-Lite)** | PyTorch Eager CUDA | RTX 3050 Laptop | `[1, 3, 640, 640]` | 6.20 ms | 6.02 ms | 7.77 ms | 9.28 ms | 161.3 FPS | 1.00x |
| **Face Detector (SCRFD-Lite)** | ONNX Runtime CUDA | RTX 3050 Laptop | `[1, 3, 640, 640]` | 5.50 ms | 5.46 ms | 5.91 ms | 6.19 ms | 181.8 FPS | 1.13x |
| **Face Detector (SCRFD-Lite)** | **ONNX Runtime CUDA (IOBinding)** | **RTX 3050 Laptop** | `[1, 3, 640, 640]` | **4.40 ms** | **4.39 ms** | **4.54 ms** | **4.58 ms** | **227.1 FPS** | **1.41x** |
| **Face Embedding (MobileFaceNet)** | PyTorch Eager (Baseline) | CPU | `[1, 3, 112, 112]` | 11.75 ms | 11.26 ms | 14.07 ms | 26.16 ms | 85.1 FPS | 1.00x |
| **Face Embedding (MobileFaceNet)** | **ONNX Runtime (`threads=4`)** | **CPU** | `[1, 3, 112, 112]` | **3.28 ms** | **3.17 ms** | **4.16 ms** | **4.63 ms** | **305.1 FPS** | **3.59x** |
| **Face Embedding (MobileFaceNet)** | PyTorch Eager CUDA | RTX 3050 Laptop | `[1, 3, 112, 112]` | 5.22 ms | 4.75 ms | 8.47 ms | 8.99 ms | 191.6 FPS | 1.00x |
| **Face Embedding (MobileFaceNet)** | ONNX Runtime CUDA | RTX 3050 Laptop | `[1, 3, 112, 112]` | 1.71 ms | 1.49 ms | 2.89 ms | 3.74 ms | 585.6 FPS | 3.06x |
| **Face Embedding (MobileFaceNet)** | **ONNX Runtime CUDA (IOBinding)** | **RTX 3050 Laptop** | `[1, 3, 112, 112]` | **1.56 ms** | **1.35 ms** | **2.65 ms** | **3.69 ms** | **639.8 FPS** | **3.34x** |
| **Batched Embedding (B=8)** | PyTorch Eager (Baseline) | CPU | `[8, 3, 112, 112]` | 75.18 ms | 70.72 ms | 103.68 ms | 126.41 ms | 106.4 FPS | 1.00x |
| **Batched Embedding (B=8)** | ONNX Runtime (`threads=4`) | CPU | `[8, 3, 112, 112]` | **42.15 ms** | 40.65 ms | 55.25 ms | 60.13 ms | **189.8 FPS** | **1.78x** |
| **Batched Embedding (B=8)** | PyTorch Eager CUDA | RTX 3050 Laptop | `[8, 3, 112, 112]` | 5.60 ms | 5.28 ms | 7.24 ms | 8.51 ms | 1,429.5 FPS | 1.00x |
| **Batched Embedding (B=8)** | **ONNX Runtime CUDA (IOBinding)** | **RTX 3050 Laptop** | `[8, 3, 112, 112]` | **4.50 ms** | **4.49 ms** | **4.62 ms** | **4.71 ms** | **1,777.1 FPS** | **1.24x** |

### Accuracy & Verification Quality
- **ArcFace Loss Backbone:** MobileFaceNet trained on WebFace600K achieves **$99.52\%$ verification accuracy on LFW** and **$94.38\%$ TAR @ FAR=$10^{-4}$ on IJB-C**.
- **Parity Error:** The exported ONNX model maintains strict numerical fidelity with PyTorch eager mode:
  $$\max |y_{\text{pytorch}} - y_{\text{onnx}}| \le 1.56 \times 10^{-7}, \quad \text{Cosine Similarity} = 1.0000000$$

---

## 4. Quickstart Guide

### Native Setup (Virtualenv)

#### 1. Clone & Initialize Environment
```bash
git clone https://github.com/your-username/E2E-Person-Alert-FaceRec-Server.git
cd E2E-Person-Alert-FaceRec-Server

# Create virtual environment (Python 3.10 - 3.12)
python -m venv .venv

# Activate environment:
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Windows GPU Users:** ONNX Runtime CUDA automatically detects and links CUDA 12 / cuDNN runtime DLLs from PyTorch (`.venv/Lib/site-packages/torch/lib`) without requiring a separate system-wide CUDA toolkit installation.

#### 3. Export Models & Generate Benchmarks
```bash
python -m recognizer.export_onnx --export-dir models/faces --runs 100
```
This generates `det_500m.onnx` and `w600k_mbf.onnx` in `models/faces/`.

#### 4. Run Tests
```bash
python tests/test_face_capture.py
python tests/test_snapshot_worker.py
python tests/test_ws_and_events.py
```

#### 5. Start the Server
```bash
uvicorn app.api_min:app --host 0.0.0.0 --port 8000 --reload
```

---

### Docker Compose Deployment

A production-ready `Dockerfile` and `docker-compose.yml` are provided.

```bash
# Build and run containerized server
docker compose up -d --build

# View real-time logs
docker compose logs -f
```

To enable GPU acceleration inside Docker, ensure `nvidia-container-toolkit` is installed on your host, and uncomment the `deploy.resources.reservations` block in `docker-compose.yml`.

---

## 5. Face Enrollment & CLI Testing

### Enrolling Known Faces
Place sample images of an authorized person in a directory (e.g., `data/faces/raw/alice/*.jpg`) and run:

```bash
python -m collector.face_capture --enroll "Alice" --images data/faces/raw/alice
```
The pipeline automatically detects faces, verifies sharpness, computes $512\text{-D}$ normalized embeddings, and appends them to `data/faces/gallery.npz`.

### Real-Time Webcam / RTSP Demo
To test detection, alignment, and recognition on a local camera or RTSP stream:

```bash
# Local USB Webcam (Index 0)
python -m collector.face_capture --source 0

# RTSP IP Camera Stream
python -m collector.face_capture --source "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main"
```

---

## 6. REST & WebSocket API Reference

### WebSocket Stream: `/ws/alerts` or `/stream`
Connect via WebSocket client (`client/ws_test.py`):
```bash
python client/ws_test.py
```

**Client Message Types:**
- `{"type": "ping"}` $\to$ Server replies with `{"type": "pong", "ts": 1730000000.0}`.
- `{"type": "subscribe", "cameras": ["front_gate"]}` $\to$ Filters alerts for specific camera IDs.

**Live Alert Event Payload Example:**
```json
{
  "type": "person_event",
  "id": "7f9a8b1c4e2d43e590218dbb96a1e342",
  "evt_id": "7f9a8b1c4e2d43e590218dbb96a1e342",
  "ts": 1730001245.82,
  "camera_id": "front_gate",
  "source": "esp32",
  "person_id": "Alice",
  "confidence": 0.8924,
  "bbox": [142.5, 88.0, 390.2, 510.4],
  "snapshot_url": "/snapshots/events/7f9a8b1c4e2d43e590218dbb96a1e342.jpg",
  "alert_type": "face_known"
}
```

### REST Endpoints

| Method | Endpoint | Description | Query / Form Parameters |
|:---|:---|:---|:---|
| `POST` | `/snapshot` | Ingests snapshot from IP camera / ESP32 | `camera_id`, `token`, `file` (multipart/form-data) |
| `GET` | `/events` | Paginated query of intrusion/face events | `page=1`, `page_size=20`, `person_id=Alice`, `start_ts=...`, `end_ts=...`, `camera_id=front_gate` |
| `GET` | `/events/recent` | Retrieves recent events list | `limit=50`, `offset=0`, `person_id=...` |
| `GET` | `/events/{evt_id}` | Retrieves detailed event metadata | Path parameter `evt_id` |
| `GET` | `/preview/latest` | Retrieves latest live camera thumbnail | `camera_id=front_gate` |
| `POST` | `/buzz` | Triggers active deterrence buzz/alarm | `camera_id`, `duration_ms` |
| `POST` | `/auth/login` | User login to retrieve JWT token | JSON `{"username": "...", "password": "..."}` |
| `POST` | `/auth/register`| Registers new dashboard user | Form `username`, `password` |

---

## 7. Project Directory Structure

```text
E2E-Person-Alert-FaceRec-Server/
├── app/
│   ├── api_min.py             # FastAPI entrypoint: REST routes, WebSockets, lifespan management
│   ├── db.py                  # SQLAlchemy engine with SQLite WAL mode and connection listener
│   ├── models.py              # ORM models (User, Event table schema)
│   ├── snapshot_worker.py     # Production async snapshot worker with SmartFrameQueue & debounce
│   └── ws_manager.py          # WebSocket ConnectionManager with heartbeat ping-pong & reaper
├── collector/
│   └── face_capture.py        # Core face pipeline: SCRFD/YuNet, Umeyama alignment, ArcFace matcher
├── recognizer/
│   ├── model.py               # PyTorch models: MobileFaceNet (512-D) and LiteSCRFDDetector
│   ├── export_onnx.py         # ONNX exporter with dynamic axes & latency/FPS benchmark suite
│   ├── dataset.py             # Dataset loaders and training data transformations
│   ├── train.py               # Model training script
│   └── eval.py                # Offline model evaluation and metric reporting
├── detector/
│   └── person_detector.py     # Standalone OpenCV / YOLO RTSP video ingestion stream worker
├── data/                      # Persistent storage (snapshots, SQLite database, enrolled gallery)
│   ├── app.db                 # SQLite database storing user credentials and alert events
│   ├── faces/gallery.npz      # Enrolled face identities and L2-normalized feature vectors
│   └── snapshots/
│       ├── preview/           # Latest per-camera preview frame thumbnails
│       └── events/            # Annotated snapshot images with bounding boxes & OSD headers
├── models/
│   └── faces/                 # Deployed ONNX models (det_500m.onnx, w600k_mbf.onnx)
├── tests/
│   ├── test_face_capture.py   # Unit tests: Umeyama alignment, vectorized cosine matching, sharpness
│   ├── test_snapshot_worker.py# Unit tests: Frame queue overflow drop, 30s debounce, SQLite persistence
│   └── test_ws_and_events.py  # Unit tests: Multi-client broadcast, dead connection cleanup, REST filtering
├── client/
│   └── ws_test.py             # Lightweight CLI WebSocket subscriber for testing live alert streams
├── configs.yaml               # System configuration: camera, YOLO, face rec thresholds, debounce
├── requirements.txt           # Python package dependencies
├── Dockerfile                 # Production Docker container build
├── docker-compose.yml         # Container orchestration configuration
└── README.md                  # Comprehensive system documentation
```

---

## 8. Configuration Reference (`configs.yaml`)

```yaml
paths:
  snapshot_dir: "data/snapshots"
  static_mount: "/static"

security:
  ingest_token: "SHARED_TOKEN_ABC"    # Shared bearer token for camera ingestion

yolo:
  device: "cuda"                      # "cuda" or "cpu"
  model: "yolov8n.pt"
  imgsz: 640
  conf_thres: 0.35

face:
  device: "auto"                      # "auto", "cuda", or "cpu"
  detector: "scrfd"                   # "scrfd", "yunet", or "yolo"
  det_model: "models/faces/det_500m.onnx"
  det_input_size: 640
  det_score_thresh: 0.50
  rec_model: "models/faces/w600k_mbf.onnx"
  rec_threshold: 0.40                 # Cosine similarity >= 0.40 -> Known
  rec_margin: 0.05                    # Best similarity - 2nd best >= margin
  min_face_size: 40                   # Minimum face bounding box size (px)
  min_sharpness: 25.0                 # Minimum Laplacian variance (blur filter)
  gallery_path: "data/faces/gallery.npz"

worker:
  max_queue_size: 15                  # Buffer capacity before drop-oldest activates
  max_latency_ms: 150                 # Stale frame drop threshold (< 150ms real-time latency)
  debounce_sec: 30.0                  # Cooldown per person per camera (seconds)
  unknown_debounce_sec: 30.0          # Cooldown for unidentified intruders
  save_annotated_snapshot: true

alarm:
  enabled: true
  map:
    front_gate: "http://esp32-front.local/alarm/on"
  cooldown_s: 10
```

---

## License

This project is licensed under the [MIT License](LICENSE).
