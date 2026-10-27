# app/snapshot_worker.py
"""
Production Real-Time Snapshot Worker for Person Detection and Face Recognition.

Features:
- Decoupled Ingestion & Inference: FastAPI I/O loop receives frames via SmartFrameQueue,
  deep learning inference runs in a dedicated background worker thread (CPU/GPU-bound).
- Smart Frame Drop: Drop-Oldest on queue overflow + Latency Guard dropping any frame
  older than 150ms to strictly guarantee real-time latency (< 150ms).
- Deep Learning Pipeline: YOLOv8 person detection + FacePipeline (SCRFD/YuNet/ArcFace)
  with facial landmarks, face alignment, and cosine similarity matching (Known/Unknown).
- Thread-safe SQLite Recording: Saves annotated snapshots to data/snapshots/events/
  and metadata (timestamp, person_id, confidence, bbox) to SQLite via SQLAlchemy.
- Debounce / Cooldown Engine: Per-camera, per-person 30s cooldown prevents alert fatigue
  when the same individual appears continuously in the camera view.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from app.db import SessionLocal
from app.models import Event

log = logging.getLogger("snapshot_worker")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    )
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]


def load_app_config(cfg_path: str | Path | None = None) -> dict:
    if cfg_path is None:
        cfg_path = BASE_DIR / "configs.yaml"
    else:
        cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        log.warning(f"Config file not found at {cfg_path}, using defaults")
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# 1. Smart Frame Queue with Overflow Drop & Async Interface
# =============================================================================
class SmartFrameQueue:
    """Thread-safe queue with Drop-Oldest on overflow and async put compatibility."""

    def __init__(self, maxsize: int = 15, max_latency_ms: float = 150.0):
        self.maxsize = maxsize
        self.max_latency_sec = max_latency_ms / 1000.0
        self._q: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self.dropped_overflow_count = 0
        self.dropped_stale_count = 0
        self.enqueued_count = 0
        self.processed_count = 0

    def put_nowait(self, item: Dict[str, Any]) -> bool:
        """Enqueue item immediately. If full, drop the oldest frame (Drop-Oldest)."""
        with self._lock:
            self.enqueued_count += 1
            if self._q.full():
                try:
                    _ = self._q.get_nowait()
                    self.dropped_overflow_count += 1
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(item)
                return True
            except queue.Full:
                return False

    async def put(self, item: Dict[str, Any]) -> None:
        """Async compatibility method for FastAPI handlers."""
        self.put_nowait(item)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch item for the inference thread."""
        item = self._q.get(block=block, timeout=timeout)
        self.processed_count += 1
        return item

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def get_stats(self) -> dict:
        return {
            "qsize": self.qsize(),
            "enqueued": self.enqueued_count,
            "processed": self.processed_count,
            "dropped_overflow": self.dropped_overflow_count,
            "dropped_stale": self.dropped_stale_count,
        }


# =============================================================================
# 2. Debounce / Alert Cooldown Manager
# =============================================================================
class AlertDebounceManager:
    """Per-camera, per-person alert debounce tracker (e.g. max 1 alert per 30s)."""

    def __init__(self, default_cooldown_sec: float = 30.0):
        self.default_cooldown_sec = default_cooldown_sec
        self._lock = threading.Lock()
        self._last_alert_times: Dict[Tuple[str, str], float] = {}
        self._last_cleanup = time.time()

    def should_alert(
        self, camera_id: str, person_id: str, now_ts: float, custom_cooldown: Optional[float] = None
    ) -> bool:
        """Return True if cooldown has elapsed for this person on this camera."""
        cooldown = custom_cooldown if custom_cooldown is not None else self.default_cooldown_sec
        key = (camera_id, str(person_id))
        with self._lock:
            last_ts = self._last_alert_times.get(key, 0.0)
            if (now_ts - last_ts) >= cooldown:
                return True
            return False

    def mark_alerted(self, camera_id: str, person_id: str, now_ts: float) -> None:
        """Record alert timestamp and periodically prune stale entries."""
        key = (camera_id, str(person_id))
        with self._lock:
            self._last_alert_times[key] = now_ts

            # Periodic cleanup every 5 minutes
            if now_ts - self._last_cleanup > 300.0:
                self._last_cleanup = now_ts
                cutoff = now_ts - 600.0
                stale_keys = [k for k, ts in self._last_alert_times.items() if ts < cutoff]
                for k in stale_keys:
                    self._last_alert_times.pop(k, None)


# =============================================================================
# 3. Detection & Face Overlay Drawing Utilities
# =============================================================================
def draw_detection_overlay(
    img_bgr: np.ndarray,
    persons: List[Dict[str, Any]],
    faces: List[Dict[str, Any]],
    info_header: str,
) -> np.ndarray:
    """Annotate snapshot image with bounding boxes, badges, landmarks and header OSD."""
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]

    # Draw person detections
    for p in persons:
        x1, y1, x2, y2 = map(int, p["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        conf = p.get("conf", 0.0)
        label = f"Person: {conf:.2f}"

        # Orange box for person
        color = (255, 128, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Draw face detections & recognition
    for f in faces:
        fx1, fy1, fx2, fy2 = map(int, f["bbox"])
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w - 1, fx2), min(h - 1, fy2)
        person_name = f.get("label", "Unknown")
        similarity = f.get("similarity", 0.0)
        status = f.get("status", "unknown")

        if status == "known":
            color = (0, 220, 0)      # Green
            text = f"{person_name} ({similarity:.2f})"
        elif status == "low_quality":
            color = (0, 215, 255)    # Yellow
            text = f"Blur/Small ({similarity:.2f})"
        else:
            color = (0, 0, 255)      # Red
            text = f"Unknown ({similarity:.2f})"

        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)

        # 5 Facial landmarks if available
        if "kps" in f and f["kps"]:
            for kx, ky in f["kps"]:
                cv2.circle(annotated, (int(kx), int(ky)), 2, (255, 255, 0), -1)

        # Badge
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by = min(h - 2, fy2 + th + 6)
        cv2.rectangle(annotated, (fx1, fy2), (fx1 + tw + 6, by), color, -1)
        cv2.putText(
            annotated, text, (fx1 + 3, by - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Top info header banner
    if info_header:
        cv2.rectangle(annotated, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.putText(
            annotated, info_header, (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


# =============================================================================
# 4. Production Snapshot Worker
# =============================================================================
class SnapshotWorker:
    def __init__(
        self,
        arg1: Any = None,
        arg2: Any = None,
        *,
        cfg: Optional[dict] = None,
        emit_async: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        """
        Flexible constructor supporting:
        - SnapshotWorker(emit_async)
        - SnapshotWorker(emit_async, cfg_path)
        - SnapshotWorker(cfg_dict, emit_async)
        """
        if callable(arg1):
            self.emit_async = arg1
            cfg_path = arg2 if isinstance(arg2, (str, Path)) else None
            self.cfg = cfg or load_app_config(cfg_path)
        elif isinstance(arg1, (dict, str, Path)):
            self.cfg = arg1 if isinstance(arg1, dict) else load_app_config(arg1)
            self.emit_async = arg2 or emit_async
        else:
            self.cfg = cfg or load_app_config()
            self.emit_async = emit_async

        # Worker & Queue configs
        worker_cfg = self.cfg.get("worker", {})
        self.max_queue_size = int(worker_cfg.get("max_queue_size", 15))
        self.max_latency_ms = float(worker_cfg.get("max_latency_ms", 150.0))
        self.debounce_sec = float(worker_cfg.get("debounce_sec", 30.0))
        self.unknown_debounce_sec = float(worker_cfg.get("unknown_debounce_sec", self.debounce_sec))
        self.save_annotated = bool(worker_cfg.get("save_annotated_snapshot", True))

        # Smart Frame Queue (Decoupling I/O and Inference)
        self.queue = SmartFrameQueue(
            maxsize=self.max_queue_size,
            max_latency_ms=self.max_latency_ms,
        )

        # Alert Debounce Manager
        self.debounce = AlertDebounceManager(default_cooldown_sec=self.debounce_sec)

        # Storage paths
        paths_cfg = self.cfg.get("paths", {})
        root = Path(paths_cfg.get("snapshot_dir", "data/snapshots"))
        self.preview_dir = root / "preview"
        self.events_dir = root / "events"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Event duration / session configs
        ev_cfg = self.cfg.get("events", {})
        self.yellow_min_sec = float(ev_cfg.get("yellow_min_sec", 15.0))
        self.red_min_sec = float(ev_cfg.get("red_min_sec", 60.0))
        self.session_gap_sec = float(ev_cfg.get("session_gap_sec", 5.0))
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # YOLOv8 Person Detector
        yolo_cfg = self.cfg.get("yolo", {})
        model_path = yolo_cfg.get("model", "yolov8n.pt")
        self.person_class_ids = set(yolo_cfg.get("person_class_ids", [0]))
        self.imgsz = int(yolo_cfg.get("imgsz", 640))
        self.conf_thres = float(yolo_cfg.get("conf_thres", 0.35))
        device = yolo_cfg.get("device", "cuda")

        log.info(f"Loading YOLO person detector ({model_path}) on device: {device}...")
        try:
            self.yolo_model = YOLO(model_path).to(device)
        except Exception as e:
            log.warning(f"Could not load YOLO on {device} ({e}), falling back to CPU")
            self.yolo_model = YOLO(model_path)

        # Face Recognition Pipeline
        self.face_pipeline = None
        self._init_face_pipeline()

        # Threading & Async coordination
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Warm up YOLO model
        self._warmup()

    def _init_face_pipeline(self):
        """Attempt to initialize FacePipeline; gracefully fall back to person-only if models absent."""
        try:
            from collector.face_capture import FacePipeline, FacePipelineConfig
            face_cfg = FacePipelineConfig.from_yaml(section="face")
            self.face_pipeline = FacePipeline.from_config(face_cfg)
            log.info("FacePipeline successfully initialized for real-time face recognition.")
        except Exception as e:
            self.face_pipeline = None
            log.warning(
                f"FacePipeline disabled (models not ready or missing: {e}). "
                "Running in Person-Detection mode."
            )

    def _warmup(self):
        """Pre-warm GPU/CPU execution contexts."""
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.yolo_model(dummy, imgsz=self.imgsz, verbose=False)
            log.info("YOLO model warmup complete.")
        except Exception as e:
            log.warning(f"Warmup warning: {e}")

    # -------------------------------------------------------------------------
    # Lifecycle & Loop Management
    # -------------------------------------------------------------------------
    def start_worker_thread(self):
        """Start the background DL inference worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._inference_loop,
                name="SnapshotWorker-InferenceThread",
                daemon=True,
            )
            self._worker_thread.start()
            log.info("SnapshotWorker inference thread started.")

    async def run_forever(self):
        """Main entry point called by FastAPI lifespan task."""
        self.loop = asyncio.get_running_loop()
        self.start_worker_thread()

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            log.info("SnapshotWorker run_forever task cancelled.")
        finally:
            self.stop()

    async def start(self):
        """Alias for run_forever."""
        await self.run_forever()

    def stop(self):
        """Gracefully stop worker thread."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        log.info("SnapshotWorker stopped.")

    # -------------------------------------------------------------------------
    # Ingestion API (called by FastAPI route / camera streaming)
    # -------------------------------------------------------------------------
    async def enqueue(
        self,
        camera_id: str,
        path: Optional[str | Path] = None,
        frame: Optional[np.ndarray] = None,
        img_bytes: Optional[bytes] = None,
        ts: Optional[float] = None,
    ) -> bool:
        """Non-blocking ingest endpoint for camera frames."""
        now_ts = ts if ts is not None else time.time()
        job = {
            "camera_id": camera_id,
            "ts": now_ts,
            "path": str(path) if path else None,
            "frame": frame,
            "img_bytes": img_bytes,
        }
        return self.queue.put_nowait(job)

    # -------------------------------------------------------------------------
    # Background Deep Learning Inference Thread Loop
    # -------------------------------------------------------------------------
    def _inference_loop(self):
        log.info("Entering inference worker loop.")
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            try:
                # 1. Smart Frame Drop: Latency Guard (< 150ms real-time limit)
                now = time.time()
                job_ts = float(job.get("ts", now))
                latency_ms = (now - job_ts) * 1000.0

                if latency_ms > self.max_latency_ms:
                    self.queue.dropped_stale_count += 1
                    if self.queue.dropped_stale_count % 25 == 1:
                        log.warning(
                            f"Dropped stale frame (Latency: {latency_ms:.1f}ms > {self.max_latency_ms}ms, "
                            f"Total dropped: {self.queue.dropped_stale_count})"
                        )
                    continue

                # 2. Process frame
                t_start = time.perf_counter()
                self._process_job(job, job_ts, latency_ms)
                _proc_time_ms = (time.perf_counter() - t_start) * 1000.0

            except Exception as e:
                log.exception(f"Error processing frame job: {e}")
            finally:
                self.queue.task_done()

    # -------------------------------------------------------------------------
    # Frame Processing & Recognition Pipeline
    # -------------------------------------------------------------------------
    def _process_job(self, job: Dict[str, Any], job_ts: float, latency_ms: float):
        camera_id = job["camera_id"]

        # Decode image
        img = None
        if job.get("frame") is not None:
            img = job["frame"]
        elif job.get("img_bytes") is not None:
            img = cv2.imdecode(
                np.frombuffer(job["img_bytes"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
        elif job.get("path") is not None:
            p = Path(job["path"])
            if p.exists():
                img = cv2.imread(str(p))

        if img is None:
            return

        # 1. YOLO Person Detection
        results = self.yolo_model(
            img,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            verbose=False,
        )[0]

        person_detections: List[Dict[str, Any]] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in self.person_class_ids and conf >= self.conf_thres:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                person_detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                    }
                )

        if not person_detections:
            self._maybe_close_session(camera_id, job_ts)
            return

        # Sort by confidence descending
        person_detections.sort(key=lambda x: x["conf"], reverse=True)
        primary_person = person_detections[0]

        # 2. Face Detection & Recognition
        face_detections: List[Dict[str, Any]] = []
        recognized_names: List[str] = []
        best_face_sim: float = 0.0

        if self.face_pipeline is not None:
            try:
                face_res_list = self.face_pipeline.process(img)
                for fr in face_res_list:
                    face_info = {
                        "bbox": fr.bbox,
                        "kps": fr.kps,
                        "label": fr.label,
                        "similarity": fr.similarity,
                        "status": fr.status.value,
                    }
                    face_detections.append(face_info)
                    if fr.status.value == "known":
                        recognized_names.append(fr.label)
                        best_face_sim = max(best_face_sim, fr.similarity)
            except Exception as e:
                log.warning(f"Face recognition error on frame: {e}")

        # Determine Primary Identified Person
        if recognized_names:
            primary_person_id = recognized_names[0]
            confidence_score = best_face_sim
            alert_type_name = "face_known"
        else:
            primary_person_id = "Unknown"
            confidence_score = primary_person["conf"]
            alert_type_name = "face_unknown" if face_detections else "person_alert"

        # 3. Debounce / Alert Cooldown Check (30 seconds per person per camera)
        cooldown = (
            self.unknown_debounce_sec
            if primary_person_id == "Unknown"
            else self.debounce_sec
        )
        should_send_alert = self.debounce.should_alert(
            camera_id=camera_id,
            person_id=primary_person_id,
            now_ts=job_ts,
            custom_cooldown=cooldown,
        )

        # 4. Session Tracking for Persistent Presence (Yellow / Red alarm)
        session_alert_type = self._update_session(camera_id, job_ts)
        if session_alert_type in ("yellow", "red"):
            alert_type_name = session_alert_type

        # 5. Handle Alert Triggering
        if should_send_alert:
            self.debounce.mark_alerted(camera_id, primary_person_id, job_ts)
            self._trigger_event(
                img_bgr=img,
                camera_id=camera_id,
                job_ts=job_ts,
                person_id=primary_person_id,
                confidence=confidence_score,
                alert_type=alert_type_name,
                primary_bbox=primary_person["bbox"],
                person_detections=person_detections,
                face_detections=face_detections,
                latency_ms=latency_ms,
            )

    # -------------------------------------------------------------------------
    # Event Persistence & WebSocket Notification
    # -------------------------------------------------------------------------
    def _trigger_event(
        self,
        img_bgr: np.ndarray,
        camera_id: str,
        job_ts: float,
        person_id: str,
        confidence: float,
        alert_type: str,
        primary_bbox: List[float],
        person_detections: List[Dict[str, Any]],
        face_detections: List[Dict[str, Any]],
        latency_ms: float,
    ):
        evt_id = uuid.uuid4().hex
        now_dt = datetime.datetime.fromtimestamp(job_ts).strftime("%Y-%m-%d %H:%M:%S")
        info_header = (
            f"CAM: {camera_id} | {now_dt} | Latency: {latency_ms:.0f}ms | Alert: {alert_type.upper()}"
        )

        # 1. Annotate snapshot with bounding boxes
        annotated_img = draw_detection_overlay(
            img_bgr=img_bgr,
            persons=person_detections,
            faces=face_detections,
            info_header=info_header,
        )

        # 2. Save annotated image to data/snapshots/events/
        snap_file = self.events_dir / f"{evt_id}.jpg"
        snapshot_url = f"/snapshots/events/{evt_id}.jpg"
        try:
            cv2.imwrite(str(snap_file), annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as e:
            log.error(f"Failed to write snapshot {snap_file}: {e}")

        # 3. Save metadata to SQLite DB (Thread-safe)
        evt_record = {
            "evt_id": evt_id,
            "ts": job_ts,
            "camera_id": camera_id,
            "source": "esp32",
            "bbox": primary_bbox,
            "snapshot_url": snapshot_url,
            "person_id": person_id,
            "confidence": float(confidence),
            "alert_type": alert_type,
        }
        self._save_event_db(evt_record)

        # 4. Dispatch WebSocket Notification to FastAPI
        payload = {
            "type": "person_event",
            "id": evt_id,
            **evt_record,
            "all_faces": face_detections,
            "payload": {
                **evt_record,
                "bbox": json.dumps(primary_bbox),
                "all_faces": face_detections,
            },
        }
        self._dispatch_event(payload)

        log.info(
            f"ALERT [{camera_id}] Person: {person_id} | Conf: {confidence:.2f} | "
            f"Type: {alert_type} | Saved: {snapshot_url}"
        )

    def _save_event_db(self, evt_data: dict) -> bool:
        """Write event record to SQLite DB in a thread-safe scoped session."""
        try:
            with SessionLocal() as db:
                db_event = Event(
                    evt_id=evt_data["evt_id"],
                    ts=evt_data["ts"],
                    camera_id=evt_data["camera_id"],
                    source=evt_data.get("source", "esp32"),
                    bbox=json.dumps(evt_data.get("bbox", [])),
                    snapshot_url=evt_data.get("snapshot_url"),
                    person_id=evt_data.get("person_id"),
                    confidence=evt_data.get("confidence"),
                    alert_type=evt_data.get("alert_type"),
                )
                db.add(db_event)
                db.commit()
                return True
        except Exception as e:
            log.error(f"SQLite DB save error: {e}")
            return False

    def _dispatch_event(self, payload: dict):
        """Safely schedule async WebSocket emit on FastAPI's event loop."""
        if self.emit_async is None:
            return

        if asyncio.iscoroutinefunction(self.emit_async):
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.emit_async(payload), self.loop)
            else:
                try:
                    asyncio.run(self.emit_async(payload))
                except Exception:
                    pass
        else:
            try:
                self.emit_async(payload)
            except Exception as e:
                log.error(f"Error in emit_async callback: {e}")

    # -------------------------------------------------------------------------
    # Session Gap & Duration Tracking
    # -------------------------------------------------------------------------
    def _update_session(self, camera_id: str, ts: float) -> Optional[str]:
        sess = self.sessions.get(camera_id)
        if sess is None:
            self.sessions[camera_id] = {
                "start_ts": ts,
                "last_ts": ts,
                "alert_type": None,
            }
            return None

        sess["last_ts"] = ts
        duration = ts - sess["start_ts"]
        if duration >= self.red_min_sec:
            return "red"
        elif duration >= self.yellow_min_sec:
            return "yellow"
        return None

    def _maybe_close_session(self, camera_id: str, now_ts: float):
        sess = self.sessions.get(camera_id)
        if not sess:
            return
        if now_ts - sess["last_ts"] >= self.session_gap_sec:
            self.sessions.pop(camera_id, None)
