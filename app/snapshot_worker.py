# app/snapshot_worker.py
from __future__ import annotations

import asyncio
import time
import uuid
import json
from pathlib import Path
from typing import Callable, Awaitable, Dict, Any

import cv2
from ultralytics import YOLO
import yaml

from app.models import Event
from app.db import SessionLocal


BASE_DIR = Path(__file__).resolve().parents[1]


def load_app_config(cfg_path: str | Path | None = None) -> dict:
    """
    Đọc cấu hình YAML dùng chung cho backend.

    Mặc định: file `configs.yaml` ở thư mục gốc project (nơi chạy `uvicorn`).
    """
    if cfg_path is None:
        cfg_path = BASE_DIR / "configs.yaml"
    else:
        cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        print(f"[SnapshotWorker] WARNING: config file not found at {cfg_path}, using defaults")
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


class SnapshotWorker:
    """
    Nhận frame từ ESP32 qua /snapshot (hoặc /burst/frame),
    chạy YOLO, cập nhật preview, và tạo Event khi phát hiện người đủ lâu.

    Worker này tự đọc cấu hình từ `configs.yaml`, KHÔNG phụ thuộc vào CFG trong api_min.
    """

    def __init__(
        self,
        emit_async: Callable[[dict], Awaitable[None]],
        cfg_path: str | Path | None = None,
    ):
        # ---- Load config từ YAML ----
        self.cfg = load_app_config(cfg_path)
        self.emit_async = emit_async
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # ---- Cấu hình YOLO ----
        yolo_cfg = self.cfg.get("yolo", {})
        model_path = yolo_cfg.get("model", "yolov8n.pt")

        # class "person" (mặc định COCO id = 0)
        self.person_class_ids = set(
            yolo_cfg.get("person_class_ids", [0])
        )

        # imgsz + conf_thres lấy từ YAML
        self.imgsz = int(yolo_cfg.get("imgsz", 640))
        self.conf_thres = float(yolo_cfg.get("conf_thres", 0.25))

        # Thiết bị (cpu / cuda) nếu có
        device = yolo_cfg.get("device", None)  # trong configs.yaml có field device
        if device is not None:
            self.model = YOLO(model_path).to(device)
        else:
            self.model = YOLO(model_path)

        # ---- Đường dẫn lưu ảnh ----
        paths_cfg = self.cfg.get("paths", {})
        root = Path(paths_cfg.get("snapshot_dir", "data/snapshots"))
        self.preview_dir = root / "preview"   # ảnh mới nhất cho mỗi camera (preview/latest)
        self.events_dir = root / "events"     # ảnh snapshot cho event
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # ---- Cấu hình ngưỡng thời gian từ YAML (events.*) ----
        ev_cfg = self.cfg.get("events", {})
        self.yellow_min_sec = float(ev_cfg.get("yellow_min_sec", 15.0))
        self.red_min_sec = float(ev_cfg.get("red_min_sec", 60.0))
        self.session_gap_sec = float(ev_cfg.get("session_gap_sec", 5.0))

        # state: mỗi camera 1 session
        # sessions[camera_id] = {
        #   "start_ts": float,
        #   "last_ts": float,
        #   "evt_id": str | None,
        #   "alert_type": str | None,  # "yellow"|"red"|None
        # }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def run_forever(self):
        """Chạy worker vĩnh viễn trong background."""
        while True:
            job = await self.queue.get()
            try:
                await self._handle_job(job)
            except Exception as e:
                print("[Worker] error:", e)
            finally:
                self.queue.task_done()

    async def _handle_job(self, job: Dict[str, Any]):
        """
        job = {
          "camera_id": str,
          "ts": float,
          "path": str,          # file ảnh vừa lưu (preview)
        }
        """
        camera_id = job["camera_id"]
        ts = float(job["ts"])
        img_path = Path(job["path"])

        # Đảm bảo file tồn tại
        if not img_path.exists():
            return

        # Đọc ảnh bằng OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            return

        # --- YOLO detect ---
        # Dùng imgsz + conf_thres từ config
        results = self.model(
            img,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            verbose=False,
        )[0]

        person_boxes: list[tuple[float, float, float, float, float]] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in self.person_class_ids and conf >= self.conf_thres:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                person_boxes.append((conf, x1, y1, x2, y2))

        # Nếu không có người -> xử lý khả năng kết thúc session
        if not person_boxes:
            self._maybe_close_session(camera_id, ts)
            return

        # Có ít nhất 1 người → chọn box có confidence cao nhất
        person_boxes.sort(key=lambda p: p[0], reverse=True)
        _, x1, y1, x2, y2 = person_boxes[0]
        bbox_json = json.dumps([x1, y1, x2, y2])

        # Cập nhật / tạo session cho camera này
        sess = self.sessions.get(camera_id)
        if sess is None:
            sess = {
                "start_ts": ts,
                "last_ts": ts,
                "evt_id": None,
                "alert_type": None,
            }
            self.sessions[camera_id] = sess
        else:
            sess["last_ts"] = ts

        duration = ts - sess["start_ts"]

        # Loại 1: < yellow_min_sec -> không tạo event
        if duration < self.yellow_min_sec:
            return

        # Đường dẫn snapshot riêng cho event
        if sess["evt_id"] is None:
            evt_id = uuid.uuid4().hex
            sess["evt_id"] = evt_id
        else:
            evt_id = sess["evt_id"]

        snap_file = self.events_dir / f"{evt_id}.jpg"
        snapshot_url = f"/snapshots/events/{evt_id}.jpg"

        # Lưu ảnh event (có thể crop theo bbox, ở đây để full frame)
        cv2.imwrite(str(snap_file), img)

        # Xác định mức cảnh báo hiện tại
        if duration >= self.red_min_sec:
            new_alert = "red"
        else:
            new_alert = "yellow"

        # Nếu đã có event:
        #  - nếu từ yellow lên red thì update
        #  - nếu đã red rồi thì chỉ update duration, không đổi loại
        if sess["alert_type"] is not None:
            # Nếu đang yellow mà giờ nên lên red
            if sess["alert_type"] == "yellow" and new_alert == "red":
                await self._update_event(
                    evt_id=evt_id,
                    ts=ts,
                    duration=duration,
                    alert_type="red",
                    bbox_json=bbox_json,
                    snapshot_url=snapshot_url,
                    camera_id=camera_id,
                )
                sess["alert_type"] = "red"
            else:
                # chỉ update duration/bbox cho event hiện tại
                await self._update_event(
                    evt_id=evt_id,
                    ts=ts,
                    duration=duration,
                    alert_type=sess["alert_type"],
                    bbox_json=bbox_json,
                    snapshot_url=snapshot_url,
                    camera_id=camera_id,
                )
        else:
            # Chưa có event -> tạo mới (yellow hoặc red)
            await self._create_event(
                evt_id=evt_id,
                ts=ts,
                duration=duration,
                alert_type=new_alert,
                bbox_json=bbox_json,
                snapshot_url=snapshot_url,
                camera_id=camera_id,
            )
            sess["alert_type"] = new_alert

    def _maybe_close_session(self, camera_id: str, now_ts: float):
        """
        Nếu quá session_gap_sec không thấy người -> kết thúc session.
        (Không xoá event, vì event đã lưu rồi)
        """
        sess = self.sessions.get(camera_id)
        if not sess:
            return
        # nếu lâu không thấy người → reset
        if now_ts - sess["last_ts"] >= self.session_gap_sec:
            self.sessions.pop(camera_id, None)

    async def _create_event(
        self,
        evt_id: str,
        ts: float,
        duration: float,
        alert_type: str,
        bbox_json: str,
        snapshot_url: str,
        camera_id: str,
    ):
        data = {
            "evt_id": evt_id,
            "ts": ts,
            "camera_id": camera_id,
            "source": "esp32",
            "bbox": bbox_json,
            "snapshot_url": snapshot_url,
            "person_id": None,
            "duration_sec": duration,
            "alert_type": alert_type,
        }

        db_event = Event(**data)
        try:
            with SessionLocal() as db:
                db.add(db_event)
                db.commit()
        except Exception as e:
            print("[Worker] DB save error (create):", e)
            return

        # Phát cho UI (Live)
        await self.emit_async({
            "type": "person_event",
            "payload": data,
        })

    async def _update_event(
        self,
        evt_id: str,
        ts: float,
        duration: float,
        alert_type: str,
        bbox_json: str,
        snapshot_url: str,
        camera_id: str,
    ):
        """
        Update duration/alert_type cho event đã tồn tại,
        và bắn lại payload cho UI (UI có thể overwrite trạng thái hiện tại).
        """
        try:
            with SessionLocal() as db:
                ev = db.query(Event).filter(Event.evt_id == evt_id).first()
                if not ev:
                    # nếu vì lý do nào đó event mất, tạo lại
                    await self._create_event(
                        evt_id, ts, duration, alert_type, bbox_json, snapshot_url, camera_id
                    )
                    return

                ev.ts = ts
                ev.duration_sec = duration
                ev.alert_type = alert_type
                ev.bbox = bbox_json
                ev.snapshot_url = snapshot_url
                db.commit()
        except Exception as e:
            print("[Worker] DB save error (update):", e)
            return

        payload = {
            "evt_id": evt_id,
            "ts": ts,
            "camera_id": camera_id,
            "source": "esp32",
            "bbox": bbox_json,
            "snapshot_url": snapshot_url,
            "person_id": None,
            "duration_sec": duration,
            "alert_type": alert_type,
        }

        await self.emit_async({
            "type": "person_event",
            "payload": payload,
        })
