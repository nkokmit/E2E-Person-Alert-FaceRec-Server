# app/snapshot_worker.py
from __future__ import annotations
import asyncio, time, uuid, json
from pathlib import Path
from typing import Callable, Awaitable, Dict, Any
from ultralytics import YOLO
import cv2

class SnapshotWorker:
    def __init__(self, cfg: dict, emit_async: Callable[[dict], Awaitable[None]]):
        self.cfg = cfg
        self.emit_async = emit_async
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.model = YOLO(cfg["yolo"]["model"])
        self.device = cfg["yolo"].get("device", "cpu")
        self.imgsz = int(cfg["yolo"].get("imgsz", 640))
        self.conf = float(cfg["yolo"].get("conf", 0.45))

    async def start(self):
        while True:
            item = await self.queue.get()
            try:
                await self._process_item(item)
            except Exception:
                pass
            finally:
                self.queue.task_done()

    async def _process_item(self, item: Dict[str, Any]):
        img_path: Path = item["img_path"]
        camera_id: str = item["camera_id"]
        ts = time.time()

        frame = cv2.imread(str(img_path))
        if frame is None:
            return
        res = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]

        # class 0 = person
        persons = [(b.xyxy[0].tolist(), float(b.conf[0].item()))
                   for b in res.boxes if int(b.cls[0].item()) == 0]

        if not persons:
            return  # Không có người -> không phát event

        xyxy, score = max(persons, key=lambda t: t[1])
        xyxy = [int(x) for x in xyxy]

        evt = {
            "id": f"evt_{uuid.uuid4().hex[:8]}",
            "ts": ts,
            "camera_id": camera_id,
            "source": "esp32",                               # từ ESP32 upload
            "bbox": xyxy,
            "snapshot_url": f"/static/snapshots/{img_path.name}",
            "person": None
        }
        await self.emit_async(evt)
