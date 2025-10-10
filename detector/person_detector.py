# filepath: detector/person_detector.py
from __future__ import annotations
import cv2, time, uuid, asyncio
from pathlib import Path
from typing import Callable, Awaitable, Optional
from ultralytics import YOLO

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class PersonDetector:
    def __init__(self, cfg: dict, emit_async: Callable[[dict], Awaitable[None]]):
        self.cfg = cfg
        self.emit_async = emit_async

        sp = Path(cfg["paths"]["snapshot_dir"])
        sp.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = sp

        cam_src = cfg["camera"]["source"]
        self.source = int(cam_src) if str(cam_src).isdigit() else cam_src
        self.imgsz = int(cfg["camera"].get("imgsz", 640))
        self.conf = float(cfg["camera"].get("conf", 0.45))
        self.stride = int(cfg["camera"].get("stride", 1))
        self.snap_every_s = float(cfg["camera"].get("snap_every_s", 3.0))
        
        self.device = cfg["yolo"].get("device", "cuda")
        self.model_name = cfg["yolo"].get("model", "yolov8n.pt")

        # state
        self.last_snap_ts = 0.0
        self.had_person = False
        self.last_bbox = None
        self.last_emit_ts = 0.0
        self.keepalive_s = 5.0
        self.iou_same = 0.6

    def _open(self):
        self.cam = cv2.VideoCapture(self.source)
        if not self.cam.isOpened():
            raise RuntimeError(f"Không mở được camera: {self.source}")
        self.model = YOLO(self.model_name)

    def _read(self):
        for _ in range(self.stride - 1):
            self.cam.read()
        return self.cam.read()

    def _maybe_snapshot(self, frame) -> Optional[str]:
        now = time.time()
        if now - self.last_snap_ts >= self.snap_every_s:
            self.last_snap_ts = now
            fn = f"person_{uuid.uuid4().hex[:8]}.jpg"
            path = self.snapshot_dir / fn
            cv2.imwrite(str(path), frame)
            return f"snapshots/{fn}"
        return None

    async def run(self, camera_id: str = "cam01"):
        self._open()
        try:
            while True:
                ok, frame = self._read()
                if not ok:
                    await asyncio.sleep(1)
                    continue

                res = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    verbose=False
                )[0]

                persons = [(b.xyxy[0].tolist(), float(b.conf[0].item()))
                           for b in res.boxes if int(b.cls[0].item()) == 0]

                now = time.time()
                if persons:
                    xyxy, score = max(persons, key=lambda t: t[1])
                    xyxy = [int(x) for x in xyxy]

                    should_emit = False
                    if not self.had_person:
                        should_emit = True
                    else:
                        iou = _iou_xyxy(xyxy, self.last_bbox) if self.last_bbox else 0.0
                        if iou < self.iou_same or (now - self.last_emit_ts >= self.keepalive_s):
                            should_emit = True

                    self.had_person, self.last_bbox = True, xyxy

                    if should_emit:
                        snap_rel = self._maybe_snapshot(frame)
                        evt = {
                            "id": f"evt_{uuid.uuid4().hex[:8]}",
                            "ts": now,
                            "camera_id": camera_id,
                            "source": "webcam",
                            "bbox": xyxy,
                            "snapshot_url": f"/static/{snap_rel}" if snap_rel else None,
                            "person": None,
                        }
                        await self.emit_async(evt)
                        self.last_emit_ts = now
                else:
                    self.had_person = False
                    self.last_bbox = None

                await asyncio.sleep(0.02)
        finally:
            if hasattr(self, "cam"):
                self.cam.release()
