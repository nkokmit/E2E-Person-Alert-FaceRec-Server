# app/detector_service.py
from __future__ import annotations
from typing import Callable, Awaitable
import asyncio

class DetectorService:

    def __init__(self, cfg: dict, emit_async: Callable[[dict], Awaitable[None]]):
        self.cfg = cfg
        self.emit_async = emit_async
        self.task: asyncio.Task | None = None
        self.camera_id = "cam01"
        self.running = False

    async def start(self):
        if self.running:
            return
        from detector.person_detector import PersonDetector
        self.det = PersonDetector(self.cfg, self.emit_async)
        self.task = asyncio.create_task(self.det.run(camera_id=self.camera_id))
        self.running = True

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
        self.task = None
        self.running = False

    async def restart(self):
        await self.stop()
        await self.start()

    async def switch_source(self, source):
        self.cfg["camera"]["source"] = source
        await self.restart()

    async def update_params(self, **kwargs):
        cam = self.cfg["camera"]
        yolo = self.cfg["yolo"]
        for k, v in kwargs.items():
            if k in cam:
                cam[k] = v
            elif k in yolo:
                yolo[k] = v
        await self.restart()

    def status(self):
        cam = self.cfg["camera"]
        y = self.cfg["yolo"]
        return {
            "running": self.running,
            "camera_id": self.camera_id,
            "source": cam.get("source"),
            "imgsz": cam.get("imgsz"),
            "conf": cam.get("conf"),
            "stride": cam.get("stride"),
            "max_fps": cam.get("max_fps"),
            "device": y.get("device"),
            "model": y.get("model"),
        }
