from __future__ import annotations
import cv2,uuid,time
from pathlib import Path
from typing import Callable,Optional,Awaitable,Any
from ultralytics import YOLO

class PersonDetector :
    def __init__(self,cfg:dict,emit_async: Callable[[dict],Awaitable[None]]):
        self.cfg = cfg
        self.emit_async = emit_async
        self.model = None
        self.cam = None
        self.last_snap_ts = 0.0

        sp = Path(cfg["paths"]["snapshot_dir"])
        sp.mkdir(parents=True,exist_ok=True)
        self.snapshot_dir = sp

        self.source = cfg["camera"]["source"]
        self.imgsz = int(cfg["camera"].get("imgsz",640))
        self.conf = float(cfg["camera"].get("conf",0.45))
        self.stride = int(cfg["camera"].get("stride",1))
        self.snap_every_s = float(cfg["camera"].get("snap_every_s",3.0))

        self.device = cfg["yolo"].get("device","cuda")
        self.model_name = cfg["yolo"].get("model","yolov8n.pt")

    def _open(self):
        self.cam = cv2.VideoCapture(self.source)
        if not self.cam.isOpened():
            raise RuntimeError("Không mở được camera/rtxp:{}".format(self.source))
        self.model = YOLO(self.model_name)
        self.model.predict(source=[cv2.UMat(640, 640)], imgsz=self.imgsz, device=self.device, verbose=False)

    def _read(self):
        for _ in range(self.stride - 1):
            self.cam.read()
        ok, frame = self.cam.read()
        return ok,frame
    
    def _maybe_snapshot(self,frame) -> Optional[str]:
        now = time.time()
        if now - self.last_snap_ts >=self.snap_every_s:
            self.last_snap_ts = now
            fn = f"person_{uuid.uuid4().hex[:8]}.jpg"
            path = self.snapshot_dir / fn
            cv2.imwrite(str(path), frame)
            return f"snapshots/{fn}"  # sẽ được mount dưới /static
        return None
    async def run(self,camera_id:str = "cam01"):
        self._open()
        try:
            while True:
                ok,frame = self._read()
                if not ok:
                    await asyncio.sleep(0.01)
                    continue
                res = self.model.predict(source = frame,imgsz = self.imgsz, device = self.device,verbose = False)[0]
                found = []
                for b in res.boxes: #boxes chua xyxy,conf,cls,xywh
                    cls_id = int(b.cls[0].item())
                    if cls_id != 0:
                        continue
                    xyxy=b.xyxy[0].tolist()
                    score = float(b.conf[0].item())
                    found.append((xyxy,score))
                if found:
                    snap_rel = self._maybe_snapshot(frame)
                    xyxy,score =  max(found,key=lambda t : t[1])
                    evt={
                        "id":"evt_{}".format(uuid.uuid4().hex[:8]),
                        "ts": time.time(),
                        "camera_id":camera_id,
                        "source":self.source,
                        "bbox":xyxy,
                        "snapshot_url":"/static/{}".format(snap_rel) if snap_rel else None,
                        "person":None,

                    }
                    await self.emit_async(evt)
        finally:
            try:
                self.cam.release()
            except Exception:
                pass


        
import asyncio          