# filepath: app/api_min.py
# Chạy: uvicorn app.api_min:app --reload

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio, time, uuid, json, os, yaml
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.db import engine, Base, get_db
from app.models import Event, Person
from app.ws_manager import WSManager
from app.detector_service import DetectorService

# ---------- Schema ----------
class PersonEvent(BaseModel):
    id: str
    ts: float
    camera_id: str
    source: str           # rtsp | ios | sim | webcam
    bbox: list[int]
    snapshot_url: str | None = None
    person: dict | None = None

ws_manager = WSManager()

# ---------- Helpers ----------
def save_event(db: Session, evt: PersonEvent):
    row = Event(
        evt_id=evt.id,
        ts=evt.ts,
        camera_id=evt.camera_id,
        source=evt.source,
        bbox=json.dumps(evt.bbox),
        snapshot_url=evt.snapshot_url,
        person_id=(evt.person and evt.person.get("id")),
    )
    db.add(row)
    db.commit()

async def emit_from_detector(app: FastAPI, evt_dict: dict):
    evt = PersonEvent(**evt_dict)
    with app.state.Session() as db:
        save_event(db, evt)
    await ws_manager.broadcast({"type": "person_event", **evt.model_dump()})

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    from app.db import SessionLocal
    app.state.Session = SessionLocal

    with open("configs.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["snapshot_dir"], exist_ok=True)
    app.mount(cfg["paths"]["static_mount"], StaticFiles(directory="data"), name="static")

    service = DetectorService(cfg, lambda evt: emit_from_detector(app, evt))
    app.state.detector_service = service
    await service.start()

    try:
        yield
    finally:
        await service.stop()

app = FastAPI(title="Mini Alerts", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],
)

# ---------- REST ----------
@app.get("/status")
def status():
    return app.state.detector_service.status()

@app.post("/detector/start")
async def start_detector():
    await app.state.detector_service.start()
    return {"ok": True, **app.state.detector_service.status()}

@app.post("/detector/stop")
async def stop_detector():
    await app.state.detector_service.stop()
    return {"ok": True, **app.state.detector_service.status()}

@app.post("/detector/switch")
async def switch_camera(source: str | int = Query(...)):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    await app.state.detector_service.switch_source(source)
    return {"ok": True, **app.state.detector_service.status()}

@app.patch("/detector/params")
async def update_params(
    imgsz: int | None = None,
    conf: float | None = None,
    stride: int | None = None,
    max_fps: float | None = None,
    model: str | None = None,
    device: str | None = None,
):
    payload = {k: v for k, v in {
        "imgsz": imgsz, "conf": conf, "stride": stride, "max_fps": max_fps,
        "model": model, "device": device
    }.items() if v is not None}
    await app.state.detector_service.update_params(**payload)
    return {"ok": True, "updated": payload, **app.state.detector_service.status()}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/events/recent")
def recent(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Event).order_by(Event.id.desc()).limit(limit).all()
    out = []
    for r in reversed(rows):
        out.append({
            "type": "person_event",
            "id": r.evt_id,
            "ts": r.ts,
            "camera_id": r.camera_id,
            "source": r.source,
            "bbox": json.loads(r.bbox),
            "snapshot_url": r.snapshot_url,
            "person": None,
        })
    return JSONResponse(out)

@app.post("/enroll")
def enroll(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    p = Person(name=name)
    db.add(p); db.commit(); db.refresh(p)
    return {"person_id": p.id, "faces": 1, "name": name}

# ---------- WS ----------
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await asyncio.sleep(60)  # giữ kết nối
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
