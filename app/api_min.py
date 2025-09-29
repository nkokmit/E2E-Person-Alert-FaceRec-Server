# filepath: app/api_min.py
# Chạy: uvicorn app.api_min:app --reload

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager, suppress
import asyncio, time, uuid, json, os, yaml
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles

from app.db import engine, Base, get_db
from app.models import Event, Person
from app.ws_manager import WSManager

# ---------- Schema ----------
class PersonEvent(BaseModel):
    id: str
    ts: float
    camera_id: str
    source: str           # rtsp | ios | sim
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
        person_id=(evt.person and evt.person.get("id"))
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

    async def detector_task():
        from detector.person_detector import PersonDetector
        det = PersonDetector(cfg, lambda evt: emit_from_detector(app, evt))
        await det.run(camera_id="cam01")

    task = asyncio.create_task(detector_task())
    try:
        yield
    finally:
        task.cancel()
        with suppress(Exception):
            await task

app = FastAPI(title="Mini Alerts", lifespan=lifespan)

# ---------- REST ----------
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/events/recent")
def recent(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Event).order_by(Event.id.desc()).limit(limit).all()
    return JSONResponse([{
        "type": "person_event",
        "id": r.evt_id,
        "ts": r.ts,
        "camera_id": r.camera_id,
        "source": r.source,
        "bbox": json.loads(r.bbox),
        "snapshot_url": r.snapshot_url,
        "person": None
    } for r in reversed(rows)])

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
