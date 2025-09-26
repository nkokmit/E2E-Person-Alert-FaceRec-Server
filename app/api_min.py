# uvicorn app.api_min:app --reload 
# .\env\Scripts\activate

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import contextlib
import asyncio, time, uuid, json
from sqlalchemy.orm import Session
from app.db import engine, Base, get_db
from app.models import Event, Person
from app.ws_manager import WSManager

# ---------- Pydantic ----------
class PersonEvent(BaseModel):
    id: str
    ts: float
    camera_id: str
    source: str           # rtsp | ios | sim
    bbox: list[int]       # [x1,y1,x2,y2]
    snapshot_url: str | None = None
    person: dict | None = None

ws_manager = WSManager()

# ---------- helpers ----------
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

async def fake_generator(app: FastAPI):
    # thay bằng detector thật sau
    while True:
        evt = PersonEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            ts=time.time(),
            camera_id="cam01",
            source="sim",
            bbox=[100, 120, 260, 420],
            snapshot_url=None,
            person=None
        )
        # lưu DB
        with app.state.Session() as db:
            save_event(db, evt)
        # phát WS
        await ws_manager.broadcast({"type": "person_event", **evt.model_dump()})
        await asyncio.sleep(3)

# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) tạo bảng
    Base.metadata.create_all(bind=engine)
    # 2) attach Session factory lên app.state
    from app.db import SessionLocal
    app.state.Session = SessionLocal # app.state lưu biến toàn cục vào app
    # 3) start background tasks
    task = asyncio.create_task(fake_generator(app))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(Exception):
            await task

app = FastAPI(title="Mini Alerts (lifespan)", lifespan=lifespan)

# ---------- REST ----------
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/events/recent")
def recent(limit: int = 50, db: Session = Depends(get_db)): # gan db qua depend vao endpoint
    q = db.query(Event).order_by(Event.id.desc()).limit(limit).all()
    out = []
    for r in reversed(q):  # trả theo thời gian tăng dần
        out.append({
            "type": "person_event",
            "id": r.evt_id,
            "ts": r.ts,
            "camera_id": r.camera_id,
            "source": r.source,
            "bbox": json.loads(r.bbox),
            "snapshot_url": r.snapshot_url,
            "person": None  # sẽ tra theo person_id sau
        })
    return JSONResponse(out)

@app.post("/enroll")
def enroll(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)): # có gì tạo ra session_local liên tục từ res,req
    # TODO: xử lý thật (RetinaFace + ArcFace). Hiện tại tạo Person rỗng.
    p = Person(name=name)
    db.add(p); db.commit(); db.refresh(p)
    return {"person_id": p.id, "faces": 1, "name": name}

# ---------- WS ----------
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            # giữ kết nối sống; client có thể gửi "ping"
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
