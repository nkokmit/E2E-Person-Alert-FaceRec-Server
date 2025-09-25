#uvicorn app.api_min:app --reload 
#.\env\Scripts\activate


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from collections import deque
import uuid, asyncio, time
from typing import List, Dict, Optional
from fastapi import UploadFile, File, Form
MAX_EVENTS = 200
EVENT_BUFFER = deque(maxlen=MAX_EVENTS)   # luôn giữ mới nhất
WS_CLIENTS: set[WebSocket] = set()

class PersonEvent(BaseModel):
    id: str
    ts: float
    camera_id: str
    source: str
    bbox: List[int]
    snapshot_url: Optional[str] = None     # tên field thống nhất
    person: Optional[Dict] = None

def push_event(evt: PersonEvent):
    EVENT_BUFFER.append(evt.model_dump())  # có ()

async def broadcast(evt_dict: dict):
    dead = [] # loại bỏ evt dis ra 
    for ws in list(WS_CLIENTS):            # lặp trên bản sao để an toàn
        try:
            await ws.send_json(evt_dict)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.discard(ws)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    task = asyncio.create_task(fake_generator())
    try:
        yield
    finally:
        # SHUTDOWN
        task.cancel()
        for ws in list(WS_CLIENTS):
            try:
                await ws.close()
            except Exception:
                pass
        WS_CLIENTS.clear()

app = FastAPI(title="Welcome to FastAPI", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/events/recent")
async def recent(limit: int = 50):
    # deque không slice được trực tiếp -> chuyển list
    data = list(EVENT_BUFFER)[-limit:]
    return JSONResponse(data)

@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    # TODO: về sau: trích face -> embed -> lưu DB
    fake_person_id = "p_" + uuid.uuid4().hex[:6]
    return {"person_id": fake_person_id, "faces": 1, "name": name}

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()                      # có ()
    WS_CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()        # có await
    except WebSocketDisconnect:
        pass
    finally:
        WS_CLIENTS.discard(ws)

async def fake_generator():
    i = 0
    while True:
        i += 1
        evt = PersonEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            ts=time.time(),
            camera_id="cam01",
            source="sim",
            bbox=[100, 120, 260, 420],
            snapshot_url=None,
            person=None if i % 3 else {"id": "p_01", "name": "Anh A"},
        )
        push_event(evt)
        await broadcast({"type": "person_event", **evt.model_dump()})
        await asyncio.sleep(3)
