# uvicorn app.api_min:app --reload --host 0.0.0.0 --port 8000

import asyncio
import hashlib
import os
import uuid
import time
import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
    Body,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    Response,
    Query,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal, engine, get_user_for_login, get_user_by_id
from app import models
from app.ws_manager import WSManager
from app.snapshot_worker import SnapshotWorker

# ====== CONFIG / INIT ======

BASE_DIR = Path(__file__).resolve().parents[1]


def load_app_config() -> dict:
    cfg_path = BASE_DIR / "configs.yaml"
    if not cfg_path.exists():
        print(f"[api_min] WARNING: config file not found at {cfg_path}, using defaults")
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


APP_CFG = load_app_config()

# thư mục snapshot lấy từ YAML
paths_cfg = APP_CFG.get("paths", {})
SNAP_ROOT = Path(paths_cfg.get("snapshot_dir", "data/snapshots"))
PREVIEW_DIR = SNAP_ROOT / "preview"
EVENTS_DIR = SNAP_ROOT / "events"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

AUTH_CFG = APP_CFG.get("auth", {})

SECRET_KEY = os.environ.get("APP_SECRET") or AUTH_CFG.get(
    "secret_key",
    "supersecret-key-change-me",  # default nếu không có env + yaml
)

TOKEN_EXP_SECS = int(AUTH_CFG.get("token_exp_secs", 7 * 24 * 3600))

# WebSocket manager + worker YOLO
ws_manager = WSManager()


async def ws_emit(msg: dict):
    await ws_manager.broadcast(msg)


snapshot_worker = SnapshotWorker(ws_emit)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====== AUTH HELPERS (KHÔNG HASH PASSWORD, chỉ dùng cho TOKEN) ======


def create_token(user_id: int) -> str:
    exp = int(time.time()) + TOKEN_EXP_SECS
    payload = f"{user_id}.{exp}"
    sig = hmac.new(
        SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{sig}"


def decode_token(token: str) -> Optional[int]:
    try:
        user_part, exp_str, sig = token.split(".")
        exp = int(exp_str)
    except Exception:
        return None
    if time.time() > exp:
        return None
    payload = f"{user_part}.{exp_str}"
    expected_sig = hmac.new(
        SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, sig):
        return None
    try:
        return int(user_part)
    except ValueError:
        return None


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_db),
):
    """Đọc header Authorization: Bearer <token>, trả về User hoặc 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:].strip()
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ====== FASTAPI APP ======


@asynccontextmanager
async def lifespan(app: FastAPI):
    # gắn shared objects
    app.state.ws_manager = ws_manager
    app.state.snapshot_worker = snapshot_worker


    # chạy worker YOLO + event
    asyncio.create_task(snapshot_worker.run_forever())

    yield
    # shutdown: không cần gì đặc biệt


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # có thể fix thành IP frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static cho ảnh snapshot
app.mount("/snapshots", StaticFiles(directory=str(SNAP_ROOT)), name="snapshots")


# ====== AUTH ROUTES ======


@app.post("/auth/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username/password required")

    # LƯU MẬT KHẨU PLAIN TEXT (chỉ dùng cho dev / nội bộ)
    user = models.User(username=username, password=password)
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="username exists")

    token = create_token(user.id)
    return {"token": token, "username": user.username}


@app.post("/auth/login")
def login(
    data: dict = Body(...),
    db: Session = Depends(get_db),
):
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    if not username or not password:
        raise HTTPException(status_code=400, detail="username/password required")

    # Dùng db.py để kiểm tra username + password plain text
    user = get_user_for_login(db, username, password)
    if not user:
        raise HTTPException(status_code=401, detail="bad credentials")

    token = create_token(user.id)
    return {"token": token, "username": user.username}


# ====== SNAPSHOT INGEST + PREVIEW ======


@app.post("/snapshot")
async def ingest_snapshot(
    request: Request,
    camera_id: str = Form(...),
    token: str = Form(...),
    file: UploadFile = File(...),
):
    # Lấy token ingest từ YAML: security.ingest_token
    expected_token = APP_CFG.get("security", {}).get("ingest_token", "SHARED_TOKEN_ABC")
    if token != expected_token:
        raise HTTPException(status_code=401, detail="bad token")

    ts = time.time()

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_DIR / f"{camera_id}.jpg"

    content = await file.read()
    with open(preview_path, "wb") as f:
        f.write(content)

    worker: SnapshotWorker = request.app.state.snapshot_worker
    await worker.queue.put(
        {
            "camera_id": camera_id,
            "ts": ts,
            "path": str(preview_path),
        }
    )

    return {"ok": True}


@app.get("/preview/latest")
def preview_latest(camera_id: str = Query(...)):
    path = PREVIEW_DIR / f"{camera_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no preview")
    return FileResponse(path, media_type="image/jpeg")


# ====== EVENTS API ======


@app.get("/events/recent")
def events_recent(
    limit: int = 50,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = (
        db.query(models.Event)
        .order_by(models.Event.ts.desc())
        .limit(limit)
        .all()
    )
    result = []
    for ev in q:
        result.append(
            {
                "id": ev.id,
                "evt_id": ev.evt_id,
                "ts": ev.ts,
                "camera_id": ev.camera_id,
                "source": ev.source,
                "bbox": ev.bbox,
                "snapshot_url": ev.snapshot_url,
                "person_id": ev.person_id,
                "duration_sec": getattr(ev, "duration_sec", None),
                "alert_type": getattr(ev, "alert_type", None),
            }
        )
    return result


# ====== BUZZ / STREAM CONTROL ======


@app.post("/buzz")
async def buzz(
    request: Request,
    camera_id: str = Form(...),
    duration_ms: int = Form(2000),
    user=Depends(get_current_user),
):
    mgr: WSManager = request.app.state.ws_manager
    await mgr.broadcast(
        {
            "cmd": "buzz",
            "camera_id": camera_id,
            "duration_ms": duration_ms,
        }
    )
    return {"ok": True}


@app.post("/stream/start")
async def stream_start(
    request: Request,
    camera_id: str = Form(...),
    user=Depends(get_current_user),
):
    mgr: WSManager = request.app.state.ws_manager
    await mgr.broadcast({"cmd": "stream_on", "camera_id": camera_id})
    return {"ok": True}


@app.post("/stream/stop")
async def stream_stop(
    request: Request,
    camera_id: str = Form(...),
    user=Depends(get_current_user),
):
    mgr: WSManager = request.app.state.ws_manager
    await mgr.broadcast({"cmd": "stream_off", "camera_id": camera_id})
    return {"ok": True}


# ====== WEBSOCKET STREAM ======


@app.websocket("/stream")
async def ws_stream(ws: WebSocket):
    mgr: WSManager = ws_manager
    await ws.accept()
    await mgr.register(ws)
    try:
        while True:
            # không xử lý nội dung ở đây, chỉ đọc để giữ kết nối
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        await mgr.unregister(ws)
    except Exception:
        await mgr.unregister(ws)


# ====== ERROR HANDLER ======


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
    )
