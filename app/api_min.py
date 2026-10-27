# uvicorn app.api_min:app --reload --host 0.0.0.0 --port 8000
import asyncio
import hashlib
import hmac
import json
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import FileResponse, JSONResponse

from app import models
from app.db import SessionLocal, get_user_by_id, get_user_for_login
from app.snapshot_worker import SnapshotWorker
from app.ws_manager import ConnectionManager, WSManager


BASE_DIR = Path(__file__).resolve().parents[1]


def load_app_config() -> dict:
    cfg_path = BASE_DIR / "configs.yaml"
    if not cfg_path.exists():
        print(f"[api_min] WARNING: config file not found at {cfg_path}, using defaults")
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


APP_CFG = load_app_config()

paths_cfg = APP_CFG.get("paths", {})
SNAP_ROOT = Path(paths_cfg.get("snapshot_dir", "data/snapshots"))
PREVIEW_DIR = SNAP_ROOT / "preview"
EVENTS_DIR = SNAP_ROOT / "events"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

AUTH_CFG = APP_CFG.get("auth", {})

SECRET_KEY = os.environ.get("APP_SECRET") or AUTH_CFG.get(
    "secret_key",
    "supersecret-key-change-me",
)

TOKEN_EXP_SECS = int(AUTH_CFG.get("token_exp_secs", 7 * 24 * 3600))

ws_cfg = APP_CFG.get("ws", {})
ws_manager = ConnectionManager(
    heartbeat_interval=float(ws_cfg.get("heartbeat_interval", 15.0)),
    heartbeat_timeout=float(ws_cfg.get("heartbeat_timeout", 45.0)),
    send_timeout=float(ws_cfg.get("send_timeout", 3.0)),
)


async def ws_emit(msg: dict):
    await ws_manager.broadcast(msg)


snapshot_worker = SnapshotWorker(ws_emit)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ws_manager = ws_manager
    app.state.snapshot_worker = snapshot_worker
    ws_manager.start_heartbeat()
    worker_task = asyncio.create_task(snapshot_worker.run_forever())
    try:
        yield
    finally:
        await ws_manager.stop_heartbeat()
        await ws_manager.disconnect_all()
        worker_task.cancel()
        with suppress(Exception):
            await worker_task


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/snapshots", StaticFiles(directory=str(SNAP_ROOT)), name="snapshots")


# =============================================================================
# Helper: Event Serialization & Query Filtering
# =============================================================================
def _parse_bbox(raw_bbox: Any) -> Any:
    if raw_bbox is None:
        return []
    if isinstance(raw_bbox, (list, dict)):
        return raw_bbox
    if isinstance(raw_bbox, str):
        try:
            return json.loads(raw_bbox)
        except Exception:
            return raw_bbox
    return raw_bbox


def _serialize_event(ev: models.Event) -> Dict[str, Any]:
    return {
        "id": ev.id,
        "evt_id": ev.evt_id,
        "ts": ev.ts,
        "camera_id": ev.camera_id,
        "source": ev.source,
        "bbox": _parse_bbox(ev.bbox),
        "snapshot_url": ev.snapshot_url,
        "person_id": ev.person_id,
        "confidence": getattr(ev, "confidence", None),
        "duration_sec": getattr(ev, "duration_sec", None),
        "alert_type": getattr(ev, "alert_type", None),
    }


def _apply_event_filters(
    q,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    person_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
):
    if start_ts is not None:
        q = q.filter(models.Event.ts >= float(start_ts))
    if end_ts is not None:
        q = q.filter(models.Event.ts <= float(end_ts))
    if person_id is not None and person_id.strip():
        q = q.filter(models.Event.person_id == person_id.strip())
    if camera_id is not None and camera_id.strip():
        q = q.filter(models.Event.camera_id == camera_id.strip())
    if alert_type is not None and alert_type.strip():
        q = q.filter(models.Event.alert_type == alert_type.strip())
    return q


# =============================================================================
# Auth & Ingest Endpoints
# =============================================================================
@app.post("/auth/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username/password required")

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

    user = get_user_for_login(db, username, password)
    if not user:
        raise HTTPException(status_code=401, detail="bad credentials")

    token = create_token(user.id)
    return {"token": token, "username": user.username}


@app.post("/snapshot")
async def ingest_snapshot(
    request: Request,
    camera_id: str = Form(...),
    token: str = Form(...),
    file: UploadFile = File(...),
):
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


# =============================================================================
# Event History REST API (Pagination + Filtering by time, person_id, camera_id)
# =============================================================================
@app.get("/events")
def list_events(
    page: int = Query(1, ge=1, description="Số trang (bắt đầu từ 1)"),
    page_size: int = Query(20, ge=1, le=200, description="Số bản ghi mỗi trang"),
    start_ts: Optional[float] = Query(None, description="Lọc từ timestamp (epoch seconds)"),
    end_ts: Optional[float] = Query(None, description="Lọc đến timestamp (epoch seconds)"),
    person_id: Optional[str] = Query(None, description="Lọc theo tên người nhận diện (vd: Hung, Unknown)"),
    camera_id: Optional[str] = Query(None, description="Lọc theo camera_id"),
    alert_type: Optional[str] = Query(None, description="Lọc theo loại cảnh báo"),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    base_q = _apply_event_filters(
        db.query(models.Event),
        start_ts=start_ts,
        end_ts=end_ts,
        person_id=person_id,
        camera_id=camera_id,
        alert_type=alert_type,
    )
    total = base_q.count()
    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 0
    offset = (page - 1) * page_size

    rows = (
        base_q.order_by(models.Event.ts.desc(), models.Event.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {
        "ok": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": [_serialize_event(ev) for ev in rows],
    }


@app.get("/events/recent")
def events_recent(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    start_ts: Optional[float] = Query(None),
    end_ts: Optional[float] = Query(None),
    person_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> List[Dict[str, Any]]:
    q = _apply_event_filters(
        db.query(models.Event),
        start_ts=start_ts,
        end_ts=end_ts,
        person_id=person_id,
        camera_id=camera_id,
        alert_type=alert_type,
    )
    rows = q.order_by(models.Event.ts.desc(), models.Event.id.desc()).offset(offset).limit(limit).all()
    return [_serialize_event(ev) for ev in rows]


@app.get("/events/{evt_id}")
def get_event_detail(
    evt_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    ev = db.query(models.Event).filter(models.Event.evt_id == evt_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="event not found")
    return {"ok": True, "event": _serialize_event(ev)}


# =============================================================================
# Control & WebSocket Streaming Endpoints
# =============================================================================
@app.post("/buzz")
async def buzz(
    request: Request,
    camera_id: str = Form(...),
    duration_ms: int = Form(2000),
    user=Depends(get_current_user),
):
    mgr: ConnectionManager = request.app.state.ws_manager
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
    mgr: ConnectionManager = request.app.state.ws_manager
    await mgr.broadcast({"cmd": "stream_on", "camera_id": camera_id})
    return {"ok": True}


@app.post("/stream/stop")
async def stream_stop(
    request: Request,
    camera_id: str = Form(...),
    user=Depends(get_current_user),
):
    mgr: ConnectionManager = request.app.state.ws_manager
    await mgr.broadcast({"cmd": "stream_off", "camera_id": camera_id})
    return {"ok": True}


async def _handle_ws_session(
    ws: WebSocket,
    client_id: Optional[str] = None,
    camera_id: Optional[str] = None,
):
    mgr: ConnectionManager = getattr(ws.app.state, "ws_manager", ws_manager)
    cameras = [c.strip() for c in camera_id.split(",") if c.strip()] if camera_id else None
    await mgr.connect(ws, client_id=client_id, cameras=cameras)
    try:
        while True:
            raw_text = await ws.receive_text()
            await mgr.handle_client_message(ws, raw_text)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await mgr.disconnect(ws)


@app.websocket("/stream")
async def ws_stream(
    ws: WebSocket,
    client_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
):
    await _handle_ws_session(ws, client_id=client_id, camera_id=camera_id)


@app.websocket("/ws/alerts")
async def ws_alerts(
    ws: WebSocket,
    client_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
):
    await _handle_ws_session(ws, client_id=client_id, camera_id=camera_id)


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
    )
