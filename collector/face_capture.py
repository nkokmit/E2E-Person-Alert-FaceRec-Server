# # app/api_min.py
# # uvicorn app.api_min:app --reload --host 0.0.0.0 --port 8000

# from fastapi import (
#     FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File,
#     Form, Query, HTTPException, Header, Request
# )
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from contextlib import asynccontextmanager, suppress
# import asyncio, time, uuid, json, os, yaml, secrets, hashlib, base64
# from datetime import datetime, timedelta, timezone
# from sqlalchemy.orm import Session
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import jwt  #
# import cv2, numpy as np
# from app.db import engine, Base, get_db                      # :contentReference[oaicite:6]{index=6}
# from app.models import Event, Person, User                   # :contentReference[oaicite:7]{index=7}
# from app.ws_manager import WSManager                         # :contentReference[oaicite:8]{index=8}
# from app.snapshot_worker import SnapshotWorker               # :contentReference[oaicite:9]{index=9}
# from app.burst_manager import BurstManager                   # file bạn đã thêm theo hướng dẫn trước

# # ===== JWT settings =====
# SECRET_KEY = os.getenv("JWT_SECRET", "dev_secret_change_me")
# ALGO = "HS256"
# ACCESS_TOKEN_EXPIRE_MIN = 7*24*60  # 7 days

# # ---------- Schema ----------
# class PersonEvent(BaseModel):
#     id: str
#     ts: float
#     camera_id: str
#     source: str           # rtsp | ios | sim | webcam | esp32
#     bbox: list[int]
#     snapshot_url: str | None = None
#     person: dict | None = None

# class TokenOut(BaseModel):
#     access_token: str
#     token_type: str = "bearer"

# class UserOut(BaseModel):
#     id: int
#     username: str

# ws_manager = WSManager()

# # ---------- Helpers ----------
# def save_event(db: Session, evt: PersonEvent):
#     row = Event(
#         evt_id=evt.id, ts=evt.ts, camera_id=evt.camera_id, source=evt.source,
#         bbox=json.dumps(evt.bbox), snapshot_url=evt.snapshot_url,
#         person_id=(evt.person and evt.person.get("id")),
#     )
#     db.add(row)
#     db.commit()

# async def emit_from_detector(app: FastAPI, evt_dict: dict):
#     evt = PersonEvent(**evt_dict)
#     with app.state.Session() as db:
#         save_event(db, evt)
#     await ws_manager.broadcast({"type": "person_event", **evt.model_dump()})

# def _load_cfg() -> dict:
#     with open("configs.yaml", "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# def _require_ingest(token: str | None, cfg: dict):
#     if not token:
#         raise HTTPException(401, "missing X-Token")
#     expect = cfg.get("security", {}).get("ingest_token")
#     if token != expect:
#         raise HTTPException(401, "invalid token")

# # ---- Password hashing (PBKDF2-HMAC-SHA256) ----
# def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
#     salt = salt or secrets.token_hex(16)
#     dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000)
#     return salt, dk.hex()

# def verify_password(password: str, salt: str, hash_hex: str) -> bool:
#     _, check = hash_password(password, salt)
#     return secrets.compare_digest(check, hash_hex)

# def create_access_token(sub: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MIN) -> str:
#     now = datetime.now(timezone.utc)
#     payload = {"sub": sub, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=expires_minutes)).timestamp())}
#     return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)

# def get_current_user(db: Session = Depends(get_db), authorization: str | None = Header(default=None)) -> User:
#     if not authorization or not authorization.lower().startswith("bearer "):
#         raise HTTPException(401, "missing bearer token")
#     token = authorization.split(" ", 1)[1]
#     try:
#         data = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
#         username = data.get("sub")
#     except Exception:
#         raise HTTPException(401, "invalid token")
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(401, "user not found")
#     return user

# # ---------- Lifespan ----------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     Base.metadata.create_all(bind=engine)
#     from app.db import SessionLocal
#     app.state.Session = SessionLocal

#     cfg = _load_cfg()
#     # static mounts
#     snap_dir = Path(cfg["paths"]["snapshot_dir"])
#     burst_dir = Path(cfg["paths"].get("burst_dir", "data/burst"))
#     os.makedirs(snap_dir, exist_ok=True)
#     os.makedirs(burst_dir, exist_ok=True)
#     app.mount(cfg["paths"]["static_mount"], StaticFiles(directory="data"), name="static")

#     # Snapshot worker
#     worker = SnapshotWorker(cfg, lambda evt: emit_from_detector(app, evt))
#     app.state.snapshot_worker = worker
#     app.state.worker_task = asyncio.create_task(worker.start())

#     # Burst manager (share YOLO model với worker)
#     app.state.burst = BurstManager(
#         cfg,
#         lambda evt: emit_from_detector(app, evt),   # emit sang WS (giống SnapshotWorker)
#         app.state.snapshot_worker.queue             # hàng đợi YOLO
#     )

#     try:
#         yield
#     finally:
#         app.state.worker_task.cancel()
#         with suppress(Exception):
#             await app.state.worker_task

# app = FastAPI(title="Mini Alerts", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )

# # ---------- AUTH ----------
# from fastapi import Body

# @app.post("/auth/register", response_model=UserOut)
# def register(username: str = Body(...), password: str = Body(...), db: Session = Depends(get_db)):
#     if db.query(User).filter(User.username == username).first():
#         raise HTTPException(409, "username already exists")
#     salt, pw_hash = hash_password(password)
#     u = User(username=username, password_salt=salt, password_hash=pw_hash)
#     db.add(u); db.commit(); db.refresh(u)
#     return UserOut(id=u.id, username=u.username)

# @app.post("/auth/login", response_model=TokenOut)
# def login(username: str = Body(...), password: str = Body(...), db: Session = Depends(get_db)):
#     u = db.query(User).filter(User.username == username).first()
#     if not u or not verify_password(password, u.password_salt, u.password_hash):
#         raise HTTPException(401, "invalid credentials")
#     token = create_access_token(sub=u.username)
#     return TokenOut(access_token=token)

# @app.get("/me", response_model=UserOut)
# def me(user: User = Depends(get_current_user)):
#     return UserOut(id=user.id, username=user.username)

# # ---------- Health / Buzz / Events ----------
# @app.get("/health")
# def health():
#     return {"ok": True}

# @app.post("/buzz")
# async def buzz(camera_id: str = Form(...), duration_ms: int = Form(2000), user: User = Depends(get_current_user)):
#     await ws_manager.broadcast({"cmd": "buzz", "camera_id": camera_id, "duration_ms": duration_ms})
#     return {"ok": True}

# @app.get("/events/recent")
# def recent(limit: int = 50, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
#     rows = db.query(Event).order_by(Event.id.desc()).limit(limit).all()
#     out = []
#     for r in reversed(rows):
#         out.append({
#             "type": "person_event",
#             "id": r.evt_id, "ts": r.ts, "camera_id": r.camera_id, "source": r.source,
#             "bbox": json.loads(r.bbox), "snapshot_url": r.snapshot_url, "person": None,
#         })
#     return JSONResponse(out)

# # ---------- WebSocket (không bắt buộc JWT, để iOS dễ subscribe) ----------
# @app.websocket("/stream")
# async def stream(ws: WebSocket):
#     await ws_manager.connect(ws)
#     try:
#         while True:
#             await asyncio.sleep(60)
#     except WebSocketDisconnect:
#         ws_manager.disconnect(ws)

# # ---------- Legacy Single Snapshot ----------
# @app.post("/snapshot")
# async def upload_snapshot(
#     camera_id: str = Form(...),
#     pir_level: int = Form(1),
#     file: UploadFile = File(...),
#     token: str = Form(...),
# ):
#     cfg = _load_cfg()
#     if token != cfg.get("security", {}).get("ingest_token"):
#         raise HTTPException(status_code=401, detail="invalid token")

#     if file.content_type not in ("image/jpeg", "image/jpg"):
#         raise HTTPException(status_code=415, detail="only jpeg")
#     raw = await file.read()
#     if len(raw) > 2_000_000:
#         raise HTTPException(status_code=413, detail="file too large")

#     snap_dir = Path(cfg["paths"]["snapshot_dir"])
#     fname = f"{camera_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
#     fpath = snap_dir / fname
#     fpath.write_bytes(raw)
#     await app.state.snapshot_worker.queue.put({"img_path": fpath, "camera_id": camera_id})
#     url = f"/static/snapshots/{fname}"
#     return {"ok": True, "camera_id": camera_id, "pir_level": pir_level, "snapshot_url": url, "size": len(raw)}

# # ---------- BURST MODE ----------
# @app.get("/burst/start")
# def burst_start(camera_id: str, request: Request, x_token: str | None = Header(default=None)):
#     cfg = _load_cfg(); _require_ingest(x_token, cfg)
#     app.state.burst.start(camera_id)
#     return {"ok": True, "camera_id": camera_id}

# @app.post("/burst/frame")
# async def burst_frame(camera_id: str, request: Request, x_token: str | None = Header(default=None)):
#     cfg = _load_cfg(); _require_ingest(x_token, cfg)
#     if request.headers.get("content-type", "").lower().startswith("image/jpeg") is False:
#         raise HTTPException(415, "Content-Type must be image/jpeg")
#     raw = await request.body()
#     fname = app.state.burst.add_frame(camera_id, raw)
#     return {"ok": True, "saved_as": fname}

# @app.get("/burst/stop")
# async def burst_stop(camera_id: str, request: Request, x_token: str | None = Header(default=None)):
#     cfg = _load_cfg(); _require_ingest(x_token, cfg)
#     info = await app.state.burst.stop_and_process(camera_id)
#     return {"ok": True, **info}
