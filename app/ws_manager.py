# app/ws_manager.py
"""
WebSocket ConnectionManager with concurrent broadcast, per-send timeout,
and automatic heartbeat (ping/pong) dead-connection cleanup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

log = logging.getLogger("ws_manager")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


@dataclass
class ClientConnection:
    ws: WebSocket
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    cameras: Optional[Set[str]] = None  # None = nhận tất cả camera

    def touch(self) -> None:
        self.last_seen = time.time()

    def accepts_camera(self, camera_id: Optional[str]) -> bool:
        if not camera_id or not self.cameras:
            return True
        return camera_id in self.cameras


class ConnectionManager:
    """Quản lý đa kết nối WebSocket thời gian thực kèm heartbeat ping-pong."""

    def __init__(
        self,
        heartbeat_interval: float = 15.0,
        heartbeat_timeout: float = 45.0,
        send_timeout: float = 3.0,
    ) -> None:
        self.heartbeat_interval = float(heartbeat_interval)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.send_timeout = float(send_timeout)

        self._connections: Dict[WebSocket, ClientConnection] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def clients(self) -> Set[WebSocket]:
        """Tương thích ngược với thuộc tính WSManager.clients cũ."""
        return set(self._connections.keys())

    @property
    def active_count(self) -> int:
        return len(self._connections)

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------
    async def connect(
        self,
        ws: WebSocket,
        client_id: Optional[str] = None,
        cameras: Optional[Iterable[str]] = None,
        accept: bool = True,
    ) -> ClientConnection:
        if accept and ws.client_state == WebSocketState.CONNECTING:
            await ws.accept()
        now = time.time()
        conn = ClientConnection(
            ws=ws,
            client_id=client_id or uuid.uuid4().hex[:12],
            connected_at=now,
            last_seen=now,
            cameras=set(cameras) if cameras else None,
        )
        async with self._lock:
            self._connections[ws] = conn
        log.info("Client connected: %s (active=%d)", conn.client_id, len(self._connections))
        return conn

    async def register(self, ws: WebSocket) -> ClientConnection:
        """Alias tương thích ngược (khi caller đã gọi ws.accept() trước)."""
        return await self.connect(ws, accept=False)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            conn = self._connections.pop(ws, None)
        if conn is not None:
            log.info("Client disconnected: %s (active=%d)", conn.client_id, len(self._connections))
        try:
            if ws.client_state != WebSocketState.DISCONNECTED:
                await asyncio.wait_for(ws.close(), timeout=1.0)
        except Exception:
            pass

    async def unregister(self, ws: WebSocket) -> None:
        """Alias tương thích ngược cho disconnect."""
        await self.disconnect(ws)

    async def disconnect_all(self) -> None:
        async with self._lock:
            sockets = list(self._connections.keys())
            self._connections.clear()
        if sockets:
            await asyncio.gather(*(self._safe_close(ws) for ws in sockets), return_exceptions=True)

    @staticmethod
    async def _safe_close(ws: WebSocket) -> None:
        try:
            if ws.client_state != WebSocketState.DISCONNECTED:
                await asyncio.wait_for(ws.close(), timeout=1.0)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Broadcast & Messaging
    # -------------------------------------------------------------------------
    async def _send_one(self, conn: ClientConnection, data: Dict[str, Any]) -> Optional[WebSocket]:
        try:
            await asyncio.wait_for(conn.ws.send_json(data), timeout=self.send_timeout)
            return None
        except Exception:
            return conn.ws

    async def broadcast(self, data: Dict[str, Any], camera_id: Optional[str] = None) -> int:
        """Phát gói tin JSON đồng thời tới toàn bộ client; tự dọn các kết nối lỗi/treo."""
        cam = camera_id or data.get("camera_id") or (
            data.get("payload", {}).get("camera_id") if isinstance(data.get("payload"), dict) else None
        )
        async with self._lock:
            targets = [c for c in self._connections.values() if c.accepts_camera(cam)]

        if not targets:
            return 0

        results = await asyncio.gather(
            *(self._send_one(c, data) for c in targets),
            return_exceptions=True,
        )

        dead: List[WebSocket] = []
        sent = 0
        for conn, res in zip(targets, results):
            if isinstance(res, Exception) or res is not None:
                dead.append(conn.ws)
            else:
                sent += 1

        for ws in dead:
            await self.disconnect(ws)
        return sent

    async def broadcast_alert(
        self,
        *,
        evt_id: str,
        ts: float,
        camera_id: str,
        snapshot_url: Optional[str],
        person_id: Optional[str] = "Unknown",
        bbox: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        alert_type: str = "person_alert",
        source: str = "esp32",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Chuẩn hóa gói tin cảnh báo xâm nhập / nhận diện khuôn mặt và broadcast."""
        event_data: Dict[str, Any] = {
            "type": "person_event",
            "id": evt_id,
            "evt_id": evt_id,
            "ts": float(ts),
            "camera_id": camera_id,
            "source": source,
            "person_id": person_id or "Unknown",
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "bbox": list(bbox) if bbox is not None else [],
            "snapshot_url": snapshot_url,
            "alert_type": alert_type,
        }
        if extra:
            event_data.update(extra)
        event_data["payload"] = {k: v for k, v in event_data.items() if k != "payload"}
        return await self.broadcast(event_data, camera_id=camera_id)

    # -------------------------------------------------------------------------
    # Heartbeat / Ping-Pong & Client Message Handling
    # -------------------------------------------------------------------------
    def touch(self, ws: WebSocket) -> None:
        conn = self._connections.get(ws)
        if conn is not None:
            conn.touch()

    async def handle_client_message(self, ws: WebSocket, raw_text: str) -> None:
        """Xử lý gói tin từ client (ping/pong hoặc đăng ký lọc camera)."""
        self.touch(ws)
        text = (raw_text or "").strip()
        if not text:
            return

        if text.lower() == "ping":
            await self._reply_pong(ws)
            return
        if text.lower() == "pong":
            return

        try:
            msg = json.loads(text)
        except Exception:
            return

        if not isinstance(msg, dict):
            return

        msg_type = str(msg.get("type", "")).lower()
        if msg_type == "ping":
            await self._reply_pong(ws)
        elif msg_type == "pong":
            pass
        elif msg_type == "subscribe":
            cams = msg.get("cameras")
            conn = self._connections.get(ws)
            if conn is not None:
                conn.cameras = set(map(str, cams)) if isinstance(cams, list) and cams else None

    async def _reply_pong(self, ws: WebSocket) -> None:
        try:
            await asyncio.wait_for(
                ws.send_json({"type": "pong", "ts": time.time()}),
                timeout=self.send_timeout,
            )
        except Exception:
            await self.disconnect(ws)

    async def sweep_once(self, now: Optional[float] = None) -> int:
        """Quét 1 lượt: ngắt các kết nối quá hạn timeout và gửi ping cho phần còn lại."""
        now_ts = now if now is not None else time.time()
        async with self._lock:
            conns = list(self._connections.values())

        if not conns:
            return 0

        expired: List[WebSocket] = []
        alive: List[ClientConnection] = []
        for c in conns:
            if (now_ts - c.last_seen) > self.heartbeat_timeout:
                expired.append(c.ws)
            else:
                alive.append(c)

        ping_msg = {"type": "ping", "ts": now_ts}
        if alive:
            results = await asyncio.gather(
                *(self._send_one(c, ping_msg) for c in alive),
                return_exceptions=True,
            )
            for c, res in zip(alive, results):
                if isinstance(res, Exception) or res is not None:
                    expired.append(c.ws)

        for ws in expired:
            await self.disconnect(ws)
        return len(expired)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.sweep_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Heartbeat loop error: %s", e)

    def start_heartbeat(self) -> asyncio.Task:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self._heartbeat_task

    async def stop_heartbeat(self) -> None:
        self._running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None


# Giữ alias WSManager để tương thích hoàn toàn với code hiện tại
WSManager = ConnectionManager
