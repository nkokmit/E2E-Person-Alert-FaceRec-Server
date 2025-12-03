# app/ws_manager.py
from typing import Set
from fastapi import WebSocket

class WSManager:
    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()

    async def register(self, ws: WebSocket):
        # api_min đã gọi ws.accept() trước khi register
        self.clients.add(ws)

    async def unregister(self, ws: WebSocket):
        try:
            await ws.close()
        except Exception:
            pass
        self.clients.discard(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)
