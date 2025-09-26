# app/ws_manager.py
from typing import Set
from fastapi import WebSocket

class WSManager:
    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    
    async def broadcast(self, data: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(data)
                print("Sent event to client:", ws.client)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)