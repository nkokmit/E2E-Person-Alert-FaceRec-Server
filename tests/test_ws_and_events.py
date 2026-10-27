"""Unit & integration tests for ConnectionManager (app/ws_manager.py) and Event History REST API."""
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, List

from starlette.websockets import WebSocketState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import models
from app.api_min import create_token, get_event_detail, list_events, events_recent
from app.db import SessionLocal
from app.ws_manager import ConnectionManager


class DummyWebSocket:
    def __init__(self, fail_send: bool = False):
        self.client_state = WebSocketState.CONNECTING
        self.fail_send = fail_send
        self.sent: List[Any] = []
        self.closed = False

    async def accept(self):
        self.client_state = WebSocketState.CONNECTED

    async def send_json(self, data: Any):
        if self.fail_send:
            raise RuntimeError("socket broken")
        self.sent.append(data)

    async def close(self):
        self.closed = True
        self.client_state = WebSocketState.DISCONNECTED


async def _test_connection_manager_broadcast_and_heartbeat():
    mgr = ConnectionManager(heartbeat_interval=0.1, heartbeat_timeout=1.0, send_timeout=0.5)

    ws1 = DummyWebSocket()
    ws2 = DummyWebSocket()
    ws_dead = DummyWebSocket(fail_send=True)

    c1 = await mgr.connect(ws1, client_id="c1", cameras=["front_gate"])
    c2 = await mgr.connect(ws2, client_id="c2")  # all cameras
    await mgr.connect(ws_dead, client_id="dead")
    assert mgr.active_count == 3

    # 1. Broadcast alert to front_gate -> ws1 & ws2 receive, ws_dead is automatically cleaned up
    sent_count = await mgr.broadcast_alert(
        evt_id="evt_001",
        ts=1730000000.0,
        camera_id="front_gate",
        snapshot_url="/snapshots/events/evt_001.jpg",
        person_id="Hung",
        bbox=[10.0, 20.0, 100.0, 200.0],
        confidence=0.93,
        alert_type="face_known",
    )
    assert sent_count == 2
    assert mgr.active_count == 2
    assert ws_dead.closed is True
    assert ws1.sent[-1]["person_id"] == "Hung"
    assert ws1.sent[-1]["snapshot_url"] == "/snapshots/events/evt_001.jpg"
    assert ws1.sent[-1]["bbox"] == [10.0, 20.0, 100.0, 200.0]

    # 2. Camera filter: broadcast to back_door -> only ws2 receives
    sent_count = await mgr.broadcast_alert(
        evt_id="evt_002",
        ts=1730000005.0,
        camera_id="back_door",
        snapshot_url="/snapshots/events/evt_002.jpg",
        person_id="Unknown",
        bbox=[0.0, 0.0, 50.0, 50.0],
    )
    assert sent_count == 1
    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 2

    # 3. Ping / pong handling
    c1.last_seen = time.time() - 10.0
    await mgr.handle_client_message(ws1, json.dumps({"type": "ping"}))
    assert ws1.sent[-1]["type"] == "pong"
    assert (time.time() - c1.last_seen) < 1.0

    # 4. Stale client eviction via sweep_once
    c2.last_seen = time.time() - 5.0  # > heartbeat_timeout (1.0s)
    evicted = await mgr.sweep_once()
    assert evicted == 1
    assert mgr.active_count == 1
    assert ws2.closed is True

    await mgr.disconnect_all()
    assert mgr.active_count == 0


def test_event_history_rest_pagination_and_filters():
    prefix = f"ut_{uuid.uuid4().hex[:8]}_"
    base_ts = 1800000000.0
    sample_rows = [
        (f"{prefix}1", base_ts + 10, "front_gate", "Hung", 0.91, [1, 2, 3, 4], "face_known"),
        (f"{prefix}2", base_ts + 20, "front_gate", "Unknown", 0.65, [5, 6, 7, 8], "face_unknown"),
        (f"{prefix}3", base_ts + 30, "back_door", "Hung", 0.88, [9, 10, 11, 12], "face_known"),
        (f"{prefix}4", base_ts + 40, "back_door", "Binh", 0.95, [13, 14, 15, 16], "face_known"),
    ]

    with SessionLocal() as db:
        try:
            for eid, ts, cam, pid, conf, bbox, atype in sample_rows:
                db.add(
                    models.Event(
                        evt_id=eid,
                        ts=ts,
                        camera_id=cam,
                        source="esp32",
                        bbox=json.dumps(bbox),
                        snapshot_url=f"/snapshots/events/{eid}.jpg",
                        person_id=pid,
                        confidence=conf,
                        alert_type=atype,
                    )
                )
            db.commit()

            # Filter by time range + person_id="Hung"
            res_hung = list_events(
                page=1,
                page_size=10,
                start_ts=base_ts,
                end_ts=base_ts + 50,
                person_id="Hung",
                camera_id=None,
                alert_type=None,
                db=db,
                user=None,
            )
            assert res_hung["ok"] is True
            assert res_hung["total"] == 2
            assert [it["evt_id"] for it in res_hung["items"]] == [f"{prefix}3", f"{prefix}1"]
            assert res_hung["items"][0]["bbox"] == [9, 10, 11, 12]

            # Pagination (page_size=2 across all 4 events in time range)
            p1 = list_events(
                page=1,
                page_size=2,
                start_ts=base_ts,
                end_ts=base_ts + 50,
                person_id=None,
                camera_id=None,
                alert_type=None,
                db=db,
                user=None,
            )
            p2 = list_events(
                page=2,
                page_size=2,
                start_ts=base_ts,
                end_ts=base_ts + 50,
                person_id=None,
                camera_id=None,
                alert_type=None,
                db=db,
                user=None,
            )
            assert p1["total"] == 4 and p1["total_pages"] == 2
            assert [it["evt_id"] for it in p1["items"]] == [f"{prefix}4", f"{prefix}3"]
            assert [it["evt_id"] for it in p2["items"]] == [f"{prefix}2", f"{prefix}1"]

            # Detail endpoint
            detail = get_event_detail(f"{prefix}4", db=db, user=None)
            assert detail["event"]["person_id"] == "Binh"

            # Recent endpoint with time filter
            rec = events_recent(
                limit=10,
                offset=0,
                start_ts=base_ts + 15,
                end_ts=base_ts + 35,
                person_id=None,
                camera_id=None,
                alert_type=None,
                db=db,
                user=None,
            )
            assert [r["evt_id"] for r in rec] == [f"{prefix}3", f"{prefix}2"]
        finally:
            db.query(models.Event).filter(models.Event.evt_id.like(f"{prefix}%")).delete(
                synchronize_session=False
            )
            db.commit()


if __name__ == "__main__":
    asyncio.run(_test_connection_manager_broadcast_and_heartbeat())
    print("PASS: _test_connection_manager_broadcast_and_heartbeat")
    test_event_history_rest_pagination_and_filters()
    print("PASS: test_event_history_rest_pagination_and_filters")
    print("All WS and Event REST tests passed!")

