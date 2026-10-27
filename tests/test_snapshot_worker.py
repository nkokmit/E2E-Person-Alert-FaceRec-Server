"""Unit tests for production SnapshotWorker and its components."""
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.snapshot_worker import (
    SmartFrameQueue,
    AlertDebounceManager,
    draw_detection_overlay,
    SnapshotWorker,
)
from app.db import SessionLocal
from app.models import Event


def test_smart_frame_queue_overflow_drop():
    """Verify that when queue is full, oldest frames are dropped and newest are preserved."""
    q = SmartFrameQueue(maxsize=3, max_latency_ms=150.0)

    # Put 5 items into queue of size 3
    for i in range(5):
        q.put_nowait({"id": i, "ts": time.time()})

    stats = q.get_stats()
    assert stats["qsize"] == 3
    assert stats["dropped_overflow"] == 2
    assert stats["enqueued"] == 5

    # Oldest (0 and 1) were dropped; items 2, 3, 4 should remain in order
    rem = [q.get()["id"] for _ in range(3)]
    assert rem == [2, 3, 4]


def test_alert_debounce_30s():
    """Verify 30s cooldown per person per camera."""
    mgr = AlertDebounceManager(default_cooldown_sec=30.0)
    t0 = 1000.0

    # Person "Hung" at front_gate
    assert mgr.should_alert("front_gate", "Hung", now_ts=t0) is True
    mgr.mark_alerted("front_gate", "Hung", now_ts=t0)

    # 5 seconds later -> in cooldown (should NOT alert)
    assert mgr.should_alert("front_gate", "Hung", now_ts=t0 + 5.0) is False

    # 29.9 seconds later -> still in cooldown
    assert mgr.should_alert("front_gate", "Hung", now_ts=t0 + 29.9) is False

    # 30.1 seconds later -> cooldown expired (should alert)
    assert mgr.should_alert("front_gate", "Hung", now_ts=t0 + 30.1) is True
    mgr.mark_alerted("front_gate", "Hung", now_ts=t0 + 30.1)

    # Another person "Stranger" at front_gate is NOT blocked
    assert mgr.should_alert("front_gate", "Stranger", now_ts=t0 + 5.0) is True

    # Same person at a different camera "back_door" is NOT blocked
    assert mgr.should_alert("back_door", "Hung", now_ts=t0 + 5.0) is True


def test_draw_detection_overlay():
    """Verify drawing bounding boxes, badges, landmarks, and header on image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = [{"bbox": [50, 50, 200, 400], "conf": 0.92}]
    faces = [
        {
            "bbox": [80, 70, 160, 180],
            "label": "Hung",
            "similarity": 0.85,
            "status": "known",
            "kps": [[95, 100], [135, 100], [115, 125], [100, 150], [130, 150]],
        }
    ]
    out = draw_detection_overlay(img, persons, faces, info_header="CAM: test | Alert: KNOWN")
    assert out.shape == img.shape
    assert not np.array_equal(out, img)  # Image has been modified with annotations


def test_snapshot_worker_db_save():
    """Verify thread-safe database insertion into SQLite."""
    evt_id = f"test_{int(time.time()*1000)}"
    worker = SnapshotWorker(emit_async=None)
    data = {
        "evt_id": evt_id,
        "ts": time.time(),
        "camera_id": "test_cam",
        "source": "esp32",
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "snapshot_url": f"/snapshots/events/{evt_id}.jpg",
        "person_id": "Hung",
        "confidence": 0.89,
        "alert_type": "face_known",
    }
    success = worker._save_event_db(data)
    assert success is True

    # Read back from DB
    with SessionLocal() as db:
        ev = db.query(Event).filter(Event.evt_id == evt_id).first()
        assert ev is not None
        assert ev.person_id == "Hung"
        assert abs(ev.confidence - 0.89) < 1e-4
        assert ev.camera_id == "test_cam"
        db.delete(ev)
        db.commit()


def run_all():
    test_smart_frame_queue_overflow_drop()
    print("PASS: test_smart_frame_queue_overflow_drop")
    test_alert_debounce_30s()
    print("PASS: test_alert_debounce_30s")
    test_draw_detection_overlay()
    print("PASS: test_draw_detection_overlay")
    test_snapshot_worker_db_save()
    print("PASS: test_snapshot_worker_db_save")
    print("All SnapshotWorker tests passed!")


if __name__ == "__main__":
    run_all()


def test_snapshot_worker_pipeline_end_to_end():
    """Verify background worker thread processing from queue."""
    emitted = []
    async def dummy_emit(msg):
        emitted.append(msg)

    worker = SnapshotWorker(emit_async=dummy_emit)
    worker.start_worker_thread()

    # Create dummy image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put frame into queue
    worker.queue.put_nowait({
        "camera_id": "test_cam_e2e",
        "ts": time.time(),
        "frame": test_img,
    })

    # Wait briefly for worker thread to process
    time.sleep(0.5)

    stats = worker.queue.get_stats()
    assert stats["processed"] >= 1
    worker.stop()
    print("PASS: test_snapshot_worker_pipeline_end_to_end")

if __name__ == "__main__":
    test_snapshot_worker_pipeline_end_to_end()

