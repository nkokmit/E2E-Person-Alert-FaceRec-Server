"""Unit tests cho phần số học của collector/face_capture.py (không cần file model)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector.face_capture import (  # noqa: E402
    ARCFACE_TEMPLATE_112, UNKNOWN_LABEL, FaceGallery, _canonicalize_kps,
    batch_sharpness, cosine_match, estimate_similarity_batch, l2_normalize, yaml,
    FacePipelineConfig,
)

rng = np.random.default_rng(0)


def test_l2_normalize():
    x = rng.normal(size=(10, 512)).astype(np.float32)
    n = l2_normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(l2_normalize(np.zeros((2, 8)))))


def test_similarity_recovers_known_transform():
    # Áp phép similarity đã biết lên template, kiểm tra Umeyama khôi phục ngược lại.
    B = 16
    ang = rng.uniform(-0.6, 0.6, B)
    s = rng.uniform(0.5, 3.0, B)
    t = rng.uniform(-50, 300, (B, 2))
    R = np.stack([np.stack([np.cos(ang), -np.sin(ang)], -1), np.stack([np.sin(ang), np.cos(ang)], -1)], 1)
    src = s[:, None, None] * np.einsum("bij,kj->bki", R, ARCFACE_TEMPLATE_112) + t[:, None, :]
    M = estimate_similarity_batch(src, ARCFACE_TEMPLATE_112)
    mapped = np.einsum("bij,bkj->bki", M[:, :, :2], src) + M[:, None, :, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE_112, atol=1e-3)


def test_similarity_matches_opencv_on_noisy_points():
    src = ARCFACE_TEMPLATE_112 * 2.3 + 40 + rng.normal(0, 2.0, (5, 2)).astype(np.float32)
    ours = estimate_similarity_batch(src[None], ARCFACE_TEMPLATE_112)[0]
    ref, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    err_ours = np.abs(src @ ours[:, :2].T + ours[:, 2] - ARCFACE_TEMPLATE_112).mean()
    err_ref = np.abs(src @ ref[:, :2].T + ref[:, 2] - ARCFACE_TEMPLATE_112).mean()
    assert err_ours <= err_ref + 0.5  # least-squares tối ưu, không tệ hơn LMEDS


def test_canonicalize_kps_swaps():
    k = np.array([[[70, 50], [40, 50], [55, 70], [68, 90], [42, 90]]], np.float32)
    c = _canonicalize_kps(k)
    assert c[0, 0, 0] < c[0, 1, 0] and c[0, 3, 0] < c[0, 4, 0]


def test_sharpness_matches_opencv():
    faces = rng.integers(0, 255, (4, 112, 112, 3), dtype=np.uint8)
    faces[0] = cv2.GaussianBlur(faces[0], (9, 9), 3)
    ours = batch_sharpness(faces)
    for i, f in enumerate(faces):
        g = f.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)  # float gray, no uint8 rounding
        ref = cv2.Laplacian(g, cv2.CV_32F)[1:-1, 1:-1].var()
        assert ours[i] == pytest.approx(ref, rel=0.02)
    assert ours[0] < ours[1]


def _naive_match(q, gal, labels, thr, margin):
    out = []
    for v in q:
        best = {}
        for e, l in zip(gal, labels):
            s = float(np.dot(v, e) / (np.linalg.norm(v) * np.linalg.norm(e)))
            best[l] = max(best.get(l, -2), s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        b, sec = ranked[0][1], (ranked[1][1] if len(ranked) > 1 else -1.0)
        out.append((ranked[0][0] if b >= thr and b - sec >= margin else UNKNOWN_LABEL, b))
    return out


def test_gallery_match_equals_naive_loop():
    g = FaceGallery(dim=64)
    centers = {n: rng.normal(size=64) for n in ["an", "binh", "chi", "dung"]}
    for n, c in centers.items():
        g.add(n, c + rng.normal(0, 0.3, (3, 64)))
    q = np.stack([centers["binh"] + rng.normal(0, 0.3, 64), centers["chi"] + rng.normal(0, 0.3, 64),
                  rng.normal(size=64)])
    q = l2_normalize(q)
    res = g.match(q, threshold=0.5, margin=0.05)
    ref = _naive_match(q, g._emb, g._labels, 0.5, 0.05)
    assert list(res.labels) == [r[0] for r in ref]
    assert np.allclose(res.scores, [r[1] for r in ref], atol=1e-5)
    assert res.labels[0] == "binh" and res.labels[1] == "chi" and res.labels[2] == UNKNOWN_LABEL


def test_empty_gallery_and_single_identity(tmp_path):
    g = FaceGallery(dim=8)
    r = g.match(l2_normalize(rng.normal(size=(2, 8))))
    assert list(r.labels) == [UNKNOWN_LABEL] * 2 and not r.known.any()
    v = l2_normalize(rng.normal(size=8))
    g.add("solo", v)
    assert g.match(v[None], threshold=0.9).labels[0] == "solo"
    g.save(tmp_path / "g.npz")
    g2 = FaceGallery.load(tmp_path / "g.npz", dim=8)
    assert len(g2) == 1 and g2.match(v[None]).labels[0] == "solo"
    assert g2.remove("solo") == 1 and len(g2) == 0


def test_config_loads_from_repo_yaml():
    cfg = FacePipelineConfig.from_yaml()
    assert cfg.detector in {"scrfd", "yunet", "yolo"} and 0 < cfg.rec_threshold < 1
