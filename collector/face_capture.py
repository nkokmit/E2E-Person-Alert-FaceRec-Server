# collector/face_capture.py
"""
Real-time face capture & recognition pipeline.

    frame (BGR) ─► detect (SCRFD | YuNet | YOLO-Face, 5 landmarks)
                ─► align (similarity transform -> ArcFace 112x112 template)
                ─► quality gate (min size, det score, sharpness)
                ─► embed (ArcFace / MobileFaceNet ONNX, batched) ─► L2 norm
                ─► cosine match vs gallery (matrix product) ─► KNOWN / UNKNOWN

Design notes
------------
* Không phụ thuộc package `insightface`: model ONNX (SCRFD det_500m/det_10g,
  ArcFace w600k_mbf/w600k_r50 trong buffalo_s / buffalo_l) được chạy trực tiếp
  bằng onnxruntime -> tự chọn CUDAExecutionProvider nếu có, fallback CPU.
* Mọi bước số học (decode anchor, similarity transform, sharpness, cosine,
  top-2, threshold) đều vectorized NumPy trên cả batch khuôn mặt.
  Vòng lặp Python còn lại duy nhất là cv2.warpAffine cho từng mặt (C++, ~µs).
* Thread-safe: ORT session.run() an toàn đa luồng; detector có state (YuNet,
  ultralytics) được bảo vệ bằng lock; gallery dùng RLock + snapshot bất biến.
* Trong FastAPI, gọi qua `await pipeline.aprocess(img)` (asyncio.to_thread)
  để không block event loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
UNKNOWN_LABEL = "Unknown"
_EPS = 1e-10

# Toạ độ 5 điểm chuẩn của ArcFace trên ảnh 112x112:
# mắt trái (ảnh), mắt phải (ảnh), mũi, khoé miệng trái, khoé miệng phải.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# =============================================================================
# Vector utilities
# =============================================================================
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = _EPS) -> np.ndarray:
    """L2-normalize theo `axis` (vectorized, an toàn với vector 0)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray, normalized: bool = False) -> np.ndarray:
    """(M,D) x (N,D) -> (M,N) cosine similarity bằng một phép GEMM (BLAS)."""
    if not normalized:
        a, b = l2_normalize(a), l2_normalize(b)
    return np.clip(a @ b.T, -1.0, 1.0)


# =============================================================================
# Detection container
# =============================================================================
@dataclass(slots=True)
class Detections:
    boxes: np.ndarray   # (N, 4) float32, x1 y1 x2 y2 (toạ độ ảnh gốc)
    scores: np.ndarray  # (N,)   float32
    kps: np.ndarray     # (N, 5, 2) float32

    @staticmethod
    def empty() -> "Detections":
        return Detections(
            np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0, 5, 2), np.float32)
        )

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def select(self, idx: np.ndarray) -> "Detections":
        return Detections(self.boxes[idx], self.scores[idx], self.kps[idx])

    @property
    def sizes(self) -> np.ndarray:
        """Cạnh ngắn của bbox (N,)."""
        wh = self.boxes[:, 2:4] - self.boxes[:, 0:2]
        return wh.min(axis=1)

    @property
    def areas(self) -> np.ndarray:
        wh = np.clip(self.boxes[:, 2:4] - self.boxes[:, 0:2], 0, None)
        return wh[:, 0] * wh[:, 1]


def _canonicalize_kps(kps: np.ndarray) -> np.ndarray:
    """Đảm bảo thứ tự [eye_L, eye_R, nose, mouth_L, mouth_R] theo trục x của ảnh.

    Mỗi detector có quy ước khác nhau (subject-left vs image-left); swap
    vectorized để alignment luôn khớp template ArcFace.
    """
    kps = kps.copy()
    for a, b in ((0, 1), (3, 4)):
        swap = kps[:, a, 0] > kps[:, b, 0]
        if swap.any():
            kps[swap, a], kps[swap, b] = kps[swap, b].copy(), kps[swap, a].copy()
    return kps


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """NMS bằng OpenCV (C++). boxes: x1y1x2y2."""
    if len(boxes) == 0:
        return np.empty((0,), np.int64)
    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, float(iou_thresh))
    return np.asarray(idx, dtype=np.int64).reshape(-1)


# =============================================================================
# ONNX Runtime helpers (CPU / GPU)
# =============================================================================
def _ensure_cuda_dlls() -> None:
    """Bổ sung đường dẫn DLL của PyTorch (cublas/cudnn) trên Windows cho onnxruntime-gpu."""
    import os
    import onnxruntime as ort

    torch_lib = Path(ort.__file__).resolve().parents[1] / "torch" / "lib"
    if torch_lib.exists():
        lib_str = str(torch_lib)
        if lib_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = lib_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_str)
            except Exception:
                pass


def resolve_providers(device: str = "auto") -> list[str]:
    """Chọn execution providers cho onnxruntime theo `device` và khả năng máy."""
    import onnxruntime as ort

    _ensure_cuda_dlls()
    available = set(ort.get_available_providers())
    device = (device or "auto").lower()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device.startswith("cuda"):
        log.warning("CUDAExecutionProvider không khả dụng -> fallback CPU")
    return ["CPUExecutionProvider"]


def create_session(model_path: str | Path, providers: Sequence[str], intra_threads: int = 0):
    import onnxruntime as ort

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model ONNX: {model_path}. Tải buffalo_s/buffalo_l từ "
            "https://github.com/deepinsight/insightface/releases (v0.7) và đặt vào models/faces/."
        )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads > 0:
        so.intra_op_num_threads = intra_threads
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=list(providers))
    log.info("Loaded %s with providers=%s", model_path.name, sess.get_providers())
    return sess


# =============================================================================
# Face detectors (all return 5 landmarks)
# =============================================================================
class FaceDetector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> Detections: ...


class SCRFDDetector:
    """SCRFD (InsightFace) qua onnxruntime. Decode anchor hoàn toàn vectorized."""

    # n_outputs -> (fmc, strides, num_anchors, has_kps)
    _LAYOUTS = {
        6: (3, (8, 16, 32), 2, False),
        9: (3, (8, 16, 32), 2, True),
        10: (5, (8, 16, 32, 64, 128), 1, False),
        15: (5, (8, 16, 32, 64, 128), 1, True),
    }

    def __init__(
        self,
        model_path: str | Path,
        input_size: int | tuple[int, int] = 640,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        intra_threads: int = 0,
    ):
        self.session = create_session(model_path, providers, intra_threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        h, w = inp.shape[2], inp.shape[3]
        # Model export với input cố định -> bắt buộc dùng đúng size đó.
        self.input_size = (w, h) if isinstance(h, int) and isinstance(w, int) else tuple(input_size)
        layout = self._LAYOUTS.get(len(self.output_names))
        if layout is None or not layout[3]:
            raise ValueError("SCRFD model phải có nhánh keypoints (5 điểm) để căn chỉnh khuôn mặt")
        self.fmc, self.strides, self.num_anchors, _ = layout
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, fh: int, fw: int, stride: int) -> np.ndarray:
        key = (fh, fw, stride)
        c = self._centers.get(key)
        if c is None:
            ys, xs = np.mgrid[:fh, :fw]
            c = (np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride).astype(np.float32)
            if self.num_anchors > 1:
                c = np.repeat(c, self.num_anchors, axis=0)
            self._centers[key] = c
        return c

    def detect(self, img_bgr: np.ndarray) -> Detections:
        h, w = img_bgr.shape[:2]
        in_w, in_h = self.input_size
        scale = min(in_w / w, in_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)  # letterbox (pad phải/dưới)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (in_w, in_h), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        batched = outs[0].ndim == 3

        boxes_l, scores_l, kps_l = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i]
            bbox = outs[i + self.fmc]
            kps = outs[i + 2 * self.fmc]
            if batched:
                scores, bbox, kps = scores[0], bbox[0], kps[0]
            scores = scores.reshape(-1)
            keep = scores >= self.score_thresh  # lọc TRƯỚC khi decode -> rẻ hơn
            if not keep.any():
                continue
            centers = self._anchor_centers(in_h // stride, in_w // stride, stride)[keep]
            d = bbox[keep] * stride
            boxes_l.append(np.concatenate([centers - d[:, :2], centers + d[:, 2:4]], axis=1))
            kps_l.append(centers[:, None, :] + (kps[keep] * stride).reshape(-1, 5, 2))
            scores_l.append(scores[keep])

        if not scores_l:
            return Detections.empty()
        boxes = np.concatenate(boxes_l).astype(np.float32) / scale
        kps = np.concatenate(kps_l).astype(np.float32) / scale
        scores = np.concatenate(scores_l).astype(np.float32)
        keep = _nms(boxes, scores, self.nms_thresh)
        return Detections(boxes[keep], scores[keep], _canonicalize_kps(kps[keep]))


class YuNetDetector:
    """OpenCV FaceDetectorYN (YuNet) – rất nhẹ, chạy tốt trên CPU, có sẵn 5 điểm mốc."""

    def __init__(
        self,
        model_path: str | Path,
        score_thresh: float = 0.6,
        nms_thresh: float = 0.3,
        top_k: int = 5000,
        use_cuda: bool = False,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy YuNet model: {model_path} "
                "(opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)"
            )
        backend = cv2.dnn.DNN_BACKEND_CUDA if use_cuda else cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CUDA if use_cuda else cv2.dnn.DNN_TARGET_CPU
        self._size = (320, 320)
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", self._size, float(score_thresh), float(nms_thresh), int(top_k), backend, target
        )
        self._lock = threading.Lock()  # detector giữ state input size

    def detect(self, img_bgr: np.ndarray) -> Detections:
        h, w = img_bgr.shape[:2]
        with self._lock:
            if (w, h) != self._size:
                self._det.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return Detections.empty()
        faces = faces.astype(np.float32)
        boxes = faces[:, :4].copy()
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        return Detections(boxes, faces[:, 14].copy(), _canonicalize_kps(faces[:, 4:14].reshape(-1, 5, 2)))


class YoloFaceDetector:
    """YOLO-Face (ultralytics pose-style, 5 keypoints, vd. yolov8n-face.pt)."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.45,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.half = bool(device and str(device).startswith("cuda"))
        self._lock = threading.Lock()

    def detect(self, img_bgr: np.ndarray) -> Detections:
        with self._lock:
            r = self.model.predict(
                img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device, half=self.half, verbose=False,
            )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty()
        if r.keypoints is None or r.keypoints.xy.shape[1] != 5:
            raise ValueError("YOLO model không trả về 5 keypoints – cần model YOLO-Face (pose 5 điểm)")
        return Detections(
            r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.conf.cpu().numpy().astype(np.float32),
            _canonicalize_kps(r.keypoints.xy.cpu().numpy().astype(np.float32)),
        )


# =============================================================================
# Face alignment (5-point similarity transform, batched Umeyama)
# =============================================================================
def estimate_similarity_batch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama (1991) vectorized cho cả batch.

    src: (B, K, 2) landmarks phát hiện được; dst: (K, 2) template.
    Trả về (B, 2, 3) ma trận affine (rotation + uniform scale + translation).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    k = src.shape[1]
    mu_s = src.mean(axis=1, keepdims=True)            # (B,1,2)
    mu_d = dst.mean(axis=0)                           # (2,)
    sc = src - mu_s
    dc = dst - mu_d
    cov = np.einsum("ki,bkj->bij", dc, sc) / k        # (B,2,2)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones((src.shape[0], 2))
    d[np.linalg.det(cov) < 0, 1] = -1.0               # tránh phản xạ
    r = (u * d[:, None, :]) @ vt                      # U diag(d) Vt
    var_s = (sc ** 2).sum(axis=(1, 2)) / k
    scale = (s * d).sum(axis=1) / np.maximum(var_s, _EPS)
    t = mu_d[None, :] - scale[:, None] * np.einsum("bij,bj->bi", r, mu_s[:, 0, :])
    m = np.concatenate([scale[:, None, None] * r, t[:, :, None]], axis=2)
    return m.astype(np.float32)


def align_faces(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Căn chỉnh tất cả khuôn mặt về template ArcFace. Trả về (B, size, size, 3) uint8."""
    if len(kps) == 0:
        return np.empty((0, size, size, 3), np.uint8)
    template = ARCFACE_TEMPLATE_112 * (size / 112.0)
    mats = estimate_similarity_batch(kps, template)
    out = np.empty((len(kps), size, size, 3), np.uint8)
    for i, m in enumerate(mats):  # warpAffine là C++; vòng lặp chỉ dispatch
        out[i] = cv2.warpAffine(img_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    return out


def batch_sharpness(faces: np.ndarray) -> np.ndarray:
    """Phương sai Laplacian (độ nét) cho cả batch (B,H,W,3) – không vòng lặp."""
    if len(faces) == 0:
        return np.empty((0,), np.float32)
    g = faces.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR->gray
    lap = (
        g[:, :-2, 1:-1] + g[:, 2:, 1:-1] + g[:, 1:-1, :-2] + g[:, 1:-1, 2:] - 4.0 * g[:, 1:-1, 1:-1]
    )
    return lap.var(axis=(1, 2))


# =============================================================================
# Embedding extractor (ArcFace / MobileFaceNet ONNX)
# =============================================================================
class ArcFaceEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        input_mean: float = 127.5,
        input_std: float = 127.5,
        max_batch: int = 32,
        flip_tta: bool = False,
        intra_threads: int = 0,
    ):
        self.session = create_session(model_path, providers, intra_threads)
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        h, w = inp.shape[2], inp.shape[3]
        self.input_size = (w, h) if isinstance(h, int) and isinstance(w, int) else (112, 112)
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.batch = self.fixed_batch or int(max_batch)
        self.dim = out.shape[-1] if isinstance(out.shape[-1], int) else 512
        self.mean, self.std = float(input_mean), float(input_std)
        self.flip_tta = flip_tta

    def _run(self, blob: np.ndarray) -> np.ndarray:
        n = blob.shape[0]
        if n <= self.batch and (self.fixed_batch is None or n == self.fixed_batch):
            return self.session.run([self.output_name], {self.input_name: blob})[0]
        chunks = []
        for s in range(0, n, self.batch):
            part = blob[s : s + self.batch]
            pad = (self.fixed_batch - len(part)) if self.fixed_batch else 0
            if pad > 0:  # model batch cố định -> pad rồi cắt
                part = np.concatenate([part, np.zeros((pad, *part.shape[1:]), part.dtype)])
            res = self.session.run([self.output_name], {self.input_name: part})[0]
            chunks.append(res[: len(res) - pad] if pad > 0 else res)
        return np.concatenate(chunks, axis=0)

    def embed(self, faces_bgr: np.ndarray) -> np.ndarray:
        """faces_bgr: (B,H,W,3) uint8 đã align -> (B, D) float32 đã L2-normalize."""
        if len(faces_bgr) == 0:
            return np.empty((0, self.dim), np.float32)
        blob = cv2.dnn.blobFromImages(
            list(faces_bgr), 1.0 / self.std, self.input_size, (self.mean,) * 3, swapRB=True
        )
        feats = self._run(blob)
        if self.flip_tta:  # cộng embedding ảnh lật ngang -> ổn định hơn ~1-2%
            feats = feats + self._run(np.ascontiguousarray(blob[..., ::-1]))
        return l2_normalize(feats)


# =============================================================================
# Gallery + cosine matching
# =============================================================================
class FaceStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    LOW_QUALITY = "low_quality"  # mặt quá mờ/nhỏ -> không phân loại để tránh báo động giả


@dataclass(slots=True)
class MatchResult:
    labels: np.ndarray       # (M,) str
    scores: np.ndarray       # (M,) best cosine similarity
    second: np.ndarray       # (M,) cosine của identity thứ 2 (-1 nếu không có)
    known: np.ndarray        # (M,) bool


def cosine_match(
    queries: np.ndarray,
    gallery: np.ndarray,
    starts: np.ndarray,
    names: np.ndarray,
    threshold: float = 0.40,
    margin: float = 0.0,
    aggregate: str = "max",
    normalized: bool = True,
) -> MatchResult:
    """So khớp M query với N embedding thuộc K identity – hoàn toàn vectorized.

    gallery: (N, D) đã sắp xếp theo identity; starts: (K,) chỉ số bắt đầu của
    mỗi identity trong gallery; names: (K,).
    Known  <=> best >= threshold  AND  (best - second) >= margin.
    """
    q = queries if normalized else l2_normalize(queries)
    m = q.shape[0]
    if m == 0 or gallery.shape[0] == 0:
        return MatchResult(
            np.full(m, UNKNOWN_LABEL, dtype=object), np.full(m, -1.0, np.float32),
            np.full(m, -1.0, np.float32), np.zeros(m, bool),
        )

    sims = np.clip(q @ gallery.T, -1.0, 1.0)                       # (M, N)
    if aggregate == "mean":
        counts = np.diff(np.append(starts, gallery.shape[0]))
        id_sims = np.add.reduceat(sims, starts, axis=1) / counts     # (M, K)
    else:
        id_sims = np.maximum.reduceat(sims, starts, axis=1)          # (M, K)

    k = id_sims.shape[1]
    if k == 1:
        best_idx = np.zeros(m, np.int64)
        best = id_sims[:, 0]
        second = np.full(m, -1.0, np.float32)
    else:
        top2 = np.argpartition(-id_sims, 1, axis=1)[:, :2]           # O(K) thay vì sort
        top2_s = np.take_along_axis(id_sims, top2, axis=1)
        order = np.argsort(-top2_s, axis=1)
        top2 = np.take_along_axis(top2, order, axis=1)
        top2_s = np.take_along_axis(top2_s, order, axis=1)
        best_idx, best, second = top2[:, 0], top2_s[:, 0], top2_s[:, 1]

    known = (best >= threshold) & ((best - second) >= margin)
    labels = np.where(known, names[best_idx].astype(object), UNKNOWN_LABEL)
    return MatchResult(labels, best.astype(np.float32), second.astype(np.float32), known)


class FaceGallery:
    """Kho embedding khuôn mặt đã đăng ký (nhiều embedding / người)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._lock = threading.RLock()
        self._emb = np.empty((0, dim), np.float32)
        self._labels = np.empty((0,), dtype=np.str_)
        self._index: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return int(self._emb.shape[0])

    @property
    def names(self) -> list[str]:
        return sorted(set(self._labels.tolist()))

    def add(self, name: str, embeddings: np.ndarray) -> None:
        emb = l2_normalize(np.atleast_2d(embeddings))
        if emb.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {emb.shape[1]} != gallery dim {self.dim}")
        with self._lock:
            self._emb = np.concatenate([self._emb, emb])
            # Không truyền dtype=np.str_ (sẽ thành '<U1' và cắt tên); để NumPy tự suy ra '<Un'.
            self._labels = np.concatenate([self._labels, np.full(len(emb), str(name))])
            self._index = None

    def remove(self, name: str) -> int:
        with self._lock:
            keep = self._labels != name
            removed = int((~keep).sum())
            self._emb, self._labels = self._emb[keep], self._labels[keep]
            self._index = None
            return removed

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gallery_sorted, starts, names) – rebuild lazily khi gallery thay đổi."""
        with self._lock:
            if self._index is None:
                order = np.argsort(self._labels, kind="stable")
                labels_sorted = self._labels[order]
                names, starts = np.unique(labels_sorted, return_index=True)
                self._index = (np.ascontiguousarray(self._emb[order]), starts, names)
            return self._index

    def match(self, queries: np.ndarray, threshold: float = 0.40, margin: float = 0.0,
              aggregate: str = "max") -> MatchResult:
        gal, starts, names = self._snapshot()
        return cosine_match(queries, gal, starts, names, threshold, margin, aggregate, normalized=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.savez_compressed(path, embeddings=self._emb, labels=self._labels)

    @classmethod
    def load(cls, path: str | Path, dim: int = 512) -> "FaceGallery":
        g = cls(dim)
        path = Path(path)
        if path.exists():
            data = np.load(path, allow_pickle=False)
            emb, labels = data["embeddings"].astype(np.float32), data["labels"].astype(np.str_)
            g.dim = emb.shape[1] if emb.size else dim
            g._emb, g._labels = l2_normalize(emb).reshape(-1, g.dim), labels
            log.info("Loaded gallery %s: %d embeddings / %d identities", path, len(g), len(g.names))
        return g


# =============================================================================
# Pipeline
# =============================================================================
@dataclass
class FaceResult:
    bbox: list[float]
    kps: list[list[float]]
    det_score: float
    label: str
    similarity: float
    status: FaceStatus
    sharpness: float
    embedding: np.ndarray | None = field(default=None, repr=False)
    aligned: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "kps": [[round(x, 1), round(y, 1)] for x, y in self.kps],
            "det_score": round(self.det_score, 4),
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "status": self.status.value,
            "sharpness": round(self.sharpness, 1),
        }


@dataclass
class FacePipelineConfig:
    device: str = "auto"                     # auto | cpu | cuda
    detector: str = "scrfd"                  # scrfd | yunet | yolo
    det_model: str = "models/faces/det_500m.onnx"
    det_input_size: int = 640
    det_score_thresh: float = 0.5
    det_nms_thresh: float = 0.4
    rec_model: str = "models/faces/w600k_mbf.onnx"
    rec_threshold: float = 0.40              # cosine; w600k_mbf ~0.35-0.45, r50 ~0.40-0.50
    rec_margin: float = 0.05                 # best - second >= margin (giảm nhầm giữa 2 người giống nhau)
    aggregate: str = "max"                   # max | mean
    flip_tta: bool = False
    rec_batch: int = 32
    min_face_size: int = 40                  # px, cạnh ngắn bbox
    min_sharpness: float = 25.0              # var(Laplacian) trên crop 112x112
    max_faces: int = 20
    intra_threads: int = 0                   # 0 = để ORT tự chọn
    gallery_path: str = "data/faces/gallery.npz"

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, section: str = "face") -> "FacePipelineConfig":
        path = Path(path) if path else BASE_DIR / "configs.yaml"
        data: dict = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = (yaml.safe_load(f) or {}).get(section, {}) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def resolve(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else BASE_DIR / pp


def build_detector(cfg: FacePipelineConfig, providers: Sequence[str]) -> FaceDetector:
    name = cfg.detector.lower()
    use_gpu = "CUDAExecutionProvider" in providers
    if name == "scrfd":
        return SCRFDDetector(cfg.resolve(cfg.det_model), cfg.det_input_size, cfg.det_score_thresh,
                             cfg.det_nms_thresh, providers, cfg.intra_threads)
    if name == "yunet":
        return YuNetDetector(cfg.resolve(cfg.det_model), cfg.det_score_thresh, cfg.det_nms_thresh,
                             use_cuda=use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    if name in ("yolo", "yolo-face", "yoloface"):
        return YoloFaceDetector(cfg.resolve(cfg.det_model), "cuda:0" if use_gpu else "cpu",
                                cfg.det_input_size, cfg.det_score_thresh, cfg.det_nms_thresh)
    raise ValueError(f"Unknown detector backend: {cfg.detector}")


class FacePipeline:
    def __init__(self, detector: FaceDetector, embedder: ArcFaceEmbedder, gallery: FaceGallery,
                 cfg: FacePipelineConfig):
        self.detector, self.embedder, self.gallery, self.cfg = detector, embedder, gallery, cfg
        self.align_size = embedder.input_size[0]

    @classmethod
    def from_config(cls, cfg: FacePipelineConfig | None = None) -> "FacePipeline":
        cfg = cfg or FacePipelineConfig.from_yaml()
        providers = resolve_providers(cfg.device)
        detector = build_detector(cfg, providers)
        embedder = ArcFaceEmbedder(cfg.resolve(cfg.rec_model), providers, max_batch=cfg.rec_batch,
                                   flip_tta=cfg.flip_tta, intra_threads=cfg.intra_threads)
        gallery = FaceGallery.load(cfg.resolve(cfg.gallery_path), dim=embedder.dim)
        pipe = cls(detector, embedder, gallery, cfg)
        pipe.warmup()
        return pipe

    def warmup(self) -> None:
        """Chạy 1 lần để khởi tạo CUDA context / cuDNN autotune trước frame thật."""
        dummy = np.zeros((480, 640, 3), np.uint8)
        self.detector.detect(dummy)
        self.embedder.embed(np.zeros((1, self.align_size, self.align_size, 3), np.uint8))

    # ---- stages -------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> Detections:
        dets = self.detector.detect(img_bgr)
        if len(dets) == 0:
            return dets
        keep = np.flatnonzero((dets.sizes >= self.cfg.min_face_size) & (dets.scores >= self.cfg.det_score_thresh))
        if len(keep) > self.cfg.max_faces:
            keep = keep[np.argsort(-dets.scores[keep])[: self.cfg.max_faces]]
        return dets.select(keep)

    def align(self, img_bgr: np.ndarray, dets: Detections) -> np.ndarray:
        return align_faces(img_bgr, dets.kps, self.align_size)

    # ---- end-to-end -----------------------------------------------------------
    def process(self, img_bgr: np.ndarray, return_embeddings: bool = False,
                return_aligned: bool = False) -> list[FaceResult]:
        dets = self.detect(img_bgr)
        n = len(dets)
        if n == 0:
            return []

        crops = self.align(img_bgr, dets)
        sharp = batch_sharpness(crops)
        good = sharp >= self.cfg.min_sharpness

        labels = np.full(n, UNKNOWN_LABEL, dtype=object)
        sims = np.full(n, -1.0, np.float32)
        status = np.full(n, FaceStatus.LOW_QUALITY, dtype=object)
        emb_all = np.zeros((n, self.embedder.dim), np.float32)

        if good.any():  # chỉ embed các mặt đạt chất lượng, 1 batch duy nhất
            emb = self.embedder.embed(crops[good])
            res = self.gallery.match(emb, self.cfg.rec_threshold, self.cfg.rec_margin, self.cfg.aggregate)
            emb_all[good] = emb
            labels[good] = res.labels
            sims[good] = res.scores
            status[good] = np.where(res.known, FaceStatus.KNOWN, FaceStatus.UNKNOWN)

        boxes, kps, scores = dets.boxes.tolist(), dets.kps.tolist(), dets.scores.tolist()
        return [
            FaceResult(
                bbox=boxes[i], kps=kps[i], det_score=scores[i], label=str(labels[i]),
                similarity=float(sims[i]), status=status[i], sharpness=float(sharp[i]),
                embedding=emb_all[i] if return_embeddings and good[i] else None,
                aligned=crops[i] if return_aligned else None,
            )
            for i in range(n)
        ]

    async def aprocess(self, img_bgr: np.ndarray, **kw) -> list[FaceResult]:
        """Dùng trong FastAPI/async worker: chạy inference ở thread pool."""
        return await asyncio.to_thread(self.process, img_bgr, **kw)

    # ---- enrollment -----------------------------------------------------------
    def enroll(self, name: str, images: Iterable[np.ndarray], save: bool = True) -> int:
        """Đăng ký 1 người từ nhiều ảnh (lấy mặt lớn nhất mỗi ảnh). Trả về số embedding thêm vào."""
        crops = []
        for img in images:
            if img is None:
                continue
            dets = self.detect(img)
            if len(dets) == 0:
                continue
            best = dets.select(np.array([int(np.argmax(dets.areas))]))
            crops.append(self.align(img, best)[0])
        if not crops:
            return 0
        crops_arr = np.stack(crops)
        crops_arr = crops_arr[batch_sharpness(crops_arr) >= self.cfg.min_sharpness]
        if len(crops_arr) == 0:
            return 0
        self.gallery.add(name, self.embedder.embed(crops_arr))
        if save:
            self.gallery.save(self.cfg.resolve(self.cfg.gallery_path))
        return len(crops_arr)


# =============================================================================
# Lazy singleton (FastAPI lifespan / Depends)
# =============================================================================
_pipeline: FacePipeline | None = None
_pipeline_lock = threading.Lock()


def get_face_pipeline(cfg: FacePipelineConfig | None = None) -> FacePipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = FacePipeline.from_config(cfg)
    return _pipeline


# =============================================================================
# CLI demo:  python -m collector.face_capture --source 0
#            python -m collector.face_capture --enroll "Hung" --images data/faces/raw/hung
# =============================================================================
def _draw(frame: np.ndarray, results: list[FaceResult]) -> None:
    colors = {FaceStatus.KNOWN: (0, 200, 0), FaceStatus.UNKNOWN: (0, 0, 255), FaceStatus.LOW_QUALITY: (0, 200, 255)}
    for r in results:
        x1, y1, x2, y2 = map(int, r.bbox)
        c = colors[r.status]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        for x, y in r.kps:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)
        cv2.putText(frame, f"{r.label} {r.similarity:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Face capture / recognition demo")
    ap.add_argument("--source", default="0", help="camera index, video file hoặc RTSP url")
    ap.add_argument("--config", default=None)
    ap.add_argument("--enroll", default=None, help="tên người cần đăng ký")
    ap.add_argument("--images", default=None, help="thư mục ảnh để đăng ký")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    pipe = FacePipeline.from_config(FacePipelineConfig.from_yaml(args.config))

    if args.enroll:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [p for p in sorted(Path(args.images).iterdir()) if p.suffix.lower() in exts]
        n = pipe.enroll(args.enroll, (cv2.imread(str(p)) for p in files))
        print(f"Enrolled {n}/{len(files)} images for '{args.enroll}'")
        return

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = pipe.process(frame)
        _draw(frame, results)
        now = time.perf_counter()
        cv2.putText(frame, f"FPS {1.0 / max(now - t_prev, 1e-6):.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        t_prev = now
        cv2.imshow("face_capture", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
