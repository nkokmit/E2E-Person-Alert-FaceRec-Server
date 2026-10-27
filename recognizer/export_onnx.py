# recognizer/export_onnx.py
"""
Pipeline chuyển đổi mô hình PyTorch (Face Detection & Face Embedding) sang ONNX
với Dynamic Batch / Dynamic Input Shape, triển khai bằng onnxruntime
(CPUExecutionProvider với intra_op_num_threads / CUDAExecutionProvider) và
Benchmark Latency (ms) & FPS trước/sau tối ưu hóa.

Chạy trực tiếp:
    python -m recognizer.export_onnx --export-dir models/faces --runs 100
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from recognizer.model import LiteSCRFDDetector, MobileFaceNet

log = logging.getLogger("export_onnx")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# =============================================================================
# 1. Export PyTorch -> ONNX với Dynamic Batch & Dynamic Spatial Shape
# =============================================================================
def export_face_embedding_onnx(
    model: nn.Module,
    onnx_path: Union[str, Path],
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """
    Export mô hình trích xuất đặc trưng khuôn mặt (MobileFaceNet / ArcFace) sang ONNX
    với Dynamic Batch axis: input (B, 3, 112, 112) -> embedding (B, 512).
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "embedding": {0: "batch_size"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    log.info("Exported & verified ONNX embedding model: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)

    if verify:
        wrapper = ORTInferenceWrapper(onnx_path, device="cpu", intra_op_num_threads=4)
        for b in (1, 4, 8):
            x = np.random.randn(b, 3, 112, 112).astype(np.float32)
            with torch.inference_mode():
                pt_out = model(torch.from_numpy(x)).numpy()
            ort_out = wrapper.run({"input": x})[0]
            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            cos_sim = float(np.sum(pt_out * ort_out, axis=1).mean())
            assert max_diff < 1e-4, f"Mismatch at batch={b}: max_diff={max_diff}"
            log.info("Parity check [Embedding B=%d]: max_abs_diff=%.2e, cosine_sim=%.7f", b, max_diff, cos_sim)

    return onnx_path


def export_face_detector_onnx(
    model: nn.Module,
    onnx_path: Union[str, Path],
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """
    Export mô hình phát hiện khuôn mặt & 5 điểm mốc (SCRFD-style) sang ONNX
    với Dynamic Batch và Dynamic Height/Width: images (B, 3, H, W).
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    output_names = [
        "score_8", "score_16", "score_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    ]
    dynamic_axes: Dict[str, Dict[int, str]] = {
        "images": {0: "batch_size", 2: "height", 3: "width"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size", 1: "num_anchors"}

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    log.info("Exported & verified ONNX detector model: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)

    if verify:
        wrapper = ORTInferenceWrapper(onnx_path, device="cpu", intra_op_num_threads=4)
        for b, h, w in ((1, 640, 640), (1, 320, 320), (2, 480, 640)):
            x = np.random.randn(b, 3, h, w).astype(np.float32)
            with torch.inference_mode():
                pt_outs = [t.numpy() for t in model(torch.from_numpy(x))]
            ort_outs = wrapper.run({"images": x})
            max_diff = max(float(np.max(np.abs(p - o))) for p, o in zip(pt_outs, ort_outs))
            assert max_diff < 1e-4, f"Detector mismatch at ({b},{h},{w}): {max_diff}"
            log.info("Parity check [Detector (%d,3,%d,%d)]: max_abs_diff=%.2e", b, h, w, max_diff)

    return onnx_path


# =============================================================================
# 2. ONNX Runtime InferenceSession Wrapper (CPU threads & CUDA optimized)
# =============================================================================
class ORTInferenceWrapper:
    """
    Production wrapper cho onnxruntime.InferenceSession:
    - Tối ưu hóa đồ thị mức cao nhất (ORT_ENABLE_ALL: Conv+BN+Activation fusion, constant folding).
    - Cấu hình CPUExecutionProvider với intra_op_num_threads / inter_op_num_threads.
    - Cấu hình CUDAExecutionProvider với cuDNN exhaustive search & default stream copy.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 1,
        cudnn_conv_algo_search: str = "EXHAUSTIVE",
        save_optimized_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        if intra_op_num_threads > 0:
            sess_options.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads > 0:
            sess_options.inter_op_num_threads = int(inter_op_num_threads)

        if save_optimized_path:
            sess_options.optimized_model_filepath = str(save_optimized_path)

        providers = self._build_providers(device, cudnn_conv_algo_search)
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.active_providers = self.session.get_providers()
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    @staticmethod
    def _build_providers(device: str, cudnn_conv_algo_search: str) -> List[Any]:
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
        available = set(ort.get_available_providers())
        dev = (device or "auto").lower()
        if dev == "cpu":
            return ["CPUExecutionProvider"]

        if ("cuda" in dev or dev == "auto") and "CUDAExecutionProvider" in available:
            cuda_opts = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": cudnn_conv_algo_search,
                "do_copy_in_default_stream": 1,
            }
            return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    def run(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(self.output_names, feed_dict)

    def create_io_binding(self, input_shape: Tuple[int, ...], device_id: int = 0):
        """Khởi tạo IOBinding với bộ nhớ đã cấp phát sẵn trên GPU (Zero-Copy)."""
        io_binding = self.session.io_binding()
        ort_input = ort.OrtValue.ortvalue_from_shape_and_type(list(input_shape), np.float32, "cuda", device_id)
        io_binding.bind_ortvalue_input(self.input_names[0], ort_input)
        for out_name in self.output_names:
            io_binding.bind_output(out_name, "cuda", device_id=device_id)
        return io_binding, ort_input

    def __call__(self, x: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        outs = self.session.run(self.output_names, {self.input_names[0]: arr})
        return outs[0] if len(outs) == 1 else outs


# =============================================================================
# 3. Benchmark Suite (Đo lường Latency ms: Mean, P50, P95, P99 & FPS)
# =============================================================================
@dataclass
class BenchmarkMetrics:
    name: str
    backend: str
    device: str
    input_shape: Tuple[int, ...]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    speedup: float = 1.0


def _measure_latency(
    fn: Callable[[], Any],
    sync_cuda: bool = False,
    warmup: int = 20,
    runs: int = 100,
) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()

    latencies_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def benchmark_model_pair(
    task_name: str,
    pt_model: nn.Module,
    onnx_path: Path,
    input_shape: Tuple[int, int, int, int],
    cpu_threads_list: Sequence[int] = (1, 4),
    warmup: int = 20,
    runs: int = 100,
) -> List[BenchmarkMetrics]:
    batch_size = input_shape[0]
    np_input = np.random.randn(*input_shape).astype(np.float32)
    results: List[BenchmarkMetrics] = []

    # 1. PyTorch CPU Baseline
    pt_cpu = pt_model.cpu().eval()
    pt_tensor_cpu = torch.from_numpy(np_input)
    with torch.inference_mode():
        mean_ms, p50, p95, p99 = _measure_latency(
            lambda: pt_cpu(pt_tensor_cpu), sync_cuda=False, warmup=warmup, runs=runs
        )
    cpu_base_ms = mean_ms
    results.append(
        BenchmarkMetrics(
            name=task_name,
            backend="PyTorch Eager (Baseline)",
            device="CPU",
            input_shape=input_shape,
            mean_ms=mean_ms,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            fps=(batch_size * 1000.0) / mean_ms,
            speedup=1.0,
        )
    )

    # 2. ONNX Runtime CPU với các cấu hình intra_op_num_threads
    for n_threads in cpu_threads_list:
        ort_cpu = ORTInferenceWrapper(onnx_path, device="cpu", intra_op_num_threads=n_threads)
        mean_ms, p50, p95, p99 = _measure_latency(
            lambda: ort_cpu(np_input), sync_cuda=False, warmup=warmup, runs=runs
        )
        results.append(
            BenchmarkMetrics(
                name=task_name,
                backend=f"ONNX Runtime (threads={n_threads})",
                device="CPU",
                input_shape=input_shape,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                fps=(batch_size * 1000.0) / mean_ms,
                speedup=cpu_base_ms / mean_ms,
            )
        )

    # 3. GPU Benchmark (nếu có CUDA) - so sánh công bằng End-to-End (Host -> GPU -> Host) & IOBinding
    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        gpu_name = torch.cuda.get_device_name(0)
        pt_gpu = pt_model.cuda().eval()

        def _pt_e2e():
            t_in = torch.from_numpy(np_input).cuda()
            out = pt_gpu(t_in)
            if isinstance(out, (tuple, list)):
                return [o.cpu().numpy() for o in out]
            return out.cpu().numpy()

        with torch.inference_mode():
            mean_ms, p50, p95, p99 = _measure_latency(
                _pt_e2e, sync_cuda=True, warmup=warmup, runs=runs
            )
        gpu_base_ms = mean_ms
        results.append(
            BenchmarkMetrics(
                name=task_name,
                backend="PyTorch Eager CUDA",
                device=gpu_name,
                input_shape=input_shape,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                fps=(batch_size * 1000.0) / mean_ms,
                speedup=1.0,
            )
        )

        ort_gpu = ORTInferenceWrapper(onnx_path, device="cuda")
        mean_ms, p50, p95, p99 = _measure_latency(
            lambda: ort_gpu(np_input), sync_cuda=False, warmup=warmup, runs=runs
        )
        results.append(
            BenchmarkMetrics(
                name=task_name,
                backend="ONNX Runtime CUDA",
                device=gpu_name,
                input_shape=input_shape,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                fps=(batch_size * 1000.0) / mean_ms,
                speedup=gpu_base_ms / mean_ms,
            )
        )

        io_binding, _ = ort_gpu.create_io_binding(input_shape)

        def _ort_iobind():
            ort_gpu.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

        mean_ms, p50, p95, p99 = _measure_latency(
            _ort_iobind, sync_cuda=False, warmup=warmup, runs=runs
        )
        results.append(
            BenchmarkMetrics(
                name=task_name,
                backend="ONNX Runtime CUDA (IOBinding)",
                device=gpu_name,
                input_shape=input_shape,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                fps=(batch_size * 1000.0) / mean_ms,
                speedup=gpu_base_ms / mean_ms,
            )
        )

    return results


def format_markdown_table(metrics: Sequence[BenchmarkMetrics]) -> str:
    lines = [
        "| Model / Task | Backend | Device | Input Shape | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (FPS) | Speedup |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m.name} | {m.backend} | {m.device} | `{list(m.input_shape)}` | "
            f"{m.mean_ms:.2f} | {m.p50_ms:.2f} | {m.p95_ms:.2f} | {m.p99_ms:.2f} | "
            f"**{m.fps:.1f}** | **{m.speedup:.2f}x** |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch Face models to ONNX & run benchmark")
    parser.add_argument("--export-dir", default="models/faces", help="Directory to save ONNX files")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Face Detector (SCRFD-style) & Face Embedder (MobileFaceNet)
    det_pt = LiteSCRFDDetector(base_c=32, num_anchors=2)
    emb_pt = MobileFaceNet(embedding_dim=512, l2_norm=True)

    det_onnx_path = export_face_detector_onnx(det_pt, export_dir / "det_500m.onnx", verify=True)
    emb_onnx_path = export_face_embedding_onnx(emb_pt, export_dir / "w600k_mbf.onnx", verify=True)

    # 2. Run Benchmarks
    all_metrics: List[BenchmarkMetrics] = []
    all_metrics.extend(
        benchmark_model_pair(
            "Face Detector (SCRFD-Lite)",
            det_pt,
            det_onnx_path,
            input_shape=(1, 3, 640, 640),
            cpu_threads_list=(1, 4, 6),
            warmup=args.warmup,
            runs=args.runs,
        )
    )
    all_metrics.extend(
        benchmark_model_pair(
            "Face Embedding (MobileFaceNet B=1)",
            emb_pt,
            emb_onnx_path,
            input_shape=(1, 3, 112, 112),
            cpu_threads_list=(1, 4, 6),
            warmup=args.warmup,
            runs=args.runs,
        )
    )
    all_metrics.extend(
        benchmark_model_pair(
            "Face Embedding (MobileFaceNet B=8)",
            emb_pt,
            emb_onnx_path,
            input_shape=(8, 3, 112, 112),
            cpu_threads_list=(1, 4, 6),
            warmup=args.warmup,
            runs=args.runs,
        )
    )

    print("\n================ BENCHMARK RESULTS ================\n")
    print(format_markdown_table(all_metrics))


if __name__ == "__main__":
    main()

