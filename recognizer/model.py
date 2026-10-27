# recognizer/model.py
"""
PyTorch reference architectures for Face Detection (multi-scale 5-landmark detector)
and Face Embedding (MobileFaceNet / ArcFace 512-D), ready for ONNX export with
dynamic batch and dynamic spatial shapes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnPReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, stride: int = 1, pad: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 1, stride: int = 1, pad: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expansion: int = 2, residual: bool = False):
        super().__init__()
        mid_c = in_c * expansion
        self.residual = residual and (stride == 1 and in_c == out_c)
        self.pw_expand = ConvBnPReLU(in_c, mid_c, kernel=1, stride=1, pad=0)
        self.dw = ConvBnPReLU(mid_c, mid_c, kernel=3, stride=stride, pad=1, groups=mid_c)
        self.pw_linear = ConvBn(mid_c, out_c, kernel=1, stride=1, pad=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_linear(self.dw(self.pw_expand(x)))
        return x + out if self.residual else out


def _make_stage(in_c: int, out_c: int, num_blocks: int, stride: int, expansion: int) -> nn.Sequential:
    layers = [DepthWiseSeparable(in_c, out_c, stride=stride, expansion=expansion, residual=False)]
    for _ in range(1, num_blocks):
        layers.append(DepthWiseSeparable(out_c, out_c, stride=1, expansion=expansion, residual=True))
    return nn.Sequential(*layers)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet (ArcFace 512-D embedding extractor) cho ảnh khuôn mặt 112x112 đã align.
    Hỗ trợ Dynamic Batch Size (B, 3, 112, 112) -> (B, 512) L2-normalized.
    """

    def __init__(self, embedding_dim: int = 512, l2_norm: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm

        self.stem = nn.Sequential(
            ConvBnPReLU(3, 64, kernel=3, stride=2, pad=1),          # 112 -> 56
            ConvBnPReLU(64, 64, kernel=3, stride=1, pad=1, groups=64),
        )
        self.stage1 = _make_stage(64, 64, num_blocks=4, stride=2, expansion=2)    # 56 -> 28
        self.stage2 = _make_stage(64, 128, num_blocks=6, stride=2, expansion=4)   # 28 -> 14
        self.stage3 = _make_stage(128, 128, num_blocks=2, stride=2, expansion=2)  # 14 -> 7
        self.conv_sep = ConvBnPReLU(128, 512, kernel=1, stride=1, pad=0)
        # Global Depthwise Conv 7x7 cho input 112x112
        self.gdconv = ConvBn(512, 512, kernel=7, stride=1, pad=0, groups=512)
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.conv_sep(x)
        x = self.gdconv(x)
        x = torch.flatten(x, 1)
        x = self.bn_out(self.fc(x))
        if self.l2_norm:
            x = F.normalize(x, p=2.0, dim=1, eps=1e-10)
        return x


class LiteSCRFDDetector(nn.Module):
    """
    Lightweight Multi-Scale Face & 5-Landmark Detector (SCRFD-style FPN strides 8, 16, 32).
    Hỗ trợ cả Dynamic Batch lẫn Dynamic Spatial Shape: (B, 3, H, W).
    """

    def __init__(self, base_c: int = 32, num_anchors: int = 2):
        super().__init__()
        self.strides = (8, 16, 32)
        self.num_anchors = num_anchors

        # Backbone
        self.stem = nn.Sequential(
            ConvBnPReLU(3, base_c, 3, 2, 1),          # /2
            DepthWiseSeparable(base_c, base_c * 2, stride=2, expansion=2),  # /4
        )
        self.c3 = _make_stage(base_c * 2, base_c * 2, num_blocks=2, stride=2, expansion=2)  # /8
        self.c4 = _make_stage(base_c * 2, base_c * 4, num_blocks=3, stride=2, expansion=2)  # /16
        self.c5 = _make_stage(base_c * 4, base_c * 8, num_blocks=2, stride=2, expansion=2)  # /32

        # Neck (FPN lateral + top-down)
        fpn_c = base_c * 2
        self.lat3 = nn.Conv2d(base_c * 2, fpn_c, 1)
        self.lat4 = nn.Conv2d(base_c * 4, fpn_c, 1)
        self.lat5 = nn.Conv2d(base_c * 8, fpn_c, 1)

        # Shared / per-stride detection heads (score: 1, bbox: 4, kps: 10)
        self.tower = nn.ModuleList([ConvBnPReLU(fpn_c, fpn_c, 3, 1, 1) for _ in range(3)])
        self.score_heads = nn.ModuleList([nn.Conv2d(fpn_c, num_anchors * 1, 1) for _ in range(3)])
        self.bbox_heads = nn.ModuleList([nn.Conv2d(fpn_c, num_anchors * 4, 1) for _ in range(3)])
        self.kps_heads = nn.ModuleList([nn.Conv2d(fpn_c, num_anchors * 10, 1) for _ in range(3)])

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        feats = (p3, p4, p5)
        b = x.shape[0]
        scores, bboxes, kps = [], [], []
        for i, f in enumerate(feats):
            t = self.tower[i](f)
            s = torch.sigmoid(self.score_heads[i](t)).permute(0, 2, 3, 1).reshape(b, -1, 1)
            box = F.relu(self.bbox_heads[i](t)).permute(0, 2, 3, 1).reshape(b, -1, 4)
            kp = self.kps_heads[i](t).permute(0, 2, 3, 1).reshape(b, -1, 10)
            scores.append(s)
            bboxes.append(box)
            kps.append(kp)
        # Trả về 9 tensor giống chuẩn SCRFD: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        return (*scores, *bboxes, *kps)

