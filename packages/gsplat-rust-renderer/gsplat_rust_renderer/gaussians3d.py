"""Native archetype construction using Rerun 0.38.1's PLY conversion rules."""
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32
from plyfile import PlyData

SH_C0: float = float(np.float32(0.5) * np.sqrt(np.float32(1.0) / np.float32(np.pi)))
SPLATS_ENTITY: str = "/world/splats"
SPLATS_VISUALIZER: str = "ComputeGaussianSplats3D"


def compute_visualizer(render_mode: Literal["default", "mip"] = "default") -> rrb.Visualizer:
    """Explicit custom-viewer choice; stock viewers cannot execute this visualizer."""
    warnings.warn("ComputeGaussianSplats3D requires the custom viewer; stock Rerun will show no splats for this explicit override.", stacklevel=2)
    descriptor = rr.ComponentDescriptor("ComputeGaussianSplats3D:render_mode", component_type="rerun.components.Text")
    return rrb.Visualizer(SPLATS_VISUALIZER, overrides=[rr.components.TextBatch([render_mode]).described(descriptor)])


def splats_from_ply(path: Path, with_sh: bool = True) -> rr.GaussianSplats3D:
    """Match native Rust PLY rules: f32 activations, +0.5 u8 rounding, normalized xyzw,
    coefficient-major f16 SH, zero padding and truncation above degree three.
    """
    vertex = PlyData.read(path)["vertex"].data
    required = ["x", "y", "z", "opacity", *[f"{prefix}_{i}" for prefix, count in (("scale", 3), ("rot", 4), ("f_dc", 3)) for i in range(count)]]
    names = set(vertex.dtype.names or ())
    if not set(required) <= names:
        raise ValueError(f"{path}: missing required 3DGS properties: {sorted(set(required) - names)}")
    centers = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    scales = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(np.float32))
    quaternions = np.column_stack([vertex[f"rot_{i}"] for i in (1, 2, 3, 0)]).astype(np.float32)
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = np.where(norm > 0.0, quaternions / np.maximum(norm, np.finfo(np.float32).tiny), np.array([0, 0, 0, 1], dtype=np.float32))
    dc = np.column_stack([vertex[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)
    c0 = np.float32(SH_C0)
    rgb = np.float32(0.5) + c0 * dc
    alpha = np.float32(1.0) / (np.float32(1.0) + np.exp(-vertex["opacity"].astype(np.float32)))
    rgba = (np.clip(np.column_stack([rgb, alpha]), 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)
    count = sum(name.startswith("f_rest_") for name in names)
    stride = count // 3 if with_sh and count % 3 == 0 else 0
    sh: Float32[np.ndarray, "n 15 3"] | None = None
    if stride:
        sh = np.zeros((len(vertex), 15, 3), dtype=np.float32)
        for channel in range(3):
            for coefficient in range(min(stride, 15)):
                name = f"f_rest_{channel * stride + coefficient}"
                if name in names:
                    sh[:, coefficient, channel] = vertex[name]
    degree = min(3, int(np.ceil(np.sqrt(min(stride, 15) + 1))) - 1)
    return rr.GaussianSplats3D(centers=centers, scales=scales, quaternions=quaternions, colors=rgba,
                             sh_coefficients=sh, spherical_harmonics_degree=degree)
