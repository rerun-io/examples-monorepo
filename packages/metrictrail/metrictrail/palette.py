"""Legend colours of every MetricTrail figure (``/tmp/fleet-artifacts/metrictrail/figures/PALETTE.md``)."""

from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float32, UInt8
from matplotlib import colormaps
from numpy import ndarray

RunLabel: TypeAlias = Literal["ours", "upstream"]
"""Which tree produced a run: our port (``packages/metrictrail`` + ``recipes/vipe-src``) or unmodified upstream ViPE."""

PAPER_RGB: tuple[int, int, int] = (91, 155, 240)
"""Paper / published numbers: blue #5B9BF0."""
UPSTREAM_RGB: tuple[int, int, int] = (245, 166, 35)
"""Upstream ViPE (unmodified) run on this machine: amber #F5A623."""
OURS_RGB: tuple[int, int, int] = (224, 64, 251)
"""Ours: the shipped path (MoGe-2 ``moge-2-vitl-normal`` through TensorRT, with surface normals; catalog layer
``metrictrail``): magenta #E040FB."""
PAPER_PORT_RGB: tuple[int, int, int] = (167, 139, 250)
"""Our torch reproduction of the paper configuration (catalog layer ``metrictrail_paper``): violet #A78BFA."""
GT_RGB: tuple[int, int, int] = (61, 220, 151)
"""Ground truth: green #3DDC97."""
BA_WINDOW_RGB: tuple[int, int, int] = (255, 224, 102)
"""DPVO local bundle-adjustment window: yellow #FFE066 (not a legend entry)."""
KEYFRAME_RGB: tuple[int, int, int] = (138, 143, 152)
"""Keyframes outside the window: grey #8A8F98 (not a legend entry)."""
RUNNING_RMSE_RGB: tuple[int, int, int] = (235, 235, 235)
"""Running ATE RMSE line: #EBEBEB (not a legend entry)."""
BACKGROUND_RGB: tuple[int, int, int] = (15, 15, 17)
"""Every view's background: #0F0F11."""


def run_rgb(label: RunLabel) -> tuple[int, int, int]:
    """Legend colour of a run label."""
    return OURS_RGB if label == "ours" else UPSTREAM_RGB


def depth_colors(depth_m: Float32[ndarray, "n"], depth_range_m: tuple[float, float]) -> UInt8[ndarray, "n 3"]:
    """Turbo colours for metric depths (near = blue, far = red), the map Rerun's ``DepthImage`` uses."""
    low, high = depth_range_m
    unit: Float32[ndarray, "n"] = np.clip((depth_m - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return (colormaps["turbo"](unit)[:, :3] * 255.0).round().astype(np.uint8)


TURBO_LUT: UInt8[ndarray, "256 3"] = (colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).round().astype(np.uint8)
""":func:`depth_colors`'s turbo map in 256 steps, for colouring whole depth maps on the GPU (``capture.turbo_rgb``)."""
