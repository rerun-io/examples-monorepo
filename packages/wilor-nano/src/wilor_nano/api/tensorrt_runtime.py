"""Tensor-name contract for the WiLoR TensorRT engines (trtkit-backed).

The engine mechanics — persistent buffers, static-batch zero-padding, stream
ordering — live in :class:`trtkit.TensorRtRuntime`; this module owns WiLoR's
binding names and its typed output contract.
"""

from __future__ import annotations

from typing import TypedDict, cast

from jaxtyping import Float
from torch import Tensor
from trtkit.base import TensorRuntime

FULL_WILOR_INPUT_NAME: str = "img_patches"
FULL_WILOR_OUTPUT_NAMES: tuple[str, ...] = ("global_orient", "hand_pose", "betas", "pred_cam", "pred_keypoints_3d", "pred_vertices")
FULL_WILOR_INPUT_SHAPE: tuple[int, int, int] = (256, 256, 3)
DETECTOR_INPUT_NAME: str = "images"
DETECTOR_OUTPUT_NAME: str = "output0"
DETECTOR_INPUT_SHAPE: tuple[int, int, int] = (3, 512, 416)


class WiLorOutput(TypedDict):
    """TensorRT full-WiLor output tensors returned before CPU/Rerun conversion."""

    global_orient: Float[Tensor, "batch 1 3"]
    hand_pose: Float[Tensor, "batch 15 3"]
    betas: Float[Tensor, "batch 10"]
    pred_cam: Float[Tensor, "batch 3"]
    pred_keypoints_3d: Float[Tensor, "batch 21 3"]
    pred_vertices: Float[Tensor, "batch 778 3"]


def run_full_wilor(runtime: TensorRuntime, crops: Float[Tensor, "batch 256 256 3"]) -> WiLorOutput:
    """Run one CUDA float16 NHWC crop batch through the full-WiLoR engine contract.

    Args:
        runtime: trtkit runtime loaded from a full-WiLoR engine.
        crops: CUDA float16 crops as ``Float[Tensor, "batch 256 256 3"]``.

    Returns:
        Typed outputs sliced to the submitted batch — views into runtime-owned
        buffers that the next call overwrites; clone anything that must survive.
    """
    return cast(WiLorOutput, runtime({FULL_WILOR_INPUT_NAME: crops}))
