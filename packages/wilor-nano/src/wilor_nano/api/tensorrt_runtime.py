"""TensorRT runners used by the optimized WiLor video path (trtkit-backed).

The engine mechanics — persistent buffers, static-batch zero-padding, stream
ordering — live in the shared :class:`trtkit.TensorRtRuntime`; these wrappers
keep WiLoR's typed call signatures and tensor-name constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import torch
from jaxtyping import Float
from torch import Tensor

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


class TensorRtFullWilorRunner:
    """Callable wrapper for the static-batch full WiLor TensorRT engine."""

    def __init__(self, engine_path: Path) -> None:
        """Create a full-WiLor TensorRT runner for the given static engine."""
        from trtkit import TensorRtRuntime

        self._runtime = TensorRtRuntime(engine_path)

    def __call__(self, inputs: Float[Tensor, "batch 256 256 3"]) -> WiLorOutput:
        """Run full-WiLor inference on NHWC crop tensors.

        Args:
            inputs: CUDA float16 crops as ``Float[Tensor, "batch 256 256 3"]``.

        Returns:
            A ``WiLorOutput`` dictionary sliced back to the original unpadded
            batch size (views into reused buffers — clone to keep past the next call).

        Raises:
            ValueError: If ``inputs`` is not a CUDA float16 tensor.
        """
        if inputs.device.type != "cuda" or inputs.dtype != torch.float16:
            raise ValueError("Full WiLor TensorRT expects CUDA float16 NHWC inputs.")
        outputs: dict[str, Tensor] = self._runtime({FULL_WILOR_INPUT_NAME: inputs})
        return cast(WiLorOutput, {name: outputs[name] for name in FULL_WILOR_OUTPUT_NAMES})


class TensorRtRawDetectorRunner:
    """Callable wrapper for the static-batch raw YOLO detector TensorRT engine."""

    def __init__(self, engine_path: Path) -> None:
        """Create a raw detector TensorRT runner for the given static engine."""
        from trtkit import TensorRtRuntime

        self._runtime = TensorRtRuntime(engine_path)

    def __call__(self, inputs: Float[Tensor, "batch 3 512 416"]) -> Tensor:
        """Run raw detector inference on NCHW detector tensors.

        Args:
            inputs: CUDA float32 detector input as
                ``Float[Tensor, "batch 3 512 416"]``.

        Returns:
            Raw detector output tensor sliced back to the original unpadded batch
            size (a view into a reused buffer — clone to keep past the next call).

        Raises:
            ValueError: If ``inputs`` is not a CUDA float32 tensor.
        """
        if inputs.device.type != "cuda" or inputs.dtype != torch.float32:
            raise ValueError("Raw detector TensorRT expects CUDA float32 NCHW inputs.")
        return self._runtime({DETECTOR_INPUT_NAME: inputs})[DETECTOR_OUTPUT_NAME]
