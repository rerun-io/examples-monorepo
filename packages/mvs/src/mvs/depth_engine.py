"""Preprocessing and accelerated execution for the multiview depth model."""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as functional
from einops import rearrange
from jaxtyping import Float, UInt8
from torch import Tensor
from trtkit import OnnxOrTrtBackendConfig, TensorRtBackendConfig, TensorRuntime, create_runtime_from_onnx

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
TRT_EXTRA_OUTPUT_PATTERNS: tuple[str, ...] = ("cost_volume/GridSample",)
"""TensorRT miscompiles the plane-sweep GridSamples when fused with their consumers
(~0.3 m mean depth error vs ONNX Runtime); materializing them as engine outputs
restores parity."""


@dataclass(frozen=True, slots=True)
class DepthInputs:
    """Batched depth queries: each current view has seven posed source views.

    Field names are exactly the ONNX graph's input names.
    """

    cur_image_b3hw: Float[Tensor, "b c=3 h=384 w=512"]
    """Normalized current RGB image."""
    src_image_bm3hw: Float[Tensor, "b m=7 c=3 h=384 w=512"]
    """Normalized source RGB images."""
    src_K_bm44: Float[Tensor, "b m=7 4 4"]
    """Source-view s1 intrinsics."""
    cur_invK_b44: Float[Tensor, "b 4 4"]
    """Inverse current-view s1 intrinsics."""
    src_cam_T_world_bm44: Float[Tensor, "b m=7 4 4"]
    """Source world-to-camera transforms."""
    cur_world_T_cam_b44: Float[Tensor, "b 4 4"]
    """Current camera-to-world transform."""


class DepthOutputs(NamedTuple):
    """Depth-model predictions."""

    depth_b1hw: Float[Tensor, "b c=1 h=192 w=256"]
    """Metric depth."""
    lowest_cost_bhw: Float[Tensor, "b h=96 w=128"]
    """Lowest plane-sweep matching cost (confidence proxy)."""


class DepthEngine:
    """ONNX Runtime or TensorRT execution for the dynamic-batch depth graph."""

    def __init__(self, onnx_path: Path, backend: OnnxOrTrtBackendConfig) -> None:
        """Load the portable model artifact through the selected trtkit backend.

        Args:
            onnx_path: Dynamic-batch ONNX depth model.
            backend: CUDA ONNX Runtime or cached TensorRT backend configuration.
        """
        if isinstance(backend, TensorRtBackendConfig):
            backend = replace(backend, extra_output_patterns=TRT_EXTRA_OUTPUT_PATTERNS)
        self._runtime: TensorRuntime = create_runtime_from_onnx(onnx_path, backend)

    def __call__(self, inputs: DepthInputs) -> DepthOutputs:
        """Predict current-view metric depth from seven posed source views."""
        outputs: dict[str, Tensor] = self._runtime({
            field.name: getattr(inputs, field.name) for field in fields(DepthInputs)
        })
        return DepthOutputs(depth_b1hw=outputs["depth_pred_s0_b1hw"], lowest_cost_bhw=outputs["lowest_cost_bhw"])


def preprocess_image(image_chw: UInt8[Tensor, "c=3 h w"]) -> Float[Tensor, "c=3 h=384 w=512"]:
    """Resize and ImageNet-normalize one RGB frame on its current device.

    Args:
        image_chw: Uint8 RGB image with shape ``3 h w``.

    Returns:
        Float32 normalized RGB image with shape ``3 384 512``.
    """
    image_13hw: Float[Tensor, "b=1 c=3 h w"] = rearrange(image_chw, "c h w -> 1 c h w").to(dtype=torch.float32).div_(255.0)
    resized_13hw: Float[Tensor, "b=1 c=3 h=384 w=512"] = functional.interpolate(
        image_13hw, size=(384, 512), mode="bicubic", align_corners=False, antialias=True
    )
    resized_3hw: Float[Tensor, "c=3 h=384 w=512"] = rearrange(resized_13hw, "1 c h w -> c h w")
    mean_311: Float[Tensor, "c=3 1 1"] = rearrange(resized_3hw.new_tensor(IMAGENET_MEAN), "c -> c 1 1")
    std_311: Float[Tensor, "c=3 1 1"] = rearrange(resized_3hw.new_tensor(IMAGENET_STD), "c -> c 1 1")
    return (resized_3hw - mean_311) / std_311


def s1_intrinsics(
    K_n33: Float[Tensor, "n 3 3"], native_wh: tuple[int, int]
) -> tuple[Float[Tensor, "n 4 4"], Float[Tensor, "n 4 4"]]:
    """Scale native camera intrinsics to the model's s1 matching resolution.

    Args:
        K_n33: Native-resolution row-major intrinsics with shape ``n 3 3``.
        native_wh: Native image width and height.

    Returns:
        Float32 homogeneous s1 intrinsics and inverses, each with shape ``n 4 4``.
    """
    K_s1_n44: Float[Tensor, "n 4 4"] = torch.zeros((len(K_n33), 4, 4), dtype=torch.float32, device=K_n33.device)
    K_s1_n44[:, :3, :3] = K_n33
    K_s1_n44[:, 3, 3] = 1.0
    K_s1_n44[:, 0] *= 256.0 / native_wh[0] / 2.0
    K_s1_n44[:, 1] *= 192.0 / native_wh[1] / 2.0
    invK_s1_n44: Float[Tensor, "n 4 4"] = torch.linalg.inv(K_s1_n44)
    return K_s1_n44, invK_s1_n44
