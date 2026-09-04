"""Shared test helpers for monoprior."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

if TYPE_CHECKING:
    from monopriors.third_party.xlens.models.net import XLensNet

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
"""Skip without a GPU; use for tests that load a checkpoint or build an engine."""

slow_cuda = pytest.mark.slow
"""Loads a checkpoint or builds a TensorRT engine; excluded from the default ``pytest`` run, select with ``-m slow``."""


def synthetic_rgb_batch(batch_size: int, image_hw: tuple[int, int]) -> UInt8[Tensor, "b h w 3"]:
    """Create deterministic RGB gradients directly on CUDA.

    Args:
        batch_size: Number of frames to create.
        image_hw: Frame height and width.

    Returns:
        uint8 CUDA RGB frames shaped ``b h w 3``.
    """
    height: int = image_hw[0]
    width: int = image_hw[1]
    x_w: Float32[Tensor, "w"] = torch.linspace(0, 255, width, device="cuda", dtype=torch.float32)
    y_h: Float32[Tensor, "h"] = torch.linspace(0, 255, height, device="cuda", dtype=torch.float32)
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    rgb_hw3: UInt8[Tensor, "h w 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)
    return rgb_hw3[None].expand(batch_size, -1, -1, -1).contiguous()


def build_random_xlens_model() -> "XLensNet":
    """The released X-Lens ViT-S architecture with seed-17 random weights (identical in every interpreter)."""
    import yaml

    from monopriors.third_party.xlens.models.net import XLensNet

    config_path: Path = Path(__file__).parents[1] / "monopriors/third_party/xlens/xlens_vits.yaml"
    config: dict[str, Any] = yaml.safe_load(config_path.read_text())
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return XLensNet(
            backbone_name=config["backbone"],
            checkpoint_dir="/nonexistent/xlens-test-checkpoints",
            head_features=config["head_features"],
            head_out_channels=tuple(config["head_out_channels"]),
            predict_mask=config["predict_mask"],
            scale_head_mode=config["scale_head_mode"],
            scale_head_num_queries=config["scale_head_num_queries"],
            scale_head_num_heads=config["scale_head_num_heads"],
            n_cam_types=config["n_cam_types"],
            use_calib_tokens=config["use_calib_tokens"],
            calib_tokens_per_type=config["calib_tokens_per_type"],
            calib_inject_types=tuple(config["calib_token_inject_types"]),
            use_distortion_bias=config["use_distortion_bias"],
            distortion_bias_layers=config["distortion_bias_layers"],
            distortion_bias_chunk_size=config["distortion_bias_chunk_size"],
            distortion_bias_hidden_dim=config["distortion_bias_hidden_dim"],
        ).eval()


def random_rig(
    views: int, image_hw: tuple[int, int], seed: int, *, with_poses: bool = True
) -> tuple[UInt8[ndarray, "s h w 3"], Float32[ndarray, "s h w 3"], Float64[ndarray, "s 4 4"] | None]:
    """Noise images, random unit rays, and optional poses spaced 0.1 m apart along x."""
    generator: np.random.Generator = np.random.default_rng(seed)
    images: UInt8[ndarray, "s h w 3"] = generator.integers(0, 256, size=(views, image_hw[0], image_hw[1], 3), dtype=np.uint8)
    rays: Float32[ndarray, "s h w 3"] = generator.normal(size=(views, image_hw[0], image_hw[1], 3)).astype(np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    cam_T_ref: Float64[ndarray, "s 4 4"] | None = None
    if with_poses:
        cam_T_ref = np.tile(np.eye(4, dtype=np.float64), (views, 1, 1))
        cam_T_ref[:, 0, 3] = np.arange(views, dtype=np.float64) * 0.1
    return images, rays, cam_T_ref
