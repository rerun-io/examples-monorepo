"""Shared NumPy conversions for single-image MoGe v2 adapters."""

import numpy as np
import torch
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.metric_depth.base_metric_depth import MetricDepthPrediction
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2NormalOutput
from monopriors.models.surface_normal.base_normal_model import SurfaceNormalPrediction


def rgb_to_cuda_batch(rgb_hw3: UInt8[np.ndarray, "h w 3"]) -> UInt8[Tensor, "1 h w 3"]:
    """Move one uint8 NumPy RGB image into a CUDA batch.

    Args:
        rgb_hw3: uint8 NumPy RGB image shaped ``h w 3``.

    Returns:
        uint8 CUDA RGB tensor shaped ``1 h w 3``.
    """
    rgb_bhw3: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(rgb_hw3).to(device="cuda")[None]
    return rgb_bhw3


def geometry_to_metric_prediction(output: MoGeV2GeometryOutput) -> MetricDepthPrediction:
    """Convert one batched geometry result to the metric-depth contract.

    Args:
        output: Owning float32 geometry tensors with a batch size of one.

    Returns:
        Float32 caller-resolution depth, binary confidence, and pixel intrinsics.

    Raises:
        ValueError: If the output batch size is not one.
    """
    if output.depth_bhw.shape[0] != 1:
        raise ValueError(f"Single-image adapters require batch size 1, got {output.depth_bhw.shape[0]}.")

    height: int = output.depth_bhw.shape[1]
    width: int = output.depth_bhw.shape[2]
    valid_hw: Bool[np.ndarray, "h w"] = output.mask_bhw[0].numpy(force=True) > 0.5
    raw_depth_hw: Float32[np.ndarray, "h w"] = output.depth_bhw[0].numpy(force=True).astype(np.float32, copy=False)
    depth_meters_hw: Float32[np.ndarray, "h w"] = np.where(valid_hw, raw_depth_hw, np.inf).astype(np.float32, copy=False)
    confidence_hw: Float32[np.ndarray, "h w"] = valid_hw.astype(np.float32)
    normalized_K_33: Float32[np.ndarray, "3 3"] = output.intrinsics_b33[0].numpy(force=True).astype(np.float32, copy=False)
    K_33: Float32[np.ndarray, "3 3"] = normalized_K_33.copy()
    K_33[0, :] *= width
    K_33[1, :] *= height
    return MetricDepthPrediction(depth_meters=depth_meters_hw, confidence=confidence_hw, K_33=K_33)


def normal_to_surface_prediction(output: MoGeV2NormalOutput | MoGeV2GeometryOutput) -> SurfaceNormalPrediction:
    """Convert one batched normal result to the surface-normal contract.

    Args:
        output: Owning float32 normal or geometry tensors with batch size one.

    Returns:
        Float32 normals zeroed outside the binary-valid mask and binary confidence.

    Raises:
        ValueError: If the output batch size is not one.
    """
    if output.normals_bhw3.shape[0] != 1:
        raise ValueError(f"Single-image adapters require batch size 1, got {output.normals_bhw3.shape[0]}.")

    valid_hw: Bool[np.ndarray, "h w"] = output.mask_bhw[0].numpy(force=True) > 0.5
    raw_normals_hw3: Float32[np.ndarray, "h w 3"] = output.normals_bhw3[0].numpy(force=True).astype(np.float32, copy=False)
    normals_hw3: Float32[np.ndarray, "h w 3"] = np.where(valid_hw[..., None], raw_normals_hw3, 0.0).astype(np.float32, copy=False)
    confidence_hw1: Float32[np.ndarray, "h w 1"] = valid_hw[..., None].astype(np.float32)
    return SurfaceNormalPrediction(normal_hw3=normals_hw3, confidence_hw1=confidence_hw1)
