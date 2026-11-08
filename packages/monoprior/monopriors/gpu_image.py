"""OpenCV image resampling, reproduced on the GPU for frames that are already there.

The catalog tools decode on NVDEC; copying each full-resolution frame to the host for one
OpenCV call and uploading the result again costs more than the model step it feeds. These
functions give the OpenCV result on the device instead.
"""

import functools

import numpy as np
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor


def area_weights(source: int, target: int) -> Float32[ndarray, "target source"]:
    """``cv2.INTER_AREA`` downscale weights along one axis.

    Each output pixel averages the source span it covers; a source pixel cut by the span's
    edge counts by its covered fraction. ``torch``'s ``mode="area"`` (adaptive average
    pooling) instead averages whole pixels, which moves real frames by up to 71 levels at
    1920 -> 896.

    Raises:
        ValueError: If ``target`` is larger than ``source`` (OpenCV upscales bilinearly there).
    """
    if not 0 < target <= source:
        raise ValueError(f"area weights downscale only, got {source} -> {target}")
    scale: float = source / target
    edges: Float64[ndarray, "edges"] = np.arange(target + 1, dtype=np.float64) * scale
    pixels: Float64[ndarray, "1 source"] = np.arange(source, dtype=np.float64)[None, :]
    overlap: Float64[ndarray, "target source"] = np.minimum(edges[1:, None], pixels + 1.0) - np.maximum(edges[:-1, None], pixels)
    return (np.clip(overlap, 0.0, None) / scale).astype(np.float32)


@functools.cache
def _device_area_weights(source: int, target: int, device: torch.device) -> Float32[Tensor, "target source"]:
    """``area_weights`` uploaded once per size and device."""
    return torch.from_numpy(area_weights(source, target)).to(device)


def area_resize(frames: UInt8[Tensor, "n 3 h w"], height: int, width: int) -> UInt8[Tensor, "n 3 height width"]:
    """Downscale uint8 frames like ``cv2.resize(..., interpolation=cv2.INTER_AREA)``, on their device.

    Two separable float32 matmuls; on robocap 1920x1080 frames the result equals OpenCV's at
    896x504 and differs by at most one level on 0.3 % of pixels at 1120x630. Float32 matmuls
    must not run as TF32 (``torch.backends.cuda.matmul.allow_tf32``, off by default).
    """
    wy: Float32[Tensor, "height h"] = _device_area_weights(frames.shape[2], height, frames.device)
    wx: Float32[Tensor, "width w"] = _device_area_weights(frames.shape[3], width, frames.device)
    resized: Float32[Tensor, "n 3 height width"] = wy @ (frames.float() @ wx.T)
    # OpenCV's integer-ratio fast path rounds half up; its general path rounds half to even.
    integer_ratio: bool = frames.shape[2] % height == 0 and frames.shape[3] % width == 0
    rounded: Float32[Tensor, "n 3 height width"] = resized.add_(0.5).floor_() if integer_ratio else resized.round_()
    return rounded.clamp_(0, 255).to(torch.uint8)
