"""OpenCV image resampling, reproduced on the GPU for frames that are already there.

The catalog tools decode on NVDEC; copying each full-resolution frame to the host for one
OpenCV call and uploading the result again costs more than the model step it feeds. These
functions give the OpenCV result on the device instead.
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int64, UInt8
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


@dataclass(frozen=True, slots=True)
class LinearRemap:
    """``cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)`` (constant zero border) as four weighted gathers.

    OpenCV 5 interpolates float maps at their exact coordinates (the 1/32 px table is only for
    fixed-point maps), so the weights here are the float32 bilinear weights of each map entry.
    Build it once per map; frames reuse it.
    """

    index: Int64[Tensor, "4 pixels"]
    """Flat source index of each pixel's four neighbours (top-left, top-right, bottom-left, bottom-right)."""
    weight: Float32[Tensor, "4 pixels"]
    """Bilinear weight of each neighbour; zero where the neighbour lies outside the source."""
    output_hw: tuple[int, int]
    """Height and width of the remapped image."""

    @classmethod
    def from_maps(
        cls, map_x: Float32[ndarray, "out_h out_w"], map_y: Float32[ndarray, "out_h out_w"], source_hw: tuple[int, int], device: torch.device
    ) -> "LinearRemap":
        """Precompute the gathers for one ``cv2.remap`` map pair over a ``source_hw`` image."""
        source_h, source_w = source_hw
        x_float: Float32[ndarray, "pixels"] = map_x.astype(np.float32).ravel()
        y_float: Float32[ndarray, "pixels"] = map_y.astype(np.float32).ravel()
        x0: Int64[ndarray, "pixels"] = np.floor(x_float).astype(np.int64)
        y0: Int64[ndarray, "pixels"] = np.floor(y_float).astype(np.int64)
        fx: Float32[ndarray, "pixels"] = x_float - x0.astype(np.float32)
        fy: Float32[ndarray, "pixels"] = y_float - y0.astype(np.float32)
        indices: list[Int64[ndarray, "pixels"]] = []
        weights: list[Float32[ndarray, "pixels"]] = []
        for dy, dx, weight in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            x: Int64[ndarray, "pixels"] = x0 + dx
            y: Int64[ndarray, "pixels"] = y0 + dy
            inside: ndarray = (x >= 0) & (x < source_w) & (y >= 0) & (y < source_h)
            indices.append(np.where(inside, y * source_w + x, 0))
            weights.append(np.where(inside, weight, np.float32(0.0)).astype(np.float32))
        return cls(
            index=torch.from_numpy(np.stack(indices)).to(device),
            weight=torch.from_numpy(np.stack(weights)).to(device),
            output_hw=(map_x.shape[0], map_x.shape[1]),
        )

    def __call__(self, frames: UInt8[Tensor, "n c h w"]) -> UInt8[Tensor, "n c out_h out_w"]:
        """Remap a batch of frames on their device."""
        flat: Float32[Tensor, "n c hw"] = frames.flatten(2).float()
        remapped: Float32[Tensor, "n c pixels"] = flat[:, :, self.index[0]] * self.weight[0]
        for k in range(1, 4):
            remapped += flat[:, :, self.index[k]] * self.weight[k]
        return remapped.round_().clamp_(0, 255).to(torch.uint8).unflatten(2, self.output_hw)
