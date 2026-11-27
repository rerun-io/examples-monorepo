"""People-mask post-processing: instance union, tile layout and placement, dilation, overlays.

Pure numpy/OpenCV/torch functions with no model or GPU, so the unit lane tests them on synthetic arrays.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TileWindow:
    """One crop of an image, in pixel rows ``y0:y1`` and columns ``x0:x1``."""

    y0: int
    """First row."""
    y1: int
    """Row after the last."""
    x0: int
    """First column."""
    x1: int
    """Column after the last."""


def union_instances(masks: Bool[ndarray, "n h w"], frame_indices: Int64[ndarray, "n"], *, n_frames: int, min_area_px: int) -> Bool[ndarray, "f h w"]:
    """OR the instance masks of each frame, dropping instances smaller than ``min_area_px``.

    Args:
        masks: Instance masks of a batch, flattened over frames.
        frame_indices: Frame of each instance in ``0..n_frames-1``.
        n_frames: Frames in the batch (frames without instances come back empty).
        min_area_px: Instances with fewer pixels are ignored.

    Returns:
        One person mask per frame.
    """
    union: Bool[ndarray, "f h w"] = np.zeros((n_frames, *masks.shape[1:]), dtype=bool)
    areas: Int64[ndarray, "n"] = masks.sum(axis=(1, 2)).astype(np.int64)
    for instance in np.flatnonzero(areas >= min_area_px):
        union[frame_indices[instance]] |= masks[instance]
    return union


def tile_windows(height: int, width: int, *, grid: int, overlap: float) -> list[TileWindow]:
    """The 2x2 overlapping tiles of an image (empty for ``grid == 1``).

    Each tile spans ``(1 + overlap) / 2`` of each side, so neighbours share ``overlap`` of the image; all tiles have
    one size, so the tiles of same-size images batch together.

    Raises:
        ValueError: If ``grid`` is not 1 or 2.
    """
    if grid == 1:
        return []
    if grid != 2:
        raise ValueError(f"tiles must be 1 or 2, got {grid}")
    tile_h: int = min(height, max(1, round(height * (1.0 + overlap) / 2.0)))
    tile_w: int = min(width, max(1, round(width * (1.0 + overlap) / 2.0)))
    return [TileWindow(y0, y0 + tile_h, x0, x0 + tile_w) for y0 in (0, height - tile_h) for x0 in (0, width - tile_w)]


def tile_to_image(
    masks: Bool[Tensor, "n th tw"], xyxy: Float32[Tensor, "n 4"], window: TileWindow, *, height: int, width: int
) -> tuple[Bool[Tensor, "n h w"], Float32[Tensor, "n 4"]]:
    """Place instance masks and boxes found in a tile into the coordinates of the whole image."""
    full: Bool[Tensor, "n h w"] = torch.zeros((masks.shape[0], height, width), dtype=torch.bool, device=masks.device)
    full[:, window.y0 : window.y1, window.x0 : window.x1] = masks
    offset: Float32[Tensor, "4"] = torch.tensor([window.x0, window.y0, window.x0, window.y0], dtype=torch.float32, device=xyxy.device)
    return full, xyxy + offset


def dilate_mask(mask: Bool[ndarray, "h w"], radius_px: int) -> Bool[ndarray, "h w"]:
    """Grow the mask by a disc of ``radius_px`` (0 returns the mask unchanged)."""
    if radius_px <= 0:
        return mask
    kernel: UInt8[ndarray, "k k"] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    grown: UInt8[ndarray, "h w"] = cv2.dilate(mask.astype(np.uint8), kernel)
    return grown.astype(bool)


def dilate_by_footprint(masks: Bool[Tensor, "n h w"], footprint: Bool[Tensor, "k k"]) -> Bool[Tensor, "n h w"]:
    """Binary dilation of a mask batch by an odd-sized ``footprint``, as one zero-padded convolution on the masks' device.

    On CUDA the conv runs in float16: a single-channel float32 conv takes a slow cuDNN path (80 ms for two 1920 px masks
    on the 5090, 3.6 ms in float16). The sums are integers <= k*k, exact in float16 while k*k <= 2048 and in float32
    always, so > 0.5 is the binary dilation either way.
    """
    dtype: torch.dtype = torch.float16 if masks.is_cuda and footprint.numel() <= 2048 else torch.float32
    weights: Tensor = footprint.to(masks.device, dtype)[None, None]
    hits: Tensor = F.conv2d(masks[:, None].to(dtype), weights, padding=footprint.shape[0] // 2)
    return hits[:, 0] > 0.5


def dilate_masks(masks: Bool[Tensor, "n h w"], radius_px: int) -> Bool[Tensor, "n h w"]:
    """``dilate_mask`` on the masks' device: the same OpenCV disc (0 returns the masks unchanged)."""
    if radius_px <= 0:
        return masks
    kernel: UInt8[ndarray, "k k"] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return dilate_by_footprint(masks, torch.from_numpy(kernel).bool())


def overlay_mask(rgb: UInt8[ndarray, "h w 3"], mask: Bool[ndarray, "h w"], *, alpha: float = 0.5) -> UInt8[ndarray, "h w 3"]:
    """Tint the masked pixels red (evidence images)."""
    tinted: UInt8[ndarray, "h w 3"] = rgb.copy()
    red: UInt8[ndarray, "3"] = np.array([255, 0, 0], dtype=np.uint8)
    tinted[mask] = np.round((1.0 - alpha) * rgb[mask] + alpha * red).astype(np.uint8)
    return tinted
