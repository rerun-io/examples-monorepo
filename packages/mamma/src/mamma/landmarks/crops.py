"""Closed-form GPU crop geometry for MammaNet — port of the validated
``_box_geometry`` / ``_gpu_crop_batch`` fast path from ``landmarks/run_ma_2d.py``
(numerically matched there against the legacy cv2/ViTDetDataset path).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Float64
from numpy import ndarray

from mamma.landmarks.config import MammaNetConfig
from mamma.landmarks.mammanet import DEFAULT_MEAN, DEFAULT_STD


def expand_to_aspect_ratio(input_wh: Float64[ndarray, "2"], target_wh: tuple[int, int]) -> Float64[ndarray, "2"]:
    """Grow ``(w, h)`` to match the target aspect ratio (original ``utils_eval``)."""
    w: float = float(input_wh[0])
    h: float = float(input_wh[1])
    w_t: float = float(target_wh[0])
    h_t: float = float(target_wh[1])
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def box_geometry(box_xyxy: Float32[ndarray, "4"], config: MammaNetConfig) -> tuple[Float64[ndarray, "2"], Float64[ndarray, "2"]]:
    """ViTDetDataset crop geometry: ``(center, bbox_size)`` for one box.

    ``bbox_size`` is the 1.2x-expanded box fitted to the 384x512 crop aspect.
    These two values also define the un-projection back to image pixels.
    """
    box: Float64[ndarray, "4"] = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
    center: Float64[ndarray, "2"] = (box[2:4] + box[0:2]) / 2.0
    wh: Float64[ndarray, "2"] = box[2:4] - box[0:2]
    bbox_size: Float64[ndarray, "2"] = expand_to_aspect_ratio(
        wh * config.bedlam_scale, target_wh=(config.crop_width, config.crop_height)
    )
    return center, bbox_size


def gpu_crop_batch(
    frames: Float32[torch.Tensor, "n 3 h w"],
    centers: Float64[ndarray, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"],
    masks: Float32[torch.Tensor, "n 1 h w"] | None,
    config: MammaNetConfig,
) -> tuple[Float32[torch.Tensor, "n 3 ch cw"], Float32[torch.Tensor, "n 1 ch cw"] | None]:
    """Affine crop + normalize on GPU via an analytic ``grid_sample`` grid.

    Args:
        frames: RGB frames, float32 in 0..255, NCHW, on the compute device.
        centers: Per-crop box centers from :func:`box_geometry`.
        bbox_sizes: Per-crop ``(w_box, h_box)`` from :func:`box_geometry`.
        masks: Optional 0..255 float masks (person masks), NCHW.
        config: Crop size constants.

    Returns:
        Normalized RGB crops ``(n, 3, 512, 384)`` and 0..1 mask crops (or None).
    """
    n: int = frames.shape[0]
    h_img: int = frames.shape[2]
    w_img: int = frames.shape[3]
    pw: int = config.crop_width
    ph: int = config.crop_height
    dev: torch.device = frames.device

    c: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=dev)
    bs: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(bbox_sizes), dtype=torch.float32, device=dev)

    ys: Float32[torch.Tensor, "ph"] = torch.arange(ph, dtype=torch.float32, device=dev)
    xs: Float32[torch.Tensor, "pw"] = torch.arange(pw, dtype=torch.float32, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    cx = c[:, 0].view(n, 1, 1)
    cy = c[:, 1].view(n, 1, 1)
    wbox = bs[:, 0].view(n, 1, 1)
    hbox = bs[:, 1].view(n, 1, 1)
    src_x: Float32[torch.Tensor, "n ph pw"] = cx + (gx.unsqueeze(0) - pw / 2.0) * (wbox / pw)
    src_y: Float32[torch.Tensor, "n ph pw"] = cy + (gy.unsqueeze(0) - ph / 2.0) * (hbox / ph)

    nx = src_x / (w_img - 1) * 2.0 - 1.0
    ny = src_y / (h_img - 1) * 2.0 - 1.0
    grid: Float32[torch.Tensor, "n ph pw 2"] = torch.stack([nx, ny], dim=-1)

    img_crops: Float32[torch.Tensor, "n 3 ph pw"] = F.grid_sample(
        frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    mean_t: Float32[torch.Tensor, "3"] = torch.as_tensor(DEFAULT_MEAN, device=dev)
    std_t: Float32[torch.Tensor, "3"] = torch.as_tensor(DEFAULT_STD, device=dev)
    img_crops = (img_crops - mean_t.view(1, 3, 1, 1)) / std_t.view(1, 3, 1, 1)

    mask_crops: Float32[torch.Tensor, "n 1 ph pw"] | None = None
    if masks is not None:
        mask_crops = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=True) / 255.0
    return img_crops, mask_crops


def unproject_joints2d(
    joints2d: Float32[torch.Tensor, "n j 3"],
    centers: Float64[ndarray, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"],
    config: MammaNetConfig,
) -> Float32[torch.Tensor, "n j 3"]:
    """Map model-output landmarks from crop coords to image pixel coords.

    The xy channels arrive in [-1, 1] (``normalize_plus_min_one``); the third
    channel (log-variance) passes through untouched. Mirrors the Pass-3 math
    in ``run_ma_2d.py``: ``v01 = (v+1)/2; px = v01*box - box/2 + center``.
    """
    out: Float32[torch.Tensor, "n j 3"] = joints2d.clone()
    if config.normalize_plus_min_one:
        out[:, :, :2] = (out[:, :, :2] + 1.0) / 2.0
    c: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(centers), dtype=out.dtype, device=out.device)
    bs: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(bbox_sizes), dtype=out.dtype, device=out.device)
    out[:, :, 0] = out[:, :, 0] * bs[:, 0:1] - bs[:, 0:1] / 2.0 + c[:, 0:1]
    out[:, :, 1] = out[:, :, 1] * bs[:, 1:2] - bs[:, 1:2] / 2.0 + c[:, 1:2]
    return out


def gpu_mask_crop_batch(
    masks: Float32[torch.Tensor, "n 1 h w"],
    centers: Float64[ndarray, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"],
    config: MammaNetConfig,
) -> Float32[torch.Tensor, "n 1 ch cw"]:
    """Mask-only variant of :func:`gpu_crop_batch` (same analytic grid).

    Used by the dual-resolution path where RGB crops sample a hi-res frame
    but masks come from the engine-resolution tracker output.
    """
    n: int = masks.shape[0]
    h_img: int = masks.shape[2]
    w_img: int = masks.shape[3]
    pw: int = config.crop_width
    ph: int = config.crop_height
    dev: torch.device = masks.device

    c: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=dev)
    bs: Float32[torch.Tensor, "n 2"] = torch.as_tensor(np.asarray(bbox_sizes), dtype=torch.float32, device=dev)
    ys: Float32[torch.Tensor, "ph"] = torch.arange(ph, dtype=torch.float32, device=dev)
    xs: Float32[torch.Tensor, "pw"] = torch.arange(pw, dtype=torch.float32, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    src_x: Float32[torch.Tensor, "n ph pw"] = c[:, 0].view(n, 1, 1) + (gx.unsqueeze(0) - pw / 2.0) * (bs[:, 0].view(n, 1, 1) / pw)
    src_y: Float32[torch.Tensor, "n ph pw"] = c[:, 1].view(n, 1, 1) + (gy.unsqueeze(0) - ph / 2.0) * (bs[:, 1].view(n, 1, 1) / ph)
    grid: Float32[torch.Tensor, "n ph pw 2"] = torch.stack([src_x / (w_img - 1) * 2.0 - 1.0, src_y / (h_img - 1) * 2.0 - 1.0], dim=-1)
    return F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=True) / 255.0
