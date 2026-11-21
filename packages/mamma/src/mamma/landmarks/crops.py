"""Closed-form GPU crop geometry for MammaNet — port of the validated
``_box_geometry`` / ``_gpu_crop_batch`` fast path from ``landmarks/run_ma_2d.py``
(numerically matched there against the legacy cv2/ViTDetDataset path).
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

from mamma.landmarks.config import MammaNetConfig
from mamma.landmarks.mammanet import DEFAULT_MEAN, DEFAULT_STD


def to_device_f32(values: ndarray | torch.Tensor, device: torch.device | str) -> Float32[torch.Tensor, "..."]:
    """Float32 device tensor from host (numpy) or device values.

    Host arrays go through ONE pinned, non-blocking copy: a pageable
    ``torch.as_tensor(..., device=cuda)`` blocks the calling thread until every
    kernel already queued on the stream has finished.
    """
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    host: torch.Tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return host.pin_memory().to(device, non_blocking=True)


@functools.cache
def _normalization(device: torch.device) -> tuple[Float32[torch.Tensor, "1 3 1 1"], Float32[torch.Tensor, "1 3 1 1"]]:
    """ImageNet mean/std on ``device``, uploaded once."""
    return (
        torch.as_tensor(DEFAULT_MEAN, device=device).view(1, 3, 1, 1),
        torch.as_tensor(DEFAULT_STD, device=device).view(1, 3, 1, 1),
    )


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


def _crop_grid(
    centers: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    height: int,
    width: int,
    config: MammaNetConfig,
    device: torch.device,
) -> Float32[torch.Tensor, "n ph pw 2"]:
    """Analytic ``grid_sample`` grid (align_corners, normalized to the ``height x width`` image) of each crop."""
    c: Float32[torch.Tensor, "n 2"] = to_device_f32(centers, device)
    bs: Float32[torch.Tensor, "n 2"] = to_device_f32(bbox_sizes, device)
    n: int = c.shape[0]
    pw: int = config.crop_width
    ph: int = config.crop_height
    ys: Float32[torch.Tensor, "ph"] = torch.arange(ph, dtype=torch.float32, device=device)
    xs: Float32[torch.Tensor, "pw"] = torch.arange(pw, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    src_x: Float32[torch.Tensor, "n ph pw"] = c[:, 0].view(n, 1, 1) + (gx.unsqueeze(0) - pw / 2.0) * (bs[:, 0].view(n, 1, 1) / pw)
    src_y: Float32[torch.Tensor, "n ph pw"] = c[:, 1].view(n, 1, 1) + (gy.unsqueeze(0) - ph / 2.0) * (bs[:, 1].view(n, 1, 1) / ph)
    return torch.stack([src_x / (width - 1) * 2.0 - 1.0, src_y / (height - 1) * 2.0 - 1.0], dim=-1)


def gpu_crop_batch(
    frames: Float32[torch.Tensor, "n 3 h w"],
    centers: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
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
    grid: Float32[torch.Tensor, "n ph pw 2"] = _crop_grid(centers, bbox_sizes, frames.shape[2], frames.shape[3], config, frames.device)
    img_crops: Float32[torch.Tensor, "n 3 ph pw"] = F.grid_sample(
        frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    mean_t, std_t = _normalization(frames.device)
    img_crops = (img_crops - mean_t) / std_t

    mask_crops: Float32[torch.Tensor, "n 1 ph pw"] | None = None
    if masks is not None:
        mask_crops = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=True) / 255.0
    return img_crops, mask_crops


def gpu_crop_frame(
    frame: UInt8[torch.Tensor, "3 h w"] | Float32[torch.Tensor, "3 h w"],
    centers: Float64[ndarray, "k 2"],
    bbox_sizes: Float64[ndarray, "k 2"],
    config: MammaNetConfig,
) -> Float32[torch.Tensor, "k 3 ch cw"]:
    """:func:`gpu_crop_batch` of ``k`` boxes in one RGB frame (uint8 or float 0..255), converting only the pixels the
    crops read.

    A native-4K frame in float32 is ~100 MB; a person's crop window is a few percent of it. The window
    is the union of the crop rectangles (``center +- size/2``) plus the two pixels bilinear reads on each
    side, clamped to the frame; sampling it at shifted centres reads the same pixels as sampling the whole
    frame (align_corners). A window clamped to nothing (boxes outside the frame) uses the whole frame.
    (A gather-based uint8 sampler, posekit's, measured 0.9 s slower over crossing_arms on the 5090.)
    """
    height, width = int(frame.shape[1]), int(frame.shape[2])
    lo: Float64[ndarray, "2"] = np.floor((centers - bbox_sizes / 2.0).min(axis=0)) - 2.0
    hi: Float64[ndarray, "2"] = np.ceil((centers + bbox_sizes / 2.0).max(axis=0)) + 3.0
    x0, x1 = int(np.clip(lo[0], 0, width)), int(np.clip(hi[0], 0, width))
    y0, y1 = int(np.clip(lo[1], 0, height)), int(np.clip(hi[1], 0, height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        x0, y0, x1, y1 = 0, 0, width, height
    window: Float32[torch.Tensor, "k 3 hh ww"] = frame[:, y0:y1, x0:x1].float().unsqueeze(0).expand(centers.shape[0], -1, -1, -1)
    crops, _ = gpu_crop_batch(window, centers - np.array([x0, y0], dtype=np.float64), bbox_sizes, None, config)
    return crops


def unproject_joints2d(
    joints2d: Float32[torch.Tensor, "n j 3"],
    centers: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
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
    c: Float32[torch.Tensor, "n 2"] = to_device_f32(centers, out.device).to(out.dtype)
    bs: Float32[torch.Tensor, "n 2"] = to_device_f32(bbox_sizes, out.device).to(out.dtype)
    out[:, :, 0] = out[:, :, 0] * bs[:, 0:1] - bs[:, 0:1] / 2.0 + c[:, 0:1]
    out[:, :, 1] = out[:, :, 1] * bs[:, 1:2] - bs[:, 1:2] / 2.0 + c[:, 1:2]
    return out


def gpu_mask_crop_batch(
    masks: Float32[torch.Tensor, "n 1 h w"],
    centers: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    bbox_sizes: Float64[ndarray, "n 2"] | Float32[torch.Tensor, "n 2"],
    config: MammaNetConfig,
) -> Float32[torch.Tensor, "n 1 ch cw"]:
    """Mask-only variant of :func:`gpu_crop_batch` (same analytic grid).

    Used by the dual-resolution path where RGB crops sample a hi-res frame
    but masks come from the engine-resolution tracker output.
    """
    grid: Float32[torch.Tensor, "n ph pw 2"] = _crop_grid(centers, bbox_sizes, masks.shape[2], masks.shape[3], config, masks.device)
    return F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=True) / 255.0
