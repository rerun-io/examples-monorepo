"""GPU-resident top-down crop generation and crop->image coordinate mapping.

Crops are produced with an analytic sampling grid plus ``grid_sample`` so CUDA
video frames flow straight into any backend without cv2/host round-trips. Two
affine conventions are supported, matching the source networks exactly:

- ``"udp"``: Unbiased Data Processing alignment (Sapiens, mmpose UDP codecs).
- ``"cv2"``: classic ``cv2.warpAffine`` top-down alignment (RTMPose/RTMW,
  mmpose SimCC pipelines).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, UInt8
from torch import Tensor

from posekit.predictions import validate_frames_rgb

CropAlign = Literal["udp", "cv2"]

# ImageNet normalization constants on the 0-255 scale (the mmpose/rtmlib/Sapiens
# family all share them; channel-order semantics are documented per model).
IMAGENET_MEAN_255: tuple[float, float, float] = (123.675, 116.28, 103.53)
IMAGENET_STD_255: tuple[float, float, float] = (58.395, 57.12, 57.375)


@dataclass(frozen=True, slots=True)
class CropSpec:
    """How a top-down model wants its person/hand crops prepared."""

    input_size: tuple[int, int]
    """Crop size as ``(width, height)``."""
    padding: float = 1.25
    """Bbox padding multiplier applied before aspect-ratio expansion."""
    align: CropAlign = "udp"
    """Affine convention used to place the crop grid."""
    bgr: bool = False
    """Whether the model consumes BGR channel order."""
    mean_rgb: tuple[float, float, float] | None = None
    """Per-channel mean (0-255 scale, in the model's channel order) or ``None`` to skip."""
    std_rgb: tuple[float, float, float] | None = None
    """Per-channel std (0-255 scale, in the model's channel order) or ``None`` to skip."""


@dataclass(frozen=True, slots=True)
class CropBatch:
    """Batched model-ready crops plus the metadata needed to map outputs back."""

    inputs: Float[Tensor, "n 3 crop_h crop_w"]
    """Normalized NCHW crops ready for a runtime."""
    frame_indices: Int[Tensor, "n"]
    """Source frame index for each crop."""
    bboxes_xyxy: Float[Tensor, "n 4"]
    """Source boxes in image-space ``xyxy`` order."""
    centers: Float[Tensor, "n 2"]
    """Image-space crop centers."""
    scales: Float[Tensor, "n 2"]
    """Padded crop extent in source-image pixels as ``(width, height)``."""
    spec: CropSpec
    """Crop convention the batch was built with."""


def bbox_xyxy_to_center_scale(
    bboxes_xyxy: Float[Tensor, "n 4"], *, aspect_wh: float, padding: float
) -> tuple[Float[Tensor, "n 2"], Float[Tensor, "n 2"]]:
    """Convert boxes to padded, aspect-expanded center/scale pairs.

    Args:
        bboxes_xyxy: Boxes in image-space ``xyxy`` order.
        aspect_wh: Target crop aspect ratio as ``width / height``.
        padding: Padding multiplier applied to the raw box extent.

    Returns:
        Image-space centers and ``(width, height)`` scales matching the crop aspect.
    """
    centers: Float[Tensor, "n 2"] = (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) * 0.5
    extents: Float[Tensor, "n 2"] = (bboxes_xyxy[:, 2:4] - bboxes_xyxy[:, 0:2]) * float(padding)
    widths: Float[Tensor, "n"] = extents[:, 0]
    heights: Float[Tensor, "n"] = extents[:, 1]
    scales: Float[Tensor, "n 2"] = torch.where(
        (widths > heights * aspect_wh)[:, None],
        torch.stack([widths, widths / aspect_wh], dim=1),
        torch.stack([heights * aspect_wh, heights], dim=1),
    )
    return centers.contiguous(), scales.contiguous()


def crop_frames(
    frames_rgb: UInt8[Tensor, "b h w 3"],
    *,
    frame_indices: Int[Tensor, "n"],
    bboxes_xyxy: Float[Tensor, "n 4"],
    spec: CropSpec,
    dtype: torch.dtype = torch.float32,
) -> CropBatch:
    """Generate batched model-ready crops from frames and boxes, fully on device.

    Args:
        frames_rgb: uint8 RGB NHWC frame batch (CUDA in production pipelines).
        frame_indices: Source frame index for each box.
        bboxes_xyxy: Boxes in image-space ``xyxy`` order.
        spec: Crop convention (size, alignment, channel order, normalization).
        dtype: Output crop dtype (match the runtime's input spec).

    Returns:
        Crop batch with normalized inputs and decode metadata.
    """
    validate_frames_rgb(frames_rgb)
    frame_indices = frame_indices.to(device=frames_rgb.device, dtype=torch.long).reshape(-1)
    bboxes: Float[Tensor, "n 4"] = bboxes_xyxy.to(device=frames_rgb.device, dtype=torch.float32).reshape(-1, 4)
    out_w: int = int(spec.input_size[0])
    out_h: int = int(spec.input_size[1])
    if int(bboxes.shape[0]) == 0:
        empty_2: Float[Tensor, "0 2"] = torch.empty((0, 2), dtype=torch.float32, device=frames_rgb.device)
        return CropBatch(torch.empty((0, 3, out_h, out_w), dtype=dtype, device=frames_rgb.device), frame_indices, bboxes, empty_2, empty_2, spec)
    centers, scales = bbox_xyxy_to_center_scale(bboxes, aspect_wh=float(out_w) / float(out_h), padding=spec.padding)
    yy, xx = _crop_meshgrid(out_h, out_w, centers.device, centers.dtype)
    if spec.align == "cv2":
        src_x = (xx[None] - float(out_w) * 0.5) * scales[:, 0, None, None] / float(out_w) + centers[:, 0, None, None]
        src_y = (yy[None] - float(out_h) * 0.5) * scales[:, 1, None, None] / float(out_h) + centers[:, 1, None, None]
    else:
        src_x = xx[None] * scales[:, 0, None, None] / float(out_w - 1) + centers[:, 0, None, None] - 0.5 * scales[:, 0, None, None]
        src_y = yy[None] * scales[:, 1, None, None] / float(out_h - 1) + centers[:, 1, None, None] - 0.5 * scales[:, 1, None, None]
    grid: Float[Tensor, "n crop_h crop_w 2"] = torch.stack(
        [
            src_x / float(int(frames_rgb.shape[2]) - 1) * 2.0 - 1.0,
            src_y / float(int(frames_rgb.shape[1]) - 1) * 2.0 - 1.0,
        ],
        dim=-1,
    ).contiguous()
    # Gather per-detection frames channels-first, converting to float once; the
    # BGR flip runs on the tiny sampled crops (grid_sample is per-channel, so
    # flipping after sampling is numerically identical to flipping the frames).
    gathered: Float[Tensor, "n 3 h w"] = frames_rgb.permute(0, 3, 1, 2)[frame_indices].float()
    with torch.backends.cudnn.flags(enabled=False):
        crops: Float[Tensor, "n 3 crop_h crop_w"] = F.grid_sample(gathered, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    if spec.bgr:
        crops = crops.flip(1)
    if spec.mean_rgb is not None:
        crops = crops - _channel_vector(spec.mean_rgb, crops.device)
    if spec.std_rgb is not None:
        crops = crops / _channel_vector(spec.std_rgb, crops.device)
    return CropBatch(crops.to(dtype=dtype).contiguous(), frame_indices, bboxes, centers, scales, spec)


def crop_coords_to_image(
    xy_crop: Float[Tensor, "n k 2"],
    *,
    centers: Float[Tensor, "n 2"],
    scales: Float[Tensor, "n 2"],
    input_size: tuple[int, int],
) -> Float[Tensor, "n k 2"]:
    """Map crop-input-space keypoints back to source-image pixels.

    Args:
        xy_crop: Keypoints in crop input coordinates (0..input_size range).
        centers: Image-space crop centers from :class:`CropBatch`.
        scales: Padded crop extents from :class:`CropBatch`.
        input_size: Crop size as ``(width, height)``.

    Returns:
        Keypoints in source-image pixel coordinates.
    """
    size: Float[Tensor, "2"] = _float32_vector((float(input_size[0]), float(input_size[1])), xy_crop.device)
    return xy_crop / size * scales[:, None, :] + centers[:, None, :] - 0.5 * scales[:, None, :]


@lru_cache(maxsize=8)
def _crop_meshgrid(
    output_height: int, output_width: int, device: torch.device, dtype: torch.dtype
) -> tuple[Float[Tensor, "crop_h crop_w"], Float[Tensor, "crop_h crop_w"]]:
    grid_y: Float[Tensor, "crop_h crop_w"]
    grid_x: Float[Tensor, "crop_h crop_w"]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, float(output_height - 1), output_height, dtype=dtype, device=device),
        torch.linspace(0.0, float(output_width - 1), output_width, dtype=dtype, device=device),
        indexing="ij",
    )
    return grid_y, grid_x


@lru_cache(maxsize=32)
def _channel_vector(values: tuple[float, float, float], device: torch.device) -> Float[Tensor, "1 3 1 1"]:
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, 3, 1, 1)


@lru_cache(maxsize=32)
def _float32_vector(values: tuple[float, ...], device: torch.device) -> Float[Tensor, "n"]:
    return torch.tensor(tuple(float(v) for v in values), dtype=torch.float32, device=device)
