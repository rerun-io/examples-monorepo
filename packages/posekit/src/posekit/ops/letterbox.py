"""GPU letterbox preprocessing for full-frame detectors (YOLOX convention)."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.predictions import validate_frames_rgb


def letterbox_frames(
    frames_rgb: UInt8[Tensor, "b h w 3"],
    *,
    output_size: tuple[int, int],
    pad_value: float = 114.0,
    bgr: bool = True,
) -> tuple[Float[Tensor, "b 3 out_h out_w"], Float[Tensor, "b"]]:
    """Resize frames into a top-left letterbox without normalization.

    Matches the mmdet/rtmlib YOLOX convention: optional RGB->BGR, aspect-preserving
    resize into the top-left corner, and constant padding (no mean/std).

    Args:
        frames_rgb: uint8 RGB NHWC frame batch.
        output_size: Detector input size as ``(height, width)``.
        pad_value: Padding fill value.
        bgr: Whether the detector consumes BGR channel order.

    Returns:
        Float32 NCHW detector inputs and the per-frame resize ratio (shared,
        since the batch shares one resolution).
    """
    validate_frames_rgb(frames_rgb)
    in_h: int = int(frames_rgb.shape[1])
    in_w: int = int(frames_rgb.shape[2])
    out_h: int = int(output_size[0])
    out_w: int = int(output_size[1])
    ratio: float = min(out_h / in_h, out_w / in_w)
    resized_h: int = int(in_h * ratio)
    resized_w: int = int(in_w * ratio)
    channels_first: Float[Tensor, "b 3 h w"] = frames_rgb.permute(0, 3, 1, 2).float()
    resized: Float[Tensor, "b 3 rh rw"] = F.interpolate(channels_first, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    if bgr:
        # Bilinear resize is per-channel, so flipping the small resized tensor
        # is numerically identical to flipping the full-resolution frames.
        resized = resized.flip(1)
    inputs: Float[Tensor, "b 3 out_h out_w"] = torch.full(
        (int(frames_rgb.shape[0]), 3, out_h, out_w), float(pad_value), dtype=torch.float32, device=frames_rgb.device
    )
    inputs[:, :, :resized_h, :resized_w] = resized
    ratios: Float[Tensor, "b"] = torch.full((int(frames_rgb.shape[0]),), ratio, dtype=torch.float32, device=frames_rgb.device)
    return inputs, ratios
