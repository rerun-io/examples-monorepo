"""Image resizing and normalization utilities for MASt3R input.

Pure image utilities with no dependency on SLAM types (Frame, config, etc.),
extracted here to avoid circular imports between ``frame`` and ``mast3r_utils``.
"""

from typing import Literal, NamedTuple

import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from dust3r.utils.image import ImgNorm
from jaxtyping import Float, Int32, UInt8
from numpy import ndarray
from torch import Tensor

from mast3r_slam.image_types import RgbNormalized


class ResizedImage(NamedTuple):
    """Result of resizing and normalizing an image for MASt3R input."""

    rgb_tensor: Float[Tensor, "1 3 h w"]
    """ImageNet-normalized RGB tensor, ready for the MASt3R encoder."""
    true_shape: Int32[ndarray, "1 2"]
    """(height, width) of the cropped image as a (1, 2) int32 array."""
    rgb_uint8: np.ndarray
    """Cropped RGB uint8 image before ImageNet normalization."""


class CropTransform(NamedTuple):
    """Scale and crop parameters mapping MASt3R frame coords back to original image coords."""

    scale_w: float
    """Horizontal scale factor: original_width / resized_width."""
    scale_h: float
    """Vertical scale factor: original_height / resized_height."""
    half_crop_w: float
    """Horizontal offset removed by center-crop (in resized pixels)."""
    half_crop_h: float
    """Vertical offset removed by center-crop (in resized pixels)."""


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    """Resize a PIL image so its longest edge matches the given size.

    Uses Lanczos interpolation when downscaling and bicubic when upscaling.

    Args:
        img: The input PIL image.
        long_edge_size: Desired length of the longest edge in pixels.

    Returns:
        The resized PIL image.
    """
    longest_edge: int = max(img.size)
    interp: PIL.Image.Resampling = (
        PIL.Image.Resampling.LANCZOS if long_edge_size < longest_edge else PIL.Image.Resampling.BICUBIC
    )
    new_size: tuple[int, int] = (
        int(round(img.size[0] * long_edge_size / longest_edge)),
        int(round(img.size[1] * long_edge_size / longest_edge)),
    )
    return img.resize(new_size, interp)


def _center_crop_224(img: PIL.Image.Image) -> PIL.Image.Image:
    """Center-crop a PIL image to the largest inscribed square.

    Args:
        img: Resized PIL image (short side >= 224).

    Returns:
        Square-cropped PIL image.
    """
    w: int
    h: int
    w, h = img.size
    cx: int = w // 2
    cy: int = h // 2
    half: int = min(cx, cy)
    return img.crop((cx - half, cy - half, cx + half, cy + half))


def _center_crop_512(img: PIL.Image.Image, square_ok: bool) -> PIL.Image.Image:
    """Center-crop a PIL image to 16-pixel-aligned dimensions.

    For non-square images (or when ``square_ok`` is True), crops to the
    largest 16-pixel-aligned rectangle.  For square images with
    ``square_ok=False``, forces a 4:3 aspect ratio.

    Args:
        img: Resized PIL image (long side == 512).
        square_ok: Allow square output when the image is square.

    Returns:
        Cropped PIL image with dimensions divisible by 16.
    """
    w: int
    h: int
    w, h = img.size
    cx: int = w // 2
    cy: int = h // 2
    halfw: int = ((2 * cx) // 16) * 8
    halfh: int | float = ((2 * cy) // 16) * 8
    if not square_ok and w == h:
        halfh = 3 * halfw / 4
    return img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))


def resize_img(
    rgb: RgbNormalized,
    size: Literal[224, 512],
    square_ok: bool = False,
) -> ResizedImage:
    """Resize and normalize an image for MASt3R input.

    Converts a float ``[0, 1]`` HWC numpy image to a PIL image, resizes it to
    the target resolution (224 with center-crop or 512 with padding-friendly
    crop), applies ``ImgNorm``, and returns the processed tensors.

    Args:
        rgb: Input RGB image in ``[0, 1]`` float range, shape ``(H, W, 3)``.
        size: Target resolution -- ``224`` (center-crop to square) or ``512``
            (resize long edge, crop to 16-pixel-aligned dimensions).
        square_ok: When ``True`` and ``size=512``, allow square output even if
            the resized image is square.

    Returns:
        A ``ResizedImage`` with the normalized tensor, true shape, and
        unnormalized uint8 image.
    """
    assert size in (224, 512)
    pil_img: PIL.Image.Image = PIL.Image.fromarray((rgb * 255).astype(np.uint8))
    if size == 224:
        # Resize short side to 224, then center-crop to square.
        orig_w: int
        orig_h: int
        orig_w, orig_h = pil_img.size
        pil_img = _resize_pil_image(pil_img, round(size * max(orig_w / orig_h, orig_h / orig_w)))
        pil_img = _center_crop_224(pil_img)
    else:
        # Resize long side to 512, then crop to 16-pixel-aligned dims.
        pil_img = _resize_pil_image(pil_img, size)
        pil_img = _center_crop_512(pil_img, square_ok)

    normalized: Float[Tensor, "1 3 h w"] = ImgNorm(pil_img)[None]
    true_shape: Int32[ndarray, "1 2"] = np.array([pil_img.size[::-1]], dtype=np.int32)
    unnormalized: np.ndarray = np.asarray(pil_img)
    return ResizedImage(rgb_tensor=normalized, true_shape=true_shape, rgb_uint8=unnormalized)


def resize_img_tensor(rgb: UInt8[Tensor, "3 H W"], size: Literal[224, 512]) -> ResizedImage:
    """``resize_img`` for a decoded frame that is already a tensor, on its device (the GPU for an NVDEC frame).

    Same geometry as :func:`resize_img`: the short side to 224 then a square centre crop, or the long side to 512
    then a 16-aligned centre crop. Antialiased bicubic stands in for PIL's Lanczos, so pixels differ from the PIL
    path by a few levels. Only the cropped uint8 image comes back to the host, for logging.
    """
    assert size in (224, 512)
    h: int = int(rgb.shape[1])
    w: int = int(rgb.shape[2])
    long_edge: int = round(size * max(w / h, h / w)) if size == 224 else size
    new_w: int = int(round(w * long_edge / max(w, h)))
    new_h: int = int(round(h * long_edge / max(w, h)))
    resized: UInt8[Tensor, "3 h w"] = (
        F.interpolate(rgb[None].float(), size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False)[0]
        .round_()
        .clamp_(0, 255)
        .to(torch.uint8)
    )
    cx: int = new_w // 2
    cy: int = new_h // 2
    if size == 224:
        half_w: int = min(cx, cy)
        half_h: int = half_w
    else:
        half_w = ((2 * cx) // 16) * 8
        half_h = ((2 * cy) // 16) * 8
        if new_w == new_h:
            half_h = 3 * half_w // 4
    cropped: UInt8[Tensor, "3 h w"] = resized[:, cy - half_h : cy + half_h, cx - half_w : cx + half_w]
    normalized: Float[Tensor, "1 3 h w"] = (cropped.float() / 255.0 - 0.5)[None] / 0.5
    true_shape: Int32[ndarray, "1 2"] = np.array([cropped.shape[1:]], dtype=np.int32)
    return ResizedImage(rgb_tensor=normalized, true_shape=true_shape, rgb_uint8=cropped.permute(1, 2, 0).cpu().numpy())


def resize_img_with_transform(
    rgb: RgbNormalized,
    size: Literal[224, 512],
    square_ok: bool = False,
) -> tuple[ResizedImage, CropTransform]:
    """Resize an image and return the crop/scale transform for coordinate mapping.

    Same as ``resize_img`` but additionally returns the ``CropTransform``
    needed to map MASt3R frame coordinates back to original image coordinates
    (used for adjusting camera intrinsics after resizing).

    Args:
        rgb: Input RGB image in ``[0, 1]`` float range, shape ``(H, W, 3)``.
        size: Target resolution (224 or 512).
        square_ok: Allow square output when the image is square.

    Returns:
        A tuple of (``ResizedImage``, ``CropTransform``).
    """
    assert size in (224, 512)
    pil_img: PIL.Image.Image = PIL.Image.fromarray((rgb * 255).astype(np.uint8))
    orig_w: int
    orig_h: int
    orig_w, orig_h = pil_img.size

    if size == 224:  # noqa: SIM108
        pil_img = _resize_pil_image(pil_img, round(size * max(orig_w / orig_h, orig_h / orig_w)))
    else:
        pil_img = _resize_pil_image(pil_img, size)
    resized_w: int
    resized_h: int
    resized_w, resized_h = pil_img.size

    if size == 224:  # noqa: SIM108
        pil_img = _center_crop_224(pil_img)
    else:
        pil_img = _center_crop_512(pil_img, square_ok)

    normalized: Float[Tensor, "1 3 h w"] = ImgNorm(pil_img)[None]
    true_shape: Int32[ndarray, "1 2"] = np.array([pil_img.size[::-1]], dtype=np.int32)
    unnormalized: np.ndarray = np.asarray(pil_img)

    scale_w: float = orig_w / resized_w
    scale_h: float = orig_h / resized_h
    half_crop_w: float = (resized_w - pil_img.size[0]) / 2
    half_crop_h: float = (resized_h - pil_img.size[1]) / 2

    return (
        ResizedImage(rgb_tensor=normalized, true_shape=true_shape, rgb_uint8=unnormalized),
        CropTransform(scale_w=scale_w, scale_h=scale_h, half_crop_w=half_crop_w, half_crop_h=half_crop_h),
    )
