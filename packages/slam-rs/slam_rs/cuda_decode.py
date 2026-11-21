"""NVDEC decoding through the shared SimpleCV reader, with grayscale batches for Rust."""

from collections.abc import Iterator
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.video_io import TorchCodecVideoReader
from torch import Tensor

from slam_rs.catalog_feed import ColorCopy, color_copy_size


def decode_gray_cuda(
    payload: bytes, resize_hw: tuple[int, int], downscale: int = 1, color_downscale: int | None = None
) -> Iterator[tuple[UInt8[ndarray, "h w"], ColorCopy | None]]:
    """Decode sequential batches, resize and convert RGB to gray on CUDA, then copy the small images to Rust's CPU input.

    Uses RGB luminance weights after NVDEC color conversion, so rounding and
    clipped chroma can differ from PyAV's direct YUV-to-gray path.
    The native dimensions after integer downscaling must match calibration;
    resizing must not make an incompatible source look valid. With ``color_downscale`` each gray frame comes with an RGB
    copy area-resampled to :func:`~slam_rs.catalog_feed.color_copy_size` on the GPU, the CPU decoder's size and filter,
    copied to the host when called; otherwise the pair's second item is None.
    """
    reader: TorchCodecVideoReader = TorchCodecVideoReader(payload, device="cuda", seek_mode="exact", resize_hw=resize_hw)
    try:
        source_resize_hw: tuple[int, int] = (max(1, reader.source_height // downscale), max(1, reader.source_width // downscale))
        if source_resize_hw != resize_hw:
            raise ValueError(f"video resolution after downscale {downscale} is {source_resize_hw}, calibration says {resize_hw}")
        weights: Float32[Tensor, "1 3 1 1"] = rearrange(torch.tensor([0.299, 0.587, 0.114], device="cuda"), "c -> 1 c 1 1")
        color_wh: tuple[int, int] | None = None if color_downscale is None else color_copy_size(resize_hw[1], resize_hw[0], color_downscale)
        for start in range(0, len(reader), 32):
            rgb: UInt8[Tensor, "b 3 h w"] = reader.get_frames_in_range(start, min(start + 32, len(reader)))
            rgb_float: Float32[Tensor, "b 3 h w"] = rgb.float()
            gray: UInt8[Tensor, "b h w"] = (rgb_float * weights).sum(dim=1).round().clamp(0, 255).to(dtype=torch.uint8)
            # One host copy per frame: a view into one batch array would leave its last holder (the consumer, behind
            # the feed's prefetch) to unmap 32 frames at once, 118 MB on a 1920 px lens, every 32 framesets.
            if color_wh is None:
                for frame in gray:
                    yield frame.cpu().numpy(), None
                continue
            small: UInt8[Tensor, "b ch cw 3"] = rearrange(
                F.interpolate(rgb_float, size=(color_wh[1], color_wh[0]), mode="area").round().clamp(0, 255).to(dtype=torch.uint8),
                "b c h w -> b h w c",
            ).contiguous()
            for frame, color in zip(gray, small, strict=True):
                yield frame.cpu().numpy(), partial(_host_copy, color)
    finally:
        reader.close()


def _host_copy(color: UInt8[Tensor, "ch cw 3"]) -> UInt8[ndarray, "ch cw 3"]:
    return color.cpu().numpy()
