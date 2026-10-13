"""Image sizing for the pipeline: sides that are multiples of 32 (FLUX and MoGe), and the outpainting border."""

import numpy as np
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from PIL import Image


def _multiple_of_32(size: int | float) -> int:
    return max(32, int(32 * round(size / 32)))  # a side never rounds to 0


def process_image(image: Image.Image, max_dimension: int = 1920, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    """Both sides multiples of 32 (LANCZOS): a long side over ``max_dimension`` is scaled down to it; a smaller image is
    rescaled into ``[min_mp, max_mp]`` megapixels (only rounded when it already is in range)."""
    width, height = image.size
    scale: float = min(max_dimension / width, max_dimension / height)
    if scale < 1.0:
        size: tuple[int, int] = (_multiple_of_32(width * scale), _multiple_of_32(height * scale))
    else:
        megapixels: float = width * height / 1_000_000
        factor: float = (min(max(megapixels, min_mp), max_mp) / megapixels) ** 0.5
        size = (_multiple_of_32(int(width * factor)), _multiple_of_32(int(height * factor)))
    return image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def add_border_and_mask(rgb: UInt8[ndarray, "h w 3"], border: float) -> tuple[UInt8[ndarray, "H W 3"], Bool[ndarray, "H W"], Int64[ndarray, "4"]]:
    """Black border of ``border`` x the image size on every side, the canvas rounded to multiples of 32.

    Returns the canvas, the mask to fill (True on the border) and where the image sits (``x y w h``).
    """
    height, width = rgb.shape[:2]
    pad_x, pad_y = int(width * border), int(height * border)
    canvas_w, canvas_h = _multiple_of_32(width + 2 * pad_x), _multiple_of_32(height + 2 * pad_y)
    canvas: UInt8[ndarray, "H W 3"] = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = rgb[: canvas_h - pad_y, : canvas_w - pad_x]
    mask: Bool[ndarray, "H W"] = np.ones((canvas_h, canvas_w), dtype=bool)
    mask[pad_y : pad_y + height, pad_x : pad_x + width] = False
    return canvas, mask, np.array([pad_x, pad_y, width, height], dtype=np.int64)
