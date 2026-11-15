"""Deterministic source-mask and center-aspect crops."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Crop:
    top: int
    left: int
    height: int
    width: int
    original_height: int
    original_width: int
    output_height: int
    output_width: int

    @property
    def scale_y(self) -> float:
        return self.output_height / self.height

    @property
    def scale_x(self) -> float:
        return self.output_width / self.width


def center_crop(image_height: int, image_width: int, output_height: int, output_width: int) -> Crop:
    aspect_ratio = output_height / output_width
    if image_height / image_width >= aspect_ratio:
        crop_width = image_width
        crop_height = min(image_height, max(1, int(round(crop_width * aspect_ratio))))
    else:
        crop_height = image_height
        crop_width = min(image_width, max(1, int(round(crop_height / aspect_ratio))))
    return Crop(
        top=(image_height - crop_height) // 2,
        left=(image_width - crop_width) // 2,
        height=crop_height,
        width=crop_width,
        original_height=image_height,
        original_width=image_width,
        output_height=output_height,
        output_width=output_width,
    )


def mask_bounds(masks: np.ndarray, threshold: float) -> tuple[int, int, int, int] | None:
    """Return union bounds as left/top inclusive and right/bottom exclusive."""

    values = np.asarray(masks)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim not in (2, 3):
        raise ValueError(f"Expected masks [H,W] or [F,H,W], got {values.shape}.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    cutoff = threshold * 255.0 if np.issubdtype(values.dtype, np.integer) else threshold
    union = values.max(axis=0) if values.ndim == 3 else values
    foreground = np.asarray(union > cutoff)
    rows = np.flatnonzero(foreground.any(axis=1))
    columns = np.flatnonzero(foreground.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)


def expand_bounds(bounds, margins, image_height: int, image_width: int):
    if bounds is None:
        return None
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    top, right, bottom, left = (float(value) for value in margins)
    box_width, box_height = xmax - xmin, ymax - ymin
    return (
        max(0, int(math.floor(xmin - left * box_width))),
        max(0, int(math.floor(ymin - top * box_height))),
        min(image_width, int(math.ceil(xmax + right * box_width))),
        min(image_height, int(math.ceil(ymax + bottom * box_height))),
    )


def crop_from_bounds(
    *,
    bounds: tuple[int, int, int, int] | None,
    image_height: int,
    image_width: int,
    output_height: int,
    output_width: int,
    margins: tuple[float, float, float, float],
    allow_upscale: bool = True,
) -> Crop:
    bounds = expand_bounds(bounds, margins, image_height, image_width)
    if bounds is None:
        return center_crop(image_height, image_width, output_height, output_width)

    xmin, ymin, xmax, ymax = bounds
    required_height = float(ymax - ymin)
    required_width = float(xmax - xmin)
    if not allow_upscale:
        required_height = max(required_height, min(output_height, image_height))
        required_width = max(required_width, min(output_width, image_width))

    aspect_ratio = output_height / output_width
    crop_height = max(required_height, required_width * aspect_ratio)
    crop_width = crop_height / aspect_ratio
    max_height = min(float(image_height), float(image_width) * aspect_ratio)
    crop_height = max(1, min(image_height, int(math.ceil(min(crop_height, max_height)))))
    crop_width = max(1, min(image_width, int(math.ceil(crop_height / aspect_ratio))))

    left_low = max(0.0, xmax - crop_width)
    left_high = min(float(xmin), image_width - crop_width)
    top_low = max(0.0, ymax - crop_height)
    top_high = min(float(ymin), image_height - crop_height)
    if left_low <= left_high:
        left = int(math.floor((left_low + left_high) / 2.0))
    else:
        left = max(0, min(int(math.floor((xmin + xmax - crop_width) / 2.0)), image_width - crop_width))
    if top_low <= top_high:
        top = int(math.floor((top_low + top_high) / 2.0))
    else:
        top = max(0, min(int(math.floor((ymin + ymax - crop_height) / 2.0)), image_height - crop_height))
    return Crop(
        top=top,
        left=left,
        height=crop_height,
        width=crop_width,
        original_height=image_height,
        original_width=image_width,
        output_height=output_height,
        output_width=output_width,
    )


def transform_intrinsics(K: np.ndarray, crop: Crop) -> np.ndarray:
    transformed = np.asarray(K, dtype=np.float64).copy()
    transformed[0, 2] -= crop.left
    transformed[1, 2] -= crop.top
    transformed[0] *= crop.scale_x
    transformed[1] *= crop.scale_y
    transformed[2, 2] = 1.0
    return transformed
