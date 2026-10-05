"""Enclosing circles from unclipped valid landmarks, and their square boxes."""

from collections.abc import Sequence

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from torch import Tensor


def enclosing_circles(points: Float32[ndarray, '*b n 2'], valid: Bool[ndarray, '*b n']) -> Float32[ndarray, '*b 3']:
    """Enclose valid points per row; rows without a valid point return three NaNs."""
    count: int = int(np.prod(points.shape[:-2]))
    rows: Float32[ndarray, 'b n 2'] = points.reshape(count, points.shape[-2], 2)
    masks: Bool[ndarray, 'b n'] = valid.reshape(count, points.shape[-2])
    circles: Float32[ndarray, 'b 3'] = np.full((count, 3), np.nan, dtype=np.float32)
    for index in range(count):
        selected: Float32[ndarray, 'n 2'] = rows[index, masks[index]]
        if len(selected):
            circle: tuple[Sequence[float], float] = cv2.minEnclosingCircle(selected)
            circles[index] = (*circle[0], circle[1])
    return circles.reshape(*points.shape[:-2], 3)


def square_boxes(circles: Float32[Tensor, '*b 3'], enlarge: float = 1.0) -> Float32[Tensor, '*b 4']:
    """Return (x0, y0, x1, y1) with half-side radius times enlarge."""
    radius: Float32[Tensor, '*b 1'] = circles[..., 2:3] * enlarge
    return torch.cat((circles[..., :2] - radius, circles[..., :2] + radius), dim=-1)
