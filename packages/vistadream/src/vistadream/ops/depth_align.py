"""Fit an inpainted view's predicted depth to the depth the scene renders there (VistaDream's smooth connect,
after https://arxiv.org/abs/2311.13384)."""

import cv2
import numpy as np
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray


def scale_shift(pred: Float32[ndarray, "k"], target: Float32[ndarray, "k"]) -> tuple[float, float]:
    """Least-squares ``target ~ pred * scale + shift`` over pairs with ``target > 0`` and ``pred < 199``; the median ratio
    (shift 0) when the fit's scale is negative; the identity with 10 pairs or fewer."""
    usable: Bool[ndarray, "k"] = (target > 0) & (pred < 199)
    if usable.sum() <= 10:
        return 1.0, 0.0
    scale, shift = np.polyfit(pred[usable], target[usable], deg=1)
    if scale < 0:
        return float(np.median(target[usable]) / (np.median(pred[usable]) + 1e-8)), 0.0
    return float(scale), float(shift)


def align_depth(rendered: Float32[ndarray, "h w"], predicted: Float32[ndarray, "h w"], holes: Bool[ndarray, "h w"]) -> Float32[ndarray, "h w"]:
    """``predicted`` scaled and shifted onto ``rendered`` where the scene is known (``~holes``), then the remaining offset
    diffused into the holes (100 rounds of a 15 x 15 box blur with the known offsets held). No holes: ``rendered``."""
    if not holes.any():
        return rendered
    known_mask: Bool[ndarray, "h w"] = ~holes
    scale, shift = scale_shift(predicted[known_mask], rendered[known_mask])
    coarse: Float64[ndarray, "h w"] = predicted.astype(np.float64) * scale + shift
    known: Float64[ndarray, "k"] = rendered[known_mask] - coarse[known_mask]
    offset: Float64[ndarray, "h w"] = np.zeros_like(coarse)
    for _ in range(100):
        offset[known_mask] = known
        offset = cv2.blur(offset, (15, 15))
    return (coarse + offset).astype(np.float32)
