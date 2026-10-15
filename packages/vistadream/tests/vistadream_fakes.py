"""CPU fakes for the unit lane (no GPU, no weights)."""

import numpy as np
import torch
from jaxtyping import Float32, Float64, UInt8
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction
from numpy import ndarray

from vistadream.ops.splats import Render
from vistadream.types import Splats


def fake_rasterize(splats: Splats, K: Float64[ndarray, "3 3"], cam_T_world: Float64[ndarray, "4 4"], width: int, height: int) -> Render:
    """Differentiable in colour: every pixel shows the mean splat colour; a camera off the origin leaves its left tenth uncovered."""
    rgb: Float32[torch.Tensor, "h w 3"] = splats.rgb.mean(0).expand(height, width, 3)
    alpha: Float32[torch.Tensor, "h w"] = torch.ones(height, width)
    if np.abs(cam_T_world[:3, 3]).max() > 1e-6:
        alpha[:, : width // 10] = 0.0
    return rgb, torch.ones(height, width), alpha


class FakePredictor(BaseRelativePredictor):
    """A gently tilted plane (no depth edges) 1-1.4 units away with every pixel valid."""

    def __call__(self, rgb: UInt8[ndarray, "h w 3"], K_33: Float32[ndarray, "3 3"] | None = None) -> RelativeDepthPrediction:
        height, width = rgb.shape[:2]
        depth: Float32[ndarray, "h w"] = (1.0 + np.linspace(0.0, 0.4, width, dtype=np.float32)[None, :].repeat(height, 0)).astype(np.float32)
        K: Float32[ndarray, "3 3"] = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float32)
        return RelativeDepthPrediction(1.0 / depth, depth, K, np.ones_like(depth))


def fake_inpaint(rgb: UInt8[ndarray, "h w 3"], mask: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
    return np.where(mask[..., None] > 0, np.uint8(128), rgb).astype(np.uint8)
