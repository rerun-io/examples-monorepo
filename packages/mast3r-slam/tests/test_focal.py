"""estimate_focal_knowing_depth recovers a pinhole's focal from its own pointmap."""

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from mast3r_slam.mast3r_utils import estimate_focal_knowing_depth


@pytest.mark.parametrize(("height", "width", "focal"), [(224, 224, 180.0), (288, 512, 420.0)])
def test_recovers_the_focal_of_a_noisy_pointmap(height: int, width: int, focal: float) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    v_hw, u_hw = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    z_hw: Float32[ndarray, "h w"] = (1.0 + 3.0 * rng.random((height, width))).astype(np.float32)
    pts_hw3: Float32[ndarray, "h w 3"] = np.stack(
        [(u_hw - width / 2) / focal * z_hw, (v_hw - height / 2) / focal * z_hw, z_hw], axis=-1
    ).astype(np.float32)
    pts_hw3 += rng.normal(0.0, 0.01, pts_hw3.shape).astype(np.float32)
    np.testing.assert_allclose(estimate_focal_knowing_depth(pts_hw3, cx=width / 2, cy=height / 2), focal, rtol=2e-3)
