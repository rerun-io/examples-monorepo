"""Natural-form Gaussian conversion contracts for fused publication."""

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray

from gauss_surf.train_gsplat.splat_values import SH_C0, inria_colors_rgba, inria_quaternions_xyzw


def test_inria_colors_match_rerun_ply_conversion() -> None:
    """Match the Gaussian archetype conversion, including u8 rounding and sigmoid endpoints."""
    source_rgb_n3: UInt8[ndarray, "n=3 3"] = np.asarray([[0, 128, 255], [128, 128, 128], [255, 128, 0]], dtype=np.uint8)
    dc_n3: Float32[ndarray, "n=3 3"] = ((source_rgb_n3.astype(np.float32) / 255.0 - 0.5) / SH_C0).astype(np.float32)
    opacity_logits_n: Float32[ndarray, "n=3"] = np.asarray([-100.0, 0.0, 100.0], dtype=np.float32)

    colors_rgba_n4: UInt8[ndarray, "n=3 4"] = inria_colors_rgba(dc_n3, opacity_logits_n)

    np.testing.assert_array_equal(colors_rgba_n4[:, :3], source_rgb_n3)
    np.testing.assert_array_equal(colors_rgba_n4[:, 3], np.asarray([0, 128, 255], dtype=np.uint8))


def test_inria_wxyz_quaternions_become_normalized_xyzw() -> None:
    quaternions_wxyz_n4: Float32[ndarray, "n=3 4"] = np.asarray(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )

    quaternions_xyzw_n4: Float32[ndarray, "n=3 4"] = inria_quaternions_xyzw(quaternions_wxyz_n4)

    np.testing.assert_allclose(quaternions_xyzw_n4, [[0.0, 0.0, 0.0, 1.0], [0.6, 0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 1.0]])
