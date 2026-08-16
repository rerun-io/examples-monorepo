"""Natural-form Gaussian values used by direct Rerun publication."""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray

SH_C0: float = 0.5 * sqrt(1.0 / pi)
"""Degree-zero real spherical-harmonic basis constant used by INRIA 3DGS."""


@dataclass(frozen=True, slots=True)
class GaussianSplats:
    """Natural-form values accepted by ``rr.GaussianSplats3D``."""

    centers_n3: Float32[ndarray, "n 3"]
    """Gaussian centers in metric world coordinates."""
    scales_n3: Float32[ndarray, "n 3"]
    """Linear per-axis standard deviations in metres."""
    quaternions_xyzw_n4: Float32[ndarray, "n 4"]
    """Normalized Rerun-layout quaternions."""
    colors_rgba_n4: UInt8[ndarray, "n 4"]
    """Unmultiplied base RGB and peak opacity."""
    sh_coefficients_n153: Float32[ndarray, "n 15 3"]
    """Coefficient-major RGB spherical harmonics of degrees one through three."""

    @property
    def count(self) -> int:
        """Return the number of Gaussian splats."""
        return len(self.centers_n3)

    def as_archetype(self) -> rr.GaussianSplats3D:
        """Build the unstable Rerun Gaussian-splat archetype."""
        return rr.GaussianSplats3D(
            centers=self.centers_n3,
            scales=self.scales_n3,
            quaternions=self.quaternions_xyzw_n4,
            colors=self.colors_rgba_n4,
            sh_coefficients=self.sh_coefficients_n153,
            spherical_harmonics_degree=3,
        )


def inria_quaternions_xyzw(quaternions_wxyz_n4: Float32[ndarray, "n 4"]) -> Float32[ndarray, "n 4"]:
    """Normalize INRIA-layout quaternions and reorder them for Rerun.

    Args:
        quaternions_wxyz_n4: Float32 unnormalized INRIA quaternions shaped ``n 4``.

    Returns:
        Float32 normalized Rerun ``xyzw`` quaternions shaped ``n 4``. A zero quaternion becomes identity.
    """
    norms_n: Float32[ndarray, "n"] = np.linalg.norm(quaternions_wxyz_n4, axis=1).astype(np.float32, copy=False)
    nonzero_n: ndarray = norms_n > 0.0
    quaternions_xyzw_n4: Float32[ndarray, "n 4"] = np.zeros_like(quaternions_wxyz_n4, dtype=np.float32)
    quaternions_xyzw_n4[:, 3] = 1.0
    quaternions_xyzw_n4[nonzero_n] = quaternions_wxyz_n4[nonzero_n][:, [1, 2, 3, 0]] / norms_n[nonzero_n, None]
    return quaternions_xyzw_n4


def inria_colors_rgba(
    f_dc_n3: Float32[ndarray, "n 3"],
    opacity_logits_n: Float32[ndarray, "n"],
) -> UInt8[ndarray, "n 4"]:
    """Convert INRIA DC coefficients and opacity logits to unmultiplied RGBA.

    Args:
        f_dc_n3: Float32 degree-zero spherical-harmonic coefficients shaped ``n 3``.
        opacity_logits_n: Float32 opacity logits shaped ``n``.

    Returns:
        Uint8 unmultiplied RGBA colors shaped ``n 4``.
    """
    if len(f_dc_n3) != len(opacity_logits_n):
        raise ValueError(f"DC/color row count {len(f_dc_n3)} does not match opacity row count {len(opacity_logits_n)}")
    rgb_01_n3: Float32[ndarray, "n 3"] = np.clip(0.5 + SH_C0 * f_dc_n3, 0.0, 1.0).astype(np.float32, copy=False)
    alpha_01_n: Float32[ndarray, "n"] = np.empty_like(opacity_logits_n, dtype=np.float32)
    nonnegative_n: ndarray = opacity_logits_n >= 0.0
    alpha_01_n[nonnegative_n] = 1.0 / (1.0 + np.exp(-opacity_logits_n[nonnegative_n]))
    exp_logits_n: Float32[ndarray, "n_negative"] = np.exp(opacity_logits_n[~nonnegative_n])
    alpha_01_n[~nonnegative_n] = exp_logits_n / (1.0 + exp_logits_n)
    rgba_01_n4: Float32[ndarray, "n 4"] = np.column_stack((rgb_01_n3, alpha_01_n)).astype(np.float32, copy=False)
    return np.clip(rgba_01_n4 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
