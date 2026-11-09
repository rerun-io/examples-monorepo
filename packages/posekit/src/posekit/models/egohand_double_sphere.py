"""Double-sphere lens adapter for Gen-HumanEgo's native images.

Projection and its analytic inverse follow Usenko et al., 3DV 2018,
https://arxiv.org/abs/1807.08957, equations 40–51.
"""

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

from posekit.cameras import CalibratedCamera


class DoubleSphereCamera:
    """Source-camera protocol used by the unchanged upstream CPU crop pipeline."""

    def __init__(self, camera: CalibratedCamera) -> None:
        self.K: Float64[ndarray, "3 3"] = camera.intrinsics
        self.width: int = camera.width
        self.height: int = camera.height
        self.model_name: str = "DOUBLE_SPHERE"
        self.xi: float = float(camera.distortion[0])
        self.alpha: float = float(camera.distortion[1])

    def project(self, rays: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 2"]:
        """Project rays into source pixels, with NaN outside the physical domain."""
        x, y, z = np.moveaxis(rays, -1, 0)
        d1: Float64[ndarray, "..."] | np.float64 = np.linalg.norm(rays, axis=-1)
        zz: Float64[ndarray, "..."] | np.float64 = self.xi * d1 + z
        d2: Float64[ndarray, "..."] | np.float64 = np.sqrt(x * x + y * y + zz * zz)
        a: float = self.alpha
        denominator: Float64[ndarray, "..."] | np.float64 = a * d2 + (1.0 - a) * zz
        w1: float = a / (1.0 - a) if a <= 0.5 else (1.0 - a) / a
        w2: float = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi * self.xi + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels: Float64[ndarray, "... 2"] = rays[..., :2] / denominator[..., None] * np.diag(self.K)[:2] + self.K[:2, 2]
        valid: Bool[ndarray, "..."] | np.bool_ = np.isfinite(d1) & (z > -w2 * d1) & (denominator > 0.0)
        return np.where(valid[..., None], pixels, np.nan)

    def unproject(self, pixels: Float64[ndarray, "... 2"]) -> Float64[ndarray, "... 3"]:
        """Invert source pixels into unit forward rays, with NaN outside the domain."""
        xy: Float64[ndarray, "... 2"] = (pixels - self.K[:2, 2]) / np.diag(self.K)[:2]
        r2: Float64[ndarray, "..."] | np.float64 = np.sum(xy * xy, axis=-1)
        a, xi = self.alpha, self.xi
        with np.errstate(divide="ignore", invalid="ignore"):
            mz: Float64[ndarray, "..."] | np.float64 = (1.0 - a * a * r2) / (a * np.sqrt(1.0 - (2.0 * a - 1.0) * r2) + 1.0 - a)
            k: Float64[ndarray, "..."] | np.float64 = (mz * xi + np.sqrt(mz * mz + (1.0 - xi * xi) * r2)) / (mz * mz + r2)
            rays: Float64[ndarray, "... 3"] = np.concatenate([k[..., None] * xy, (k * mz - xi)[..., None]], axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return np.where((np.isfinite(rays).all(-1) & (rays[..., 2] > 0.0))[..., None], rays, np.nan)
