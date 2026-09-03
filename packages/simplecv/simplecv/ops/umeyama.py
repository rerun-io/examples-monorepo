"""Umeyama similarity alignment between two row-aligned point sets."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Float64
from numpy import ndarray

MIN_CORRESPONDENCES: int = 3
"""Fewest row-aligned point pairs that determine a 3D similarity; two points leave the rotation about their axis free."""


@dataclass(slots=True, frozen=True)
class SimilarityTransform:
    """Similarity mapping source coordinates into destination coordinates.

    ``dst ≈ scale * (dst_R_src @ src.T).T + dst_t_src``.
    """

    dst_R_src: Float64[ndarray, "3 3"]
    """Rotation taking source-frame vectors into the destination frame."""
    dst_t_src: Float64[ndarray, "3"]
    """Translation of the source origin, expressed in the destination frame."""
    scale: float
    """Isotropic scale; ``1.0`` for a rigid alignment."""

    def apply(self, src_points: Float[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        """Map source-frame points into the destination frame."""
        return self.scale * np.einsum("ij,nj->ni", self.dst_R_src, src_points.astype(np.float64, copy=False)) + self.dst_t_src

    def as_matrix(self) -> Float64[ndarray, "4 4"]:
        """The similarity as a homogeneous ``dst_T_src`` matrix (scale folded into the rotation block)."""
        dst_T_src: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        dst_T_src[:3, :3] = self.scale * self.dst_R_src
        dst_T_src[:3, 3] = self.dst_t_src
        return dst_T_src


def umeyama_alignment(
    src_points: Float[ndarray, "n 3"],
    dst_points: Float[ndarray, "n 3"],
    *,
    allow_scaling: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """Least-squares similarity aligning ``src_points`` onto ``dst_points`` (Umeyama 1991).

    The estimate runs in float64 whatever the input dtype: the SVD and the scale
    ratio lose precision in float32, and callers pass float32 detections.

    Args:
        src_points: Source XYZ coordinates.
        dst_points: Target XYZ coordinates, row-aligned with ``src_points``.
        allow_scaling: Estimate an isotropic scale; otherwise the scale is fixed at 1.
        eps: Smallest source variance accepted before the problem counts as degenerate.

    Returns:
        The similarity such that ``dst_points ≈ scale * (dst_R_src @ src_points.T).T + dst_t_src``.

    Raises:
        ValueError: If the shapes differ, fewer than three pairs are given, or the source variance is at most ``eps``.
    """
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Source and destination shapes must match; got {src_points.shape} vs {dst_points.shape}")
    if src_points.shape[0] < MIN_CORRESPONDENCES:
        raise ValueError(f"Umeyama alignment needs at least {MIN_CORRESPONDENCES} correspondences; got {src_points.shape[0]}")
    src: Float64[ndarray, "n 3"] = src_points.astype(np.float64, copy=False)
    dst: Float64[ndarray, "n 3"] = dst_points.astype(np.float64, copy=False)
    src_mean: Float64[ndarray, "3"] = np.mean(src, axis=0)
    dst_mean: Float64[ndarray, "3"] = np.mean(dst, axis=0)
    src_centered: Float64[ndarray, "n 3"] = src - src_mean
    dst_centered: Float64[ndarray, "n 3"] = dst - dst_mean
    covariance: Float64[ndarray, "3 3"] = (dst_centered.T @ src_centered) / float(src.shape[0])
    svd: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"], Float64[ndarray, "3 3"]] = np.linalg.svd(covariance)
    u: Float64[ndarray, "3 3"] = svd[0]
    singular_values: Float64[ndarray, "3"] = svd[1]
    vt: Float64[ndarray, "3 3"] = svd[2]
    reflection: Float64[ndarray, "3"] = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        reflection[2] = -1.0
    dst_R_src: Float64[ndarray, "3 3"] = u @ np.diag(reflection) @ vt
    src_variance: float = float(np.mean(np.sum(src_centered**2, axis=1)))
    if src_variance <= eps:
        raise ValueError(f"Source variance too small for stable Umeyama estimation (variance={src_variance})")
    scale: float = float(np.sum(singular_values * reflection) / src_variance) if allow_scaling else 1.0
    dst_t_src: Float64[ndarray, "3"] = dst_mean - scale * (dst_R_src @ src_mean)
    return SimilarityTransform(dst_R_src=dst_R_src, dst_t_src=dst_t_src, scale=scale)
