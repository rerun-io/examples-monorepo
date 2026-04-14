"""Python wrapper for the CUDA bundle adjustment solver.

The heavy lifting is performed by ``_cuda_ba.forward`` which runs
Gauss-Newton iterations with a Schur complement to jointly optimize camera
poses and 3-D patch depths.

This module also re-exports two CUDA utility functions:

- ``neighbors`` -- find co-visible patch/frame neighbourhoods.
- ``reproject`` -- project 3-D patches into specified target frames.
- ``solve_system`` -- sparse PGO solve for loop closure.
"""

from jaxtyping import Float, Int
from torch import Tensor

from dpvo import _cuda_ba

neighbors = _cuda_ba.neighbors
"""Query the co-visibility neighbourhood of patches across frames."""

reproject = _cuda_ba.reproject
"""Reproject 3-D patches into target camera frames using current poses."""

solve_system = _cuda_ba.solve_system
"""Sparse Jacobian-based solve for pose-graph optimization (loop closure PGO)."""


def BA(
    poses: Float[Tensor, "..."],
    patches: Float[Tensor, "..."],
    intrinsics: Float[Tensor, "..."],
    target: Float[Tensor, "..."],
    weight: Float[Tensor, "..."],
    lmbda: Float[Tensor, "1"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    t0: int,
    t1: int,
    M: int = -1,
    iterations: int = 2,
    eff_impl: bool = False,
) -> None:
    """Run Gauss-Newton bundle adjustment with Schur complement in CUDA.

    Updates ``poses`` **in-place** (via ``poses.data``) by minimizing the
    reprojection error of the given patches against their target 2-D
    coordinates, weighted by ``weight``.

    Args:
        poses: Camera poses to optimize (modified in-place).
        patches: 3-D patch representations (inverse depth parameterisation).
        intrinsics: Per-frame camera intrinsics ``[fx, fy, cx, cy]``.
        target: Target 2-D reprojection coordinates for each edge.
        weight: Per-edge confidence weights.
        lmbda: Levenberg-Marquardt damping factor.
        ii: Source frame indices for each edge.
        jj: Target frame indices for each edge.
        kk: Patch indices for each edge.
        t0: Start of the active keyframe window (poses before ``t0`` are
            held fixed).
        t1: End of the active keyframe window (exclusive).
        iterations: Number of Gauss-Newton iterations to run.
    """
    _cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)
