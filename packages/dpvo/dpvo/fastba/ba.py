"""Python wrapper for the configured bundle adjustment solver.

This module also re-exports three backend utility functions:

- ``neighbors`` -- find co-visible patch/frame neighbourhoods.
- ``reproject`` -- project 3-D patches into specified target frames.
- ``solve_system`` -- sparse PGO solve for loop closure.
"""

from jaxtyping import Float, Int
from torch import Tensor

import os
from typing import Any


def _load_backend() -> Any:
    """Load the configured fastba backend.

    ``auto`` and ``mojo`` use the standalone Mojo package. CUDA is only used
    when ``DPVO_FASTBA_BACKEND=cuda`` is set explicitly.
    """
    backend = os.environ.get("DPVO_FASTBA_BACKEND", "auto").lower()
    if backend not in {"auto", "mojo", "cuda"}:
        raise ValueError(
            "DPVO_FASTBA_BACKEND must be one of 'auto', 'mojo', or 'cuda', "
            f"got {backend!r}"
        )

    if backend in {"auto", "mojo"}:
        from dpvo_fastba_mojo import backend as mojo_ba

        return mojo_ba

    from dpvo import _cuda_ba

    return _cuda_ba


_ba_backend = _load_backend()

neighbors = _ba_backend.neighbors
"""Query the co-visibility neighbourhood of patches across frames."""

reproject = _ba_backend.reproject
"""Reproject 3-D patches into target camera frames using current poses."""

solve_system = _ba_backend.solve_system
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
        M: Patches per frame (used for E-block allocation when
            ``eff_impl=True``).  Set to -1 for sliding-window BA.
        iterations: Number of Gauss-Newton iterations to run.
        eff_impl: If ``True``, use efficient E-block (block_e.cu) for
            global BA with many patches.  If ``False``, use the dense
            E matrix path for sliding-window BA.
    """
    _ba_backend.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)
