"""Python wrapper for the native bundle adjustment solver.

The heavy lifting is performed by the backend's ``forward`` (CUDA:
``_cuda_ba.forward``), which runs Gauss-Newton iterations with a Schur
complement to jointly optimize camera poses and 3-D patch depths, plus an
optional robust log-inverse-depth prior on the patches (:class:`BADepthPrior`).

Every op picks its backend from the input tensor's device through
:mod:`dpvo.kernels`. This module also exposes two backend utilities:

- ``neighbors`` -- find co-visible patch/frame neighbourhoods.
- ``reproject`` -- project 3-D patches into specified target frames.

and ``solve_system``, loop closure's pose-graph step (:mod:`dpvo.mojo_backend.pgo`).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from dpvo.kernels import ba_kernels
from dpvo.mojo_backend.pgo import solve_system as solve_system


@dataclass(frozen=True, slots=True)
class BADepthPrior:
    """Robust log-inverse-depth unary on every BA patch (ViPE MetricTrail kernel).

    For each patch in the BA the kernel adds ``r = log(prior) - log(d)`` with
    ``d`` the patch-centre inverse depth, weighted by ``lam * weight * huber(r)``.
    ``disps``/``weights`` cover every buffer slot (``N*M``) and are indexed by
    the global flat patch index. A zero or NaN weight, or a non-positive or
    non-finite prior, skips that patch. ``lam == 0`` keeps the prior out of the
    solve (no kernel launch), exactly like passing ``None``.
    """

    disps: Float32[Tensor, "n_patches"]
    """Prior inverse depth per buffer patch slot."""
    weights: Float32[Tensor, "n_patches"]
    """Per-patch prior weight (``0`` disables that patch)."""
    lam: float
    """Global prior weight, ``>= 0``; ``0`` disables the prior."""
    huber: float = 0.3
    """Huber threshold on ``|r|`` in log inverse depth, ``> 0``."""

    def __post_init__(self) -> None:
        if self.disps.shape != self.weights.shape:
            raise ValueError(f"depth prior disps {tuple(self.disps.shape)} and weights {tuple(self.weights.shape)} differ in shape")
        if not self.lam >= 0.0:
            raise ValueError(f"depth prior lam must be >= 0, got {self.lam}")
        if not self.huber > 0.0:
            raise ValueError(f"depth prior huber must be > 0, got {self.huber}")


def BA(
    poses: Float32[Tensor, "1 n_frames 7"],
    patches: Float32[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float32[Tensor, "1 n_frames 4"],
    target: Float32[Tensor, "1 n_edges 2"],
    weight: Float32[Tensor, "1 n_edges 2"],
    lmbda: Float32[Tensor, "1"],
    ii: Int64[Tensor, "n_edges"],
    jj: Int64[Tensor, "n_edges"],
    kk: Int64[Tensor, "n_edges"],
    t0: int,
    t1: int,
    M: int = -1,
    iterations: int = 2,
    eff_impl: bool = False,
    *,
    depth_prior: BADepthPrior | None = None,
) -> None:
    """Run Gauss-Newton bundle adjustment with Schur complement.

    Updates ``poses`` (via ``poses.data``) and ``patches`` **in place** by
    minimizing the reprojection error of the given patches against their
    target 2-D coordinates, weighted by ``weight``, plus the optional depth
    prior.

    Args:
        poses: Camera poses to optimize (modified in place).
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
        depth_prior: Optional per-patch inverse-depth prior; ``None`` (or
            ``lam == 0``) runs plain BA.

    Raises:
        ValueError: ``depth_prior`` does not hold one value per patch slot.
    """
    prior: Float32[Tensor, "n_prior"]
    prior_weight: Float32[Tensor, "n_prior"]
    if depth_prior is None:
        prior = torch.empty(0, dtype=torch.float32, device=patches.device)
        prior_weight = prior
        lam: float = 0.0
        huber: float = 0.3
    else:
        if depth_prior.disps.shape[0] != patches.shape[1]:
            raise ValueError(f"depth prior has {depth_prior.disps.shape[0]} values, BA has {patches.shape[1]} patch slots")
        prior = depth_prior.disps.contiguous()
        prior_weight = depth_prior.weights.contiguous()
        lam = depth_prior.lam
        huber = depth_prior.huber
    ba_kernels(patches.device).forward(
        poses.data, patches, intrinsics, target, weight, lmbda, prior, prior_weight, lam, huber, ii, jj, kk, M, t0, t1, iterations, eff_impl
    )


def neighbors(ii: Int64[Tensor, "n_edges"], jj: Int64[Tensor, "n_edges"]) -> tuple[Int64[Tensor, "n_edges"], Int64[Tensor, "n_edges"]]:
    """Link each edge to the previous and next observation of the same patch.

    Args:
        ii: Patch (source) index per edge.
        jj: Target frame index per edge.

    Returns:
        ``(ix, jx)``: per edge, the edge index of the previous / next
        observation of the same patch ordered by ``jj``, or ``-1``.
    """
    linked: Sequence[Tensor] = ba_kernels(ii.device).neighbors(ii, jj)
    ix: Int64[Tensor, "n_edges"] = linked[0]
    jx: Int64[Tensor, "n_edges"] = linked[1]
    return ix, jx


def reproject(
    poses: Float32[Tensor, "1 n_frames 7"],
    patches: Float32[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float32[Tensor, "1 n_frames 4"],
    ii: Int64[Tensor, "n_edges"],
    jj: Int64[Tensor, "n_edges"],
    kk: Int64[Tensor, "n_edges"],
) -> Float32[Tensor, "1 n_edges 2 ps ps"]:
    """Reproject every pixel of patch ``kk`` from frame ``ii`` into frame ``jj``.

    Args:
        poses: Camera poses (world-to-camera, ``[t, q]``).
        patches: Patch pixel coordinates and inverse depths.
        intrinsics: Per-frame camera intrinsics ``[fx, fy, cx, cy]``.
        ii: Source frame index per edge.
        jj: Target frame index per edge.
        kk: Patch index per edge.

    Returns:
        Reprojected ``(x, y)`` pixel coordinates per edge and patch pixel.
    """
    return ba_kernels(poses.device).reproject(poses, patches, intrinsics, ii, jj, kk)
