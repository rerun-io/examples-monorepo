"""MASt3R-SLAM's Gauss-Newton pose optimisation (paper §3.3) on the Mojo kernels, with the signatures of the former
CUDA extension's `gauss_newton_rays` and `gauss_newton_calib`.

Each iteration, as the extension's host loop: the step kernel (`mojo/gn.mojo`) sums Jᵀ W J and Jᵀ W r per edge on
the GPU; the CPU assembles and solves the pose system in float64 on one thread (`mojo/solve.mojo`, the extension's
Eigen step); lietorch-mojo's `pose_retr` retracts every pose but the first by dx = -H⁻¹ g. The loop stops after
max_iter iterations or when |dx| < delta_thresh. Twc is updated in place.
"""

import torch
from jaxtyping import Bool, Float32, Int64
from lietorch_mojo.pose_retr import pose_retr
from torch import Tensor

from mast3r_slam.mojo_backend._launch import launch

NUM_FIX: int = 1
"""The first pose of the problem stays fixed, as in the CUDA extension."""
SUMS: int = 119
"""The float32 sums per edge that the step kernel writes (gn.mojo's SUMS: 105 Hessian entries, 7 + 7 gradient)."""
THREADS: int = 128
RAYS_TRACKING_HUBER: float = 1.345
"""Huber threshold the rays kernel fixes (``_huber`` in ``gn.mojo``): :func:`track_pose_rays` cannot take another."""
RAYS_TRACKING_REL_ERROR: float = 1e-3
"""The only ``tracking.rel_error`` :func:`track_pose_rays` accepts: it stops on the step norm alone, never on the cost."""
"""gn.mojo's THREADS."""


def _gauss_newton(
    step: str,
    scalars: tuple[int | float, ...],
    Twc: Tensor,
    tensors: tuple[Tensor, ...],
    ii: Tensor,
    jj: Tensor,
    max_iter: int,
    delta_thresh: float,
) -> Tensor:
    """tensors = (Xs, Cs, K, idx_ii2jj, valid_match, Q); ii / jj are keyframe ids, and the rows of Twc / Xs / Cs
    follow their sorted unique values."""
    Xs, Cs, K, idx_ii2jj, valid_match, Q = tensors
    unique_kf: Tensor = torch.unique(torch.cat([ii, jj]), sorted=True)
    ii_row: Tensor = torch.searchsorted(unique_kf, ii)
    jj_row: Tensor = torch.searchsorted(unique_kf, jj)
    ii_opt: Tensor = (ii_row - NUM_FIX).cpu()
    jj_opt: Tensor = (jj_row - NUM_FIX).cpu()
    num_points: int = Xs.shape[1]
    num_edges: int = ii.shape[0]
    # About 2,048 blocks in all, and at least 8 points per thread.
    blocks_per_edge: int = max(1, min(-(-2048 // num_edges), -(-num_points // (THREADS * 8))))
    sums: Tensor = torch.empty(num_edges, SUMS, device=Twc.device)
    dx_host: Tensor = torch.zeros(Xs.shape[0] - NUM_FIX, 7)
    dx: Tensor = dx_host.to(Twc.device)
    for _ in range(max_iter):
        sums.zero_()
        launch(step, (blocks_per_edge, *scalars), (Twc, Xs, Cs, K, ii_row, jj_row, idx_ii2jj, valid_match, Q, sums))
        launch("gn_solve", (), (sums.cpu(), ii_opt, jj_opt, dx_host))
        dx = dx_host.to(Twc.device)
        pose_retr(Twc, dx, NUM_FIX)
        if float(dx_host.norm()) < delta_thresh:
            break
    return dx


def gauss_newton_rays(
    Twc: Float32[Tensor, "p 8"],
    Xs: Float32[Tensor, "p n 3"],
    Cs: Float32[Tensor, "p n 1"],
    ii: Int64[Tensor, "e"],
    jj: Int64[Tensor, "e"],
    idx_ii2jj: Int64[Tensor, "e n"],
    valid_match: Bool[Tensor, "e n 1"],
    Q: Float32[Tensor, "e n 1"],
    sigma_ray: float,
    sigma_dist: float,
    C_thresh: float,
    Q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> Float32[Tensor, "p_opt 7"]:
    """Optimise Twc (all poses but the first) on the ray-direction and distance residuals of the matches; returns
    the last increment."""
    # Rays have no intrinsics: Twc stands in for K, whose pointer the rays kernel never reads.
    scalars = (0, 0, 0, sigma_ray, sigma_dist, C_thresh, Q_thresh, 0.0)
    return _gauss_newton("gn_rays_step", scalars, Twc, (Xs, Cs, Twc, idx_ii2jj, valid_match, Q), ii, jj, max_iter, delta_thresh)


def track_pose_rays(
    Twc: Float32[Tensor, "2 8"],
    Xs: Float32[Tensor, "2 n 3"],
    valid: Bool[Tensor, "1 n 1"],
    Q: Float32[Tensor, "1 n 1"],
    sigma_ray: float,
    sigma_dist: float,
    max_iter: int,
    delta_thresh: float,
) -> int:
    """Tracking's pose solve (``FrameTracker.opt_pose_ray_dist_sim3``) on the rays kernels: a two-pose problem.

    Pose 0 is the keyframe, fixed; pose 1 is the frame relative to it, optimised in place. Row k of ``Xs[1]`` is
    the frame point matched to keyframe point ``Xs[0][k]``; ``valid`` already holds the confidence and quality
    tests, so the kernel's own thresholds are switched off. Returns the number of iterations run.

    Raises:
        torch.linalg.LinAlgError: When the system is not positive definite, as ``torch.linalg.cholesky`` did.
    """
    device: torch.device = Twc.device
    num_points: int = Xs.shape[1]
    rows: Tensor = torch.arange(2, device=device)
    idx: Tensor = torch.arange(num_points, device=device).view(1, num_points)
    cs: Tensor = Xs.new_ones(2, num_points, 1)
    ii_opt: Tensor = torch.tensor([-NUM_FIX])
    jj_opt: Tensor = torch.tensor([0])
    blocks: int = max(1, min(2048, -(-num_points // (THREADS * 8))))
    sums: Tensor = torch.empty(1, SUMS, device=device)
    dx_host: Tensor = torch.zeros(1, 7)
    scalars: tuple[int | float, ...] = (blocks, 0, 0, 0, sigma_ray, sigma_dist, 0.0, -1.0, 0.0)
    for iteration in range(max_iter):
        sums.zero_()
        launch("gn_rays_step", scalars, (Twc, Xs, cs, Twc, rows[:1], rows[1:], idx, valid, Q, sums))
        if not launch("gn_solve", (), (sums.cpu(), ii_opt, jj_opt, dx_host)):
            raise torch.linalg.LinAlgError("tracking pose solve: the system is not positive definite")
        pose_retr(Twc, dx_host.to(device), NUM_FIX)
        if float(dx_host.norm()) < delta_thresh:
            return iteration + 1
    return max_iter


def gauss_newton_calib(
    Twc: Float32[Tensor, "p 8"],
    Xs: Float32[Tensor, "p n 3"],
    Cs: Float32[Tensor, "p n 1"],
    K: Float32[Tensor, "3 3"],
    ii: Int64[Tensor, "e"],
    jj: Int64[Tensor, "e"],
    idx_ii2jj: Int64[Tensor, "e n"],
    valid_match: Bool[Tensor, "e n 1"],
    Q: Float32[Tensor, "e n 1"],
    height: int,
    width: int,
    pixel_border: int,
    z_eps: float,
    sigma_pixel: float,
    sigma_depth: float,
    C_thresh: float,
    Q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> Float32[Tensor, "p_opt 7"]:
    """Optimise Twc (all poses but the first) on the pixel and log-depth residuals of the matches under the
    pinhole K; returns the last increment."""
    scalars = (height, width, pixel_border, sigma_pixel, sigma_depth, C_thresh, Q_thresh, z_eps)
    return _gauss_newton("gn_calib_step", scalars, Twc, (Xs, Cs, K.contiguous(), idx_ii2jj, valid_match, Q), ii, jj, max_iter, delta_thresh)
