"""Bundle adjustment for multi-camera calibration: ``refine_rig`` and the undistort-then-refine rounds around it
(``refine_through_lenses``)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from simplecv.ops.triangulate import triangulate_weighted
from simplecv.types.rig import RigCameras
from torch import Tensor

from exo_calib.ops.correspondences import ObservationSet, build_observations, observations_at, under_supported_view_indices
from exo_calib.ops.eval import camera_centers, rig_geometry
from exo_calib.ops.keypoints import undistort_views
from exo_calib.ops.triangulation import (
    RobustTriangulationResult,
    reprojection_errors_px,
    triangulate_points,
    triangulate_robust,
    valid_observation_mask,
)
from exo_calib.types import PINHOLE, Lens


@dataclass(slots=True, frozen=True)
class PosePrior:
    """Soft anchors on the camera centres.

    Reprojection-only BA leaves a 7-DOF similarity gauge (fixing one pose still
    leaves global scale free); weak centre priors pin every gauge direction
    without fighting real signal, and keep the refinement a *refinement* of the
    Stage A metric rig.
    """

    centers: Float64[ndarray, "v 3"]
    """Float64 world-frame centres the cameras are anchored to, shape ``(v, 3)``."""
    sigma_m: float
    """Prior standard deviation in metres, shared by every camera."""


def mean_reprojection_error_px(
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n_points 3"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
) -> float:
    """Compute confidence-weighted mean reprojection error in pixels.

    Args:
        obs: Sparse pixel observations for the points.
        points_xyz: Float64 world points with shape ``(n_points, 3)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Confidence-weighted mean pixel error, or NaN when no observation is valid.
    """
    reprojection_error: Float64[ndarray, "n_obs"] = reprojection_errors_px(obs, points_xyz, intrinsics, cam_T_world)
    finite_mask: Bool[ndarray, "n_obs"] = np.isfinite(reprojection_error)
    if not finite_mask.any():
        return float("nan")
    weights: Float64[ndarray, "r"] = obs.obs_conf[finite_mask]
    return float(np.sum(weights * reprojection_error[finite_mask]) / np.sum(weights))


@dataclass(slots=True, frozen=True)
class RigFit:
    """A refined rig: poses and one focal scale shared by every camera."""

    cam_T_world: Float64[ndarray, "v 4 4"]
    """Refined Float64 world-to-camera transforms with shape ``(v, 4, 4)``."""
    intrinsics: Float64[ndarray, "v 3 3"]
    """The input intrinsics with both focal axes times ``focal_scale``."""
    focal_scale: float
    """The one factor every camera's focal was multiplied by."""


MAX_LOG_FOCAL_CHANGE: float = 0.5
"""Bound on the shared log-focal update (a factor of 0.61-1.65): Stage A's focal on a fisheye rig was up to 40 % off."""


def skew(w: Float64[Tensor, "v 3"]) -> Float64[Tensor, "v 3 3"]:
    zero: Float64[Tensor, " v"] = torch.zeros_like(w[:, 0])
    return torch.stack((zero, -w[:, 2], w[:, 1], w[:, 2], zero, -w[:, 0], -w[:, 1], w[:, 0], zero), dim=1).reshape(-1, 3, 3)


def refine_rig(
    cam_T_world: Float64[ndarray, "v 4 4"],
    intrinsics: Float64[ndarray, "v 3 3"],
    obs: ObservationSet,
    *,
    pose_prior: PosePrior | None,
    robust_scale_px: float,
    fixed_pose_indices: tuple[int, ...] = (),
    max_iterations: int = 30,
    device: Literal["cuda", "cpu"] = "cuda",
) -> RigFit:
    """Refine every free camera's pose and one shared focal scale by Levenberg-Marquardt on the Cauchy reprojection error.

    The points are not unknowns: every evaluation re-triangulates them from the current rig (the weighted DLT of
    ``triangulate_points``), so only ``6v + 1`` parameters remain and one forward-mode ``jacfwd`` gives the Jacobian. A
    focal moving with the points held fixed only trades against them; this way the points follow the focal.

    Args:
        cam_T_world: Float64 initial world-to-camera transforms with shape ``(v, 4, 4)``.
        intrinsics: Float64 initial intrinsics with shape ``(v, 3, 3)``; their focal ratios are kept.
        obs: Pinhole-pixel observations (undistort raw pixels first, see ``exo_calib.ops.lens``).
        pose_prior: Soft camera-centre anchors; ``None`` leaves the similarity gauge free.
        robust_scale_px: Cauchy scale in pixels.
        fixed_pose_indices: Cameras whose pose stays as given.
        max_iterations: Levenberg-Marquardt iteration cap.
        device: Torch device the solve runs on.

    Returns:
        The refined poses, the scaled intrinsics and the shared focal scale.
    """
    dtype: torch.dtype = torch.float64
    n_views, n_points = cam_T_world.shape[0], obs.point_frame_idx.size
    free: list[int] = [v for v in range(n_views) if v not in fixed_pose_indices]
    valid: Bool[ndarray, " n_obs"] = valid_observation_mask(obs, n_views, n_points=n_points)
    view: Int64[Tensor, " m"] = torch.as_tensor(obs.obs_view_idx[valid], device=device)
    point: Int64[Tensor, " m"] = torch.as_tensor(obs.obs_point_idx[valid], device=device)
    observed: Float64[Tensor, "m 2"] = torch.as_tensor(obs.obs_xy[valid], dtype=dtype, device=device)
    conf: Float64[Tensor, " m"] = torch.as_tensor(obs.obs_conf[valid], dtype=dtype, device=device)
    root_conf: Float64[Tensor, "m 1"] = conf.sqrt()[:, None]
    grid: Float64[Tensor, "v n 2"] = torch.full((n_views, n_points, 2), torch.nan, dtype=dtype, device=device)
    grid[view, point] = observed
    weight: Float64[Tensor, "v n"] = torch.zeros((n_views, n_points), dtype=dtype, device=device)
    weight[view, point] = conf**2  # triangulate_points' weight
    rotation0: Float64[Tensor, "v 3 3"] = torch.as_tensor(cam_T_world[:, :3, :3], dtype=dtype, device=device)
    center0: Float64[Tensor, "v 3"] = torch.as_tensor(camera_centers(cam_T_world), dtype=dtype, device=device)
    K0: Float64[Tensor, "v 3 3"] = torch.as_tensor(intrinsics, dtype=dtype, device=device)
    focal_block: Float64[Tensor, "3 3"] = torch.zeros((3, 3), dtype=dtype, device=device)
    focal_block[:2, :2] = 1.0  # fx, skew, fy; the principal point stays
    free_mask: Float64[Tensor, "v 1"] = torch.zeros((n_views, 1), dtype=dtype, device=device)
    free_mask[free] = 1.0
    free_index: Int64[Tensor, " v"] = torch.as_tensor([free.index(v) if v in free else 0 for v in range(n_views)], device=device)

    def rig(x: Float64[Tensor, " p"]) -> tuple[Float64[Tensor, "v 3 4"], Float64[Tensor, "v 3 3"], Float64[Tensor, "v 3"]]:
        pose: Float64[Tensor, "v 6"] = x[:-1].reshape(-1, 6)[free_index] * free_mask
        rotation: Float64[Tensor, "v 3 3"] = torch.linalg.matrix_exp(skew(pose[:, :3])) @ rotation0
        center: Float64[Tensor, "v 3"] = center0 + pose[:, 3:]
        world_to_cam: Float64[Tensor, "v 3 4"] = torch.cat((rotation, -(rotation @ center[:, :, None])), dim=2)
        K: Float64[Tensor, "v 3 3"] = K0 * (1.0 + (torch.exp(x[-1]) - 1.0) * focal_block)
        return world_to_cam, K, pose[:, 3:]

    def residuals(x: Float64[Tensor, " p"]) -> Float64[Tensor, " r"]:
        world_to_cam, K, center_shift = rig(x)
        points, ok = triangulate_weighted(grid, weight, K @ world_to_cam)
        homogeneous: Float64[Tensor, "m 4"] = torch.cat((torch.where(ok[:, None], points, 0.0)[point], torch.ones_like(observed[:, :1])), dim=1)
        in_camera: Float64[Tensor, "m 3"] = torch.einsum("mij,mj->mi", world_to_cam[view], homogeneous)
        projected: Float64[Tensor, "m 3"] = torch.einsum("mij,mj->mi", K[view], in_camera)
        usable: Bool[Tensor, "m 1"] = (ok[point] & (in_camera[:, 2] > 0.0))[:, None]
        error: Float64[Tensor, "m 2"] = torch.where(usable, projected[:, :2] / projected[:, 2:] - observed, 0.0) * root_conf
        prior: Float64[Tensor, " q"] = (center_shift[free] / pose_prior.sigma_m * robust_scale_px).reshape(-1) if pose_prior else error[:0, 0]
        return torch.cat((error.reshape(-1), prior))

    n_error: int = 2 * int(view.numel())

    def cost(r: Float64[Tensor, " r"]) -> Float64[Tensor, ""]:
        squared: Float64[Tensor, " m"] = r[:n_error].reshape(-1, 2).pow(2).sum(dim=1)
        return robust_scale_px**2 * torch.log1p(squared / robust_scale_px**2).sum() + r[n_error:].pow(2).sum()

    x: Float64[Tensor, " p"] = torch.zeros(6 * len(free) + 1, dtype=dtype, device=device)
    r: Float64[Tensor, " r"] = residuals(x)
    current: Float64[Tensor, ""] = cost(r)
    damping: float = 1e-3
    for _iteration in range(max_iterations):
        jacobian: Float64[Tensor, "r p"] = torch.func.jacfwd(residuals)(x)
        irls: Float64[Tensor, " r"] = torch.ones_like(r)  # Cauchy as reweighted least squares; the prior stays quadratic
        irls[:n_error] = (1.0 / (1.0 + r[:n_error].reshape(-1, 2).pow(2).sum(dim=1) / robust_scale_px**2)).repeat_interleave(2)
        normal: Float64[Tensor, "p p"] = jacobian.T @ (irls[:, None] * jacobian)
        gradient: Float64[Tensor, " p"] = jacobian.T @ (irls * r)
        for _attempt in range(10):
            candidate: Float64[Tensor, " p"] = x - torch.linalg.solve(normal + damping * torch.diag(normal.diagonal().clamp_min(1e-9)), gradient)
            candidate[-1] = candidate[-1].clamp(-MAX_LOG_FOCAL_CHANGE, MAX_LOG_FOCAL_CHANGE)
            candidate_r: Float64[Tensor, " r"] = residuals(candidate)
            if (candidate_cost := cost(candidate_r)) < current:
                break
            damping *= 4.0
        else:
            break  # no step lowers the cost: converged
        improvement: float = float((current - candidate_cost) / current)
        x, r, current, damping = candidate, candidate_r, candidate_cost, max(damping / 3.0, 1e-7)
        if improvement < 1e-7:
            break
    world_to_cam, K, _ = rig(x)
    refined: Float64[ndarray, "v 4 4"] = np.repeat(np.eye(4)[None], n_views, axis=0)
    refined[:, :3] = world_to_cam.detach().cpu().numpy()
    refined[list(fixed_pose_indices)] = cam_T_world[list(fixed_pose_indices)]  # bit-exact, not exp(0) @ R
    return RigFit(refined, K.detach().cpu().numpy(), float(torch.exp(x[-1])))


def one_focal_per_camera_model(intrinsics: Float64[ndarray, "v 3 3"], resolution_wh: Int64[ndarray, "v 2"]) -> Float64[ndarray, "v 3 3"]:
    """Every camera of one model (one resolution) at its cameras' median focal, the principal points kept (Kineo's shared
    intrinsics). Stage A's per-camera focals differ by 1-5 % within a model for no physical reason, and ``refine_rig`` keeps
    their ratios; one focal per model starts the shared focal scale where a single value can reach (focal error 1-4 % to
    0.2-1 % on the tune segments). The resolution stands for the model: two phone models that record at one resolution get
    one focal, and that ratio stays."""
    shared: Float64[ndarray, "v 3 3"] = intrinsics.copy()
    for resolution in np.unique(resolution_wh, axis=0):
        same: Bool[ndarray, " v"] = (resolution_wh == resolution).all(axis=1)
        shared[same, 0, 0] = shared[same, 1, 1] = float(np.median(intrinsics[same][:, [0, 1], [0, 1]]))
    return shared


MIN_PAIR_CONFIDENCE: float = 0.90
"""Kineo pair rule: geometric-mean confidence a view pair must exceed (Kineo's default is 0.75)."""
MIN_VIEWS: int = 2
"""Minimum observing cameras per correspondence point."""
REPROJECTION_THRESHOLD_PX: float = 100.0
"""Per-observation gate for robust triangulation; wide, so the Cauchy refinement sees every useful observation."""
ROBUST_SCALE_PX: float = 1.0
LENS_ROUNDS: int = 3
"""Most undistort-then-refine rounds when a camera has a lens: undistortion needs the focal the refinement moves (Ego-Exo4D's
GoPros settle by the third)."""
FOCAL_SETTLED: float = 1e-3
"""A round that moves the focal less than this ends the rounds: the next would undistort the same pixels."""
POSE_PRIOR_SIGMA_RATIO: float = 0.06
"""Soft prior anchoring camera centers to Stage A, as a fraction of the rig's median pairwise camera distance;
pins the similarity gauge without freezing rigs of any size."""
MINIMUM_VIEW_SUPPORT_RATIO: float = 0.10
"""Fix views whose robust-observation count is below this fraction of the median count."""


@dataclass(slots=True, frozen=True)
class LensRefinement:
    """A rig refined through its lenses, with the observations it kept and their points."""

    intrinsics: Float64[ndarray, "v 3 3"]
    cam_T_world: Float64[ndarray, "v 4 4"]
    obs: ObservationSet
    """The last round's robust observations, at ``xy``."""
    xy: Float64[ndarray, "v t 133 2"]
    """Every view's keypoints in its pinhole pixels at the final focal."""
    points_xyz: Float64[ndarray, "n 3"]
    """``obs`` triangulated through the refined rig."""
    init_reprojection_px: float
    """The first round's observations reprojected through the Stage A rig."""
    ba_reprojection_px: list[float]
    """After each round's refinement."""


def refine_through_lenses(
    lenses: Sequence[Lens],
    kp_xy: Float64[ndarray, "v t 133 2"],
    conf: Float64[ndarray, "v t 133"],
    init: RigCameras,
    *,
    device: Literal["cuda", "cpu"],
) -> LensRefinement:
    """Refine every pose and one focal scale of the Stage A rig ``init`` from multi-view keypoints (raw pixels, through each
    camera's lens; ``conf`` 0 = unused): Kineo-style correspondences, robust triangulation against ``init``, then ``refine_rig``
    from one focal per camera model, with the camera centres softly held at Stage A's. A rig with a lens repeats
    undistort-then-refine with the new focal until the focal settles (``FOCAL_SETTLED``) or ``LENS_ROUNDS`` ran."""
    pairwise: Float64[ndarray, "v v"] = rig_geometry(init.cam_T_world).pairwise_distance_m
    prior = PosePrior(centers=camera_centers(init.cam_T_world), sigma_m=POSE_PRIOR_SIGMA_RATIO * float(np.median(pairwise[np.triu_indices(len(lenses), k=1)])))
    has_lens: bool = any(lens != PINHOLE for lens in lenses)
    K: Float64[ndarray, "v 3 3"] = one_focal_per_camera_model(init.intrinsics, init.resolution_wh)
    before_px: list[float] = []  # each round's reprojection on the Stage A rig, then after its refinement
    after_px: list[float] = []
    for round_index in range(LENS_ROUNDS if has_lens else 1):
        xy: Float64[ndarray, "v t 133 2"] = undistort_views(lenses, kp_xy, K)
        candidates: ObservationSet = build_observations(xy, conf, min_pair_conf=MIN_PAIR_CONFIDENCE, min_views=MIN_VIEWS)
        robust: RobustTriangulationResult = triangulate_robust(candidates, K, init.cam_T_world, reproj_threshold_px=REPROJECTION_THRESHOLD_PX)
        obs: ObservationSet = robust.obs
        before_px.append(mean_reprojection_error_px(obs, robust.points_xyz, K, init.cam_T_world))
        fixed: tuple[int, ...] = under_supported_view_indices(obs.obs_view_idx, len(lenses), MINIMUM_VIEW_SUPPORT_RATIO)
        fit: RigFit = refine_rig(init.cam_T_world, K, obs, pose_prior=prior, robust_scale_px=ROBUST_SCALE_PX, fixed_pose_indices=fixed, device=device)
        K = fit.intrinsics
        after_px.append(mean_reprojection_error_px(obs, triangulate_points(obs, K, fit.cam_T_world), K, fit.cam_T_world))
        print(
            f"round {round_index}: kept {obs.obs_point_idx.size}/{candidates.obs_point_idx.size} observations; reprojection {before_px[-1]:.2f} -> "
            f"{after_px[-1]:.2f} px; focal scale {K[0, 0, 0] / init.intrinsics[0, 0, 0]:.4f}"
        )
        if abs(fit.focal_scale - 1.0) < FOCAL_SETTLED:
            break
    if has_lens:  # the pixels of the final focal
        xy = undistort_views(lenses, kp_xy, K)
        obs = observations_at(xy, obs)
    return LensRefinement(K, fit.cam_T_world, obs, xy, triangulate_points(obs, K, fit.cam_T_world), before_px[0], after_px)
