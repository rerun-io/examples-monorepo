"""The model-based pose fit of MEgATrack §3.5: Levenberg–Marquardt over θ on the UmeTrack hand model.

The energy E = E_2D + w_1·E_dist + w_2·E_temporal and its Jacobian live in ``handtrack.fit.solver``. This module holds the
per-frame solve: a batched LM with one damping per hand, the joint angles kept inside the model's ``joint_limits`` (plus
a 5° margin) by projected steps with an active set, and ``initial_pose`` for hands with no previous pose.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import permutations, product
from typing import Literal, TypeAlias

import torch
from jaxtyping import Bool, Float32, Float64, Int64
from simplecv.umetrack_temp.generic_hand_model_torch import LANDMARK, HandModelTorch
from torch import Tensor

from handtrack.fit.observations import MAX_VIEWS, HandObservation
from handtrack.fit.solver import (
    DAMPING_GROWTH,
    FIT_JOINTS,
    MAX_DAMPING,
    MIN_DAMPING_FACTOR,
    PARAMETERS,
    POSE_PARAMETERS,
    Prior,
    Problem,
    Theta,
    Views,
    energy_terms,
    fit_limits,
    linearize,
    local_landmarks,
    marquardt_scaling,
    no_prior,
    normal_equations,
    orthonormalize,
    poses_from_theta,
    predicted_reduction,
    repeat_views,
    retract,
    select,
    stack_views,
    take,
    theta_from_poses,
)
from handtrack.geometry.camera import CameraRig, project
from handtrack.hand.pose import HandPose

JOINT_LIMIT_MARGIN_RAD: float = math.radians(5.0)
"""UmeTrack's ground truth exceeds the model's ``joint_limits`` by up to exactly 5° (7% of joint angles on a testing recording)."""
PALM: tuple[int, ...] = (
    LANDMARK.WRIST_JOINT,
    LANDMARK.INDEX_PROXIMAL_FRAME,
    LANDMARK.MIDDLE_PROXIMAL_FRAME,
    LANDMARK.RING_PROXIMAL_FRAME,
    LANDMARK.PINKY_PROXIMAL_FRAME,
    LANDMARK.PALM_CENTER,
)
"""Wrist, the four finger MCPs and the palm centre: fixed in the wrist frame (the palm centre moves < 3 mm)."""
_PALM_MASK: Float32[Tensor, "21"] = torch.zeros(21).index_fill(0, torch.tensor([int(i) for i in PALM]), 1.0)
"""1 at the ``PALM`` keypoints: multiply a per-keypoint weight by it to keep only the palm."""
MIN_ALIGN_POINTS: int = 3
"""Keypoints a view needs before its palm is aligned rigidly (Kabsch needs three)."""


def _cube_rotations() -> Float32[Tensor, "24 3 3"]:
    """The 24 rotations that map the cube onto itself: the signed permutation matrices with determinant +1."""
    signed: list[Float32[Tensor, "3 3"]] = [
        torch.eye(3)[list(order)] * torch.tensor(signs)[:, None] for order in permutations(range(3)) for signs in product((1.0, -1.0), repeat=3)
    ]
    return torch.stack([matrix for matrix in signed if torch.linalg.det(matrix) > 0])


_CUBE_ROTATIONS: Float32[Tensor, "24 3 3"] = _cube_rotations()
"""``initial_pose`` tries the first ``rotation_hypotheses`` of them, in this order."""


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Weights and solver settings of the pose fit."""

    dist_weight: float = 0.04
    """w_1 on E_dist, whose residuals are millimetres (E_2D's are pixels): 5 mm of d_rel error weighs as much as 1 px."""
    temporal_weight: float = 20.0
    """w_2 on E_temporal: rotation and joint angles in radians, translation in ``temporal_translation_unit_m``."""
    temporal_translation_unit_m: float = 0.1
    """Decimetres: 1 dm of wrist travel weighs as much as 1 rad of rotation, and either moves the fingertips about 1 dm.

    On six real testing recordings at 30 fps (1.5 px, 5 mm d_rel noise) decimetres give the lowest MKPE; centimetres lag
    fast hands (15.9 vs 6.3 mm MKPE in one view), metres smooth less.
    """
    joint_limit_margin_rad: float = JOINT_LIMIT_MARGIN_RAD
    """The joint limits are widened by this much on both sides."""
    max_iterations: int = 10
    """LM iterations from a previous pose (it converges in 3-4 on real motion at 30 fps)."""
    init_iterations: int = 40
    """LM iterations for each stage of ``initial_pose``."""
    relative_tolerance: float = 1e-3
    """Stop when an accepted step lowers the energy by less than this fraction of it (or by less than ``absolute_tolerance``)."""
    init_relative_tolerance: float = 1e-6
    """The same for the stages of ``initial_pose``, which start far from the solution."""
    absolute_tolerance: float = 1e-6
    """In the energy's units (pixels²)."""
    initial_damping: float = 1e-3
    """λ at the first iteration, relative to diag(JᵀJ) (Marquardt scaling)."""
    rotation_hypotheses: int = 24
    """Wrist rotations tried by ``initial_pose`` besides the per-view palm alignment (the rotation group of the cube; 0 = none).

    They matter only when d_rel is poor (40 mm of noise, one view: 197 vs 185 of 200 acquisitions) and double the cost.
    """
    full_fit_hypotheses: int = 2
    """How many of the best palm fits ``initial_pose`` refines over all 26 degrees of freedom."""
    finger_starts: int = 2
    """Finger poses each refined palm fit starts from: 1 = the neutral pose, 2 = also the open hand."""


DEFAULT_CONFIG: FitConfig = FitConfig()

Termination: TypeAlias = Literal["tolerance", "iterations", "damping", "no_evidence", "non_finite"]


@dataclass(frozen=True, slots=True)
class FitResult:
    """The fitted θ of one hand and its energy terms (unweighted)."""

    pose: HandPose
    """Unbatched θ."""
    e_2d: float
    """Pixels²."""
    e_dist: float
    """Millimetres²."""
    e_temporal: float
    """0 when there was no previous pose."""
    energy: float
    """E_2D + w_1·E_dist + w_2·E_temporal."""
    iterations: int
    """LM iterations of the final solve."""
    converged: bool
    """True only when the convergence tolerance was reached."""
    termination: Termination = "iterations"
    """Why the solve stopped; acquisition without evidence returns a neutral pose and NaN energy."""


@dataclass(frozen=True, slots=True)
class _Solved:
    """One batched LM solve: the final θ, its weighted energy, the iterations run and whether the stopping rule fired."""

    theta: Theta
    energy: Float32[Tensor, "b"]
    iterations: Int64[Tensor, "b"]
    converged: Bool[Tensor, "b"]
    termination: list[Termination]


def _problem(model: HandModelTorch, phi: float, views: Views, prior: Prior, config: FitConfig) -> Problem:
    return Problem(
        model=model,
        views=views,
        prior=prior,
        phi=phi,
        dist_weight=config.dist_weight,
        temporal_weight=config.temporal_weight,
        temporal_translation_unit_m=config.temporal_translation_unit_m,
    )


def _levenberg_marquardt(problem: Problem, theta: Theta, free: Int64[Tensor, "k"], config: FitConfig) -> _Solved:
    """Batched LM, one damping per hand: Marquardt scaling, Nielsen's damping update, projected steps with an active set.

    At most ``config.max_iterations``, stopping at ``config.relative_tolerance``. Each iteration is one batched linearisation,
    at the candidate; a rejected candidate keeps the previous one.
    """
    limits: Float32[Tensor, "20 2"] = fit_limits(problem.model, config.joint_limit_margin_rad)
    theta = replace(theta, angles=theta.angles.clamp(limits[:, 0], limits[:, 1]))
    b: int = theta.translation.shape[0]
    residual, jacobian = linearize(problem, theta, free)
    energy: Float32[Tensor, "b"] = (residual * residual).sum(-1)
    damping: Float64[Tensor, "b"] = torch.full((b,), config.initial_damping, dtype=torch.float64)
    growth: Float64[Tensor, "b"] = torch.full((b,), DAMPING_GROWTH, dtype=torch.float64)
    non_finite: Bool[Tensor, "b"] = ~torch.isfinite(energy)
    done: Bool[Tensor, "b"] = non_finite.clone()
    converged: Bool[Tensor, "b"] = torch.zeros(b, dtype=torch.bool)
    count: Int64[Tensor, "b"] = torch.zeros(b, dtype=torch.int64)
    for _ in range(config.max_iterations):
        if bool(done.all()):
            break
        # Failed rows must not poison the independent solves of the remaining hands.
        residual = torch.where(done[:, None], 0.0, residual)
        jacobian = torch.where(done[:, None, None], 0.0, jacobian)
        hessian, gradient, mask = normal_equations(theta, residual, jacobian, free, limits)
        scaling: Float64[Tensor, "b k"] = marquardt_scaling(torch.diagonal(hessian, dim1=1, dim2=2), mask)
        system: Float64[Tensor, "b k k"] = hessian + torch.diag_embed(damping[:, None] * scaling + (1.0 - mask))
        step: Float64[Tensor, "b k"] = -torch.linalg.solve(system, gradient)
        full_step: Float32[Tensor, "b 27"] = torch.zeros((b, PARAMETERS), dtype=torch.float32)
        full_step[:, free] = step.to(torch.float32)
        candidate: Theta = retract(theta, full_step, limits)
        predicted: Float64[Tensor, "b"] = predicted_reduction(theta, candidate, free, step, hessian, gradient)
        new_residual, new_jacobian = linearize(problem, candidate, free)
        new_energy: Float32[Tensor, "b"] = (new_residual * new_residual).sum(-1)
        reduction: Float32[Tensor, "b"] = energy - new_energy
        failed: Bool[Tensor, "b"] = ~torch.isfinite(new_energy) & ~done
        non_finite = non_finite | failed
        accept: Bool[Tensor, "b"] = (reduction > 0) & ~done & ~failed
        ratio: Float64[Tensor, "b"] = reduction.to(torch.float64) / predicted.clamp(min=1e-30)
        small: Bool[Tensor, "b"] = reduction <= config.relative_tolerance * energy + config.absolute_tolerance
        count = count + (~done).to(torch.int64)
        theta = select(accept, candidate, theta)
        residual = torch.where(accept[:, None], new_residual, residual)
        jacobian = torch.where(accept[:, None, None], new_jacobian, jacobian)
        energy = torch.where(accept, new_energy, energy)
        damping = torch.where(accept, damping * torch.clamp(1.0 - (2.0 * ratio - 1.0) ** 3, min=MIN_DAMPING_FACTOR), damping * growth)
        growth = torch.where(accept, torch.full_like(growth, DAMPING_GROWTH), growth * DAMPING_GROWTH)
        converged = converged | (accept & small)
        done = done | converged | non_finite | (damping > MAX_DAMPING)
        if bool(done.all()):
            break
    termination: list[Termination] = [
        "non_finite" if non_finite[i] else "tolerance" if converged[i] else "damping" if done[i] else "iterations" for i in range(b)
    ]
    return _Solved(theta=theta, energy=energy, iterations=count, converged=converged, termination=termination)


def _results(problem: Problem, solved: _Solved, carried: Float32[Tensor, "b 2"]) -> list[FitResult]:
    rotation: Float32[Tensor, "b 3 3"] = solved.theta.rotation.clone()
    finite_rotation: Bool[Tensor, "b"] = torch.isfinite(rotation).all(dim=(1, 2))
    rotation[finite_rotation] = orthonormalize(rotation[finite_rotation])
    theta: Theta = replace(solved.theta, rotation=rotation)
    e_2d, e_dist, e_temporal = energy_terms(problem, theta)
    energy: Float32[Tensor, "b"] = e_2d + problem.dist_weight * e_dist + problem.temporal_weight * e_temporal
    return [
        FitResult(
            pose=pose,
            e_2d=float(e_2d[i]),
            e_dist=float(e_dist[i]),
            e_temporal=float(e_temporal[i]),
            energy=float(energy[i]),
            iterations=int(solved.iterations[i]),
            converged=bool(solved.converged[i]) and bool(torch.isfinite(energy[i])),
            termination=solved.termination[i] if torch.isfinite(energy[i]) else "non_finite",
        )
        for i, pose in enumerate(poses_from_theta(theta, carried))
    ]


def fit_pose(
    model: HandModelTorch, phi: float, hands: Sequence[HandObservation], previous: Sequence[HandPose | None], config: FitConfig = DEFAULT_CONFIG
) -> list[FitResult]:
    """Fit θ for each hand of one frame, in one batch.

    Args:
        model: The hand model, already at the subject's size: the profile for a known hand, ``scaled_hand_model(generic, ϕ)`` otherwise.
        phi: ϕ of ``model`` relative to the generic hand; it converts KeyNet's d_rel back to millimetres in E_dist.
        hands: One observation per hand.
        previous: θ(t−1) per hand, or None. A previous pose is the start and the E_temporal target; without one the hand
            starts from ``initial_pose`` and has no temporal term.
        config: Weights and solver settings.

    Returns:
        One result per hand, in order.
    """
    if len(hands) != len(previous):
        raise ValueError(f"{len(hands)} hands but {len(previous)} previous poses")
    results: dict[int, FitResult] = {}
    warm: list[int] = [i for i, pose in enumerate(previous) if pose is not None]
    cold: list[int] = [i for i, pose in enumerate(previous) if pose is None]
    if warm:
        poses: list[HandPose] = [pose for pose in previous if pose is not None]
        start: Theta = theta_from_poses(poses)
        prior: Prior = Prior(rotation=start.rotation, translation=start.translation, angles=start.angles, present=torch.ones(len(warm)))
        problem: Problem = _problem(model, phi, stack_views([hands[i] for i in warm]), prior, config)
        solved: _Solved = _levenberg_marquardt(problem, start, torch.arange(POSE_PARAMETERS), config)
        carried: Float32[Tensor, "b 2"] = torch.stack([pose.joint_angles[FIT_JOINTS:] for pose in poses])
        results.update(zip(warm, _results(problem, solved, carried), strict=True))
    if cold:
        results.update(zip(cold, initial_pose(model, phi, [hands[i] for i in cold], config), strict=True))
    return [results[i] for i in range(len(hands))]


def _direction(theta_xy: Float32[Tensor, "*batch 2"]) -> Float32[Tensor, "*batch 3"]:
    """The unit ray whose angle-scaled coordinates (Fisheye62's θ-normalised point) are ``theta_xy``."""
    angle: Float32[Tensor, "*batch"] = (theta_xy * theta_xy).sum(-1).add(1e-12).sqrt()
    return torch.cat([theta_xy * (torch.sin(angle) / angle)[..., None], torch.cos(angle)[..., None]], dim=-1)


def _unproject(cameras: CameraRig, pixels: Float32[Tensor, "c n 2"]) -> Float32[Tensor, "c n 3"]:
    """Unit rays in each camera's frame; Fisheye62 is inverted by Newton's method on ``project`` (central-difference 2x2 Jacobian)."""
    normalized: Float32[Tensor, "c n 2"] = (pixels - cameras.principal[:, None, :]) / cameras.focal[:, None, :]
    if cameras.fisheye62 is None:
        rays: Float32[Tensor, "c n 3"] = torch.cat([normalized, torch.ones_like(normalized[..., :1])], dim=-1)
        return rays / rays.norm(dim=-1, keepdim=True)
    step: float = 1e-3
    offsets: Float32[Tensor, "5 1 1 2"] = torch.tensor([[0.0, 0.0], [step, 0.0], [0.0, step], [-step, 0.0], [0.0, -step]])[:, None, None, :]
    theta_xy: Float32[Tensor, "c n 2"] = normalized
    for _ in range(8):
        evaluated: Float32[Tensor, "5 c n 2"] = project(cameras, _direction(theta_xy + offsets))
        jacobian: Float32[Tensor, "c n 2 2"] = torch.stack([evaluated[1] - evaluated[3], evaluated[2] - evaluated[4]], dim=-1) / (2.0 * step)
        theta_xy = theta_xy - torch.linalg.solve(jacobian, (evaluated[0] - pixels)[..., None])[..., 0]
    return _direction(theta_xy)


def _kabsch(
    source: Float32[Tensor, "b n 3"], target: Float32[Tensor, "b n 3"], weights: Float32[Tensor, "b n"]
) -> tuple[Float32[Tensor, "b 3 3"], Float32[Tensor, "b 3"]]:
    """The rigid (R, t) minimising Σ w‖R·source + t − target‖²."""
    total: Float32[Tensor, "b 1"] = weights.sum(-1, keepdim=True).clamp(min=1e-9)
    source_mean: Float32[Tensor, "b 3"] = (weights[..., None] * source).sum(1) / total
    target_mean: Float32[Tensor, "b 3"] = (weights[..., None] * target).sum(1) / total
    covariance: Float32[Tensor, "b 3 3"] = torch.einsum("bn,bni,bnj->bij", weights, source - source_mean[:, None], target - target_mean[:, None])
    u, _, vh = torch.linalg.svd(covariance)
    sign: Float32[Tensor, "b"] = torch.sign(torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
    correction: Float32[Tensor, "b 3 3"] = torch.diag_embed(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1))
    rotation: Float32[Tensor, "b 3 3"] = vh.transpose(1, 2) @ correction @ u.transpose(1, 2)
    return rotation, target_mean - torch.einsum("bij,bj->bi", rotation, source_mean)


def _wrist_hypotheses(
    model: HandModelTorch, phi: float, views: Views, neutral: Float32[Tensor, "20"], hypotheses: int
) -> tuple[Theta, Bool[Tensor, "b h"]]:
    """World-from-wrist guesses per hand: the palm aligned to each view's rays placed at their d_rel distances, then a rotation grid.

    Each view's observed palm keypoints (all its observed keypoints when fewer than 3 palm points are seen) are unprojected
    to rays and placed at d̄ + ϕ·d̂_i along them, with d̄ chosen so their spread matches the model's palm; Kabsch aligns
    the neutral palm to them. The grid rotations keep the palm centroid of the best view.
    """
    b: int = views.mirror.shape[0]
    local: Float32[Tensor, "b 21 3"] = local_landmarks(model, neutral.expand(1, b, FIT_JOINTS), views.mirror)[0]
    observed: Float32[Tensor, "b 2 21"] = (views.weights > 0).to(torch.float32)
    unprojected: Float32[Tensor, "b 2 21 3"] = _unproject(views.cameras, views.keypoints_px.reshape(b * MAX_VIEWS, 21, 2)).reshape(
        b, MAX_VIEWS, 21, 3
    )
    rays: Float32[Tensor, "b 2 21 3"] = torch.nan_to_num(unprojected) * observed[..., None]
    palm: Float32[Tensor, "b 2 21"] = observed * _PALM_MASK
    enough_palm: Bool[Tensor, "b 2 1"] = palm.sum(-1, keepdim=True) >= MIN_ALIGN_POINTS
    used: Float32[Tensor, "b 2 21"] = torch.where(enough_palm, palm, observed)
    count: Float32[Tensor, "b 2"] = used.sum(-1)
    valid: Bool[Tensor, "b 2"] = count >= MIN_ALIGN_POINTS
    total: Float32[Tensor, "b 2 1"] = count[..., None].clamp(min=1.0)

    def centred(points: Float32[Tensor, "b 2 21 3"]) -> Float32[Tensor, "b 2 21 3"]:
        """``points`` minus their mean over the ``used`` keypoints of each view."""
        return points - (used[..., None] * points).sum(2, keepdim=True) / total[..., None]

    offsets: Float32[Tensor, "b 2 21 1"] = (phi * views.d_rel_mm / 1000.0)[..., None]
    a: Float32[Tensor, "b 2 21 3"] = centred(rays)
    c: Float32[Tensor, "b 2 21 3"] = centred(offsets * rays)
    model_points: Float32[Tensor, "b 2 21 3"] = local[:, None].expand(b, MAX_VIEWS, 21, 3)
    model_centred: Float32[Tensor, "b 2 21 3"] = centred(model_points)
    spread: Float32[Tensor, "b 2"] = (used * (model_centred * model_centred).sum(-1)).sum(-1)
    qa: Float32[Tensor, "b 2"] = (used * (a * a).sum(-1)).sum(-1).clamp(min=1e-12)
    qb: Float32[Tensor, "b 2"] = (used * (a * c).sum(-1)).sum(-1)
    qc: Float32[Tensor, "b 2"] = (used * (c * c).sum(-1)).sum(-1) - spread
    mean_distance: Float32[Tensor, "b 2"] = ((-qb + (qb * qb - qa * qc).clamp(min=0.0).sqrt()) / qa).clamp(0.1, 2.0)
    points_cam: Float32[Tensor, "b 2 21 3"] = (mean_distance[..., None, None] + offsets) * rays
    cam_rotation, cam_translation = _kabsch(model_points.reshape(-1, 21, 3), points_cam.reshape(-1, 21, 3), used.reshape(-1, 21))
    world_from_cam: Float32[Tensor, "b 2 3 3"] = views.cam_from_world[..., :3, :3].transpose(-1, -2)
    cam_origin: Float32[Tensor, "b 2 3"] = -torch.einsum("bvij,bvj->bvi", world_from_cam, views.cam_from_world[..., :3, 3])
    aligned_rotation: Float32[Tensor, "b 2 3 3"] = world_from_cam @ cam_rotation.reshape(b, MAX_VIEWS, 3, 3)
    aligned_translation: Float32[Tensor, "b 2 3"] = (
        torch.einsum("bvij,bvj->bvi", world_from_cam, cam_translation.reshape(b, MAX_VIEWS, 3)) + cam_origin
    )
    best_view: Int64[Tensor, "b"] = torch.where(valid, count, torch.full_like(count, -1.0)).argmax(-1)
    rows: Int64[Tensor, "b"] = torch.arange(b)
    anchor_cam: Float32[Tensor, "b 3"] = (used[rows, best_view][..., None] * points_cam[rows, best_view]).sum(1) / total[rows, best_view]
    anchor: Float32[Tensor, "b 3"] = torch.einsum("bij,bj->bi", world_from_cam[rows, best_view], anchor_cam) + cam_origin[rows, best_view]
    model_anchor: Float32[Tensor, "b 3"] = (used[rows, best_view][..., None] * local).sum(1) / total[rows, best_view]
    grid: Float32[Tensor, "g 3 3"] = _CUBE_ROTATIONS[:hypotheses]
    grid_translation: Float32[Tensor, "b g 3"] = anchor[:, None] - torch.einsum("gij,bj->bgi", grid, model_anchor)
    rotation: Float32[Tensor, "b h 3 3"] = torch.cat([aligned_rotation, grid.expand(b, -1, 3, 3)], dim=1)
    translation: Float32[Tensor, "b h 3"] = torch.cat([aligned_translation, grid_translation], dim=1)
    h: int = rotation.shape[1]
    usable: Bool[Tensor, "b h"] = torch.cat([valid, valid.any(-1, keepdim=True).expand(b, grid.shape[0])], dim=1)
    theta: Theta = Theta(
        rotation=rotation.reshape(b * h, 3, 3),
        translation=translation.reshape(b * h, 3),
        angles=neutral.expand(b * h, FIT_JOINTS),
        scale=torch.ones(b * h),
    )
    return theta, usable


def _geodesic(a: Float32[Tensor, "*batch 3 3"], b: Float32[Tensor, "*batch 3 3"]) -> Float32[Tensor, "*batch"]:
    cosine: Float32[Tensor, "*batch"] = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.arccos(cosine.clamp(-1.0, 1.0))


def initial_pose(model: HandModelTorch, phi: float, hands: Sequence[HandObservation], config: FitConfig = DEFAULT_CONFIG) -> list[FitResult]:
    """θ for hands with no previous pose, which is every acquisition (a DetNet box and a zero keypoint input).

    1. Wrist hypotheses: the neutral palm aligned to each view's unprojected palm keypoints at their d_rel distances, and
       the 24 rotations of the cube about the palm centroid.
    2. A rigid LM (6 DoF) of each hypothesis on the palm keypoints, which do not move with the fingers.
    3. The full LM (26 DoF, fingers from the neutral pose) from the best palm fits, keeping the lowest energy.

    A hand with no usable hypothesis is not refined: it returns a neutral pose, NaN energies, and
    ``converged=False, termination="no_evidence"``. The tracker treats it as not acquired.

    The neutral pose is midway between the joint limits (UmeTrack's ``neutral_joint_angles`` with ``lower_factor=0.5``).
    """
    b: int = len(hands)
    views: Views = stack_views(hands)
    limits: Float32[Tensor, "20 2"] = model.joint_limits[:FIT_JOINTS]
    neutral: Float32[Tensor, "20"] = 0.5 * (limits[:, 0] + limits[:, 1])
    hypotheses, usable = _wrist_hypotheses(model, phi, views, neutral, config.rotation_hypotheses)
    has_evidence: Bool[Tensor, "b"] = usable.any(dim=1)
    if not bool(has_evidence.all()):
        valid: list[int] = torch.nonzero(has_evidence).flatten().tolist()
        acquired: list[FitResult] = initial_pose(model, phi, [hands[i] for i in valid], config) if valid else []
        by_hand: dict[int, FitResult] = dict(zip(valid, acquired, strict=True))
        return [
            by_hand[i]
            if i in by_hand
            else FitResult(
                pose=HandPose(torch.eye(3), torch.zeros(3), torch.cat([neutral, torch.zeros(2)])),
                e_2d=math.nan,
                e_dist=math.nan,
                e_temporal=math.nan,
                energy=math.nan,
                iterations=0,
                converged=False,
                termination="no_evidence",
            )
            for i in range(b)
        ]
    h: int = usable.shape[1]
    stage: FitConfig = replace(config, max_iterations=config.init_iterations, relative_tolerance=config.init_relative_tolerance)
    palm_only: Float32[Tensor, "b 2 21"] = views.weights * _PALM_MASK
    enough_palm: Bool[Tensor, "b 1 1"] = ((palm_only > 0).sum(-1) >= MIN_ALIGN_POINTS).any(-1)[:, None, None]
    rigid_views: Views = replace(repeat_views(views, h), weights=torch.where(enough_palm, palm_only, views.weights).repeat_interleave(h, dim=0))
    rigid_problem: Problem = _problem(model, phi, rigid_views, no_prior(b * h), config)
    rigid: _Solved = _levenberg_marquardt(rigid_problem, hypotheses, torch.arange(6), stage)
    rigid_energy: Float32[Tensor, "b h"] = torch.where(usable, rigid.energy.reshape(b, h), torch.full((b, h), torch.inf))
    order: Int64[Tensor, "b h"] = rigid_energy.argsort(dim=-1)
    rotations: Float32[Tensor, "b h 3 3"] = rigid.theta.rotation.reshape(b, h, 3, 3)
    rows: Int64[Tensor, "b"] = torch.arange(b)
    best: Float32[Tensor, "b 3 3"] = rotations[rows, order[:, 0]]
    # Refine the best palm fits whose wrist rotations differ by more than 30° from the best one (the next best otherwise).
    distinct: Bool[Tensor, "b h"] = (_geodesic(best[:, None], rotations) > math.radians(30.0)) & torch.isfinite(rigid_energy)
    ranked: Float32[Tensor, "b h"] = torch.where(distinct, rigid_energy, torch.full_like(rigid_energy, torch.inf))
    ranked[rows, order[:, 0]] = -torch.inf
    k: int = min(config.full_fit_hypotheses, h)
    chosen: Int64[Tensor, "b k"] = ranked.argsort(dim=-1)[:, :k]
    chosen = torch.where(ranked.gather(1, chosen) < torch.inf, chosen, order[:, :k])
    # Each refined wrist starts its fingers from every finger pose in turn: the neutral pose, and the open hand if configured.
    finger_starts: Float32[Tensor, "s 20"] = torch.stack([neutral, torch.zeros_like(neutral).clamp(limits[:, 0], limits[:, 1])])[
        : config.finger_starts
    ]
    n: int = k * finger_starts.shape[0]
    wrists: Theta = take(rigid.theta, (rows[:, None] * h + chosen).repeat_interleave(finger_starts.shape[0], dim=1).reshape(-1))
    start: Theta = replace(wrists, angles=finger_starts.repeat(b * k, 1))
    full_problem: Problem = _problem(model, phi, repeat_views(views, n), no_prior(b * n), config)
    full: _Solved = _levenberg_marquardt(full_problem, start, torch.arange(POSE_PARAMETERS), stage)
    winner: Int64[Tensor, "b"] = rows * n + torch.where(torch.isfinite(full.energy), full.energy, torch.inf).reshape(b, n).argmin(-1)
    solved: _Solved = _Solved(
        theta=take(full.theta, winner),
        energy=full.energy[winner],
        iterations=full.iterations[winner],
        converged=full.converged[winner],
        termination=[full.termination[i] for i in winner.tolist()],
    )
    problem: Problem = _problem(model, phi, views, no_prior(b), config)
    return _results(problem, solved, torch.zeros((b, 2)))
