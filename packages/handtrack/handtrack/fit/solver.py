"""The energy of MEgATrack §3.5 and its linearisation, shared by the pose fit and the scale calibration.

E(θ) = E_2D + w_1·E_dist + w_2·E_temporal, with

- E_2D = Σ_ij w_ij ‖Π_j(p_i(θ)) − p̂_ij‖², in pixels of each camera's own image, Π_j its own lens model;
- E_dist = Σ_ij w_ij w_0j ((dist_j(p_i) − dist_j(p_0)) − ϕ·(d̂_ij − d̂_0j))², in millimetres, p_0 the wrist;
- E_temporal = ‖θ − θ(t−1)‖², a constant-position prior: the wrist rotation as the chordal distance ½‖R − R(t−1)‖²_F
  (≈ the squared angle in radians), the wrist translation in ``Problem.temporal_translation_unit_m``, the joint angles in radians.

The parameters of one hand are a local rotation increment R ← R·exp([δω]×) (3), the wrist translation (3), joint angles
0-19 (20), and a factor on the model's rest geometry (1; only the scale calibration frees it): 27 in all. Joint angles 20-21
are not skinned and stay fixed. The Jacobian comes from central differences through the skinning (``pose.landmarks``) and
the lens model; all perturbations of all hands are one batched evaluation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from jaxtyping import Bool, Float32, Float64, Int64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch, hat
from torch import Tensor

from handtrack.fit.observations import DIST_REFERENCE, MAX_VIEWS, HandObservation, ViewObservation
from handtrack.geometry.camera import CameraRig, project, transform_points
from handtrack.hand.pose import HandPose, Side, landmarks

DAMPING_GROWTH: float = 2.0
"""Nielsen's ν: λ grows by it after a rejected step, and it doubles on each rejection in a row."""
MIN_DAMPING_FACTOR: float = 1.0 / 3.0
"""λ shrinks by max(1/3, 1 − (2ρ − 1)³) after an accepted step with gain ratio ρ."""
MAX_DAMPING: float = 1e10
"""λ beyond which a solve gives up (its steps have shrunk to nothing)."""
FIT_JOINTS: int = 20
"""Joint angles 0-19 are fitted; the skinning ignores 20-21."""
ANGLE_OFFSET: int = 6
"""Index of joint angle 0 in the parameter vector."""
POSE_PARAMETERS: int = ANGLE_OFFSET + FIT_JOINTS
"""The 26 degrees of freedom of θ."""
SCALE_PARAMETER: int = POSE_PARAMETERS
"""Index of the scale factor on the model's rest geometry."""
PARAMETERS: int = POSE_PARAMETERS + 1
RESIDUALS_2D: int = MAX_VIEWS * 21 * 2
RESIDUALS_DIST: int = MAX_VIEWS * 21
_STEPS: Float32[Tensor, "27"] = torch.tensor([3e-3] * 3 + [3e-4] * 3 + [3e-3] * FIT_JOINTS + [3e-3])
"""Central-difference steps: radians for rotations and joint angles, metres for the translation, and the scale factor."""


@dataclass(frozen=True, slots=True)
class Views:
    """The observations of b hands stacked for the solver, two views per hand; a missing view repeats view 0 with weight 0."""

    cameras: CameraRig
    """b·2 cameras, hand-major."""
    cam_from_world: Float32[Tensor, "b 2 4 4"]
    """cam_from_rig · rig_from_world per view."""
    keypoints_px: Float32[Tensor, "b 2 21 2"]
    """In each camera's own image pixels; 0 where the weight is 0."""
    weights: Float32[Tensor, "b 2 21"]
    """0 where a keypoint is not observed, and throughout a padded view."""
    d_rel_mm: Float32[Tensor, "b 2 21"]
    """Millimetres of the generic hand; 0 where the weight is 0."""
    mirror: Float32[Tensor, "b"]
    """+1 for a left hand, -1 for a right hand: the sign of the wrist frame's x axis."""


@dataclass(frozen=True, slots=True)
class Theta:
    """θ for a batch of hands, plus the factor on the model's rest geometry (1 outside the calibration)."""

    rotation: Float32[Tensor, "b 3 3"]
    """World from wrist."""
    translation: Float32[Tensor, "b 3"]
    """Metres."""
    angles: Float32[Tensor, "b 20"]
    """Joint angles 0-19, radians."""
    scale: Float32[Tensor, "b"]
    """The factor on the model's rest geometry."""


@dataclass(frozen=True, slots=True)
class Prior:
    """θ(t−1) per hand, in ``Theta``'s units; ``present`` is 0 where there is none."""

    rotation: Float32[Tensor, "b 3 3"]
    translation: Float32[Tensor, "b 3"]
    angles: Float32[Tensor, "b 20"]
    present: Float32[Tensor, "b"]


@dataclass(frozen=True, slots=True)
class Problem:
    """Everything the residuals depend on besides θ."""

    model: HandModelTorch
    views: Views
    prior: Prior
    phi: float
    """ϕ, which turns the predicted d_rel back into millimetres."""
    dist_weight: float
    """w_1."""
    temporal_weight: float
    """w_2."""
    temporal_translation_unit_m: float
    """The unit of the wrist translation in E_temporal."""


def stack_views(hands: Sequence[HandObservation]) -> Views:
    padded: list[tuple[ViewObservation, ...]] = [
        hand.views + (replace(hand.views[0], weights=torch.zeros_like(hand.views[0].weights)),) * (MAX_VIEWS - len(hand.views)) for hand in hands
    ]
    flat: list[ViewObservation] = [view for views in padded for view in views]
    cameras: CameraRig = CameraRig.cat([view.camera for view in flat])
    world_from_rig: Float32[Tensor, "n 4 4"] = torch.stack([view.world_from_rig for view in flat])
    rig_from_world: Float32[Tensor, "n 4 4"] = torch.zeros_like(world_from_rig)
    rig_from_world[:, :3, :3] = world_from_rig[:, :3, :3].transpose(-1, -2)
    rig_from_world[:, :3, 3] = -torch.einsum("nji,nj->ni", world_from_rig[:, :3, :3], world_from_rig[:, :3, 3])
    rig_from_world[:, 3, 3] = 1.0
    b: int = len(hands)
    weights: Float32[Tensor, "b 2 21"] = torch.stack([view.weights for view in flat]).reshape(b, MAX_VIEWS, 21)
    seen: Bool[Tensor, "b 2 21"] = weights > 0
    # A keypoint with weight 0 may hold anything, NaN included; zero it so it cannot poison the residuals.
    keypoints_px: Float32[Tensor, "b 2 21 2"] = torch.stack([view.keypoints_px for view in flat]).reshape(b, MAX_VIEWS, 21, 2)
    d_rel_mm: Float32[Tensor, "b 2 21"] = torch.stack([view.d_rel_mm for view in flat]).reshape(b, MAX_VIEWS, 21)
    return Views(
        cameras=cameras,
        cam_from_world=(cameras.cam_from_rig @ rig_from_world).reshape(b, MAX_VIEWS, 4, 4),
        keypoints_px=torch.where(seen[..., None], keypoints_px, 0.0),
        weights=weights,
        d_rel_mm=torch.where(seen, d_rel_mm, 0.0),
        mirror=torch.tensor([1.0 if hand.side == Side.LEFT else -1.0 for hand in hands]),
    )


def repeat_views(views: Views, n: int) -> Views:
    """Each hand's views ``n`` times in a row, one copy per hypothesis."""
    b: int = views.mirror.shape[0]
    index: Int64[Tensor, "q"] = (torch.arange(b).repeat_interleave(n)[:, None] * MAX_VIEWS + torch.arange(MAX_VIEWS)).reshape(-1)
    return Views(
        cameras=views.cameras.select(index.tolist()),
        cam_from_world=views.cam_from_world.repeat_interleave(n, dim=0),
        keypoints_px=views.keypoints_px.repeat_interleave(n, dim=0),
        weights=views.weights.repeat_interleave(n, dim=0),
        d_rel_mm=views.d_rel_mm.repeat_interleave(n, dim=0),
        mirror=views.mirror.repeat_interleave(n, dim=0),
    )


def theta_from_poses(poses: Sequence[HandPose]) -> Theta:
    """Unbatched poses stacked into θ at scale 1."""
    return Theta(
        rotation=torch.stack([pose.rotation for pose in poses]),
        translation=torch.stack([pose.translation for pose in poses]),
        angles=torch.stack([pose.joint_angles[:FIT_JOINTS] for pose in poses]),
        scale=torch.ones(len(poses)),
    )


def poses_from_theta(theta: Theta, carried: Float32[Tensor, "b 2"]) -> list[HandPose]:
    """The inverse of ``theta_from_poses``: one unbatched pose per hand, joint angles 20-21 restored from ``carried``."""
    return [HandPose(theta.rotation[i], theta.translation[i], torch.cat([theta.angles[i], carried[i]])) for i in range(carried.shape[0])]


def take(theta: Theta, index: Int64[Tensor, "q"]) -> Theta:
    return Theta(rotation=theta.rotation[index], translation=theta.translation[index], angles=theta.angles[index], scale=theta.scale[index])


def select(mask: Bool[Tensor, "b"], chosen: Theta, other: Theta) -> Theta:
    """``chosen`` where ``mask`` holds, ``other`` elsewhere."""
    return Theta(
        rotation=torch.where(mask[:, None, None], chosen.rotation, other.rotation),
        translation=torch.where(mask[:, None], chosen.translation, other.translation),
        angles=torch.where(mask[:, None], chosen.angles, other.angles),
        scale=torch.where(mask, chosen.scale, other.scale),
    )


def no_prior(b: int) -> Prior:
    return Prior(rotation=torch.eye(3).expand(b, 3, 3), translation=torch.zeros((b, 3)), angles=torch.zeros((b, FIT_JOINTS)), present=torch.zeros(b))


def _so3_exp(v: Float32[Tensor, "b 3"]) -> Float32[Tensor, "b 3 3"]:
    """Rodrigues' formula with its small-angle series (simplecv's ``so3_exp_map`` clamps the angle instead: not bit-identical)."""
    angle_sq: Float32[Tensor, "b"] = (v * v).sum(-1)
    angle: Float32[Tensor, "b"] = angle_sq.sqrt()
    small: Bool[Tensor, "b"] = angle < 1e-4
    safe: Float32[Tensor, "b"] = torch.where(small, torch.ones_like(angle), angle)
    a: Float32[Tensor, "b"] = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(safe) / safe)
    c: Float32[Tensor, "b"] = torch.where(small, 0.5 - angle_sq / 24.0, (1.0 - torch.cos(safe)) / (safe * safe))
    k: Float32[Tensor, "b 3 3"] = hat(v)
    return torch.eye(3, dtype=v.dtype, device=v.device) + a[:, None, None] * k + c[:, None, None] * (k @ k)


def orthonormalize(rotation: Float32[Tensor, "b 3 3"]) -> Float32[Tensor, "b 3 3"]:
    """The nearest rotation (polar decomposition), removing the drift of repeated float32 updates."""
    u, _, vh = torch.linalg.svd(rotation)
    sign: Float32[Tensor, "b"] = torch.sign(torch.linalg.det(u @ vh))
    return u @ torch.diag_embed(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)) @ vh


def local_landmarks(model: HandModelTorch, angles: Float32[Tensor, "f b 20"], mirror: Float32[Tensor, "b"]) -> Float32[Tensor, "f b 21 3"]:
    """Landmarks in the wrist frame, metres, with the right hand's x axis mirrored (as ``HandPose.world_from_wrist_mm``)."""
    f: int = angles.shape[0]
    b: int = angles.shape[1]
    identity: HandPose = HandPose(
        rotation=torch.eye(3, dtype=angles.dtype, device=angles.device).expand(f, b, 3, 3),
        translation=torch.zeros((f, b, 3), dtype=angles.dtype, device=angles.device),
        joint_angles=torch.cat([angles, torch.zeros_like(angles[..., :2])], dim=-1),
    )
    local: Float32[Tensor, "f b 21 3"] = landmarks(model, identity, Side.LEFT)
    signs: Float32[Tensor, "b 3"] = torch.stack([mirror, torch.ones_like(mirror), torch.ones_like(mirror)], dim=-1)
    return local * signs[:, None, :]


def residuals(problem: Problem, theta: Theta, delta: Float32[Tensor, "f b 27"]) -> Float32[Tensor, "f b m"]:
    """The weighted residual vector [E_2D | E_dist | E_temporal] at θ ⊕ δ for f perturbations δ.

    The rotation enters to first order, R·(I + [δω]×), which has the same derivative at δ = 0 as R·exp([δω]×). δ[0] must
    leave the joint angles unperturbed (``linearize`` and ``energy_terms`` do): its skinned hand serves every other
    perturbation that moves no joint angle, so only the ones that do are skinned besides it.
    """
    views: Views = problem.views
    f: int = delta.shape[0]
    b: int = delta.shape[1]
    rotation: Float32[Tensor, "f b 3 3"] = theta.rotation + theta.rotation @ hat(delta[..., 0:3].reshape(-1, 3)).reshape(f, b, 3, 3)
    translation: Float32[Tensor, "f b 3"] = theta.translation + delta[..., 3:6]
    angles: Float32[Tensor, "f b 20"] = theta.angles + delta[..., ANGLE_OFFSET:POSE_PARAMETERS]
    scale: Float32[Tensor, "f b"] = theta.scale + delta[..., SCALE_PARAMETER]
    skin: Bool[Tensor, "f"] = delta[..., ANGLE_OFFSET:POSE_PARAMETERS].abs().amax(dim=(1, 2)) > 0
    skin[0] = True
    computed: Float32[Tensor, "n b 21 3"] = local_landmarks(problem.model, angles[skin], views.mirror)
    skinned: Float32[Tensor, "f b 21 3"] = computed[0].expand(f, b, 21, 3).clone()
    skinned[skin] = computed
    local: Float32[Tensor, "f b 21 3"] = skinned * scale[..., None, None]
    points_world: Float32[Tensor, "f b 21 3"] = torch.einsum("fbij,fbnj->fbni", rotation, local) + translation[:, :, None, :]
    points_cam: Float32[Tensor, "f b 2 21 3"] = transform_points(
        views.cam_from_world.expand(f, b, MAX_VIEWS, 4, 4), points_world[:, :, None].expand(f, b, MAX_VIEWS, 21, 3)
    )
    pixels: Float32[Tensor, "f b 2 21 2"] = project(views.cameras, points_cam.reshape(f, b * MAX_VIEWS, 21, 3)).reshape(f, b, MAX_VIEWS, 21, 2)
    two_d: Float32[Tensor, "f b 2 21 2"] = views.weights.sqrt()[..., None] * (pixels - views.keypoints_px)
    distance_mm: Float32[Tensor, "f b 2 21"] = points_cam.norm(dim=-1) * 1000.0
    reference: int = DIST_REFERENCE
    model_mm: Float32[Tensor, "f b 2 21"] = distance_mm - distance_mm[..., reference : reference + 1]
    observed_mm: Float32[Tensor, "b 2 21"] = problem.phi * (views.d_rel_mm - views.d_rel_mm[..., reference : reference + 1])
    dist_weight: Float32[Tensor, "b 2 21"] = problem.dist_weight * views.weights * views.weights[..., reference : reference + 1]
    dist: Float32[Tensor, "f b 2 21"] = dist_weight.sqrt() * (model_mm - observed_mm)
    prior: Prior = problem.prior
    temporal: Float32[Tensor, "f b 32"] = (
        torch.cat(
            [
                (rotation - prior.rotation).reshape(f, b, 9) / math.sqrt(2.0),
                (translation - prior.translation) / problem.temporal_translation_unit_m,
                angles - prior.angles,
            ],
            dim=-1,
        )
        * (problem.temporal_weight * prior.present).sqrt()[:, None]
    )
    return torch.cat([two_d.reshape(f, b, RESIDUALS_2D), dist.reshape(f, b, RESIDUALS_DIST), temporal], dim=-1)


def linearize(problem: Problem, theta: Theta, free: Int64[Tensor, "k"]) -> tuple[Float32[Tensor, "b m"], Float32[Tensor, "b m k"]]:
    """The residuals and their Jacobian over the ``free`` parameters, by central differences in one batched evaluation.

    Forward-mode AD (``torch.autograd.forward_ad``) gives the exact Jacobian but runs 5x slower: every op that mixes a dual
    tensor with a constant goes through torch's Python zero-tensor path (~90 us per op). In float32 the central
    differences agree with it to ~2e-4 (median) and ~7e-4 (worst column), relative.
    """
    b: int = theta.translation.shape[0]
    k: int = free.shape[0]
    steps: Float32[Tensor, "k"] = _STEPS[free]
    delta: Float32[Tensor, "f b 27"] = torch.zeros((2 * k + 1, b, PARAMETERS), dtype=theta.translation.dtype, device=theta.translation.device)
    delta[1 + torch.arange(k), :, free] = steps[:, None]
    delta[1 + k + torch.arange(k), :, free] = -steps[:, None]
    evaluated: Float32[Tensor, "f b m"] = residuals(problem, theta, delta)
    jacobian: Float32[Tensor, "k b m"] = (evaluated[1 : k + 1] - evaluated[k + 1 :]) / (2.0 * steps[:, None, None])
    return evaluated[0], jacobian.permute(1, 2, 0).contiguous()


def energy_terms(problem: Problem, theta: Theta) -> tuple[Float32[Tensor, "b"], Float32[Tensor, "b"], Float32[Tensor, "b"]]:
    """Unweighted E_2D, E_dist and E_temporal (a term with weight 0 reads 0)."""
    squared: Float32[Tensor, "b m"] = residuals(problem, theta, torch.zeros((1, theta.translation.shape[0], PARAMETERS)))[0] ** 2
    e_2d: Float32[Tensor, "b"] = squared[:, :RESIDUALS_2D].sum(-1)
    dist: Float32[Tensor, "b"] = squared[:, RESIDUALS_2D : RESIDUALS_2D + RESIDUALS_DIST].sum(-1)
    temporal: Float32[Tensor, "b"] = squared[:, RESIDUALS_2D + RESIDUALS_DIST :].sum(-1)
    e_dist: Float32[Tensor, "b"] = dist / problem.dist_weight if problem.dist_weight > 0 else torch.zeros_like(dist)
    e_temporal: Float32[Tensor, "b"] = temporal / problem.temporal_weight if problem.temporal_weight > 0 else torch.zeros_like(temporal)
    return e_2d, e_dist, e_temporal


def fit_limits(model: HandModelTorch, margin_rad: float) -> Float32[Tensor, "20 2"]:
    """The bounds the fit keeps joint angles 0-19 in: the model's ``joint_limits`` widened by ``margin_rad`` on both sides."""
    return model.joint_limits[:FIT_JOINTS] + torch.tensor([-margin_rad, margin_rad], dtype=model.joint_limits.dtype)


def retract(theta: Theta, step: Float32[Tensor, "b 27"], limits: Float32[Tensor, "20 2"]) -> Theta:
    """θ ⊕ step: the rotation on SO(3), the joint angles clamped to their limits."""
    return Theta(
        rotation=theta.rotation @ _so3_exp(step[:, 0:3]),
        translation=theta.translation + step[:, 3:6],
        angles=torch.clamp(theta.angles + step[:, ANGLE_OFFSET:POSE_PARAMETERS], limits[:, 0], limits[:, 1]),
        scale=theta.scale + step[:, SCALE_PARAMETER],
    )


def free_mask(theta: Theta, gradient: Float64[Tensor, "b k"], free: Int64[Tensor, "k"], limits: Float32[Tensor, "20 2"]) -> Float64[Tensor, "b k"]:
    """1 for a free parameter, 0 for a joint angle held at a limit that the gradient pushes it through (the active set)."""
    b: int = theta.angles.shape[0]
    is_angle: Bool[Tensor, "k"] = (free >= ANGLE_OFFSET) & (free < POSE_PARAMETERS)
    joint: Int64[Tensor, "k"] = (free - ANGLE_OFFSET).clamp(0, FIT_JOINTS - 1)
    angles: Float32[Tensor, "b k"] = theta.angles[:, joint]
    at_lower: Bool[Tensor, "b k"] = (angles <= limits[joint, 0] + 1e-6) & (gradient > 0)
    at_upper: Bool[Tensor, "b k"] = (angles >= limits[joint, 1] - 1e-6) & (gradient < 0)
    held: Bool[Tensor, "b k"] = is_angle.expand(b, -1) & (at_lower | at_upper)
    return (~held).to(torch.float64)


def normal_equations(
    theta: Theta, residual: Float32[Tensor, "b m"], jacobian: Float32[Tensor, "b m k"], free: Int64[Tensor, "k"], limits: Float32[Tensor, "20 2"]
) -> tuple[Float64[Tensor, "b k k"], Float64[Tensor, "b k"], Float64[Tensor, "b k"]]:
    """JᵀJ and Jᵀr in float64 with the rows and columns of held joint angles zeroed, and the ``free_mask`` that held them."""
    j64: Float64[Tensor, "b m k"] = jacobian.to(torch.float64)
    hessian: Float64[Tensor, "b k k"] = j64.transpose(1, 2) @ j64
    gradient: Float64[Tensor, "b k"] = torch.einsum("bmk,bm->bk", j64, residual.to(torch.float64))
    mask: Float64[Tensor, "b k"] = free_mask(theta, gradient, free, limits)
    return hessian * mask[:, :, None] * mask[:, None, :], gradient * mask, mask


def marquardt_scaling(diagonal: Float64[Tensor, "b k"], mask: Float64[Tensor, "b k"]) -> Float64[Tensor, "b k"]:
    """D of (JᵀJ + λD)δ = −Jᵀr: diag(JᵀJ), floored so an unobserved parameter stays damped, and 1 for held parameters."""
    floor: Float64[Tensor, "b 1"] = 1e-9 * diagonal.amax(-1, keepdim=True).clamp(min=1e-12)
    return diagonal.clamp(min=floor) * mask + (1.0 - mask)


def predicted_reduction(
    theta: Theta,
    candidate: Theta,
    free: Int64[Tensor, "k"],
    step: Float64[Tensor, "b k"],
    hessian: Float64[Tensor, "b k k"],
    gradient: Float64[Tensor, "b k"],
) -> Float64[Tensor, "b"]:
    """Quadratic energy decrease for the applied step, including projection onto the joint limits."""
    is_angle: Bool[Tensor, "k"] = (free >= ANGLE_OFFSET) & (free < POSE_PARAMETERS)
    effective: Float64[Tensor, "b k"] = step.clone()
    effective[:, is_angle] = (candidate.angles - theta.angles)[:, free[is_angle] - ANGLE_OFFSET].to(torch.float64)
    return -(2.0 * (effective * gradient).sum(-1) + torch.einsum("bk,bkl,bl->b", effective, hessian, effective))


def undamped_inverse(hessian: Float64[Tensor, "b k k"], mask: Float64[Tensor, "b k"]) -> Float64[Tensor, "b k k"]:
    """Inverse curvature for a Gauss–Newton stop check, scaled by column norms and allowing unobserved parameters.

    Args:
        hessian: Float64[Tensor, "b k k"], active-set normal matrix.
        mask: Float64[Tensor, "b k"], one for free parameters and zero for held angles.
    """
    units: Float64[Tensor, "b k"] = marquardt_scaling(torch.diagonal(hessian, dim1=1, dim2=2), mask).rsqrt()
    normalized: Float64[Tensor, "b k k"] = hessian * units[:, :, None] * units[:, None, :]
    return torch.linalg.pinv(normalized, hermitian=True) * units[:, :, None] * units[:, None, :]
