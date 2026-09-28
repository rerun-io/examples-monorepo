"""Hand-scale calibration (MEgATrack §3.6): one scale ϕ, shared by both hands, from frames seen in stereo.

min over θ_1..θ_n and ϕ of Σ_t E_2D(θ_t, ϕ), where the model is the input model with its rest geometry scaled by ϕ.
The Jacobian is block-diagonal in the θ_t plus one dense column for ϕ, so each LM step eliminates the θ_t blocks
(Schur complement on ϕ) and costs O(n): one 26x26 solve per block and a scalar equation for ϕ. The joint solve lets
the optimiser trade the hand's distance against its size, which is what §3.6 says makes the estimate work; alternating
between the poses and ϕ would crawl along that valley.

Offline protocol (§5.1): calibrate on the first 100 frames of a sequence, then track the whole sequence again with
``scaled_hand_model(generic_hand_model(), ϕ)`` and ϕ.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from jaxtyping import Bool, Float32, Float64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import HandObservation
from handtrack.fit.pose_fit import DEFAULT_CONFIG, FitConfig, fit_pose
from handtrack.fit.solver import (
    FIT_JOINTS,
    PARAMETERS,
    POSE_PARAMETERS,
    Problem,
    Theta,
    energy_terms,
    fit_limits,
    free_mask,
    linearize,
    marquardt_scaling,
    no_prior,
    orthonormalize,
    retract,
    stack_views,
    theta_from_poses,
)
from handtrack.hand.pose import HandPose


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Settings of the joint LM over every θ_t and ϕ."""

    iterations: int = 30
    """On a real recording (200 stereo blocks, 1.5 px) the solve converges in 26 iterations; ϕ moves 5e-4 after the 10th."""
    relative_tolerance: float = 1e-5
    """Stop when an accepted step lowers Σ E_2D by less than this fraction of it."""
    initial_damping: float = 1e-3
    min_view_keypoints: int = 10
    """A view counts towards stereo when at least this many of its keypoints have weight > 0."""
    initial_fit: FitConfig = DEFAULT_CONFIG
    """The per-frame fits (at the input model's size) that give the starting θ_t when none are passed."""


DEFAULT_CALIBRATION: CalibrationConfig = CalibrationConfig()


@dataclass(frozen=True, slots=True)
class ScaleCalibration:
    """The calibrated scale and the poses it was solved with."""

    phi: float
    """The scale of the subject's hand relative to the input model (ϕ itself when the input is the generic model)."""
    poses: tuple[HandPose, ...]
    """θ_t of every stereo observation used, in input order."""
    used: tuple[int, ...]
    """Indices of those observations in the input."""
    e_2d: float
    """Σ_t E_2D at the solution, pixels²."""
    blocks: int
    """Number of stereo observations (hand, frame) in the solve."""
    iterations: int
    converged: bool


def scaled_hand_model(model: HandModelTorch, scale: float) -> HandModelTorch:
    """The hand enlarged by ``scale`` about its wrist: rest joints, rest landmarks and mesh times ``scale`` (UmeTrack's ``scaled_hand_model``)."""
    return replace(
        model,
        joint_rest_positions=model.joint_rest_positions * scale,
        landmark_rest_positions=model.landmark_rest_positions * scale,
        mesh_vertices=model.mesh_vertices * scale,
    )


def calibrate_scale(
    model: HandModelTorch,
    hands: Sequence[HandObservation],
    initial: Sequence[HandPose | None] | None = None,
    config: CalibrationConfig = DEFAULT_CALIBRATION,
) -> ScaleCalibration:
    """Solve for one ϕ shared by every observation in ``hands`` (left and right, any frames) and a θ_t per observation.

    Args:
        model: The model to scale, normally ``generic_hand_model()``.
        hands: Observations of either hand in any frames; only those with at least two views of ``min_view_keypoints``
            observed keypoints enter the solve.
        initial: Optional starting θ_t per observation (e.g. the tracker's poses at ϕ = 1); a missing one is fitted with
            ``fit_pose`` at the input model's size from the neutral initialiser.
        config: Solver settings.
    """
    if initial is not None and len(initial) != len(hands):
        raise ValueError(f"{len(hands)} observations but {len(initial)} initial poses")
    used: list[int] = [
        i for i, hand in enumerate(hands) if sum(int((view.weights > 0).sum()) >= config.min_view_keypoints for view in hand.views) >= 2
    ]
    if not used:
        raise ValueError("no observation is seen in stereo")
    blocks: list[HandObservation] = [hands[i] for i in used]
    starts: list[HandPose | None] = [None if initial is None else initial[i] for i in used]
    missing: list[int] = [j for j, pose in enumerate(starts) if pose is None]
    if missing:
        fitted = fit_pose(model, 1.0, [blocks[j] for j in missing], [None] * len(missing), config.initial_fit)
        for j, result in zip(missing, fitted, strict=True):
            starts[j] = result.pose
    poses: list[HandPose] = [pose for pose in starts if pose is not None]
    theta: Theta = theta_from_poses(poses)
    n: int = len(blocks)
    problem: Problem = Problem(
        model=model, views=stack_views(blocks), prior=no_prior(n), phi=1.0, dist_weight=0.0, temporal_weight=0.0, temporal_translation_unit_m=1.0
    )
    limits: Float32[Tensor, "20 2"] = fit_limits(model, config.initial_fit.joint_limit_margin_rad)
    free = torch.arange(PARAMETERS)
    residual, jacobian = linearize(problem, theta, free)
    energy: float = float((residual * residual).sum())
    damping: float = config.initial_damping
    growth: float = 2.0
    converged: bool = False
    iterations: int = 0
    for _ in range(config.iterations):
        iterations += 1
        j64: Float64[Tensor, "n m 27"] = jacobian.to(torch.float64)
        hessian: Float64[Tensor, "n 27 27"] = j64.transpose(1, 2) @ j64
        gradient: Float64[Tensor, "n 27"] = torch.einsum("nmk,nm->nk", j64, residual.to(torch.float64))
        mask: Float64[Tensor, "n 27"] = free_mask(theta, gradient, free, limits)
        hessian = hessian * mask[:, :, None] * mask[:, None, :]
        gradient = gradient * mask
        pose_block: Float64[Tensor, "n 26 26"] = hessian[:, :POSE_PARAMETERS, :POSE_PARAMETERS]
        coupling: Float64[Tensor, "n 26"] = hessian[:, :POSE_PARAMETERS, POSE_PARAMETERS]
        pose_mask: Float64[Tensor, "n 26"] = mask[:, :POSE_PARAMETERS]
        scaling: Float64[Tensor, "n 26"] = marquardt_scaling(torch.diagonal(pose_block, dim1=1, dim2=2), pose_mask)
        damped: Float64[Tensor, "n 26 26"] = pose_block + torch.diag_embed(damping * scaling + (1.0 - pose_mask))
        solved: Float64[Tensor, "n 26 2"] = torch.linalg.solve(damped, torch.stack([coupling, gradient[:, :POSE_PARAMETERS]], dim=-1))
        scale_curvature: float = float(hessian[:, POSE_PARAMETERS, POSE_PARAMETERS].sum())
        schur: float = scale_curvature * (1.0 + damping) - float((coupling * solved[..., 0]).sum())
        scale_step: float = (-float(gradient[:, POSE_PARAMETERS].sum()) + float((coupling * solved[..., 1]).sum())) / schur
        pose_step: Float64[Tensor, "n 26"] = -(solved[..., 1] + solved[..., 0] * scale_step)
        step: Float64[Tensor, "n 27"] = torch.cat([pose_step, torch.full((n, 1), scale_step, dtype=torch.float64)], dim=-1)
        predicted: float = -float(2.0 * (step * gradient).sum() + torch.einsum("nk,nkl,nl->", step, hessian, step))
        candidate: Theta = retract(theta, step.to(torch.float32), limits)
        new_residual, new_jacobian = linearize(problem, candidate, free)
        new_energy: float = float((new_residual * new_residual).sum())
        reduction: float = energy - new_energy
        if reduction > 0:
            ratio: float = reduction / max(predicted, 1e-30)
            theta, residual, jacobian, energy = candidate, new_residual, new_jacobian, new_energy
            damping *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
            growth = 2.0
            if reduction <= config.relative_tolerance * (energy + reduction):
                converged = True
                break
        else:
            damping *= growth
            growth *= 2.0
            if damping > 1e10:
                converged = True
                break
    rotation: Float32[Tensor, "n 3 3"] = orthonormalize(theta.rotation)
    e_2d: Float32[Tensor, "n"] = energy_terms(problem, replace(theta, rotation=rotation))[0]
    carried: Float32[Tensor, "n 2"] = torch.stack([pose.joint_angles[FIT_JOINTS:] for pose in poses])
    stereo_ok: Bool[Tensor, ""] = torch.isfinite(theta.scale).all()
    if not bool(stereo_ok):
        raise ValueError("the scale calibration diverged")
    return ScaleCalibration(
        phi=float(theta.scale[0]),
        poses=tuple(HandPose(rotation[i], theta.translation[i], torch.cat([theta.angles[i], carried[i]])) for i in range(n)),
        used=tuple(used),
        e_2d=float(e_2d.sum()),
        blocks=n,
        iterations=iterations,
        converged=converged,
    )
