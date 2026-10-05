"""Hand-scale calibration (MEgATrack §3.6): one scale ϕ, shared by both hands, from frames seen in stereo.

min over θ_1..θ_n and ϕ of Σ_t E_2D(θ_t, ϕ), where the model is the input model with its rest geometry scaled by ϕ.
The Jacobian is block-diagonal in the θ_t plus one dense column for ϕ, so each LM step eliminates the θ_t blocks
(Schur complement on ϕ) and costs O(n): one 26x26 solve per block and a scalar equation for ϕ. The joint solve lets
the optimiser trade the hand's distance against its size, which is what §3.6 says makes the estimate work; alternating
between the poses and ϕ would crawl along that valley.

Offline protocol (§5.1): calibrate on the first 100 frames of a sequence, then track the whole sequence again with
``scaled_hand_model(generic_hand_model(), ϕ)`` and ϕ.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float32, Float64, Int64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import HandObservation
from handtrack.fit.pose_fit import DEFAULT_CONFIG, FitConfig, FitResult, Termination, fit_pose
from handtrack.fit.solver import (
    DAMPING_GROWTH,
    FIT_JOINTS,
    MAX_DAMPING,
    MIN_DAMPING_FACTOR,
    PARAMETERS,
    POSE_PARAMETERS,
    Problem,
    Theta,
    energy_terms,
    fit_limits,
    linearize,
    marquardt_scaling,
    no_prior,
    normal_equations,
    orthonormalize,
    poses_from_theta,
    predicted_reduction,
    retract,
    stack_views,
    theta_from_poses,
    undamped_inverse,
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
    """λ at the first iteration, relative to diag(JᵀJ) (Marquardt scaling) of each θ_t block."""
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
    """LM iterations run."""
    converged: bool
    """True when the convergence tolerance was reached or damping stopped at a stationary solution."""
    termination: Termination = "iterations"
    """Why the solve stopped; non-finite energy returns NaN phi and converged=False."""


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
        fitted: list[FitResult] = fit_pose(model, 1.0, [blocks[j] for j in missing], [None] * len(missing), config.initial_fit)
        for j, result in zip(missing, fitted, strict=True):
            starts[j] = result.pose
    poses: list[HandPose] = [pose for pose in starts if pose is not None]
    theta: Theta = theta_from_poses(poses)
    n: int = len(blocks)
    problem: Problem = Problem(
        model=model, views=stack_views(blocks), prior=no_prior(n), phi=1.0, dist_weight=0.0, temporal_weight=0.0, temporal_translation_unit_m=1.0
    )
    limits: Float32[Tensor, "20 2"] = fit_limits(model, config.initial_fit.joint_limit_margin_rad)
    theta = replace(theta, angles=theta.angles.clamp(limits[:, 0], limits[:, 1]))
    free: Int64[Tensor, "27"] = torch.arange(PARAMETERS)
    residual, jacobian = linearize(problem, theta, free)
    energy: float = float((residual * residual).sum())
    damping: float = config.initial_damping
    growth: float = DAMPING_GROWTH
    converged: bool = False
    termination: Termination = "iterations"
    iterations: int = 0
    if not math.isfinite(energy):
        termination = "non_finite"
    for _ in range(config.iterations):
        if termination == "non_finite":
            break
        iterations += 1
        hessian, gradient, mask = normal_equations(theta, residual, jacobian, free, limits)
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
        candidate: Theta = retract(theta, step.to(torch.float32), limits)
        predicted: float = float(predicted_reduction(theta, candidate, free, step, hessian, gradient).sum())
        new_residual, new_jacobian = linearize(problem, candidate, free)
        new_energy: float = float((new_residual * new_residual).sum())
        if not math.isfinite(new_energy):
            termination = "non_finite"
            break
        reduction: float = energy - new_energy
        if reduction > 0:
            ratio: float = reduction / max(predicted, 1e-30)
            converged = reduction <= config.relative_tolerance * energy
            theta, residual, jacobian, energy = candidate, new_residual, new_jacobian, new_energy
            damping *= max(MIN_DAMPING_FACTOR, 1.0 - (2.0 * ratio - 1.0) ** 3)
            growth = DAMPING_GROWTH
        else:
            damping *= growth
            growth *= DAMPING_GROWTH
        if converged:
            termination = "tolerance"
            break
        if damping > MAX_DAMPING:
            # Recompute at the retained solution: the last step may have been accepted.
            hessian, gradient, mask = normal_equations(theta, residual, jacobian, free, limits)
            pose_block = hessian[:, :POSE_PARAMETERS, :POSE_PARAMETERS]
            coupling = hessian[:, :POSE_PARAMETERS, POSE_PARAMETERS]
            pose_mask = mask[:, :POSE_PARAMETERS]
            scaling = marquardt_scaling(torch.diagonal(pose_block, dim1=1, dim2=2), pose_mask)
            inverse: Float64[Tensor, "n 26 26"] = undamped_inverse(pose_block, pose_mask)
            solved = inverse @ torch.stack([coupling, gradient[:, :POSE_PARAMETERS]], dim=-1)
            scale_curvature = float(hessian[:, POSE_PARAMETERS, POSE_PARAMETERS].sum())
            scale_gradient: float = float(gradient[:, POSE_PARAMETERS].sum())
            schur = scale_curvature - float((coupling * solved[..., 0]).sum())
            gn_prediction: float = math.inf
            if schur > torch.finfo(torch.float64).eps * scale_curvature:
                scale_step = (-scale_gradient + float((coupling * solved[..., 1]).sum())) / schur
                pose_step = -(solved[..., 1] + solved[..., 0] * scale_step)
                step = torch.cat([pose_step, torch.full((n, 1), scale_step, dtype=torch.float64)], dim=-1)
                candidate = retract(theta, step.to(torch.float32), limits)
                gn_prediction = float(predicted_reduction(theta, candidate, free, step, hessian, gradient).sum())
            scaled_gradient_sq: float = max(
                float((gradient[:, :POSE_PARAMETERS].square() / scaling).amax()),
                scale_gradient**2 / max(scale_curvature, 1e-30),
            )
            # The residuals are float32: a gradient whose squared normalized size is
            # below their energy resolution cannot justify another energy-decreasing step.
            converged = abs(gn_prediction) <= config.relative_tolerance * energy or scaled_gradient_sq <= torch.finfo(torch.float32).eps * energy
            termination = "stationary" if converged else "damping"
            break
    if not bool(torch.isfinite(theta.scale).all()):
        raise ValueError("the scale calibration diverged")
    if termination != "non_finite":
        theta = replace(theta, rotation=orthonormalize(theta.rotation))
    e_2d: Float32[Tensor, "n"] = energy_terms(problem, theta)[0]
    if not bool(torch.isfinite(e_2d).all()):
        termination = "non_finite"
    carried: Float32[Tensor, "n 2"] = torch.stack([pose.joint_angles[FIT_JOINTS:] for pose in poses])
    return ScaleCalibration(
        phi=math.nan if termination == "non_finite" else float(theta.scale[0]),
        poses=tuple(poses_from_theta(theta, carried)),
        used=tuple(used),
        e_2d=float(e_2d.sum()),
        blocks=n,
        iterations=iterations,
        converged=converged and termination != "non_finite",
        termination=termination,
    )
