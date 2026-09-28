"""Regression coverage for fit feasibility and explicit solver failures."""

import math
from dataclasses import replace
from typing import Literal

import pytest
import torch
from jaxtyping import Float32, Float64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from test_fit_synthetic import _views, _visible_scene
from torch import Tensor

from handtrack.fit.observations import HandObservation, ViewObservation, observe
from handtrack.fit.pose_fit import FitConfig, FitResult, Termination, fit_pose
from handtrack.fit.scale import CalibrationConfig, ScaleCalibration, calibrate_scale
from handtrack.fit.solver import Theta, fit_limits, predicted_reduction, retract, theta_from_poses
from handtrack.hand.pose import HandPose, Side, generic_hand_model


@pytest.mark.parametrize("calibration", [False, True])
def test_start_is_projected_before_any_iteration(calibration: bool) -> None:
    model: HandModelTorch = generic_hand_model()
    pose, _ = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(1), (0, 1))
    config: FitConfig = FitConfig(max_iterations=0)
    limits: Float32[Tensor, "20 2"] = fit_limits(model, config.joint_limit_margin_rad)
    angles: Float32[Tensor, "22"] = pose.joint_angles.clone()
    angles[0] = limits[0, 1] + 0.1
    angles[1] = limits[1, 0] - 0.1
    start: HandPose = replace(pose, joint_angles=angles)
    observation: HandObservation = observe(model, start, Side.LEFT, 1.0, _views((0, 1)))
    result: FitResult | ScaleCalibration
    if calibration:
        result = calibrate_scale(model, [observation], [start], CalibrationConfig(iterations=0))
        fitted: HandPose = result.poses[0]
    else:
        result = fit_pose(model, 1.0, [observation], [start], config)[0]
        fitted: HandPose = result.pose
        assert result.e_temporal == pytest.approx(0.02, abs=1e-6)
    assert bool((fitted.joint_angles[:20] >= limits[:, 0]).all())
    assert bool((fitted.joint_angles[:20] <= limits[:, 1]).all())
    assert torch.equal(start.joint_angles, angles)


def test_acquisition_without_evidence_preserves_other_hands() -> None:
    model: HandModelTorch = generic_hand_model()
    _, observed = _visible_scene(model, Side.RIGHT, torch.Generator().manual_seed(1), (0, 1))
    missing: HandObservation = replace(observed, views=tuple(replace(view, weights=torch.zeros(21)) for view in observed.views))
    valid: FitResult = fit_pose(model, 1.0, [observed], [None])[0]
    failed, acquired = fit_pose(model, 1.0, [missing, observed], [None, None])
    assert not failed.converged
    assert failed.termination == "no_evidence"
    assert math.isnan(failed.energy)
    assert failed.iterations == 0
    assert torch.equal(failed.pose.rotation, torch.eye(3))
    assert torch.equal(failed.pose.translation, torch.zeros(3))
    assert torch.equal(failed.pose.joint_angles[:20], model.joint_limits[:20].mean(-1))
    assert acquired.energy == pytest.approx(valid.energy, abs=1e-6)
    torch.testing.assert_close(acquired.pose.joint_angles, valid.pose.joint_angles)


@pytest.mark.parametrize("other_parameter", [3, 26])
def test_projected_prediction_uses_the_joint_motion_applied(other_parameter: int) -> None:
    model: HandModelTorch = generic_hand_model()
    pose, _ = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(1), (0, 1))
    limits: Float32[Tensor, "20 2"] = fit_limits(model, FitConfig().joint_limit_margin_rad)
    theta: Theta = theta_from_poses([pose])
    theta.angles[0, 0] = limits[0, 1] - 0.25
    full_step: Float32[Tensor, "1 27"] = torch.zeros(1, 27)
    full_step[0, [other_parameter, 6]] = 1.0
    candidate: Theta = retract(theta, full_step, limits)
    # For q(x,y) = 2x² + 2y² - 4x - 4y, applied (1, 1/4) reduces q by 23/8.
    prediction: Float64[Tensor, "1"] = predicted_reduction(
        theta,
        candidate,
        torch.tensor([other_parameter, 6]),
        torch.ones(1, 2, dtype=torch.float64),
        2.0 * torch.eye(2, dtype=torch.float64)[None],
        torch.full((1, 2), -2.0, dtype=torch.float64),
    )
    assert float(prediction[0]) == pytest.approx(2.875)


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("iterations, termination", [(0, "iterations"), (1, "damping")])
def test_termination_does_not_claim_convergence(calibration: bool, iterations: int, termination: Termination) -> None:
    model: HandModelTorch = generic_hand_model()
    pose, observation = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(1), (0, 1))
    result: FitResult | ScaleCalibration
    if calibration:
        result = calibrate_scale(model, [observation], [pose], CalibrationConfig(iterations=iterations, initial_damping=1e30))
    else:
        result = fit_pose(model, 1.0, [observation], [pose], FitConfig(max_iterations=iterations, initial_damping=1e30))[0]
    assert not result.converged
    assert result.termination == termination


@pytest.mark.parametrize("field", ["keypoints_px", "d_rel_mm"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_observed_keypoint_must_be_finite(field: Literal["keypoints_px", "d_rel_mm"], value: float) -> None:
    model: HandModelTorch = generic_hand_model()
    _, observation = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(1), (0, 1))
    view: ViewObservation = observation.views[0]
    values: Float32[Tensor, "21 2"] | Float32[Tensor, "21"] = getattr(view, field).clone()
    values[7] = value
    with pytest.raises(ValueError, match=rf"{field}.*keypoint 7"):
        replace(view, weights=torch.ones(21), **{field: values})
    weights: Float32[Tensor, "21"] = torch.ones(21)
    weights[7] = 0.0
    replace(view, weights=weights, **{field: values})


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("field", ["rotation", "translation", "joint_angles"])
def test_non_finite_start_is_a_failure(calibration: bool, field: Literal["rotation", "translation", "joint_angles"]) -> None:
    model: HandModelTorch = generic_hand_model()
    pose, observation = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(1), (0, 1))
    values: Float32[Tensor, "3 3"] | Float32[Tensor, "3"] | Float32[Tensor, "22"] = getattr(pose, field).clone()
    values.flatten()[0] = math.nan
    bad_pose: HandPose = replace(pose, **{field: values})
    result: FitResult | ScaleCalibration
    if calibration:
        result = calibrate_scale(model, [observation], [bad_pose])
        assert math.isnan(result.phi)
    else:
        result, good = fit_pose(model, 1.0, [observation, observation], [bad_pose, pose])
        assert math.isfinite(good.energy)
        assert good.termination != "non_finite"
    assert not result.converged
    assert result.termination == "non_finite"
    assert result.iterations == 0
