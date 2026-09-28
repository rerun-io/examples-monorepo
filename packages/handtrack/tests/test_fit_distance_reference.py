"""Distance evidence uses the first observed wrist or palm reference in each view."""

from dataclasses import replace

import pytest
import torch
from jaxtyping import Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from test_fit_synthetic import _landmark_error_mm, _perturbed, _visible_scene
from torch import Tensor

from handtrack.fit.observations import HandObservation, ViewObservation
from handtrack.fit.pose_fit import FitConfig, FitResult, fit_pose
from handtrack.hand.pose import HandPose, Side, generic_hand_model


@pytest.mark.parametrize("wrist_weight, palm_weight, expected", [(0.0, 0.25, 10.0), (0.5, 0.25, 20.0), (0.0, 0.0, 0.0)])
def test_distance_reference_uses_first_observed_wrist_or_palm(wrist_weight: float, palm_weight: float, expected: float) -> None:
    model: HandModelTorch = generic_hand_model()
    truth, observation = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(3), (0,))
    view: ViewObservation = observation.views[0]
    weights: Float32[Tensor, "21"] = torch.zeros(21)
    weights[0], weights[5], weights[20] = 0.4, wrist_weight, palm_weight
    distances: Float32[Tensor, "21"] = view.d_rel_mm.clone()
    distances[0] += 10.0
    distances[weights == 0] = torch.nan
    changed: HandObservation = replace(observation, views=(replace(view, weights=weights, d_rel_mm=distances),))
    result: FitResult = fit_pose(model, 1.0, [changed], [truth], FitConfig(max_iterations=0, temporal_weight=0.0))[0]
    # Only the fingertip has a 10 mm error: 10² × 0.4 × reference weight.
    assert result.e_dist == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("cameras", [(0,), (0, 1)])
@pytest.mark.parametrize("side", list(Side))
def test_exact_distances_recover_pose_without_observed_wrist(cameras: tuple[int, ...], side: Side) -> None:
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(1)
    truth, observation = _visible_scene(model, side, generator, cameras)
    assert all(view.weights[5] > 0 and view.weights[20] > 0 for view in observation.views)
    missing_wrist: list[ViewObservation] = []
    for view in observation.views:
        weights: Float32[Tensor, "21"] = view.weights.clone()
        weights[5] = 0.0
        distances: Float32[Tensor, "21"] = view.d_rel_mm.clone()
        distances[5] = torch.nan
        missing_wrist.append(replace(view, weights=weights, d_rel_mm=distances))
    fallback: HandObservation = replace(observation, views=tuple(missing_wrist))
    start: HandPose = _perturbed(truth, generator, angle_rad=0.25, translation_m=0.02, joint_rad=0.2)
    config: FitConfig = FitConfig(temporal_weight=0.0, max_iterations=40)
    observed, recovered = fit_pose(model, 1.0, [observation, fallback], [start, start], config)
    observed_error: float = _landmark_error_mm(model, observed.pose, truth, side)
    recovered_error: float = _landmark_error_mm(model, recovered.pose, truth, side)
    # Compare recovered geometry: float32 precision can stop LM by damping at an exact fit.
    assert observed_error < 0.05
    assert recovered_error < 0.05
    assert recovered_error <= observed_error + 0.05


def test_stereo_views_choose_distance_references_independently() -> None:
    model: HandModelTorch = generic_hand_model()
    truth, observation = _visible_scene(model, Side.LEFT, torch.Generator().manual_seed(3), (0, 1))
    changed: list[ViewObservation] = []
    for view, wrist_weight in zip(observation.views, (0.0, 0.5), strict=True):
        weights: Float32[Tensor, "21"] = torch.zeros(21)
        weights[0], weights[5], weights[20] = 0.4, wrist_weight, 0.25
        distances: Float32[Tensor, "21"] = view.d_rel_mm.clone()
        distances[0] += 10.0
        changed.append(replace(view, weights=weights, d_rel_mm=distances))
    result: FitResult = fit_pose(
        model, 1.0, [replace(observation, views=tuple(changed))], [truth], FitConfig(max_iterations=0, temporal_weight=0.0)
    )[0]
    assert result.e_dist == pytest.approx(30.0, abs=0.004)
