"""Numpy-only public binding tests, using a synthetic hand (no model asset)."""

import numpy as np
import pytest
from numpy.typing import NDArray

import handfit


def test_rejects_incompatible_model_topology() -> None:
    with pytest.raises(ValueError, match="topology"):
        handfit.HandFitter(
            np.zeros((20, 3), dtype=np.float32),
            np.zeros((20, 3), dtype=np.float32),
            np.zeros((21, 3), dtype=np.float32),
            np.zeros((21, 3), dtype=np.float32),
            np.zeros((20, 2), dtype=np.float32),
            np.zeros((21, 3), dtype=np.int64),
            np.zeros((4, 22), dtype=np.int64),
            np.array([1, 0.04, 20, 0.1, 0.087, 10, 0.001, 0.000001, 0.001], dtype=np.float64),
        )


def synthetic_fitter() -> tuple[handfit.HandFitter, NDArray[np.float32]]:
    """A rigid synthetic palm: joints have zero axes, so only its wrist moves."""
    indices = np.array([[f, f, f] for f in [4, 7, 10, 13, 16, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 1, 14, 15]] + [[1, 8, 5]], dtype=np.int64)
    parent = [255, 0, 1, 2, 255, 4, 5, 6, 255, 8, 9, 10, 255, 12, 13, 14, 255, 16, 17, 18, 255, 20]
    frame = [255, 2, 3, 4, 255, 5, 6, 7, 255, 8, 9, 10, 255, 11, 12, 13, 255, 14, 15, 16, 255, 0]
    child = [1, 2, 3, 255, 5, 6, 7, 255, 9, 10, 11, 255, 13, 14, 15, 255, 17, 18, 19, 255, 21, 255]
    sibling = [4, 255, 255, 255, 8, 255, 255, 255, 12, 255, 255, 255, 16, 255, 255, 255, 20, 255, 255, 255, 255, 255]
    rng = np.random.default_rng(3)
    rest = rng.uniform(-50, 50, (21, 3)).astype(np.float32)
    weights = np.zeros((21, 3), dtype=np.float32)
    weights[:, 0] = 1
    fitter = handfit.HandFitter(
        np.zeros((20, 3), dtype=np.float32),
        np.zeros((20, 3), dtype=np.float32),
        rest,
        weights,
        np.tile(np.array([-1, 1], dtype=np.float32), (20, 1)),
        indices,
        np.array([parent, frame, child, sibling], dtype=np.int64),
        np.array([1, 0.04, 0, 0.1, 0, 30, 1e-9, 1e-10, 1e-3], dtype=np.float64),
    )
    fitter.add_camera(np.eye(4, dtype=np.float32), np.array([500, 510], dtype=np.float32), np.array([320, 240], dtype=np.float32))
    return fitter, rest


def frame_inputs(rest: NDArray[np.float32], hands: int = 2) -> list[NDArray[np.float32] | NDArray[np.int64]]:
    sides = np.arange(hands, dtype=np.int64) % 2
    rotations = np.tile(np.eye(3, dtype=np.float32), (hands, 1, 1))
    translations = np.tile(np.array([0.015, -0.013, 0.62], dtype=np.float32), (hands, 1))
    angles = np.zeros((hands, 22), dtype=np.float32)
    angles[:, 20:] = [0.4, -0.5]
    cameras = np.tile(np.array([0, -1], dtype=np.int64), (hands, 1))
    world = np.tile(np.eye(4, dtype=np.float32), (hands, 2, 1, 1))
    pixels = np.full((hands, 2, 21, 2), np.nan, dtype=np.float32)
    weights = np.zeros((hands, 2, 21), dtype=np.float32)
    distances = np.full((hands, 2, 21), np.nan, dtype=np.float32)
    for h, side in enumerate(sides):
        points = rest / 1000 * np.array([-1 if side else 1, 1, 1], dtype=np.float32) + np.array([0.02, -0.01, 0.6], dtype=np.float32)
        pixels[h, 0] = points[:, :2] / points[:, 2:] * [500, 510] + [320, 240]
        weights[h, 0] = 1
        distances[h, 0] = np.linalg.norm(points, axis=-1) * 1000
    return [sides, rotations, translations, angles, cameras, world, pixels, weights, distances]


@pytest.mark.parametrize("central_difference", [False, True])
def test_synthetic_fit_recovers_wrist_and_preserves_carried_angles(central_difference: bool) -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest)
    results = fitter.fit(*inputs, central_difference=central_difference)
    for i, result in enumerate(results):
        np.testing.assert_allclose(result.translation, [0.02, -0.01, 0.6], atol=2e-6)
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=2e-5)
        np.testing.assert_array_equal(result.joint_angles[20:], inputs[3][i, 20:])
        assert result.energy < 1e-7
        assert result.converged
        assert np.linalg.det(result.rotation) == pytest.approx(1, abs=1e-6)
        alone = fitter.fit(*(a[i : i + 1] for a in inputs), central_difference=central_difference)[0]
        np.testing.assert_array_equal(result.translation, alone.translation)
        assert result.iterations == alone.iterations


def test_masked_nan_wrist_uses_palm_distance_reference() -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest, 1)
    inputs[7][0, 0, 5] = 0
    inputs[6][0, 0, 5] = np.nan
    inputs[8][0, 0, 5] = np.nan
    output = fitter.fit(*inputs)[0]
    np.testing.assert_allclose(output.translation, [0.02, -0.01, 0.6], atol=2e-6)
    assert np.isfinite(output.energy)


def test_no_evidence_and_invalid_shapes() -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest, 1)
    inputs[7].fill(0)
    result = fitter.fit(*inputs)[0]
    assert result.termination == "stationary"
    assert result.converged
    assert result.energy == 0.0
    np.testing.assert_array_equal(result.translation, inputs[2][0])
    inputs[1] = np.zeros((1, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="rotations"):
        fitter.fit(*inputs)


def test_nonfinite_start_does_not_poison_other_hand() -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest)
    inputs[2][0, 0] = np.nan
    outputs = fitter.fit(*inputs)
    assert outputs[0].termination == "non_finite"
    assert not outputs[0].converged
    np.testing.assert_allclose(outputs[1].translation, [0.02, -0.01, 0.6], atol=2e-6)


def test_mixed_warm_cold_and_insufficient_evidence() -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest, 4)
    inputs[7][2] = 0
    inputs[7][3] = 0
    inputs[7][3, 0, :2] = 1
    outputs = fitter.fit(*inputs, has_previous=np.array([True, False, False, False]))
    for output in outputs[:2]:
        np.testing.assert_allclose(output.translation, [0.02, -0.01, 0.6], atol=2e-6)
        assert output.converged
    np.testing.assert_array_equal(outputs[0].joint_angles[20:], [np.float32(0.4), np.float32(-0.5)])
    np.testing.assert_array_equal(outputs[1].joint_angles[20:], [0, 0])
    assert len(outputs[1].rigid_iterations) == 26
    assert len(outputs[1].chosen_hypotheses) == 2
    assert len(outputs[1].full_iterations) == 4
    for output in outputs[2:]:
        assert output.termination == "no_evidence"
        assert not output.converged
        assert output.iterations == 0
        assert np.isnan([output.e_2d, output.e_dist, output.e_temporal, output.energy]).all()
        np.testing.assert_array_equal(output.rotation, np.eye(3))
        np.testing.assert_array_equal(output.translation, [0, 0, 0])
        np.testing.assert_array_equal(output.joint_angles, np.zeros(22))
    again = fitter.fit(*inputs, has_previous=np.array([True, False, False, False]))
    assert outputs[1].chosen_hypotheses == again[1].chosen_hypotheses
    assert outputs[1].winner == again[1].winner
    np.testing.assert_array_equal(outputs[1].translation, again[1].translation)


def test_cold_alignment_falls_back_to_observed_fingers() -> None:
    fitter, rest = synthetic_fitter()
    inputs = frame_inputs(rest, 1)
    inputs[7].fill(0)
    inputs[7][0, 0, :5] = 1
    output = fitter.fit(*inputs, has_previous=np.array([False]))[0]
    np.testing.assert_allclose(output.translation, [0.02, -0.01, 0.6], atol=2e-6)
    assert output.energy < 1e-7
    assert output.converged
