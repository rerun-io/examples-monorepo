"""Cross-validation must retain observed constraints in each camera."""

import numpy as np
import pytest

from selfcap_curate.exo_refine_catalog import Inputs, balanced_folds, heldout_hands
from selfcap_curate.factory import QuestBlock, optical_targets


def test_sparse_visibility_is_present_in_both_training_folds() -> None:
    visible = np.array([[True, True], [True, False], [True, True], [True, False]])
    folds = balanced_folds(visible)
    for fold in (0, 1):
        assert visible[folds == fold].any(axis=0).all()


def test_insufficient_constraints_fail_instead_of_fitting_an_empty_fold() -> None:
    with pytest.raises(ValueError, match="two"):
        balanced_folds(np.array([[True, False], [True, True]]))


def test_borrowed_camera50_is_the_physical_left_target() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.001, -0.012, -0.073]
    targets = optical_targets(QuestBlock({}, {"1": transform}, "faulty-left-borrowed-cam50"))
    assert list(targets) == ["quest0"]
    np.testing.assert_array_equal(targets["quest0"], transform)


def test_healthy_targets_keep_factory_baseline() -> None:
    transform = np.eye(4)
    transform[0, 3] = 0.063
    targets = optical_targets(QuestBlock({}, {"1": transform}, "healthy"))
    np.testing.assert_array_equal(targets["quest0"], np.eye(4))
    np.testing.assert_array_equal(targets["quest1"], transform)


def test_clustered_hand_frames_populate_both_training_folds(fixture_sync) -> None:
    from selfcap_curate.exo_landmarks import Landmarks
    from selfcap_curate.exo_refine import RefinementProblem
    from selfcap_curate.records import HandTrack, PhoneDetections

    times = np.arange(.98, 1.11, .01)
    hand = HandTrack(times, np.ones((len(times), 21, 3)), np.ones(len(times), dtype=bool), np.ones(len(times)))
    detection = PhoneDetections(np.array([1.01, 1.03, 1.05, 1.07]), np.ones((4, 133, 2)), np.ones((4, 133)))
    problem = RefinementProblem((), (), Landmarks([], "test", "synthetic", "2026-10-30"), np.zeros((0, 3)), np.zeros((0, 3, 3)))
    inputs = Inputs(problem, ("phone",), (detection,), {"left": hand, "right": hand}, fixture_sync(), (), (0.0,), (0.0,), 0.0)
    for fold in (0, 1):
        assert len(heldout_hands(inputs, fold, train=True)[0].world) == 60
        assert len(heldout_hands(inputs, fold, train=False)[0].world) == 60


def test_partial_refinement_preserves_original_rig_ids() -> None:
    from selfcap_curate.exo_calibration import Camera
    from selfcap_curate.exo_refine import Refinement

    camera = Camera(np.zeros(11))
    result = Refinement((camera, camera), np.zeros(3), "labels", "fixed", camera_rigs=(1, 3))
    assert result.camera_for_rig(0) is None
    assert result.camera_for_rig(1) is camera
    assert result.camera_for_rig(2) is None
    assert result.camera_for_rig(3) is camera


def test_missing_and_null_initial_cameras_are_excluded() -> None:
    from selfcap_curate.calib_layer import CalibSession
    from selfcap_curate.exo_calibration import Camera, PhoneCalibration, SessionCalibration
    from selfcap_curate.exo_label import calibrated_rigs
    from selfcap_curate.records import OakFit

    camera = Camera(np.zeros(11))
    state = CalibSession('test', SessionCalibration('test', {'A': PhoneCalibration(True, camera),
                         'B': PhoneCalibration(False), 'D': PhoneCalibration(True, camera)}),
                         OakFit('test', False), QuestBlock({}, {}, 'unknown-layout'), {})
    assert calibrated_rigs(state, ('A', 'B', 'C', 'D')) == (0, 3)
