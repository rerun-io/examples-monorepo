# SPDX-License-Identifier: Apache-2.0
"""Behaviour of `colsfm.trajectory`, the audit's one trajectory scorer.

The point of these tests is the distinction the module exists to keep: an SE(3)
fit holds the scale at 1.0 and an SIM(3) fit estimates it, both are always
reported, and `would_be_scale` says what the free fit would have chosen. A tool
that scored a monocular model with only the rigid number, or a rig model with
only the similarity one, would be answering a different question.
"""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from conftest import build_reconstruction
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.alignment import Positions, RigidAlignment, RigTrack, rig_track_from_frames_meta
from colsfm.frames_meta import FramesMeta
from colsfm.geometry import Vector3
from colsfm.trajectory import (
    MIN_ALIGNMENT_SAMPLES,
    TrajectoryScore,
    image_center_score,
    indices_of_timestamps,
    rig_track_from_reconstruction,
    score_positions,
    score_track_against_ground_truth,
)

SQUARE_XYZ: Positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
"""Four coplanar points: enough to determine a fit, small enough to reason about."""


def test_an_identical_trajectory_scores_zero_under_both_alignments() -> None:
    """No displacement, no scale change: every residual is zero and the scale is one."""
    score: TrajectoryScore = score_positions(SQUARE_XYZ, SQUARE_XYZ)

    assert score.num_matched == 4
    assert score.rigid_rmse_millimeters == pytest.approx(0.0, abs=1e-9)
    assert score.rigid_max_millimeters == pytest.approx(0.0, abs=1e-9)
    assert score.similarity_rmse_millimeters == pytest.approx(0.0, abs=1e-9)
    assert score.would_be_scale == pytest.approx(1.0)


def test_a_rescaled_trajectory_separates_the_two_alignments() -> None:
    """This is the whole distinction: SIM(3) absorbs a global scale, SE(3) must not."""
    score: TrajectoryScore = score_positions(SQUARE_XYZ, 2.0 * SQUARE_XYZ)

    assert score.would_be_scale == pytest.approx(2.0)
    assert score.similarity_rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert score.rigid_rmse_millimeters > 100.0


def test_a_translated_trajectory_scores_zero_under_both(rigid_shift_meters: float = 3.0) -> None:
    """A rigid motion is absorbed by both fits; only the scale tells them apart."""
    score: TrajectoryScore = score_positions(SQUARE_XYZ, SQUARE_XYZ + rigid_shift_meters)

    assert score.rigid_rmse_millimeters == pytest.approx(0.0, abs=1e-9)
    assert score.similarity_rmse_millimeters == pytest.approx(0.0, abs=1e-9)
    assert score.would_be_scale == pytest.approx(1.0)


def test_residuals_are_reported_in_millimetres() -> None:
    """One point pushed a millimetre out reads as a fraction of a millimetre, not of a metre.

    The fit absorbs most of a single point's displacement, so the residual is
    below the millimetre that caused it — but it is in that decade, which is the
    unit this test exists to pin.
    """
    displaced: Positions = SQUARE_XYZ.copy()
    displaced[0, 0] += 0.001

    score: TrajectoryScore = score_positions(SQUARE_XYZ, displaced)

    assert score.rigid_max_millimeters == pytest.approx(0.637335, abs=1e-6)
    assert score.rigid_rmse_millimeters == pytest.approx(0.395275, abs=1e-6)


def test_too_few_samples_score_as_not_a_number() -> None:
    """Below three samples a three-dimensional fit is undetermined, not zero-error."""
    score: TrajectoryScore = score_positions(SQUARE_XYZ[:2], SQUARE_XYZ[:2])

    assert score.num_matched == 2
    assert np.isnan(score.rigid_rmse_millimeters)
    assert np.isnan(score.similarity_rmse_millimeters)
    assert np.isnan(score.would_be_scale)
    assert MIN_ALIGNMENT_SAMPLES == 3


def test_a_track_scored_against_itself_matches_every_sample(three_samples: FramesMeta) -> None:
    """The timestamp join keeps every sample when both tracks carry the same stamps."""
    track: RigTrack = rig_track_from_frames_meta(three_samples)

    score: TrajectoryScore = score_track_against_ground_truth(track, track)

    assert score.num_matched == len(track)
    assert score.rigid_rmse_millimeters == pytest.approx(0.0, abs=1e-9)


def test_the_timestamp_join_drops_samples_the_reference_lacks(three_samples: FramesMeta) -> None:
    """A reference covering one sample scores against that sample alone."""
    track: RigTrack = rig_track_from_frames_meta(three_samples)
    single: RigTrack = track.take(np.asarray([0], dtype=np.int64))

    assert score_track_against_ground_truth(track, single).num_matched == 1


def test_a_reconstruction_reproduces_the_rig_track_it_was_built_from(three_samples: FramesMeta) -> None:
    """`rig_track_from_reconstruction` inverts what `build_reconstruction` composed."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)

    recovered: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples)
    expected: RigTrack = rig_track_from_frames_meta(three_samples)

    np.testing.assert_array_equal(recovered.timestamps_microseconds, expected.timestamps_microseconds)
    np.testing.assert_allclose(recovered.world_t_rig, expected.world_t_rig, atol=1e-9)


def test_a_rigid_alignment_carries_through_to_the_rig_track(three_samples: FramesMeta) -> None:
    """A rigid fit commutes with the camera-to-vehicle composition, so it maps the track whole.

    The check is independent of how the function composes: whatever it does to a
    camera pose, a scale-free rotation and translation of the world must come out
    as the same rotation and translation of the rig positions.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    plain: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples)
    quarter_turn_about_z: Float64[ndarray, "3 3"] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    moved: RigidAlignment = RigidAlignment(
        target_R_source=quarter_turn_about_z,
        target_t_source=np.array([5.0, -2.0, 0.5], dtype=np.float64),
        rmse_meters=0.0,
        max_error_meters=0.0,
        would_be_scale=1.0,
    )

    aligned: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples, moved)

    np.testing.assert_allclose(aligned.world_t_rig, moved.apply(plain.world_t_rig), atol=1e-9)


def scaling_alignment(scale: float) -> RigidAlignment:
    """A fit that only rescales the world, so a track's response to scale can be isolated.

    Args:
        scale: The similarity scale to apply.

    Returns:
        An alignment with no rotation and no translation.
    """
    return RigidAlignment(
        target_R_source=np.eye(3, dtype=np.float64),
        target_t_source=np.zeros(3, dtype=np.float64),
        rmse_meters=0.0,
        max_error_meters=0.0,
        would_be_scale=scale,
        scale=scale,
    )


def test_a_scaling_alignment_reaches_the_camera_poses(three_samples: FramesMeta) -> None:
    """A SIM(3) fit's scale is applied to the camera poses, before the rig composition.

    The rig positions are *not* simply multiplied by the scale: the
    camera-to-vehicle offset is a calibrated length no fit rescales. What the
    scale does reach is the camera centre, so the track's response to it is
    affine — doubling and tripling move the track by the same step. An
    implementation that ignored the scale would move it by nothing.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    plain: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples)
    doubled: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples, scaling_alignment(2.0))
    tripled: RigTrack = rig_track_from_reconstruction(reconstruction, three_samples, scaling_alignment(3.0))

    step_meters: Float64[ndarray, "n 3"] = doubled.world_t_rig - plain.world_t_rig
    np.testing.assert_allclose(tripled.world_t_rig - doubled.world_t_rig, step_meters, atol=1e-9)
    assert np.linalg.norm(step_meters, axis=1).min() > 1.0


def test_image_centres_score_against_the_poses_the_model_carries(three_samples: FramesMeta) -> None:
    """A model scored against its own camera centres has no residual and unit scale."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    centres: dict[str, Vector3] = {
        image.name: np.asarray(image.cam_from_world().inverse().translation, dtype=np.float64)
        for image in reconstruction.images.values()
    }

    score, similarity = image_center_score(reconstruction, centres)

    assert score.num_matched == len(centres)
    assert score.rigid_rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert similarity.scale == pytest.approx(1.0)


def test_image_centres_of_a_monocular_model_recover_its_missing_scale(three_samples: FramesMeta) -> None:
    """A model at half scale is a zero-residual SIM(3) fit and a large SE(3) one."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    centres: dict[str, Vector3] = {
        image.name: 2.0 * np.asarray(image.cam_from_world().inverse().translation, dtype=np.float64)
        for image in reconstruction.images.values()
    }

    score, similarity = image_center_score(reconstruction, centres)

    assert similarity.scale == pytest.approx(2.0)
    assert score.would_be_scale == pytest.approx(2.0)
    assert score.similarity_rmse_millimeters == pytest.approx(0.0, abs=1e-3)
    assert score.rigid_rmse_millimeters > score.similarity_rmse_millimeters


def test_image_centres_refuse_a_model_with_too_little_ground_truth(three_samples: FramesMeta) -> None:
    """Two matched images cannot determine a fit, and silence would be worse than an error."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    names: list[str] = sorted(image.name for image in reconstruction.images.values())[:2]
    centres: dict[str, Vector3] = {name: np.zeros(3, dtype=np.float64) for name in names}

    with pytest.raises(ValueError, match="carry ground truth"):
        image_center_score(reconstruction, centres)


def test_selecting_no_timestamps_keeps_every_sample(three_samples: FramesMeta) -> None:
    """`None` means "the track's own samples", the default of every audit comparison."""
    track: RigTrack = rig_track_from_frames_meta(three_samples)

    np.testing.assert_array_equal(indices_of_timestamps(track, None), np.arange(len(track)))


def test_selecting_timestamps_keeps_them_in_track_order(three_samples: FramesMeta) -> None:
    """The common sample set is a filter, not a reordering."""
    track: RigTrack = rig_track_from_frames_meta(three_samples)
    stamps: Int64[ndarray, "n"] = track.timestamps_microseconds
    keep: set[int] = {int(stamps[-1]), int(stamps[0])}

    selected: Int64[ndarray, "m"] = indices_of_timestamps(track, keep)

    np.testing.assert_array_equal(selected, [0, len(track) - 1])


def test_selecting_an_absent_timestamp_selects_nothing(three_samples: FramesMeta) -> None:
    """A timestamp no sample carries contributes no index rather than an error."""
    track: RigTrack = rig_track_from_frames_meta(three_samples)
    positions: Float64[ndarray, "m"] = indices_of_timestamps(track, {-1})

    assert positions.size == 0
