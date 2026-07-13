"""Behavioral tests for the ARKitScenes PromptDA API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rerun_prompt_da.apis.prompt_da_arkitscenes import (
    filter_depth_for_fusion,
    k_matrix_from_flat,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)


def test_stride_for_uses_nearest_native_frame_interval() -> None:
    """Choose the closest whole-frame stride without dropping below one."""
    assert stride_for(60.0, 10.0) == 6
    assert stride_for(60.0, 60.0) == 1
    assert stride_for(60.0, 7.0) == 9
    assert stride_for(60.0, 120.0) == 1


def test_k_matrix_from_flat_decodes_column_major_intrinsics() -> None:
    """Restore Rerun's flattened column-major camera matrix."""
    flat = np.array([1600.0, 0.0, 0.0, 0.0, 1590.0, 0.0, 950.0, 710.0, 1.0])
    assert_array_equal(k_matrix_from_flat(flat), np.array([[1600.0, 0.0, 950.0], [0.0, 1590.0, 710.0], [0.0, 0.0, 1.0]]))


@pytest.mark.parametrize(
    ("quaternion", "expected_rotation"),
    [
        (np.array([0.0, 0.0, 0.0, 1.0]), np.eye(3)),
        (
            np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_world_t_cam_from_pose_builds_rigid_transform(quaternion: np.ndarray, expected_rotation: np.ndarray) -> None:
    """Combine xyzw rotation and translation into a homogeneous pose."""
    transform = world_t_cam_from_pose(np.array([1.0, 2.0, 3.0]), quaternion)
    assert_allclose(transform[:3, :3], expected_rotation, atol=1e-7)
    assert_array_equal(transform[:3, 3], np.array([1.0, 2.0, 3.0]))
    assert_array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_filter_depth_for_fusion_masks_low_confidence_and_far_depth() -> None:
    """Keep only medium-or-better depth within the fusion range."""
    depth = np.array([[1000, 2000, 3000, 5000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000]], dtype=np.uint16)
    confidence = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    filtered = filter_depth_for_fusion(depth, confidence, 4.0)
    assert_array_equal(
        filtered,
        np.array([[0, 0, 3000, 0], [0, 0, 3000, 4000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000]], dtype=np.uint16),
    )
    assert filtered.dtype == np.uint16


def test_segments_to_process_selects_explicit_segment_for_replacement() -> None:
    """Allow explicitly selected segments even when PromptDA already exists."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base", "promptda"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    assert segments_to_process(rows, "one", False) == ["one"]


def test_segments_to_process_rejects_unknown_explicit_segment() -> None:
    """Show available ids when an explicit segment does not exist."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    with pytest.raises(SystemExit, match="one.*two"):
        segments_to_process(rows, "missing", False)


def test_segments_to_process_all_skips_existing_promptda_layers() -> None:
    """Process only segments that do not already carry the target layer."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base", "promptda"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    assert segments_to_process(rows, None, True) == ["two"]


@pytest.mark.parametrize(("video_id", "process_all"), [(None, False), ("one", True)])
def test_segments_to_process_requires_exactly_one_selection_mode(video_id: str | None, process_all: bool) -> None:
    """Reject missing and ambiguous segment selection modes."""
    with pytest.raises(SystemExit, match="exactly one"):
        segments_to_process([], video_id, process_all)


def test_resilient_decoder_turns_decode_errors_into_skippable_none() -> None:
    """A per-sample AV1 decode failure becomes a None frame, not a crash."""
    from unittest.mock import patch

    import av
    from rerun.experimental.dataloader import VideoFrameDecoder

    from rerun_prompt_da.apis.prompt_da_arkitscenes import ResilientVideoFrameDecoder

    decoder = ResilientVideoFrameDecoder(codec="av1", keyframe_interval=300, fps_estimate=60.0)
    error = av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
    with patch.object(VideoFrameDecoder, "decode", side_effect=error):
        assert decoder.decode(object(), 0, "segment") is None
    # The poisoned context must stay pinned (releasing it can deadlock dav1d teardown).
    assert decoder.decode_failures == [error]
