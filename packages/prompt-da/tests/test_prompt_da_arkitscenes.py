"""Behavioral tests for the ARKitScenes PromptDA API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("pyarrow", reason="ARKitScenes catalog deps live in the prompt-da-catalog envs")
pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog deps live in the prompt-da-catalog envs")

from rerun_prompt_da.apis.arkitscenes_shared import (  # noqa: E402
    filter_depth_for_fusion,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)
from rerun_prompt_da.apis.prompt_da_arkitscenes import (  # noqa: E402
    orientation_quarter_turns_from_segment_row,
    portrait_from_segment_row,
    rotate_landscape_to_portrait,
    rotate_portrait_to_landscape,
)


def test_portrait_property_parses_catalog_list_value() -> None:
    """Interpret the first catalog property value instead of list truthiness."""
    assert portrait_from_segment_row({"property:portrait:value": ["True"]}) is True
    assert portrait_from_segment_row({"property:portrait:value": ["False"]}) is False


def test_portrait_property_is_required() -> None:
    """Reject segments whose required orientation metadata is absent."""
    with pytest.raises(KeyError, match="portrait"):
        portrait_from_segment_row({})


def test_orientation_quarter_turns_parses_catalog_list_value() -> None:
    """Preserve ingest's stored rotation direction for portrait inference."""
    assert orientation_quarter_turns_from_segment_row({"property:orientation_quarter_turns_ccw:value": ["3"]}) == 3


def test_portrait_rotation_round_trip_preserves_asymmetric_layout() -> None:
    """Undo ingest's CCW portrait bake for inference and restore it afterward."""
    portrait = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.uint16)
    landscape = rotate_portrait_to_landscape(portrait, 1)
    assert_array_equal(landscape, np.array([[30, 20, 10], [31, 21, 11]], dtype=np.uint16))
    assert_array_equal(rotate_landscape_to_portrait(landscape, 1), portrait)
    assert_array_equal(rotate_landscape_to_portrait(rotate_portrait_to_landscape(portrait, 3), 3), portrait)


def test_stride_for_uses_nearest_native_frame_interval() -> None:
    """Choose the closest whole-frame stride without dropping below one."""
    assert stride_for(60.0, 10.0) == 6
    assert stride_for(60.0, 60.0) == 1
    assert stride_for(60.0, 7.0) == 9
    assert stride_for(60.0, 120.0) == 1


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
    assert segments_to_process(rows, "one", False, "promptda") == ["one"]


def test_segments_to_process_rejects_unknown_explicit_segment() -> None:
    """Show available ids when an explicit segment does not exist."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    with pytest.raises(SystemExit, match="one.*two"):
        segments_to_process(rows, "missing", False, "promptda")


def test_segments_to_process_all_skips_existing_promptda_layers() -> None:
    """Process only segments that do not already carry the target layer."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base", "promptda"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    assert segments_to_process(rows, None, True, "promptda") == ["two"]


@pytest.mark.parametrize(("video_id", "process_all"), [(None, False), ("one", True)])
def test_segments_to_process_requires_exactly_one_selection_mode(video_id: str | None, process_all: bool) -> None:
    """Reject missing and ambiguous segment selection modes."""
    with pytest.raises(SystemExit, match="exactly one"):
        segments_to_process([], video_id, process_all, "promptda")


def test_resilient_decoder_turns_decode_errors_into_skippable_none() -> None:
    """A per-sample AV1 decode failure becomes a None frame, not a crash."""
    from unittest.mock import patch

    import av
    import pyarrow as pa
    from rerun.experimental.dataloader import VideoFrameDecoder

    from rerun_prompt_da.apis.prompt_da_arkitscenes import ResilientVideoFrameDecoder

    decoder = ResilientVideoFrameDecoder(codec="av1", keyframe_interval=300, fps_estimate=60.0)
    error = av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
    with patch.object(VideoFrameDecoder, "decode", side_effect=error):
        assert decoder.decode(pa.chunked_array([pa.array([b"packet"])]), 0, "segment") is None
    # Retain the observed failure for diagnostics. The dav1d_flush deadlock specifically needs an
    # errored, un-drained context finalized at interpreter shutdown; prompt del + gc is safe.
    assert decoder.decode_failures == [error]
