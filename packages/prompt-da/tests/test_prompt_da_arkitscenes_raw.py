"""Behavioral tests for raw-disk ARKitScenes PromptDA inference."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

pytest.importorskip("pyarrow", reason="ARKitScenes catalog deps live in the prompt-da-catalog envs")
pytest.importorskip("torchcodec", reason="raw video decode deps live in the prompt-da-catalog envs")
pytest.importorskip("arkitscenes_download", reason="ARKitScenes ingest deps live in the prompt-da-catalog envs")

from rerun_prompt_da.apis.prompt_da_arkitscenes_raw import (  # noqa: E402
    filter_depth_for_fusion,
    nearest_timestamped_path,
    retained_video_times,
    rotate_depth_for_catalog,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)


def test_stride_for_uses_nearest_native_frame_interval() -> None:
    """Choose the closest whole-frame stride without dropping below one."""
    assert stride_for(60.0, 10.0) == 6
    assert stride_for(60.0, 120.0) == 1


def test_nearest_timestamped_path_enforces_pairing_tolerance() -> None:
    """Pair the closest sensor sample and reject one beyond two milliseconds."""
    paths = [Path("frame_1.000.png"), Path("frame_1.010.png"), Path("frame_1.020.png")]
    assert nearest_timestamped_path(paths, 1.0119, tolerance_s=0.002) == paths[1]
    assert nearest_timestamped_path(paths, 1.0121, tolerance_s=0.002) is None
    assert nearest_timestamped_path(paths, 1.016, tolerance_s=None) == paths[2]


def test_retained_video_times_snap_only_the_first_near_epoch_sample() -> None:
    """Match ingest's first-sample snap while preserving later source times."""
    assert retained_video_times(np.array([9.9, 10.1, 10.2]), epoch=10.0, wide_drop=1).tolist() == [0.0, pytest.approx(0.2)]
    assert retained_video_times(np.array([10.3, 10.4]), epoch=10.0, wide_drop=0).tolist() == [pytest.approx(0.3), pytest.approx(0.4)]


def test_filter_depth_for_fusion_masks_confidence_and_far_depth() -> None:
    """Zero low-confidence and over-range values while retaining the threshold."""
    depth = np.array([[1000, 4000], [4001, 3000]], dtype=np.uint16)
    confidence = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    assert_array_equal(filter_depth_for_fusion(depth, confidence, 4.0), np.array([[0, 4000], [0, 3000]], dtype=np.uint16))


def test_world_t_cam_from_pose_round_trips_pose_components() -> None:
    """Preserve xyzw rotation and translation in the homogeneous transform."""
    translation = np.array([1.0, -2.0, 3.0])
    quaternion = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()
    world_t_cam = world_t_cam_from_pose(translation, quaternion)
    assert_allclose(world_t_cam[:3, 3], translation)
    assert_allclose(Rotation.from_matrix(world_t_cam[:3, :3]).as_quat(), quaternion)
    assert_allclose(world_t_cam @ np.linalg.inv(world_t_cam), np.eye(4), atol=1e-12)


@pytest.mark.parametrize(("video_id", "process_all"), [(None, False), ("one", True)])
def test_segments_to_process_requires_exactly_one_mode(tmp_path: Path, video_id: str | None, process_all: bool) -> None:
    """Reject missing and ambiguous segment selection modes."""
    with pytest.raises(SystemExit, match="exactly one"):
        segments_to_process([], video_id, process_all, tmp_path)


def test_process_all_skips_existing_layer_and_missing_raw_data(tmp_path: Path) -> None:
    """Select only unfinished catalog segments that also exist on local disk."""
    (tmp_path / "Training" / "ready").mkdir(parents=True)
    rows = [
        {"rerun_segment_id": "done", "rerun_layer_names": ["base", "promptda_raw"]},
        {"rerun_segment_id": "ready", "rerun_layer_names": ["base"]},
        {"rerun_segment_id": "missing", "rerun_layer_names": ["base"]},
    ]
    assert segments_to_process(rows, None, True, tmp_path) == ["ready"]


def test_catalog_depth_rotation_reverses_unbaking() -> None:
    """Rotate landscape prediction back to the catalog's baked orientation."""
    baked = np.arange(6, dtype=np.uint16).reshape(3, 2)
    for quarter_turns in range(4):
        landscape = np.ascontiguousarray(np.rot90(baked, -quarter_turns))
        assert_array_equal(rotate_depth_for_catalog(landscape, quarter_turns), baked)
