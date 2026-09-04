"""Pure tests for the Robocap catalog application's time, config, and logging contracts."""

from typing import Any

import cv2
import numpy as np
import pytest
import rerun as rr
import torch
from jaxtyping import Float32, Int32, UInt8
from numpy import ndarray
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.rerun_logging import person_color
from posekit.skeletons import COCO_17
from scipy.spatial.transform import Rotation

from lamptrack.apis.lamp_catalog import (
    Config,
    _log_camera_observations,
    _log_person,
    best_detection_window,
    build_time_grid,
    interpolate_pose,
    log_static_context,
)
from lamptrack.models.lamp import PersonState


def _batches_by_component(archetype: Any) -> dict[str, Any]:
    """Index one archetype's component batches by their component name."""
    return {batch.component_descriptor().component: batch for batch in archetype.as_component_batches()}


def _spy_on_rerun_log(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, dict[str, Any]]]:
    """Capture every ``rr.log`` call instead of writing to a recording."""
    logged: list[tuple[str, Any, dict[str, Any]]] = []
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **kwargs: logged.append((entity_path, archetype, kwargs)))
    return logged


def _person_state(track_id: int) -> PersonState:
    """Build a one-step smoothed window with distinct joint positions."""
    return PersonState(
        track_id=track_id,
        timestamps_ns=np.asarray([0], dtype=np.int64),
        joints_world=np.arange(72, dtype=np.float32).reshape(1, 24, 3),
        betas=np.zeros(10, dtype=np.float32),
        root_T=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
        rotations=np.tile(np.eye(3, dtype=np.float32), (1, 24, 1, 1)),
    )


def test_catalog_defaults_select_outward_robocap_cameras() -> None:
    """The default stream order is the settled four-view LAMP order."""
    config = Config()

    assert config.segment_id == "robocap__f408193e6447b3b0__s00000029"
    assert config.cams == ("cam_00", "cam_01", "cam_04", "cam_05")
    assert config.fps == 10.0
    assert config.start_s == 1152.0
    assert config.max_seconds == 120.0


def test_catalog_rejects_more_than_four_cameras_even_when_four_are_distinct() -> None:
    """Runtime validation rejects malformed CLI camera lists, not only duplicates."""
    with pytest.raises(ValueError, match="four distinct"):
        Config(cams=("cam_00", "cam_01", "cam_04", "cam_05", "cam_00"))


def test_build_time_grid_uses_shared_video_overlap_and_absolute_start() -> None:
    """Sampling begins at the requested video-time second within all streams."""
    video_times = [
        np.arange(437, 1001, dtype="timedelta64[s]").astype("timedelta64[ns]"),
        np.arange(438, 1002, dtype="timedelta64[s]").astype("timedelta64[ns]"),
    ]

    grid = build_time_grid(video_times, fps=2.0, start_s=880.0, max_seconds=2.0)

    assert np.array_equal(grid, np.asarray([880.0, 880.5, 881.0, 881.5], dtype=np.float64) * 1e9)


def test_interpolate_pose_interpolates_translation_and_rotation() -> None:
    """Rig motion uses linear translation and quaternion slerp."""
    times = np.asarray([0, 2_000_000_000], dtype=np.int64)
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[1, :3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    poses[1, :3, 3] = np.asarray([2.0, 4.0, 6.0])

    midpoint = interpolate_pose(times, poses, 1_000_000_000)

    assert np.allclose(midpoint[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(midpoint[:3, :3], Rotation.from_euler("z", 45.0, degrees=True).as_matrix())
    assert np.array_equal(midpoint[3], [0.0, 0.0, 0.0, 1.0])


def test_best_detection_window_uses_earliest_maximum() -> None:
    """The 120-second selection is deterministic when two windows tie."""
    sample_seconds = np.arange(600.0, 901.0)
    counts = np.zeros_like(sample_seconds, dtype=np.int64)
    counts[(sample_seconds >= 650.0) & (sample_seconds < 660.0)] = 2
    counts[(sample_seconds >= 780.0) & (sample_seconds < 790.0)] = 2

    start_s, total = best_detection_window(sample_seconds, counts, window_seconds=120.0)

    assert start_s == 600.0
    assert total == 20


def test_static_context_logs_one_annotation_context_per_camera_and_for_people(monkeypatch: pytest.MonkeyPatch) -> None:
    """Annotation contexts are static setup, logged once, never per frameset."""
    logged = _spy_on_rerun_log(monkeypatch)

    log_static_context(("cam_00", "cam_01"))

    contexts = [(entity_path, kwargs) for entity_path, archetype, kwargs in logged if isinstance(archetype, rr.AnnotationContext)]
    assert [entity_path for entity_path, _ in contexts] == [
        "world/people",
        "world/rig_00/cam_00/pinhole",
        "world/rig_00/cam_01/pinhole",
    ]
    assert all(kwargs["static"] for _, kwargs in contexts), "contexts must not be re-sent on every frameset"


def test_log_person_draws_annotated_joints_a_trail_and_a_translucent_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """People render like the replay tool: annotation-context edges and a 50% mesh."""
    logged = _spy_on_rerun_log(monkeypatch)
    state: PersonState = _person_state(track_id=3)
    vertices: Float32[ndarray, "4 3"] = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=np.float32
    )
    faces: Int32[ndarray, "4 3"] = np.asarray([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)
    trails: dict[int, list[Float32[ndarray, "3"]]] = {}

    _log_person(state, vertices, faces, trails)
    _log_person(state, vertices, faces, trails)

    entities = [entity_path for entity_path, _, _ in logged]
    assert entities == [
        "world/people/3/joints",
        "world/people/3/pelvis_trail",
        "world/people/3/mesh",
    ] * 2, "no separate skeleton entity: the SMPL annotation context draws the edges"

    joints = _batches_by_component(logged[0][1])
    assert joints["Points3D:keypoint_ids"].as_arrow_array().to_pylist() == list(range(24))
    assert joints["Points3D:class_ids"].as_arrow_array().to_pylist() == [0]

    trail = _batches_by_component(logged[4][1])
    assert len(trail["LineStrips3D:strips"].as_arrow_array().to_pylist()[0]) == 2, "the pelvis trail grows one point per frameset"

    mesh = _batches_by_component(logged[2][1])
    normals: Float32[ndarray, "4 3"] = np.asarray(mesh["Mesh3D:vertex_normals"].as_arrow_array().to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(normals, vertices / np.sqrt(np.float32(3.0)), atol=1e-6)
    red, green, blue = person_color(3)
    assert mesh["Mesh3D:albedo_factor"].as_arrow_array().to_pylist() == [(red << 24) | (green << 16) | (blue << 8) | 128]


def test_log_camera_observations_halves_the_preview_and_its_overlays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection runs at full resolution while the logged preview is half size."""
    logged = _spy_on_rerun_log(monkeypatch)
    image: UInt8[ndarray, "1080 1920 3"] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxes = BoxDetections(
        xyxy=torch.asarray([[10.0, 20.0, 110.0, 220.0]], dtype=torch.float32),
        scores=torch.asarray([0.9], dtype=torch.float32),
        frame_indices=torch.zeros(1, dtype=torch.int64),
        track_ids=torch.asarray([7], dtype=torch.int64),
    )
    keypoints = Keypoints2d(
        xy=torch.arange(34, dtype=torch.float32).reshape(1, 17, 2) * 2.0,
        scores=torch.ones((1, 17), dtype=torch.float32),
        frame_indices=torch.zeros(1, dtype=torch.int64),
        skeleton=COCO_17,
    )

    _log_camera_observations("cam_00", image, boxes, keypoints, keypoint_conf_min=0.5)

    root = "world/rig_00/cam_00/pinhole/preview"
    assert [entity_path for entity_path, _, _ in logged] == [
        f"{root}/image",
        f"{root}/detections",
        f"{root}/detections/person_7/bbox",
        f"{root}/detections/person_7/keypoints",
    ]

    blob = _batches_by_component(logged[0][1])["EncodedImage:blob"].as_arrow_array().to_pylist()[0]
    assert cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (540, 960, 3)

    assert isinstance(logged[1][1], rr.Clear)
    assert logged[1][1].is_recursive.as_arrow_array().to_pylist() == [True]

    box = _batches_by_component(logged[2][1])
    assert box["Boxes2D:centers"].as_arrow_array().to_pylist() == [[30.0, 60.0]]
    assert box["Boxes2D:half_sizes"].as_arrow_array().to_pylist() == [[25.0, 50.0]]

    points = _batches_by_component(logged[3][1])
    positions: Float32[ndarray, "17 2"] = np.asarray(points["Points2D:positions"].as_arrow_array().to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(positions, np.arange(34, dtype=np.float32).reshape(17, 2))
    assert points["Points2D:keypoint_ids"].as_arrow_array().to_pylist() == list(range(17))
    assert points["Points2D:class_ids"].as_arrow_array().to_pylist() == [0]
