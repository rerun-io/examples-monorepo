"""Reference-clock and matched-joint comparison contracts."""

import numpy as np
import pytest

pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")

from posekit.apis.egohand_comparison import EgoCamera, project_ego


def test_reference_projection_interpolates_pose_and_rejects_gaps() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.0, 0.08, 0.4]
    camera = EgoCamera(
        4,
        0,
        np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        np.zeros(14),
        (100, 100),
        np.eye(4),
        np.array([0.0, 0.08, 0.4]),
        poses,
    )
    points = np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32)
    result = project_ego(camera, np.array([0.04, 0.2]), points)
    np.testing.assert_allclose(result[0, 0], [46.0, 50.0], atol=1e-6)
    assert np.isnan(result[1]).all()


def test_quest_clock_applies_offset_minus_hand_latency() -> None:
    import pyarrow as pa
    from serde.json import to_json

    from posekit.apis.egohand_comparison import ComparisonClock, quest_pixels
    from posekit.models.egohand_geometry import OAKCamera

    points = np.zeros((3, 133, 3), dtype=np.float32)
    points[:, 91:, 2] = 1.0
    points[:, 91:, 0] = np.array([0.0, 0.02, 0.04])[:, None]
    table = pa.table(
        {
            "capture_time": pa.array([0, 20_000_000, 40_000_000], type=pa.duration("ns")),
            "video_time": pa.array([0, 50_000_000, 100_000_000], type=pa.duration("ns")),
            "/world/gt/coco133_xyz:Points3D:positions": points.tolist(),
            "/world/rig_04:Transform3D:mat3x3": [np.eye(3).reshape(-1).tolist()] * 3,
            "/world/rig_04:Transform3D:translation": [[0.0, 0.0, 0.0]] * 3,
        }
    )
    clock = pa.table({"property:time_sync:record_json": [[to_json(ComparisonClock({"quest_pose": 100.0}, [], 48.0))]]})
    camera = OAKCamera(np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]), np.zeros(14), 100, 100)
    actual = quest_pixels(table, clock, camera, np.array([0.062]))
    np.testing.assert_allclose(actual[..., 0], 51.0, atol=1e-5)
    np.testing.assert_allclose(actual[..., 1], 50.0, atol=1e-5)


def test_errors_use_identical_joint_slots_for_both_models() -> None:
    from posekit.apis.egohand_comparison import summarize

    quest = np.zeros((1, 2, 21, 2), dtype=np.float32)
    ego = np.full_like(quest, np.nan)
    sapiens = np.full_like(quest, np.nan)
    ego[0, 0, 0] = [3.0, 4.0]
    sapiens[0, 0, 0] = [0.0, 10.0]
    ego[0, 0, 1] = [100.0, 100.0]  # absent in Sapiens, so absent in BOTH summaries
    report = summarize(ego, sapiens, quest)
    assert report.pairs == 1
    assert report.egohand_median_px == 5.0
    assert report.sapiens_median_px == 10.0
