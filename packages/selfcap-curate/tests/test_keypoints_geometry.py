"""CPU triangulation rejects bad views, missing confidence and degenerate rays."""

import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from selfcap_curate.keypoints import (
    DISPLAY_CONFIDENCE,
    Camera,
    Keypoints2D,
    Keypoints3D,
    SelectedView,
    ViewDetections,
    project,
    select_wearer,
    smooth_keypoints,
    triangulate,
    triangulate_episode,
    write_keypoints,
)


def cameras():
    result = []
    for x, y in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]:
        pose = np.eye(4)
        pose[:3, 3] = [x, y, 0.0]
        result.append(
            Camera(np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]]), pose, (-0.12, 0.02), f"/world/rig_{len(result):02d}/cam_00")
        )
    return result


def test_weighted_ransac_keeps_consensus_and_missing_joints() -> None:
    cams = cameras()
    truth = np.array([[0.2, -0.3, 4.0], [0.7, 0.4, 5.0], [-0.4, 0.3, 3.0]])
    uv = np.stack([project(c, truth)[:, :2] for c in cams])
    uv[3, 0] += [140.0, -90.0]
    scores = np.ones((4, 3))
    scores[1:, 2] = 0.1
    scores[0, 0], scores[1, 0] = 0.6, 0.8
    result = triangulate(cams, uv, scores)
    np.testing.assert_allclose(result.xyz[:2], truth[:2], atol=1e-4)
    assert result.n_views.tolist() == [3, 4, 0]
    assert np.isnan(result.xyz[2]).all()
    # Mean RTMW score of the agreeing views; the outlier view 3 of joint 0 does not count.
    np.testing.assert_allclose(result.confidence, [0.8, 1.0, 0.0], atol=1e-6)


def test_wearer_and_degenerate_rays() -> None:
    cams = cameras()
    anchors = np.array([[0.2, -0.3, 4.0], [0.1, 0.0, 4.0], [0.4, 0.0, 4.0]])
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [1000.0, 400.0, 1400.0, 900.0], [np.nan] * 4])
    assert select_wearer(cams[0], anchors, boxes, np.array([0.9, 0.7, 0.0])) == 1
    point = np.array([[0.0, 0.0, 4.0]])
    uv = np.stack([project(cams[0], point)[:, :2]] * 2)
    assert triangulate([cams[0], cams[0]], uv, np.ones((2, 1))).n_views[0] == 0


def test_nonfinite_observations_and_behind_camera_are_missing() -> None:
    cams = cameras()
    truth = np.array([[0.1, 0.2, 4.0], [0.2, 0.1, 3.0], [0.3, 0.1, 4.0]])
    uv = np.stack([project(c, truth)[:, :2] for c in cams])
    scores = np.ones((4, 3))
    uv[2, 0], uv[3, 0], scores[3, 1], scores[:, 2] = np.nan, np.inf, np.inf, np.nan
    result = triangulate(cams, uv, scores)
    assert result.n_views.tolist() == [2, 3, 0]
    np.testing.assert_allclose(result.xyz[:2], truth[:2], atol=1e-4)
    behind = np.array([[0.0, 0.0, -4.0]])
    assert triangulate(cams, np.stack([project(c, behind)[:, :2] for c in cams]), np.ones((4, 1))).n_views[0] == 0


def test_appending_all_missing_frames_preserves_calibration_and_empty_rows(tmp_path) -> None:

    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="selfcap__synthetic", send_properties=False) as recording:
        recording.save(base)
        recording.send_property("calib", rr.AnyValues(method="unchanged", exo_cameras=1))
    original = base.read_bytes()
    observed = ViewDetections(np.zeros((2, 1, 4)), np.zeros((2, 1)), np.full((2, 1, 133, 2), np.nan), np.zeros((2, 1, 133)))
    data = Keypoints2D(np.array([0.0, 0.1]), {"phone": observed}, 10.0)
    camera = cameras()[0]
    result = triangulate_episode(data, {"phone": camera}, np.tile([0.0, 0.0, 4.0], (2, 3, 1)))
    output = tmp_path / "complete.rrd"
    write_keypoints(base, output, result)
    assert base.read_bytes() == original
    rows = [c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static]
    assert len(rows) == 3  # 3D, the phone's RTMW overlay and the phone's projection
    assert all(batch.num_rows == 2 for batch in rows)
    assert all("frame_index" not in batch.schema.names for batch in rows)
    points = next(batch for batch in rows if "Points3D:positions" in batch.schema.names)
    assert [len(row) for row in points.column("Points3D:positions").to_pylist()] == [133, 133]
    assert points.column("simplecv.KeypointConfidence3D:confidences").to_pylist() == [[0.0] * 133] * 2
    assert {str(c.entity_path) for c in RrdReader(output).stream() if str(c.entity_path).startswith("/__properties")} == {
        "/__properties/calib",
        "/__properties/projections",
    }


def three_d(times, xyz, confidence, selected, cams, rate) -> Keypoints3D:
    return Keypoints3D(times, xyz.astype(np.float32), np.full(xyz.shape[:2], 4, np.uint8), confidence.astype(np.float32), selected, cams, rate)


def test_smoothing_cuts_jitter_keeps_gaps_and_leaves_slow_plans_raw() -> None:
    rng = np.random.default_rng(0)
    times = np.arange(100) / 25.0
    truth = np.stack([np.sin(times), np.cos(times), 3.0 + 0.1 * times], axis=1)
    xyz = np.repeat((truth + rng.normal(0.0, 0.01, truth.shape))[:, None], 133, axis=1)
    xyz[40:43, 5] = np.nan  # a gap is never bridged
    xyz[:, 6] = np.nan
    xyz[10:14, 6] = truth[10:14]  # a run shorter than the window stays raw
    raw = three_d(times, xyz, np.ones((100, 133)), {}, {}, 25.0)
    smooth = smooth_keypoints(raw)

    def jitter(points):
        return np.nanmedian(np.linalg.norm(points[2:] - 2 * points[1:-1] + points[:-2], axis=-1))

    assert jitter(smooth.xyz[:, 0]) < 0.4 * jitter(raw.xyz[:, 0])
    assert np.nanmedian(np.linalg.norm(smooth.xyz[:, 0] - truth, axis=1)) < np.nanmedian(np.linalg.norm(raw.xyz[:, 0] - truth, axis=1))
    np.testing.assert_array_equal(np.isfinite(smooth.xyz), np.isfinite(raw.xyz))
    np.testing.assert_array_equal(smooth.xyz[10:14, 6], raw.xyz[10:14, 6])
    slow = three_d(times, xyz, np.ones((100, 133)), {}, {}, 2.0)
    np.testing.assert_array_equal(smooth_keypoints(slow).xyz, slow.xyz)


def test_writer_hides_low_confidence_joints_and_projects_into_each_phone(tmp_path) -> None:
    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="selfcap__synthetic", send_properties=False) as recording:
        recording.save(base)
        recording.send_property("calib", rr.AnyValues(method="unchanged", exo_cameras=1))
    camera = cameras()[0]
    xyz = np.tile(np.linspace([-0.3, -0.3, 3.5], [0.3, 0.3, 4.5], 133), (2, 1, 1))
    confidence = np.full((2, 133), 0.9)
    confidence[:, 5] = DISPLAY_CONFIDENCE - 0.05
    uv = np.tile(project(camera, xyz[0])[:, :2], (2, 1, 1)).astype(np.float32)
    scores = np.full((2, 133), 0.9)
    scores[:, 7] = 0.5
    result = three_d(np.array([0.0, 0.04]), xyz, confidence, {"phone": SelectedView(uv, scores)}, {"phone": camera}, 25.0)
    output = tmp_path / "complete.rrd"
    write_keypoints(base, output, result)

    def row(entity, kind):
        batch = next(c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static and str(c.entity_path) == entity)
        return np.asarray(batch.column(f"{kind}:positions").to_pylist()[0], dtype=float)

    points = row("/world/pred/rtmw_triangulated/coco133_xyz", "Points3D")
    assert np.isnan(points[5]).all() and np.isfinite(np.delete(points, 5, axis=0)).all()
    overlay = row("/world/rig_00/cam_00/pinhole/pred/rtmw/coco133_uv", "Points2D")
    assert np.isnan(overlay[7]).all() and np.isfinite(np.delete(overlay, 7, axis=0)).all()
    projected = row("/world/rig_00/cam_00/pinhole/coco133_uv_projected", "Points2D")
    assert np.isnan(projected[5]).all()
    np.testing.assert_allclose(np.delete(projected, 5, axis=0), np.delete(uv[0], 5, axis=0), atol=1e-3)
