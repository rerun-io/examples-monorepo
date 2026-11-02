"""CPU triangulation rejects bad views, missing confidence and degenerate rays."""

import numpy as np
import pytest
import rerun as rr
from dataforge.schema import SHOWN_MIN_CONFIDENCE
from rerun.chunk import RrdReader

from selfcap_curate.keypoints import (
    HEADSET_COVERED,
    Camera,
    EgoCamera,
    Keypoints2D,
    Keypoints3D,
    SelectedView,
    ViewDetections,
    project,
    reprojection_confidence,
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
    scores[0, 0], scores[1, 0] = 0.7, 0.8
    result = triangulate(cams, uv, scores)
    np.testing.assert_allclose(result.xyz[:2], truth[:2], atol=1e-4)
    assert result.n_views.tolist() == [3, 4, 0]
    assert np.isnan(result.xyz[2]).all()
    # Kineo's pairwise score: view 3 of joint 0 saw the joint 166 px away (s = exp(-40 x 166/900) ~ 6e-4), so its three
    # pairs score ~0 and pull the mean down; joint 1's four views agree exactly at full score.
    miss = np.exp(-40.0 * np.hypot(140.0, 90.0) / 900.0)
    pairs = np.sqrt([0.7 * 0.8, 0.7 * 1.0, 0.8 * 1.0, 0.7 * miss, 0.8 * miss, 1.0 * miss])
    np.testing.assert_allclose(result.confidence, [pairs.sum() / 6.0, 1.0, 0.0], atol=1e-4)


def test_confidence_is_kineos_pairwise_reprojection_score() -> None:
    """Kineo (arXiv 2510.24464, eq. 11-14): each view's miss over its focal length, s = exp(-lambda e), each pair the
    geometric mean of w s, averaged over the pairs of views that saw the joint. lambda = 40: an 8.9 px miss at f = 890 px
    gives s = exp(-0.4)."""
    errors = np.array([[[0.0], [0.0], [8.9]]])
    scores = np.array([[[0.9], [0.8], [1.0]]])
    seen = np.ones((1, 3, 1), bool)
    s3 = np.exp(-0.4)
    expected = (np.sqrt(0.9 * 0.8) + np.sqrt(0.9 * s3) + np.sqrt(0.8 * s3)) / 3.0
    np.testing.assert_allclose(reprojection_confidence(errors, scores, np.full(3, 890.0), seen), [[expected]], rtol=1e-6)
    seen[:, 1:] = False
    assert reprojection_confidence(errors, scores, np.full(3, 890.0), seen)[0, 0] == 0.0  # one view: no pair


def test_views_that_disagree_a_little_all_count() -> None:
    """A hidden joint each phone guesses slightly differently (two phones 6 px one way, two the other, as RTMW-x's ears on
    f28dc7af): the point comes from all four phones, between the guesses, not from whichever pair agrees exactly."""
    cams = cameras()
    truth = np.array([[0.2, -0.3, 4.0]])
    uv = np.stack([project(c, truth)[:, :2] for c in cams])
    uv[:2, 0, 0] += 6.0
    uv[2:, 0, 0] -= 6.0
    result = triangulate(cams, uv, np.ones((4, 1)))
    assert result.n_views[0] == 4
    assert np.linalg.norm(result.xyz[0] - truth[0]) < 0.005


def test_wearer_and_degenerate_rays() -> None:
    cams = cameras()
    anchors = np.array([[0.2, -0.3, 4.0], [0.1, 0.0, 4.0], [0.4, 0.0, 4.0]])
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [1000.0, 400.0, 1400.0, 900.0], [np.nan] * 4])
    assert select_wearer(cams[0], anchors, boxes, np.array([0.9, 0.7, 0.0])) == 1
    point = np.array([[0.0, 0.0, 4.0]])
    uv = np.stack([project(cams[0], point)[:, :2]] * 2)
    assert triangulate([cams[0], cams[0]], uv, np.ones((2, 1))).n_views[0] == 0


def test_two_phones_facing_each_other_cannot_fix_depth_alone() -> None:
    """Rays from phones on opposite sides lie on one line (here 1.4 degrees apart as lines, 178.6 as rays): the point can
    slide along it and both still agree, so that pair alone gives no point. A third phone at 90 degrees fixes it."""
    lens = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    poses = [np.eye(4), np.eye(4), np.eye(4)]
    poses[1][:3, :3], poses[1][:3, 3] = np.diag([-1.0, 1.0, -1.0]), [0.0, 0.0, 8.0]  # at z = 8, looking back down -z
    poses[2][:3, :3], poses[2][:3, 3] = np.c_[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], [4.0, 0.0, 4.0]  # looking down -x
    cams = [Camera(lens, pose, (0.0, 0.0), f"/world/rig_{i:02d}/cam_00") for i, pose in enumerate(poses)]
    truth = np.array([[0.05, 0.0, 4.0]])
    uv = np.stack([project(c, truth)[:, :2] for c in cams])
    assert triangulate(cams[:2], uv[:2], np.ones((2, 1))).n_views[0] == 0
    result = triangulate(cams, uv, np.ones((3, 1)))
    assert result.n_views[0] == 3
    np.testing.assert_allclose(result.xyz[0], truth[0], atol=1e-4)


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


@pytest.mark.parametrize(
    ("model_args", "properties"),
    [({}, {"calib", "projections"}), ({"model": "sapiens2"}, {"calib", "keypoints_sapiens2"})],
    ids=["default-rtmw", "sapiens2"],
)
def test_appending_all_missing_frames_preserves_calibration_and_empty_rows(tmp_path, model_args: dict[str, str], properties: set[str]) -> None:
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
    write_keypoints(base, output, result, **model_args)
    assert base.read_bytes() == original
    rows = [c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static and not str(c.entity_path).endswith("/skeleton")]
    assert len(rows) == 5  # 3D, geometric support, raw 2D, its empty display copy and projection
    assert all(batch.num_rows == 2 for batch in rows)
    assert all("frame_index" not in batch.schema.names for batch in rows)
    points = next(batch for batch in rows if "Points3D:positions" in batch.schema.names)
    assert [len(row) for row in points.column("Points3D:positions").to_pylist()] == [133, 133]
    assert points.column("simplecv.KeypointConfidence3D:confidences").to_pylist() == [[0.0] * 133] * 2
    assert {str(c.entity_path) for c in RrdReader(output).stream() if str(c.entity_path).startswith("/__properties")} == {
        f"/__properties/{name}" for name in properties
    }


def three_d(times, xyz, confidence, selected, cams, rate) -> Keypoints3D:
    return Keypoints3D(times, xyz.astype(np.float32), np.full(xyz.shape[:2], 4, np.uint8), confidence.astype(np.float32), selected, cams, rate)


def test_smoothing_cuts_jitter_without_lag_keeps_gaps_and_leaves_slow_plans_raw() -> None:
    rng = np.random.default_rng(0)
    times = np.arange(100) / 25.0
    truth = np.stack([np.sin(times), np.cos(times), 3.0 + 0.1 * times], axis=1)
    xyz = np.repeat((truth + rng.normal(0.0, 0.01, truth.shape))[:, None], 133, axis=1)
    xyz[40:43, 5] = np.nan  # a gap is never bridged
    xyz[:, 6] = np.repeat(np.linspace(0.0, 2.0, 100)[:, None], 3, axis=1)  # a steady 0.5 m/s ramp
    raw = three_d(times, xyz, np.ones((100, 133)), {}, {}, 25.0)
    smooth = smooth_keypoints(raw)

    def jitter(points):
        return np.nanmedian(np.linalg.norm(points[2:] - 2 * points[1:-1] + points[:-2], axis=-1))

    assert jitter(smooth.xyz[:, 0]) < 0.4 * jitter(raw.xyz[:, 0])
    assert np.nanmedian(np.linalg.norm(smooth.xyz[:, 0] - truth, axis=1)) < np.nanmedian(np.linalg.norm(raw.xyz[:, 0] - truth, axis=1))
    np.testing.assert_array_equal(np.isfinite(smooth.xyz), np.isfinite(raw.xyz))
    # Forward and backward lags cancel: away from the ends a steady motion comes back where it was.
    np.testing.assert_allclose(smooth.xyz[20:80, 6], raw.xyz[20:80, 6], atol=1e-3)
    slow = three_d(times, xyz, np.ones((100, 133)), {}, {}, 2.0)
    np.testing.assert_array_equal(smooth_keypoints(slow).xyz, slow.xyz)


@pytest.mark.parametrize(
    ("model_args", "xyz_path", "uv_suffix", "projection_name"),
    [
        ({}, "/world/pred/rtmw_triangulated/coco133_xyz", "pred/rtmw/coco133_uv", "coco133_uv_projected"),
        ({"model": "sapiens2"}, "/world/gt/sapiens2/coco133_xyz", "sapiens2/coco133_uv", "sapiens2/coco133_uv_projected"),
    ],
    ids=["default-rtmw", "sapiens2"],
)
def test_writer_keeps_low_confidence_3d_and_projects_into_each_phone(
    tmp_path, model_args: dict[str, str], xyz_path: str, uv_suffix: str, projection_name: str
) -> None:
    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="selfcap__synthetic", send_properties=False) as recording:
        recording.save(base)
        recording.send_property("calib", rr.AnyValues(method="unchanged", exo_cameras=1))
    camera = cameras()[0]
    xyz = np.tile(np.linspace([-0.3, -0.3, 3.5], [0.3, 0.3, 4.5], 133), (2, 1, 1))
    confidence = np.full((2, 133), 0.9)
    confidence[:, [3, 4, 6]] = [0.7, 0.85, 1.0]
    confidence[:, 5] = SHOWN_MIN_CONFIDENCE - 0.05
    assert {0, 1, 2, 23, 90} <= set(HEADSET_COVERED.tolist()) and not {3, 4, 5, 7, 91} & set(HEADSET_COVERED.tolist())
    uv = np.tile(project(camera, xyz[0])[:, :2], (2, 1, 1)).astype(np.float32)
    scores = np.full((2, 133), 0.9)
    scores[:, [3, 4, 6]] = [0.7, 0.85, 1.0]
    scores[:, 7] = 0.5
    result = three_d(np.array([0.0, 0.04]), xyz, confidence, {"phone": SelectedView(uv, scores)}, {"phone": camera}, 25.0)
    output = tmp_path / "complete.rrd"
    ego = EgoCamera(4, 0, camera.K, np.zeros(14), (1920, 1080), np.eye(4), result.times, np.stack([camera.world_T_cam] * 2))
    write_keypoints(base, output, result, ego=(ego,), quest_hands=(result.times, result.xyz), **model_args)

    def row(entity, kind):
        batch = next(c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static and str(c.entity_path) == entity)
        return np.asarray(batch.column(f"{kind}:positions").to_pylist()[0], dtype=float)

    # Low-confidence 3D stays shown; the 2D entity keeps every detection and the headset-covered joints.
    hidden3d = HEADSET_COVERED
    points = row(xyz_path, "Points3D")
    assert np.isnan(points[hidden3d]).all() and np.isfinite(np.delete(points, hidden3d, axis=0)).all()
    overlay = row(f"/world/rig_00/cam_00/pinhole/{uv_suffix}", "Points2D")
    assert np.isfinite(overlay).all()
    projected = row(f"/world/rig_00/cam_00/pinhole/{projection_name}", "Points2D")
    assert np.isnan(projected[hidden3d]).all()
    np.testing.assert_allclose(np.delete(projected, hidden3d, axis=0), np.delete(uv[0], hidden3d, axis=0), atol=1e-3)

    # Scores 0.7, 0.85, 1.0: on the detector scale (0.7-1.0) red, yellow, green; on the 3D scale (0.2-1.0) yellow-green to green.
    on_3d_scale, on_detector_scale = [[191, 255, 0], [95, 255, 0], [0, 255, 0]], [[255, 0, 0], [255, 255, 0], [0, 255, 0]]
    for entity, kind, expected in (
        (xyz_path, "Points3D", on_3d_scale),
        (f"/world/rig_00/cam_00/pinhole/{uv_suffix}", "Points2D", on_detector_scale),
        (f"/world/rig_00/cam_00/pinhole/{projection_name}", "Points2D", on_3d_scale),
        (f"/world/rig_04/cam_00/pinhole/{projection_name}", "Points2D", on_3d_scale),
    ):
        batch = next(c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static and str(c.entity_path) == entity)
        packed = np.asarray(batch.column(f"{kind}:colors").to_pylist()[0], dtype=np.uint32)[[3, 4, 6]]
        rgb = np.array([list(int(value).to_bytes(4, "big")[:3]) for value in packed])
        np.testing.assert_allclose(rgb, expected, atol=1)
        np.testing.assert_array_equal(
            np.asarray(batch.column(next(name for name in batch.schema.names if name.endswith(":confidences"))).to_pylist()[0])[[3, 4, 6]],
            np.array([0.7, 0.85, 1.0], dtype=np.float32),
        )
    quest = next(
        c.to_record_batch()
        for c in RrdReader(output).stream()
        if not c.is_static and str(c.entity_path) == "/world/rig_04/cam_00/pinhole/quest_coco133_uv_projected"
    )
    assert "Points2D:colors" not in quest.schema.names  # Its blue annotation colour stays in control.


def test_display_writer_preserves_coco_ids_with_confidence_colours(tmp_path) -> None:
    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="display", send_properties=False) as recording:
        recording.save(base)
        recording.send_property("calib", rr.AnyValues(method="unchanged"))
    camera = cameras()[0]
    xyz = np.tile([0.1, 0.2, 4.0], (2, 133, 1))
    result = three_d(np.array([0.0, 0.02]), xyz, np.ones((2, 133)), {}, {"phone": camera}, 25.0)
    output = tmp_path / "display.rrd"
    write_keypoints(base, output, result)
    rows = {str(c.entity_path): c.to_record_batch() for c in RrdReader(output).stream() if not c.is_static and any(name.endswith(":positions") for name in c.to_record_batch().schema.names)}
    assert "/world/rig_00/cam_00/pinhole/skeleton" not in rows
    for path, kind in (("/world/skeleton", "Points3D"),):
        batch = rows[path]
        assert any(name.endswith(":confidences") for name in batch.schema.names)
        assert any(name.endswith(":average_confidence") for name in batch.schema.names)
        assert "display_confidence" not in batch.schema.names
        points = np.asarray(batch.column(f"{kind}:positions").to_pylist()[0])
        assert len(points) == 133
        assert np.isnan(points[np.r_[0:3, 9:11, 13:91]]).all()
        assert np.isfinite(points[np.r_[3:9, 11:13, 91:133]]).all()
    assert not any("LineStrips" in name for c in RrdReader(output).stream() for name in c.to_record_batch().schema.names)
    static = [c.to_record_batch() for c in RrdReader(output).stream() if c.is_static and str(c.entity_path) == "/world/skeleton"]
    radii = [batch["Points3D:radii"].to_pylist() for batch in static if "Points3D:radii" in batch.schema.names]
    assert radii[-1] == [[-3.0]]  # UI points, independent of camera distance.
