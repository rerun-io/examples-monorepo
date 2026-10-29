"""CPU triangulation rejects bad views, missing confidence and degenerate rays."""

import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from selfcap_curate.keypoints import Camera, Keypoints2D, ViewDetections, project, select_wearer, triangulate, triangulate_episode, write_keypoints


def cameras():
    result = []
    for x, y in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]:
        pose = np.eye(4)
        pose[:3, 3] = [x, y, 0.0]
        result.append(Camera(np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]]), pose, (-0.12, 0.02), "camera"))
    return result


def test_weighted_ransac_keeps_consensus_and_missing_joints() -> None:
    cams = cameras()
    truth = np.array([[0.2, -0.3, 4.0], [0.7, 0.4, 5.0], [-0.4, 0.3, 3.0]])
    uv = np.stack([project(c, truth)[:, :2] for c in cams])
    uv[3, 0] += [140.0, -90.0]
    scores = np.ones((4, 3))
    scores[1:, 2] = 0.1
    result = triangulate(cams, uv, scores)
    np.testing.assert_allclose(result.xyz[:2], truth[:2], atol=1e-4)
    assert result.n_views.tolist() == [3, 4, 0]
    assert np.isnan(result.xyz[2]).all()


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
    assert len(rows) == 2
    assert all(batch.num_rows == 2 for batch in rows)
    assert {str(c.entity_path) for c in RrdReader(output).stream() if str(c.entity_path).startswith("/__properties")} == {"/__properties/calib"}
