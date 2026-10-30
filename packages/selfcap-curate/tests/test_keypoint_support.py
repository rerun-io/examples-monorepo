"""Detector scores gate inputs; camera inliers describe geometric support after smoothing."""

import numpy as np
import pytest
from test_keypoints_geometry import cameras

from selfcap_curate.keypoints import Keypoints3D, SelectedView, project, smooth_keypoints, triangulate


def test_low_score_view_is_not_support_and_bad_view_is_identified() -> None:
    cams = cameras()
    point = np.array([[0.2, -0.3, 4.0]])
    uv = np.stack([project(camera, point)[:, :2] for camera in cams])
    uv[3] += 100.0
    scores = np.array([[0.299], [0.8], [0.9], [0.99]])
    result = triangulate(cams, uv, scores)
    np.testing.assert_allclose(result.xyz, point, atol=1e-5)
    np.testing.assert_array_equal(result.inliers[:, 0], [False, True, True, False])
    assert result.n_views.tolist() == [2]
    assert result.max_reprojection_px[0] < 1e-3


def test_smoothing_labels_temporal_estimates_without_inventing_support() -> None:
    cams = cameras()[:2]
    times = np.arange(8) * 0.04
    points = np.tile([0.0, 0.0, 4.0], (8, 133, 1)).astype(np.float32)
    points[4:, :, 0] = 0.3
    selected = {
        str(i): SelectedView(np.stack([project(camera, frame.astype(np.float64))[:, :2] for frame in points]).astype(np.float32), np.ones((8, 133)))
        for i, camera in enumerate(cams)
    }
    result = Keypoints3D(
        times,
        points,
        np.full((8, 133), 2, np.uint8),
        np.ones((8, 133), np.float32),
        selected,
        dict(zip(selected, cams, strict=True)),
        25.0,
        np.ones((8, 2, 133), bool),
        np.zeros((8, 133), np.float32),
    )
    smooth = smooth_keypoints(result)
    assert np.all((smooth.n_views == 0) | (smooth.n_views == 2))
    assert np.all(smooth.n_views[:, 91:] == 2)  # Hands retain their previous treatment.
    assert np.nanmax(smooth.max_reprojection_px) <= 8.0
    for i, camera in enumerate(cams):
        projected = np.stack([project(camera, frame.astype(np.float64))[:, :2] for frame in smooth.xyz])
        error = np.linalg.norm(projected - selected[str(i)].xy, axis=-1)
        assert np.nanmax(error[smooth.inliers[:, i]]) <= 8.0
    assert smooth.bridged.any()
    assert np.isfinite(smooth.xyz).all()


@pytest.mark.parametrize("model", ["rtmw", "sapiens2"])
def test_support_columns_are_stored_beside_each_3d_joint(tmp_path, model) -> None:
    import rerun as rr
    from rerun.chunk import RrdReader

    from selfcap_curate.keypoints import write_keypoints

    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="support", send_properties=False) as recording:
        recording.save(base)
        recording.send_property("calib", rr.AnyValues(method="test"))
    cams = cameras()[:2]
    result = Keypoints3D(np.array([0.0]), np.full((1, 133, 3), [0.0, 0.0, 4.0], np.float32),
                         np.full((1, 133), 2, np.uint8), np.ones((1, 133), np.float32), {},
                         {"a": cams[0], "b": cams[1]}, 25.0, np.ones((1, 2, 133), bool), np.full((1, 133), 0.25, np.float32))
    output = tmp_path / "support.rrd"
    write_keypoints(base, output, result, model=model)
    batch = next(c.to_record_batch() for c in RrdReader(output).stream() if "support_view_count" in c.to_record_batch().schema.names)
    assert batch.column("support_view_count").to_pylist() == [[2] * 133]
    assert batch.column("support_max_reprojection_px").to_pylist() == [[0.25] * 133]
    assert batch.column("support_inlier_mask").to_pylist() == [[[True, True]] * 133]
