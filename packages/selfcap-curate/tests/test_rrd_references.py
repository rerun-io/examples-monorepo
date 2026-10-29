"""Local golden streams retain exact detector values and sub-nanosecond times."""

import numpy as np
from frozen_v2 import load_keypoint_reference, load_keypoints2d, load_oak_detections, load_oak_reference
from rrd_references import KeypointReference, OakReference, reference_tables, write_keypoint_reference, write_oak_reference

from selfcap_curate.keypoints import Keypoints2D, ViewDetections
from selfcap_curate.records import OakDetections


def test_gpu_reference_adapters_roundtrip_exact_arrays(tmp_path):
    times = np.array([np.nextafter(0.04, 1.0), np.nextafter(0.08, 0.0)])
    xy = np.arange(532, dtype=np.float32).reshape(2, 133, 2).astype(np.float64)
    xy[0, 1] = np.nan
    scores = np.full((2, 133), float(np.float16(0.7)))
    oak = OakReference(np.array([2, 4], dtype=np.int64), 50.0, times, OakDetections(times + 0.001, xy, scores), 192, 160)
    path = tmp_path / "oak.rrd"
    write_oak_reference(path, "selfcap__synthetic", "/world/rig_03/cam_00", oak)
    loaded = load_oak_reference(path)
    assert (loaded.fps, loaded.width, loaded.height) == (50.0, 192, 160)
    np.testing.assert_array_equal(loaded.indices, oak.indices)
    np.testing.assert_array_equal(loaded.times_s, times)
    actual = load_oak_detections(path)
    for field in ("video_time", "xy", "scores"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(oak.detections, field))
    boxes = np.array([[[0.125, 1.75, 4.125, 5.5], [np.nan] * 4]] * 2)
    view = ViewDetections(boxes, np.array([[0.875, np.nan]] * 2), np.stack([xy, xy], axis=1), np.stack([scores, scores], axis=1))
    keypoints = Keypoints2D(times, {"Phone": view}, 3.0)
    reference = KeypointReference(keypoints, {"Phone": times + 0.0000000001}, {"Phone": times + 0.02}, 1)
    path = tmp_path / "keypoints.rrd"
    write_keypoint_reference(path, "selfcap__synthetic", {"Phone": "/world/rig_00/cam_00"}, reference)
    loaded = load_keypoint_reference(path)
    assert loaded.schema_version == 1
    assert loaded.detections.rate == 3.0
    np.testing.assert_array_equal(loaded.pts_s["Phone"], reference.pts_s["Phone"])
    np.testing.assert_array_equal(loaded.video_times_s["Phone"], reference.video_times_s["Phone"])
    actual = load_keypoints2d(path)
    np.testing.assert_array_equal(actual.times, keypoints.times)
    assert tuple(actual.views) == ("Phone",)
    for field in ("boxes", "box_scores", "xy", "scores"):
        np.testing.assert_array_equal(getattr(actual.views["Phone"], field), getattr(view, field))

    # Padded-person corners remain exact metadata, but must not poison viewer bounds.
    boxes_table = reference_tables(path)["/world/rig_00/cam_00/pinhole/pred/rtmw_persons/boxes"]
    assert np.isfinite(np.asarray(boxes_table["Boxes2D:half_sizes"].to_pylist())).all()
