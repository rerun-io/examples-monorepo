"""GT pairing, alignment and the frame / source-frame / timestamp timelines."""

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.ground_truth import AlignedRun, GtContext, align_to_gt, protocol_for
from metrictrail.outputs import time_columns


def similarity_moved(world_T_cam: Float64[ndarray, "n 4 4"], scale: float) -> Float64[ndarray, "n 4 4"]:
    """The same poses in another world frame related by a known similarity (an up-to-scale SLAM result)."""
    rotation: Float64[ndarray, "3 3"] = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    moved: Float64[ndarray, "n 4 4"] = world_T_cam.copy()
    moved[:, :3, :3] = rotation @ world_T_cam[:, :3, :3]
    moved[:, :3, 3] = scale * world_T_cam[:, :3, 3] @ rotation.T + np.array([0.5, -1.0, 2.0])
    return moved


def test_align_to_gt_recovers_a_known_similarity(tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    context, world_T_cam = tum_like_context
    estimate: Float64[ndarray, "n 4 4"] = similarity_moved(world_T_cam, 0.5)
    ids: Int64[ndarray, "n"] = np.arange(60, dtype=np.int64)
    aligned: AlignedRun = align_to_gt(estimate, ids, context)
    assert aligned.rows.shape[0] == 60  # every 30 Hz frame is within 20 ms of a 100 Hz GT pose
    assert aligned.sim3.scale == pytest.approx(2.0, rel=1e-3)
    assert aligned.sim3.rmse_m < 0.01  # frame vs GT stamps differ by up to 5 ms of motion (float32 GT too)
    assert aligned.se3.rmse_m > 10 * aligned.sim3.rmse_m  # scale 1 cannot fit a half-size trajectory
    assert np.sqrt(np.mean(aligned.sim3_err_m**2)) == pytest.approx(aligned.sim3.rmse_m, rel=1e-9)


def test_align_to_gt_skips_lost_frames(tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    context, world_T_cam = tum_like_context
    estimate: Float64[ndarray, "n 4 4"] = world_T_cam.copy()
    estimate[[3, 7]] = np.nan
    aligned: AlignedRun = align_to_gt(estimate, np.arange(60, dtype=np.int64), context)
    assert 3 not in aligned.rows.tolist() and 7 not in aligned.rows.tolist()
    assert aligned.rows.shape[0] == 58


def test_time_columns_map_rows_to_source_frames_and_vslamlab_stamps(tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    context, _ = tum_like_context
    source_frame_ids: Int64[ndarray, "n"] = np.arange(0, 60, 5, dtype=np.int64)  # a stride-5 run (7-Scenes style)
    rows: Int64[ndarray, "k"] = np.array([0, 2, 11], dtype=np.int64)
    columns = time_columns(rows, source_frame_ids, context.frames.timestamps_ns)
    assert [column.timeline_name() for column in columns] == ["frame", "source_frame", "video_time"]
    arrays = [column.as_arrow_array().to_numpy() for column in columns]
    assert arrays[0].tolist() == [0, 2, 11]
    assert arrays[1].tolist() == [0, 10, 55]
    assert arrays[2].astype("int64").tolist() == context.frames.timestamps_ns[[0, 10, 55]].tolist()
    assert [column.timeline_name() for column in time_columns(rows, source_frame_ids, None)] == ["frame", "source_frame"]


def test_protocol_for_benchmarks() -> None:
    assert protocol_for("tum").dataset == "rgbdtum"
    assert protocol_for("kitti_calibrated").dataset == "kitti"
    with pytest.raises(ValueError, match="no VSLAM-LAB paper protocol"):
        protocol_for("lamaria")
