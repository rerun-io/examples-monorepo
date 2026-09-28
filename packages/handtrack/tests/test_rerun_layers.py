from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from dataforge import schema
from dataforge.writing import atomic_recording
from fake_track import fake_detnet, fake_track, synthetic_rig, synthetic_truth
from jaxtyping import Bool, Float32
from numpy import ndarray
from rerun.chunk import RrdReader

from handtrack import rerun_layers
from handtrack.results import BoxSource, SegmentTrack


def _rows_by_entity(path: Path) -> dict[str, list[pa.RecordBatch]]:
    reader: RrdReader = RrdReader(path)
    batches: dict[str, list[pa.RecordBatch]] = {}
    for chunk in reader.stream(store=reader.recordings()[0]).to_chunks():
        batches.setdefault(str(chunk.entity_path), []).append(chunk.to_record_batch())
    return batches


def _instances(batches: list[pa.RecordBatch], component: str) -> list[int]:
    """Instance count of every temporal row of ``component``, in time order."""
    rows: list[tuple[int, int]] = []
    for batch in batches:
        if component in batch.schema.names and schema.TIMELINE in batch.schema.names:
            times: list[int] = batch.column(schema.TIMELINE).cast(pa.int64()).to_pylist()
            values: list[list[object] | None] = batch.column(component).to_pylist()
            rows.extend((time, 0 if value is None else len(value)) for time, value in zip(times, values, strict=True))
    return [count for _, count in sorted(rows)]


def test_enclosing_squares_are_the_squares_of_the_smallest_enclosing_circles() -> None:
    # Four points on a circle of radius 10 around (100, 50), plus one inside: the circle is exactly that one.
    points: Float32[ndarray, "2 5 2"] = np.array(
        [[[110, 50], [90, 50], [100, 60], [100, 40], [101, 51]], [[0, 0], [4, 0], [2, 0], [2, 0], [900, 900]]], dtype=np.float32
    )
    valid: Bool[ndarray, "2 5"] = np.array([[True] * 5, [True, True, True, True, False]])
    boxes: Float32[ndarray, "2 4"] = rerun_layers.enclosing_squares(points, valid)
    np.testing.assert_allclose(boxes[0], [90, 40, 110, 60], atol=1e-3)
    np.testing.assert_allclose(boxes[1], [0, -2, 4, 2], atol=1e-3)
    enlarged: Float32[ndarray, "2 4"] = rerun_layers.enclosing_squares(points, valid, enlarge=1.2)
    np.testing.assert_allclose(enlarged[0], [88, 38, 112, 62], atol=1e-3)
    assert np.isnan(rerun_layers.enclosing_squares(points, np.zeros((2, 5), dtype=bool))).all()


def test_camera_pixels_puts_a_point_on_the_optical_axis_at_the_principal_point() -> None:
    rig = synthetic_rig()
    cam_from_rig: Float32[ndarray, "4 4"] = rig.cam_from_rig[1].numpy()
    point_rig: Float32[ndarray, "3"] = cam_from_rig[:3, :3].T @ (np.array([0.0, 0.0, 0.4], dtype=np.float32) - cam_from_rig[:3, 3])
    world_from_rig: Float32[ndarray, "1 4 4"] = np.eye(4, dtype=np.float32)[None]
    world_from_rig[0, :3, 3] = [1.0, 2.0, 3.0]
    points: Float32[ndarray, "1 1 1 3"] = (point_rig + world_from_rig[0, :3, 3]).reshape(1, 1, 1, 3).astype(np.float32)
    pixels: Float32[ndarray, "1 4 1 1 2"] = rerun_layers.camera_pixels(rig, world_from_rig, points)
    np.testing.assert_allclose(pixels[0, 1, 0, 0], rig.principal[1].numpy(), atol=1e-3)
    behind: Float32[ndarray, "1 1 1 3"] = (world_from_rig[0, :3, 3] - 5 * (point_rig / np.linalg.norm(point_rig))).reshape(1, 1, 1, 3).astype(np.float32)
    assert np.isnan(rerun_layers.camera_pixels(rig, world_from_rig, behind)[0, 1]).all()


def test_keypoint_error_is_the_mean_landmark_distance_in_mm_where_both_exist() -> None:
    truth = synthetic_truth(12)
    gt: Float32[ndarray, "f 2 21 3"] = rerun_layers.gt_landmarks(truth)
    track: SegmentTrack = fake_track(truth)
    shifted: Float32[ndarray, "f 2 21 3"] = gt + np.array([0.003, 0.004, 0.0], dtype=np.float32)
    error: Float32[ndarray, "f 2"] = rerun_layers.keypoint_error_mm(shifted, gt, track.tracked)
    both: Bool[ndarray, "f 2"] = track.tracked & truth.present
    np.testing.assert_allclose(error[both], 5.0, atol=1e-3)
    assert np.isnan(error[~both]).all()


def test_ground_truth_on_track_frames_selects_by_video_time() -> None:
    truth = synthetic_truth(20)
    picked = truth.on_frames(truth.video_time_ns[5:9])
    np.testing.assert_array_equal(picked.frame_index, [5, 6, 7, 8])
    np.testing.assert_array_equal(picked.translation, truth.translation[5:9])


def test_handtrack_layer_writes_every_entity_on_the_segment_recording(tmp_path: Path) -> None:
    truth = synthetic_truth(40)
    track: SegmentTrack = fake_track(truth)
    target: Path = tmp_path / f"{truth.segment}.rrd"
    with atomic_recording(target, recording_id=truth.segment, send_properties=False) as recording:
        rerun_layers.write_handtrack_layer(recording, track, truth, truth.model)
    reader: RrdReader = RrdReader(target)
    assert [entry.recording_id for entry in reader.recordings()] == [truth.segment]
    rows: dict[str, list[pa.RecordBatch]] = _rows_by_entity(target)
    assert "/__properties/RecordingInfo" not in rows
    for side in ("left", "right"):
        assert rerun_layers.pred_mesh_path(side) in rows
        assert rerun_layers.pred_keypoints3d_path(side) in rows
        assert rerun_layers.error_path(side) in rows
        assert rerun_layers.tracked_path(side) in rows
        for camera in range(4):
            assert rerun_layers.pred_keypoints2d_path(camera, side) in rows
            assert rerun_layers.presence_path(camera, side) in rows
            assert rerun_layers.gt_box_path(camera, side) in rows
    # One row per frame on the box entities; a frame without a box is an empty row that clears the previous one.
    for camera in range(4):
        for side_index, side in enumerate(("left", "right")):
            counts: list[int] = _instances(rows[rerun_layers.pred_box_path(camera, side)], "Boxes2D:centers")
            expected: list[int] = (track.box_source[:, camera, side_index] != BoxSource.NONE).astype(int).tolist()
            assert counts == expected
    keypoint_rows: list[int] = _instances(rows[rerun_layers.pred_keypoints3d_path("left")], "Points3D:positions")
    assert keypoint_rows == (track.tracked[:, 0] * 21).tolist()


def test_detnet_layer_writes_boxes_on_the_reserved_schema_paths(tmp_path: Path) -> None:
    truth = synthetic_truth(16)
    detections: SegmentTrack = fake_detnet(truth)
    target: Path = tmp_path / f"{truth.segment}.rrd"
    with atomic_recording(target, recording_id=truth.segment, send_properties=False) as recording:
        rerun_layers.write_detnet_layer(recording, detections)
    rows: dict[str, list[pa.RecordBatch]] = _rows_by_entity(target)
    for camera in range(4):
        for side_index, side in enumerate(("left", "right")):
            path: str = schema.boxes_path(0, camera, f"{side}_hand")
            counts: list[int] = _instances(rows[path], "Boxes2D:centers")
            assert counts == np.isfinite(detections.box[:, camera, side_index, 0]).astype(int).tolist()
            assert rerun_layers.detnet_presence_path(camera, side) in rows


def test_prediction_model_scales_the_generic_hand_for_the_unknown_hand() -> None:
    truth = synthetic_truth(4)
    track: SegmentTrack = fake_track(truth)
    assert rerun_layers.prediction_model(track, truth) is truth.model
    unknown: SegmentTrack = replace(track, meta=replace(track.meta, hand_mode="unknown", hand_scale=1.1))
    model = rerun_layers.prediction_model(unknown, truth)
    generic = rerun_layers.generic_numpy_model()
    np.testing.assert_allclose(model.mesh_vertices, generic.mesh_vertices * 1.1, rtol=1e-6)
    np.testing.assert_allclose(model.landmark_rest_positions, generic.landmark_rest_positions * 1.1, rtol=1e-6)


def test_a_partial_run_clears_its_overlays_on_the_next_frame_and_ground_truth_covers_the_segment(tmp_path: Path) -> None:
    truth = synthetic_truth(40)
    full: SegmentTrack = fake_track(truth, drop=(0.0, 0.0))
    run: SegmentTrack = replace(full, **{name: getattr(full, name)[:20] for name in ("video_time_ns", "frame_index", "tracked", "rotation", "translation", "joint_angles", "landmarks", "box", "box_source", "keypoints_2d", "presence", "detnet_camera", "detnet_presence", "fit_energy")})
    target: Path = tmp_path / f"{truth.segment}.rrd"
    with atomic_recording(target, recording_id=truth.segment, send_properties=False) as recording:
        rerun_layers.write_handtrack_layer(recording, run, truth, truth.model)
    rows: dict[str, list[pa.RecordBatch]] = _rows_by_entity(target)
    boxes: list[int] = _instances(rows[rerun_layers.pred_box_path(1, "left")], "Boxes2D:centers")
    assert len(boxes) == 21 and boxes[-1] == 0
    assert _instances(rows[rerun_layers.pred_keypoints3d_path("left")], "Points3D:positions")[-1] == 0
    assert len(_instances(rows[rerun_layers.gt_box_path(1, "left")], "Boxes2D:centers")) == 40
    assert len(_instances(rows[rerun_layers.error_path("left")], "Scalars:scalars")) == 20


def test_a_hand_model_that_misses_the_tracked_landmarks_is_refused(tmp_path: Path) -> None:
    truth = synthetic_truth(8)
    track: SegmentTrack = fake_track(truth)
    with atomic_recording(tmp_path / "layer.rrd", recording_id=truth.segment, send_properties=False) as recording, pytest.raises(ValueError, match="misses"):
        rerun_layers.write_handtrack_layer(recording, track, truth, rerun_layers.scaled_model(truth.model, 1.1))
