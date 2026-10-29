"""The shared projections writer: one property, one projected entity per camera, confidence masking."""

from pathlib import Path

import numpy as np
from conftest import read_chunks
from simplecv.rerun_custom_types import confidence_scores_to_rgb

from dataforge import hands, schema, writing


def test_log_projections_writes_property_and_each_camera(tmp_path: Path) -> None:
    positions = np.full((1, 133, 2), np.nan, dtype=np.float32)
    positions[0, 0] = [50.0, 50.0]
    target = tmp_path / "projected.rrd"
    times = np.array([123], dtype=np.int64)
    frames = np.array([7], dtype=np.int64)
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        hands.log_projections(
            recording,
            [((0, 2), positions), ((1, 0), positions)],
            times_ns=times,
            frame_indices=frames,
            confidence=np.full((1, 133), 0.75, dtype=np.float32),
            camera_model="FISHEYE62",
            calibration_source="factory",
        )
    chunks = read_chunks(target)
    fields = {
        key: value for chunk in chunks if chunk.entity_path == "/__properties/projections" for row in chunk.to_record_batch().to_pylist() for key, value in row.items()
    }
    assert fields["derived_from"] == ["coco133_xyz"]
    assert fields["camera_model"] == ["FISHEYE62"]
    assert fields["calibration_source"] == ["factory"]
    temporal = [c for c in chunks if not c.is_static]
    assert {str(c.entity_path) for c in temporal} == {schema.coco133_uv_projected_path(0, 2), schema.coco133_uv_projected_path(1, 0)}
    for chunk in temporal:
        batch = chunk.to_record_batch()
        scores = batch.column(next(n for n in batch.schema.names if n.endswith(":confidences"))).to_pylist()[0]
        # A NaN pixel carries no confidence, whatever the source score.
        assert scores == [0.75] + [0.0] * 132
        assert batch.column("frame_index").to_pylist() == [7]
        assert batch.column("video_time").cast("int64").to_pylist() == [123]


def test_coco_hands_fill_their_coco133_slots_and_nothing_else() -> None:
    joints = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
    dense = hands.coco133_from_coco_hands(joints)
    assert dense.shape == (133, 3) and dense.dtype == np.float32
    np.testing.assert_array_equal(dense[91:112], joints[0])
    np.testing.assert_array_equal(dense[112:133], joints[1])
    assert np.isnan(dense[:91]).all()


def test_keypoint_columns_are_bounded_and_preserve_every_row(tmp_path: Path) -> None:
    """Long sequences keep columns near the object-store target on both clocks."""
    count = 2400
    times = np.arange(count, dtype=np.int64) * 33_333_333
    frames = np.arange(count, dtype=np.int64) + 7
    positions = np.arange(count * 133 * 3, dtype=np.float32).reshape(count, 133, 3)
    target = tmp_path / "bounded.rrd"
    with writing.atomic_recording(target, recording_id="bounded", send_properties=False) as recording:
        hands.log_keypoints3d(recording, times_ns=times, frame_indices=frames, positions=positions, confidence=None)
        hands.log_keypoints2d(recording, "/pixels", times_ns=times, frame_indices=frames, positions=positions[..., :2], confidence=None)
    for path, dimension in ((schema.coco133_xyz_path(), 3), ("/pixels", 2)):
        batches = [c.to_record_batch() for c in read_chunks(target) if not c.is_static and str(c.entity_path) == path]
        assert sorted(b.num_rows for b in batches) == ([96, 768, 768, 768] if dimension == 3 else [492, 954, 954])
        assert all(b.nbytes < 2 * 1024 * 1024 for b in batches)
        batches.sort(key=lambda b: b.column("frame_index")[0].as_py())
        assert [f for b in batches for f in b.column("frame_index").to_pylist()] == frames.tolist()
        assert [t for b in batches for t in b.column("video_time").cast("int64").to_pylist()] == times.tolist()
        actual = np.concatenate([np.asarray(b.column(next(n for n in b.schema.names if n.endswith(":positions"))).to_pylist()) for b in batches])
        np.testing.assert_array_equal(actual, positions[..., :dimension])


def test_keypoint_descriptors_are_written_once_per_entity_and_recording(tmp_path: Path) -> None:
    """Repeated batches need one descriptor; a fresh recording needs its own."""
    for run in range(2):
        target = tmp_path / f"static-{run}.rrd"
        with writing.atomic_recording(target, recording_id="same-id", send_properties=False) as recording:
            for frame in range(3):
                recording.flush()
                hands.log_keypoints3d(
                    recording, times_ns=np.array([frame], dtype=np.int64), frame_indices=np.array([frame], dtype=np.int64),
                    positions=np.ones((1, 133, 3), dtype=np.float32), confidence=None, describe=frame == 0,
                )
                hands.log_keypoints2d(
                    recording, "/pixels", times_ns=np.array([frame], dtype=np.int64), frame_indices=np.array([frame], dtype=np.int64),
                    positions=np.ones((1, 133, 2), dtype=np.float32), confidence=None, describe=frame == 0,
                )
        for path in (schema.coco133_xyz_path(), "/pixels"):
            descriptors = [c for c in read_chunks(target) if c.is_static and str(c.entity_path) == path]
            assert len(descriptors) == 1


def test_2d_rows_are_coloured_by_confidence_like_3d(tmp_path: Path) -> None:
    """2D keypoints carry the same red -> yellow -> green confidence colours as 3D; no frame_index when None."""
    times = np.array([0, 40_000_000], dtype=np.int64)
    confidence = np.tile(np.linspace(0.0, 1.0, 133, dtype=np.float32), (2, 1))
    target = tmp_path / "coloured.rrd"
    with writing.atomic_recording(target, recording_id="coloured", send_properties=False) as recording:
        hands.log_keypoints2d(recording, "/pixels", times_ns=times, frame_indices=None, positions=np.zeros((2, 133, 2), np.float32), confidence=confidence)
    batch = next(c.to_record_batch() for c in read_chunks(target) if not c.is_static and str(c.entity_path) == "/pixels")
    assert "frame_index" not in batch.schema.names
    colours = np.asarray(batch.column("Points2D:colors").to_pylist()[0], dtype=np.uint32)
    expected = confidence_scores_to_rgb(confidence[:1, :, None])[0]
    np.testing.assert_array_equal(colours, [int.from_bytes(bytes([*rgb, 255]), "big") for rgb in expected.astype(np.uint8).tolist()])
