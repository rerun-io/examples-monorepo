"""Shared writers preserve confidence and clear unavailable geometry."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from conftest import read_chunks

from dataforge import hands, meshes, objects, writing


@pytest.mark.parametrize("shipped", [None, 0.25, 0.0])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_dense_confidence_rule(tmp_path: Path, shipped: float | None, dimensions: int) -> None:
    xyz = np.full((1, 133, 3), np.nan, dtype=np.float32)
    xyz[0, 91] = [1.0, 2.0, 3.0]
    confidence = None if shipped is None else np.full((1, 133), shipped, dtype=np.float32)
    target = tmp_path / "hands.rrd"
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        times = np.array([0], dtype=np.int64)
        if dimensions == 3:
            hands.log_keypoints3d(recording, times_ns=times, frame_indices=times, positions=xyz, confidence=confidence)
        else:
            hands.log_keypoints2d(recording, 1, 0, times_ns=times, frame_indices=times, positions=xyz[:, :, :2], confidence=confidence)
    batch = next(c.to_record_batch() for c in read_chunks(target) if not c.is_static)
    positions = batch.column(next(n for n in batch.schema.names if "positions" in n)).to_pylist()[0]
    scores = batch.column(next(n for n in batch.schema.names if n.endswith(":confidences"))).to_pylist()[0]
    assert positions[91] == [1.0, 2.0, 3.0][:dimensions]
    assert scores[91] == (1.0 if shipped is None else shipped)
    assert np.isnan(positions[0]).all()
    assert scores[0] == 0.0


def test_mesh_batch_writes_empty_rows(tmp_path: Path) -> None:
    target = tmp_path / "mesh.rrd"
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        meshes.log_mesh_batch(
            recording, "/mesh", times_ns=np.arange(3, dtype=np.int64), frame_indices=np.arange(3, dtype=np.int64),
            vertices=np.ones((2, 3, 3), dtype=np.float32), trusted=[True, False, True]
        )
    batch = read_chunks(target)[0].to_record_batch()
    rows = batch.column(next(n for n in batch.schema.names if "vertex_positions" in n)).to_pylist()
    assert [len(row) for row in rows] == [3, 0, 3]


def test_object_visibility_transitions(tmp_path: Path) -> None:
    target = tmp_path / "object.rrd"
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        objects.log_object_mesh(
            recording,
            "toy",
            times_ns=np.arange(5, dtype=np.int64),
            frame_indices=np.arange(5, dtype=np.int64),
            asset=rr.Asset3D(contents=b"test", media_type="model/gltf-binary"),
            confidence=np.array([1.0, 0.5, 0.9, 0.8, 0.0], dtype=np.float32),
            posed=np.array([False, True, True, True, True]),
            trust_threshold=0.5,
        )
    batch = next(c.to_record_batch() for c in read_chunks(target) if any("albedo_factor" in n for n in c.to_record_batch().schema.names))
    assert batch.column("frame_index").to_pylist() == [0, 2, 4]
    assert batch.column(next(n for n in batch.schema.names if "albedo_factor" in n)).to_pylist() == [[0xFFFFFF00], [0xFFFFFFFF], [0xFFFFFF00]]


def test_object_pose_is_sparse_but_confidence_is_dense(tmp_path: Path) -> None:
    target = tmp_path / "pose.rrd"
    transforms = np.tile(np.eye(4, dtype=np.float64), (4, 1, 1))
    transforms[2, :3, 3] = [1.0, 2.0, 3.0]
    transforms[3] = np.nan
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        objects.log_object_pose(
            recording,
            "toy",
            times_ns=np.arange(4, dtype=np.int64),
            frame_indices=np.arange(4, dtype=np.int64),
            transforms=transforms,
            confidence=np.array([0.0, 0.5, 0.9, 1.0], dtype=np.float64),
        )
    batches = [chunk.to_record_batch() for chunk in read_chunks(target)]
    poses = next(batch for batch in batches if "Transform3D:translation" in batch.schema.names)
    assert poses.column("frame_index").to_pylist() == [0, 1, 2]
    assert poses.column("Transform3D:translation").to_pylist() == [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0]]]
    confidence = next(batch for batch in batches if "Scalars:scalars" in batch.schema.names)
    assert confidence.column("Scalars:scalars").to_pylist() == [[0.0], [0.5], [0.9], [1.0]]


@pytest.mark.parametrize("dimensions", [2, 3])
def test_two_hands_mapping_and_absent_hand(dimensions: int) -> None:
    landmarks = np.ones((2, 21, dimensions), dtype=np.float32)
    landmarks[0, 5] = 2.0
    landmarks[0, 6] = 4.0
    positions, confidence = hands.coco133_from_hands(landmarks, np.array([0.25, 0.75], dtype=np.float32))
    assert positions.shape == (133, dimensions)
    np.testing.assert_array_equal(positions[9], positions[91])
    np.testing.assert_array_equal(positions[10], positions[112])
    np.testing.assert_array_equal(positions[92], np.full(dimensions, 3.0, dtype=np.float32))
    np.testing.assert_array_equal(confidence[[9, 10, 92, 113]], [0.25, 0.75, 0.25, 0.75])
    landmarks[1] = np.nan
    positions, confidence = hands.coco133_from_hands(landmarks, np.array([0.25, 0.75], dtype=np.float32))
    positions, confidence = hands.confidence_rule(positions[None], confidence[None])
    assert np.isnan(positions[0, 112:]).all()
    assert np.isnan(positions[0, 10]).all()
    assert not confidence[0, 112:].any()
    assert confidence[0, 10] == 0.0
    assert confidence[0, 0] == 0.0


def test_hand_parameters_keep_arrow_types_and_both_clocks(tmp_path: Path) -> None:
    import pyarrow as pa

    target = tmp_path / "parameters.rrd"
    times = np.array([123456789], dtype=np.int64)
    indices = np.array([7], dtype=np.int64)
    angles = np.arange(22, dtype=np.float32).reshape(1, 22)
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        hands.log_joint_angles(recording, "left", times_ns=times, frame_indices=indices, angles=angles)
        hands.log_hand_confidence(recording, "left", times_ns=times, frame_indices=indices, confidence=np.array([0.123456789012345], dtype=np.float64))
    batches = [chunk.to_record_batch() for chunk in read_chunks(target)]
    joints = next(batch for batch in batches if "joint_angles" in batch.schema.names)
    assert joints.column("joint_angles").type == pa.list_(pa.list_(pa.float32(), 22))
    assert joints.column("joint_angles").to_pylist() == [[list(range(22))]]
    confidence = next(batch for batch in batches if "Scalars:scalars" in batch.schema.names)
    assert confidence.column("Scalars:scalars").type == pa.list_(pa.float64())
    assert confidence.column("Scalars:scalars").to_pylist() == [[0.123456789012345]]
    for batch in batches:
        assert batch.column("video_time").cast(pa.int64()).to_pylist() == [123456789]
        assert batch.column("frame_index").to_pylist() == [7]
