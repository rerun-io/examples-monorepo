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
        indexes = [rr.TimeColumn("video_time", duration=[0.0])]
        if dimensions == 3:
            hands.log_keypoints3d(recording, indexes, xyz, confidence)
        else:
            hands.log_keypoints2d(recording, "/world/rig/cam/pinhole/coco133_uv", indexes, xyz[:, :, :2], confidence)
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
            recording, "/mesh", [rr.TimeColumn("frame_index", sequence=[0, 1, 2])], np.ones((2, 3, 3), dtype=np.float32), [True, False, True]
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
            np.arange(5, dtype=np.int64),
            np.arange(5, dtype=np.int64),
            rr.Asset3D(contents=b"test", media_type="model/gltf-binary"),
            np.array([0.0, 0.5, 0.9, 0.8, 0.0], dtype=np.float32),
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
            np.arange(4, dtype=np.int64),
            np.arange(4, dtype=np.int64),
            transforms,
            np.array([0.0, 0.5, 0.9, 1.0], dtype=np.float64),
            trust_threshold=0.5,
        )
    batches = [chunk.to_record_batch() for chunk in read_chunks(target)]
    poses = next(batch for batch in batches if "Transform3D:translation" in batch.schema.names)
    assert poses.column("frame_index").to_pylist() == [2]
    assert poses.column("Transform3D:translation").to_pylist() == [[[1.0, 2.0, 3.0]]]
    confidence = next(batch for batch in batches if "Scalars:scalars" in batch.schema.names)
    assert confidence.column("Scalars:scalars").to_pylist() == [[0.0], [0.5], [0.9], [1.0]]
