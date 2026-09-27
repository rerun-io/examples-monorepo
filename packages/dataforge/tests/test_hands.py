"""The shared projections writer: one property, one projected entity per camera, confidence masking."""

from pathlib import Path

import numpy as np
from conftest import read_chunks

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
