"""The colsfm layer's growing cloud: each point on `video_time` at its first observing rig frame, one row per rig frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import RrdReader

from colsfm.rerun_layer import log_points_by_first_view


def test_points_are_logged_in_one_row_per_first_seen_time_with_their_colours(tmp_path: Path) -> None:
    rrd: Path = tmp_path / "layer.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    xyz: np.ndarray = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rgb: np.ndarray = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    log_points_by_first_view(recording, "/world/runs/colsfm/points_growing", xyz, rgb, np.array([20, 10, 20], dtype=np.int64), "video_time")
    recording.flush()
    del recording

    rows: dict[int, list[list[float]]] = {}
    static: list[str] = []
    for chunk in RrdReader(rrd).store().stream():
        if chunk.is_static:
            static.append(str(chunk.entity_path))
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "Points3D:positions" in batch.schema.names:
            times: list[int] = batch.column("video_time").cast(pa.int64()).to_pylist()
            rows |= dict(zip(times, batch.column("Points3D:positions").to_pylist(), strict=True))
    assert rows == {10: [[1.0, 0.0, 0.0]], 20: [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]}
    # The radius is one static value for the whole cloud, not a column per point.
    assert "/world/runs/colsfm/points_growing" in static
