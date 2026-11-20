"""The colsfm layer's position difference: how far each optimised rig frame lies from the prior pose it started from."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.chunk import RrdReader

from colsfm.rerun_layer import log_position_difference


def test_the_difference_is_the_distance_at_each_rig_frame_both_runs_hold(tmp_path: Path) -> None:
    """A rig frame the prior and the optimised run share gets the distance between its two positions, in metres; a frame
    only one of them holds gets no row."""
    rrd: Path = tmp_path / "layer.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    prior_t_ns: Int64[ndarray, " n"] = np.array([10, 20, 30], dtype=np.int64)
    prior_xyz: Float64[ndarray, "n 3"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    t_ns: Int64[ndarray, " m"] = np.array([10, 15, 20, 40], dtype=np.int64)
    xyz: Float64[ndarray, "m 3"] = np.array([[0.0, 0.03, 0.04], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
    log_position_difference(recording, "/world/runs/colsfm/position_difference", prior_t_ns, prior_xyz, t_ns, xyz, "video_time")
    recording.flush()
    del recording

    rows: dict[int, float] = {}
    for chunk in RrdReader(rrd).store().stream():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if not chunk.is_static and "Scalars:scalars" in batch.schema.names:
            rows |= {t: values[0] for t, values in zip(batch.column("video_time").cast(pa.int64()).to_pylist(), batch.column("Scalars:scalars").to_pylist(), strict=True)}
    assert set(rows) == {10, 20}
    np.testing.assert_allclose([rows[10], rows[20]], [0.05, 0.0], atol=1e-12)
