"""Landmark records retain their source and produce visible catalog annotations."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import RrdReader
from serde.json import from_json, to_json

from selfcap_curate.display_style import log_display_skeleton
from selfcap_curate.exo_landmarks import Landmark, Landmarks, write_landmarks


def test_landmarks_round_trip_and_layer(tmp_path: Path) -> None:
    # A display written first must not establish the generic landmark confidence dtype.
    with rr.RecordingStream("display", recording_id="display", send_properties=False) as recording:
        recording.save(tmp_path / "display.rrd")
        log_display_skeleton(recording, "/world/skeleton", np.zeros((1,133,3),np.float32), np.ones((1,133),np.float32), np.array([0],np.int64))
    record = Landmarks([Landmark(2, 3.0, "phone1", 123.0, 456.0, 0.65)], "agent-model", "visual aperture", "2026-10-29")
    assert from_json(Landmarks, to_json(record)) == record
    path = tmp_path / "labels.rrd"
    write_landmarks(path, "selfcap__2025-12-03__d31bf1da__ep001", record)
    chunks = list(RrdReader(path).stream())
    points = next(c.to_record_batch() for c in chunks if not c.is_static)
    assert points.column("video_time").cast("int64").to_pylist() == [3_000_000_000]
    assert points.column("Points2D:labels").to_pylist() == [["phone1"]]
    assert points.schema.field("confidence").type.value_type == pa.float64()
    assert str(next(c for c in chunks if c.is_static).entity_path) == "/__properties/exo_landmarks"
