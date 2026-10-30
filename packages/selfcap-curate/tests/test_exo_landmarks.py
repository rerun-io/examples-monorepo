"""Landmark records retain their source and produce visible catalog annotations."""

from pathlib import Path

from rerun.chunk import RrdReader
from serde.json import from_json, to_json

from selfcap_curate.exo_landmarks import Landmark, Landmarks, write_landmarks


def test_landmarks_round_trip_and_layer(tmp_path: Path) -> None:
    record = Landmarks([Landmark(2, 3.0, "phone1", 123.0, 456.0, 0.65)], "agent-model", "visual aperture", "2026-10-29")
    assert from_json(Landmarks, to_json(record)) == record
    path = tmp_path / "labels.rrd"
    write_landmarks(path, "selfcap__2025-12-03__d31bf1da__ep001", record)
    chunks = list(RrdReader(path).stream())
    points = next(c.to_record_batch() for c in chunks if not c.is_static)
    assert points.column("video_time").cast("int64").to_pylist() == [3_000_000_000]
    assert points.column("Points2D:labels").to_pylist() == [["phone1"]]
    assert str(next(c for c in chunks if c.is_static).entity_path) == "/__properties/exo_landmarks"
