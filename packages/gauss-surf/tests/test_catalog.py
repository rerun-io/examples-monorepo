"""Behavior contracts for composable catalog segment reads."""

from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch
from rerun.catalog import DatasetEntry

from gauss_surf.catalog import SegmentReader


class _CollectedSegmentTable:
    """Minimal catalog segment-table query used by reader tests."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._table: pa.Table = pa.Table.from_pylist(rows)
        self.collect_count: int = 0

    def collect(self) -> list[pa.RecordBatch]:
        """Return Arrow batches and record how often the table was read."""
        self.collect_count += 1
        return self._table.to_batches()


class _Dataset(DatasetEntry):
    """Minimal dataset boundary for segment-row behavior."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Create a dataset-entry test double without an SDK backend."""
        self.table: _CollectedSegmentTable = _CollectedSegmentTable(rows)

    def segment_table(self) -> _CollectedSegmentTable:
        """Return the fake segment-table query."""
        return self.table


def _row(orientation: Any = 0, layers: list[str] | None = None) -> dict[str, Any]:
    """Build one segment-table row with catalog-shaped cells."""
    return {
        "rerun_segment_id": "segment-a",
        "rerun_layer_names": layers if layers is not None else ["base", "frame_selection"],
        "property:capture:orientation_quarter_turns_ccw": orientation,
    }


def test_segment_reader_returns_and_caches_the_unique_row() -> None:
    """Repeated reads do not rescan the catalog segment table."""
    dataset: _Dataset = _Dataset([_row()])
    reader: SegmentReader = SegmentReader(dataset, "segment-a")

    assert reader.row()["rerun_segment_id"] == "segment-a"
    assert reader.row()["rerun_segment_id"] == "segment-a"
    assert dataset.table.collect_count == 1


def test_segment_reader_absent_id_exits_cleanly_with_available_ids() -> None:
    """A missing segment is a clean CLI error rather than an index failure."""
    reader: SegmentReader = SegmentReader(_Dataset([_row()]), "missing")

    with pytest.raises(SystemExit, match="available ids: segment-a"):
        reader.row()


def test_segment_reader_rejects_duplicate_segment_rows() -> None:
    """Ambiguous catalog identity is refused before a stage reads data."""
    reader: SegmentReader = SegmentReader(_Dataset([_row(), _row()]), "segment-a")

    with pytest.raises(SystemExit, match="2 rows"):
        reader.row()


def test_segment_reader_reports_missing_upstream_layers() -> None:
    """Layer prerequisites share one clean error path across stages."""
    reader: SegmentReader = SegmentReader(_Dataset([_row(layers=["base"])]), "segment-a")

    with pytest.raises(SystemExit, match=r"missing required layers: \['frame_selection', 'promptda'\]"):
        reader.require_layers(("frame_selection", "promptda"))


@pytest.mark.parametrize("stage_context", ["MoGe normal inference", "ultrawide rectification"])
def test_none_orientation_cell_exits_cleanly_for_both_stage_contexts(stage_context: str) -> None:
    """A catalog ``[None]`` orientation cell never reaches ``int(None)``."""
    reader: SegmentReader = SegmentReader(_Dataset([_row(orientation=[None])]), "segment-a")

    with pytest.raises(SystemExit, match="missing required catalog property"):
        reader.require_zero_orientation(stage_context)


def test_segment_reader_accepts_scalar_and_singleton_zero_orientation() -> None:
    """Both catalog encodings of the supported orientation remain valid."""
    for orientation in (0, [0], np.asarray([0])):
        reader: SegmentReader = SegmentReader(_Dataset([_row(orientation=orientation)]), "segment-a")
        reader.require_zero_orientation("test stage")


def test_segment_reader_rejects_nonzero_orientation_with_stage_context() -> None:
    """Unsupported orientation errors name the stage that imposed the limit."""
    reader: SegmentReader = SegmentReader(_Dataset([_row(orientation=[1])]), "segment-a")

    with pytest.raises(SystemExit, match="test stage supports only zero quarter-turns"):
        reader.require_zero_orientation("test stage")


def test_decode_frames_refuses_timestamp_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared NVDEC seam verifies all requested timestamps exactly once."""

    class _Decoder:
        """Deterministic stand-in for the GPU decoder."""

        def __init__(self, *_args: Any) -> None:
            self.times: np.ndarray = np.asarray([10, 20], dtype=np.int64).astype("timedelta64[ns]")

        def decode_at(self, _timestamp: np.timedelta64, _video_id: str) -> torch.Tensor:
            """Return one RGB tensor so exactness validation can run."""
            return torch.zeros((3, 2, 2), dtype=torch.uint8)

    monkeypatch.setattr("gauss_surf.catalog.SegmentNvdecDecoder", _Decoder)
    reader: SegmentReader = SegmentReader(_Dataset([_row()]), "segment-a")
    drifted_n: np.ndarray = np.asarray([10, 21], dtype=np.int64).astype("timedelta64[ns]")

    with pytest.raises(ValueError, match="no exact video-packet match"):
        list(reader.decode_frames("video/wide", drifted_n, fps=60.0, device=torch.device("cpu")))


def test_decode_frames_preserves_requested_timestamp_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact decoding yields one frame for each requested timestamp in request order."""

    class _Decoder:
        """Timestamp-valued stand-in for the GPU decoder."""

        def __init__(self, *_args: Any) -> None:
            self.times: np.ndarray = np.asarray([10, 20], dtype=np.int64).astype("timedelta64[ns]")

        def decode_at(self, timestamp: np.timedelta64, _video_id: str) -> torch.Tensor:
            """Encode the requested nanosecond value into the returned frame."""
            value: int = int(timestamp.astype(np.int64))
            return torch.full((3, 1, 1), value, dtype=torch.uint8)

    monkeypatch.setattr("gauss_surf.catalog.SegmentNvdecDecoder", _Decoder)
    reader: SegmentReader = SegmentReader(_Dataset([_row()]), "segment-a")
    requested_n: np.ndarray = np.asarray([20, 10], dtype=np.int64).astype("timedelta64[ns]")

    frames: list[torch.Tensor] = list(reader.decode_frames("video/wide", requested_n, fps=60.0, device=torch.device("cpu")))

    assert [int(frame[0, 0, 0]) for frame in frames] == [20, 10]
