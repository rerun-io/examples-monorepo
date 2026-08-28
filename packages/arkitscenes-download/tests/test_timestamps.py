"""Behavior checks for exact catalog timestamp matching."""

import numpy as np
import pyarrow as pa
import pytest
from jaxtyping import Int64, Shaped
from numpy import ndarray
from numpy.testing import assert_array_equal

from arkitscenes_download.ingest.paths import TIMELINE
from arkitscenes_download.ingest.timestamps import match_exact_timestamps, table_timestamps


def test_table_timestamps_returns_the_timeline_column_as_timedelta_ns() -> None:
    """Read a chunked catalog timeline column into one ``timedelta64[ns]`` array."""
    first_chunk: pa.Array = pa.array([100, 200], type=pa.duration("ns"))
    second_chunk: pa.Array = pa.array([300], type=pa.duration("ns"))
    table: pa.Table = pa.Table.from_batches(
        [
            pa.record_batch({TIMELINE: first_chunk}),
            pa.record_batch({TIMELINE: second_chunk}),
        ]
    )

    times_n: Shaped[ndarray, "3"] = table_timestamps(table)

    assert times_n.dtype == np.dtype("timedelta64[ns]")
    assert_array_equal(times_n, np.array([100, 200, 300], dtype="timedelta64[ns]"))


def test_match_exact_timestamps_maps_packet_positions() -> None:
    """Map each chosen timestamp to its exact packet position."""
    packet_times_n: Shaped[ndarray, "4"] = np.array([100, 200, 300, 400], dtype="timedelta64[ns]")
    chosen_times_n: Shaped[ndarray, "3"] = np.array([100, 300, 400], dtype="timedelta64[ns]")

    packet_indices_n: Int64[ndarray, "3"] = match_exact_timestamps(packet_times_n, chosen_times_n)

    assert packet_indices_n.dtype == np.int64
    assert_array_equal(packet_indices_n, np.array([0, 2, 3], dtype=np.int64))


@pytest.mark.parametrize("chosen_ns", [250, 500])
def test_match_exact_timestamps_rejects_missing_packet_times(chosen_ns: int) -> None:
    """Reject missing timestamps both within and beyond the packet range."""
    packet_times_n: Shaped[ndarray, "4"] = np.array([100, 200, 300, 400], dtype="timedelta64[ns]")
    chosen_times_n: Shaped[ndarray, "1"] = np.array([chosen_ns], dtype="timedelta64[ns]")

    with pytest.raises(ValueError, match="no exact video-packet match"):
        match_exact_timestamps(packet_times_n, chosen_times_n)
