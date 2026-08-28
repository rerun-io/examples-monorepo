"""Timeline helpers shared by every stage that joins catalog rows to video packets."""

from typing import TypeAlias

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int64, Shaped
from numpy import ndarray

from arkitscenes_download.ingest.paths import TIMELINE

TimedeltaNs: TypeAlias = Shaped[ndarray, " n"]
"""One-dimensional ``timedelta64[ns]`` timestamps."""


def table_timestamps(table: pa.Table) -> TimedeltaNs:
    """Return an Arrow table timeline as ``timedelta64[ns]`` values."""
    values_n: ndarray = table.column(TIMELINE).combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(values_n, dtype="timedelta64[ns]")


def match_exact_timestamps(packet_times_n: TimedeltaNs, target_times_n: TimedeltaNs) -> Int64[ndarray, "n_targets"]:
    """Resolve target times to exact packet indices without nearest-neighbor drift.

    Args:
        packet_times_n: Decoder packet timestamps shaped ``n_packets``.
        target_times_n: Target timestamps shaped ``n_targets``.

    Returns:
        int64 packet indices shaped ``n_targets``.

    Raises:
        ValueError: If either input is not one-dimensional, the packet timeline
            is unsorted, or any target time is absent from the packet timeline.
    """
    if packet_times_n.ndim != 1 or target_times_n.ndim != 1:
        raise ValueError("Packet and target timestamps must be one-dimensional")
    packet_ns_n: TimedeltaNs = np.asarray(packet_times_n, dtype="timedelta64[ns]")
    target_ns_n: TimedeltaNs = np.asarray(target_times_n, dtype="timedelta64[ns]")
    if np.any(packet_ns_n[1:] < packet_ns_n[:-1]):
        raise ValueError("Packet timestamps must be sorted")
    matched_indices_n: Int64[ndarray, "n_targets"] = np.searchsorted(packet_ns_n, target_ns_n, side="left").astype(np.int64)
    in_bounds_n: Bool[ndarray, "n_targets"] = matched_indices_n < len(packet_ns_n)
    exact_n: Bool[ndarray, "n_targets"] = np.zeros(len(target_ns_n), dtype=np.bool_)
    exact_n[in_bounds_n] = packet_ns_n[matched_indices_n[in_bounds_n]] == target_ns_n[in_bounds_n]
    if not np.all(exact_n):
        first_missing: np.timedelta64 = target_ns_n[int(np.flatnonzero(~exact_n)[0])]
        raise ValueError(f"target timestamp {first_missing} has no exact video-packet match")
    return matched_indices_n
