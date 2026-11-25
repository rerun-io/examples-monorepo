"""The walk path every run writer shares: the gradient, the trail's one-row-per-pose rule, the whole path, the endpoints."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")
import rerun.chunk as rrc  # noqa: E402

from simplecv.rerun_walk_path import (  # noqa: E402
    ENDPOINTS,
    TIMELINE,
    TRAIL,
    TRAJECTORY,
    segments,
    send_trail,
    send_walk_path,
    walk_colors,
)

RUN: str = "/world/runs/test"
"""The run every test writes under."""


def _store(tmp_path: Path, times_ns: Int64[ndarray, " n"], positions: Float64[ndarray, "n 3"], color: tuple[int, int, int] | None) -> rrc.ChunkStore:
    target: Path = tmp_path / "walk.rrd"
    with rr.RecordingStream("simplecv-test", recording_id="walk") as recording:
        recording.save(target)
        send_walk_path(RUN, times_ns, positions, color, recording=recording)
    return rrc.ChunkStore.from_chunks(list(rrc.RrdReader(target).stream().collect().stream()))


def _rows(store: rrc.ChunkStore, column: str) -> pa.Table:
    table: pa.Table = store.reader(index=TIMELINE).to_arrow_table().sort_by(TIMELINE)
    return table.select([TIMELINE, column]).drop_null()


def _walk(count: int) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 3"]]:
    step: Float64[ndarray, " n"] = np.arange(count, dtype=np.float64)
    return np.arange(count, dtype=np.int64) * 1_000_000, np.column_stack([step, np.sin(step), np.zeros(count)])


def test_the_walk_gradient_runs_green_through_yellow_to_red() -> None:
    """The one pin of the gradient slam-rs, colsfm and dataforge's layouts share."""
    t_ns: Int64[ndarray, " n"] = np.array([0, 25, 50, 75, 100], dtype=np.int64)
    np.testing.assert_array_equal(walk_colors(t_ns, 0, 100), [[38, 255, 38], [147, 255, 38], [255, 255, 38], [255, 147, 38], [255, 38, 38]])


def test_the_leaf_names_are_dataforge_s() -> None:
    """dataforge's ``schema.trail_path`` / ``trajectory_path`` end in these leaves (dataforge depends on simplecv, not back)."""
    assert (TRAIL, TRAJECTORY, ENDPOINTS, TIMELINE) == ("trail", "trajectory", "endpoints", "video_time")


def test_the_trail_has_one_row_per_pose_the_first_a_zero_length_segment(tmp_path: Path) -> None:
    """One rule for every writer: row ``k`` at ``t_ns[k]`` is the step from pose ``k - 1`` to ``k``, so a cursor window
    over the trail holds as many rows as the pose track; each row takes the gradient at its own pose."""
    times_ns, positions = _walk(6)
    store: rrc.ChunkStore = _store(tmp_path, times_ns, positions, None)
    strips: pa.Table = _rows(store, f"{RUN}/{TRAIL}:LineStrips3D:strips")
    assert strips.column(0).cast(pa.int64()).to_pylist() == times_ns.tolist()
    drawn: list[list[list[float]]] = [row[0] for row in strips.column(1).to_pylist()]
    np.testing.assert_allclose(drawn[0], [positions[0], positions[0]])
    for k in range(1, 6):
        np.testing.assert_allclose(drawn[k], [positions[k - 1], positions[k]])
    colors: list[int] = [row[0] for row in _rows(store, f"{RUN}/{TRAIL}:LineStrips3D:colors").column(1).to_pylist()]
    expected = walk_colors(times_ns, int(times_ns[0]), int(times_ns[-1]))
    assert colors == [(int(r) << 24) | (int(g) << 16) | (int(b) << 8) | 255 for r, g, b in expected]


def test_a_flat_trail_carries_its_colour_and_ui_radius_once_static(tmp_path: Path) -> None:
    times_ns, positions = _walk(4)
    target: Path = tmp_path / "flat.rrd"
    with rr.RecordingStream("simplecv-test", recording_id="flat") as recording:
        recording.save(target)
        send_trail(f"{RUN}/{TRAIL}", times_ns, positions, (10, 20, 30), radius_ui=3.0, recording=recording)
    store = rrc.ChunkStore.from_chunks(list(rrc.RrdReader(target).stream().collect().stream()))
    static: pa.Table = store.reader(index=None).to_arrow_table()
    assert static.column(f"{RUN}/{TRAIL}:LineStrips3D:colors").to_pylist() == [[(10 << 24) | (20 << 16) | (30 << 8) | 255]]
    assert static.column(f"{RUN}/{TRAIL}:LineStrips3D:radii").to_pylist() == [[-3.0]], "a negative radius is ui points"
    assert len(_rows(store, f"{RUN}/{TRAIL}:LineStrips3D:strips")) == 4


def test_the_whole_path_and_the_end_dot_come_once_at_the_last_pose(tmp_path: Path) -> None:
    times_ns, positions = _walk(5)
    store: rrc.ChunkStore = _store(tmp_path, times_ns, positions, (1, 2, 3))
    whole: pa.Table = _rows(store, f"{RUN}/{TRAJECTORY}:LineStrips3D:strips")
    assert whole.column(0).cast(pa.int64()).to_pylist() == [times_ns[-1]]
    np.testing.assert_allclose(np.asarray(whole.column(1).to_pylist()[0]), segments(positions))
    end: pa.Table = _rows(store, f"{RUN}/{ENDPOINTS}/end:Points3D:positions")
    assert end.column(0).cast(pa.int64()).to_pylist() == [times_ns[-1]]
    static: pa.Table = store.reader(index=None).to_arrow_table()
    np.testing.assert_allclose(static.column(f"{RUN}/{ENDPOINTS}/start:Points3D:positions").to_pylist()[0], [positions[0]])
    assert len(static.column(f"{RUN}/{ENDPOINTS}/start_ring:LineStrips3D:strips").to_pylist()[0][0]) == 49


def test_a_path_of_one_pose_writes_nothing(tmp_path: Path) -> None:
    times_ns, positions = _walk(1)
    store: rrc.ChunkStore = _store(tmp_path, times_ns, positions, None)
    assert not any(column.startswith(RUN) for column in store.reader(index=None).to_arrow_table().column_names)
