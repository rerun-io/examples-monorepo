"""A growing cloud (colsfm's `points_growing`): one entity per block of `GROWING_BLOCK_S` under `points_growing/blocks` (one row: its
points, at the block's last first view) and `points_growing/recent`, one row per rig frame (each point at its first
observing rig frame).

A layout draws the blocks as they are (latest-at) and the recent rows over the last `GROWING_BLOCK_S`: together they hold
every point seen by the cursor. A block's latest-at result does not change as the cursor moves, so the viewer keeps its
processed points; a range query's result changes on every frame of playback, and the viewer processes all its rows again."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader

from simplecv.rerun_growing_points import log_points_by_first_view

ENTITY: str = "/world/runs/colsfm/points_growing"
RECENT: str = f"{ENTITY}/recent"
BLOCKS: str = f"{ENTITY}/blocks/"


def logged_rows(rrd: Path) -> tuple[dict[str, dict[int, list[list[float]]]], set[str]]:
    """Per entity, the positions of each row by its `video_time`, and the entities with a static chunk."""
    rows: dict[str, dict[int, list[list[float]]]] = {}
    static: set[str] = set()
    for chunk in RrdReader(rrd).store().stream():
        if chunk.is_static:
            static.add(str(chunk.entity_path))
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "Points3D:positions" in batch.schema.names:
            times: list[int] = batch.column("video_time").cast(pa.int64()).to_pylist()
            rows.setdefault(str(chunk.entity_path), {}).update(zip(times, batch.column("Points3D:positions").to_pylist(), strict=True))
    return rows, static


def write(tmp_path: Path, xyz: Float64[ndarray, "p 3"], seen_ns: Int64[ndarray, " p"], block_ns: int) -> Path:
    rrd: Path = tmp_path / "layer.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    rgb: UInt8[ndarray, "p 3"] = np.full((len(xyz), 3), 200, dtype=np.uint8)
    log_points_by_first_view(recording, ENTITY, xyz, rgb, seen_ns, "video_time", block_ns=block_ns)
    recording.flush()
    del recording
    return rrd


def test_each_block_is_one_entity_with_its_points_at_its_last_first_view_and_recent_rows_one_per_rig_frame(tmp_path: Path) -> None:
    xyz: Float64[ndarray, "p 3"] = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    rows, static = logged_rows(write(tmp_path, xyz, np.array([20, 10, 20, 45, 70], dtype=np.int64), block_ns=30))
    # Blocks of 30 ns from the first view (10): [10, 40) at 20, [40, 70) at 45, [70, 100) at 70.
    assert {entity: r for entity, r in rows.items() if entity.startswith(BLOCKS)} == {
        f"{BLOCKS}0000": {20: [[1.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]]},
        f"{BLOCKS}0001": {45: [[3.0, 0, 0]]},
        f"{BLOCKS}0002": {70: [[4.0, 0, 0]]},
    }
    assert rows[RECENT] == {10: [[1.0, 0, 0]], 20: [[0.0, 0, 0], [2.0, 0, 0]], 45: [[3.0, 0, 0]], 70: [[4.0, 0, 0]]}
    # The recent rows' radius is one static value; the blocks carry none, so a layout sizes them per view (view defaults).
    assert {entity for entity in static if entity.startswith(ENTITY)} == {RECENT}


def test_blocks_by_latest_at_and_recent_rows_over_one_block_show_exactly_the_points_seen_by_the_cursor(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    seen_ns: Int64[ndarray, " p"] = np.sort(rng.integers(1_000, 400_000, size=600)).astype(np.int64)
    xyz: Float64[ndarray, "p 3"] = np.column_stack([np.arange(600, dtype=np.float64), np.zeros(600), np.zeros(600)])
    block_ns: int = 37_000
    rows, _ = logged_rows(write(tmp_path, xyz, seen_ns, block_ns=block_ns))
    blocks: list[dict[int, list[list[float]]]] = [r for entity, r in rows.items() if entity.startswith(BLOCKS)]
    assert all(len(block) == 1 for block in blocks), "one row per block: latest-at never changes once it shows"
    for cursor in range(0, 420_000, 997):
        shown: set[float] = {p[0] for block in blocks for t, ps in block.items() if t <= cursor for p in ps}
        shown |= {p[0] for t, ps in rows[RECENT].items() if cursor - block_ns <= t <= cursor for p in ps}
        assert shown == set(xyz[seen_ns <= cursor, 0].tolist()), cursor
    assert len(blocks) <= 400_000 // block_ns + 1


def test_a_cloud_without_points_logs_nothing(tmp_path: Path) -> None:
    """A run that triangulated nothing still writes its layer (rig, path): the growing cloud is then absent, not an error."""
    rows, static = logged_rows(write(tmp_path, np.zeros((0, 3)), np.zeros(0, dtype=np.int64), block_ns=30))
    assert not rows and not {entity for entity in static if entity.startswith(ENTITY)}

