"""The live TSDF surface pattern: temporal super-blocks while running, a clear and one mesh at the run's end."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from simplecv.rerun_live_mesh import BLOCKS, FULL, LiveMeshLog, MeshArrays, merge_meshes


def _triangle(x: float, colored: bool = True) -> MeshArrays:
    vertices = np.array([[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    colors = np.full((3, 3), 200, dtype=np.uint8) if colored else None
    normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (3, 1))
    return vertices, triangles, colors, normals


EMPTY: MeshArrays = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.float32))


def test_merge_offsets_triangles_and_drops_colours_unless_all_have_them() -> None:
    vertices, triangles, colors, normals = merge_meshes([_triangle(0.0), _triangle(5.0)])
    assert vertices.shape == (6, 3) and normals.shape == (6, 3)
    assert triangles.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert colors is not None and colors.shape == (6, 3)
    assert merge_meshes([_triangle(0.0), _triangle(5.0, colored=False)])[2] is None


def _log_run(rrd: Path, group: int) -> None:
    rec = rr.RecordingStream("live-mesh-test", recording_id="test")
    rec.save(rrd)
    log = LiveMeshLog("world/mesh", rec, group=group)
    rec.set_time("frame", sequence=1)
    log.log_changed([((0, 0, 0), _triangle(0.0)), ((1, 0, 0), _triangle(1.0)), ((2, 0, 0), _triangle(2.0))])
    rec.set_time("frame", sequence=2)
    log.log_changed([((2, 0, 0), EMPTY)])  # block 2's surface vanished
    rec.set_time("frame", sequence=3)
    log.finish(merge_meshes([_triangle(0.0), _triangle(1.0)]))
    rec.flush()
    rec.disconnect()


def _rows(rrd: Path) -> list[tuple[str, bool, list[int], list[str]]]:
    """``(entity, static, frame times, component columns)`` per chunk."""
    rows: list[tuple[str, bool, list[int], list[str]]] = []
    for chunk in RrdReader(rrd).stream():
        path: str = str(chunk.entity_path)
        if not path.startswith("/world/mesh"):
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        frames: list[int] = batch.column("frame").to_pylist() if "frame" in batch.schema.names else []
        rows.append((path, chunk.is_static, frames, [name for name in batch.schema.names if ":" in name]))
    return rows


def test_blocks_are_temporal_super_blocks_and_the_end_is_one_mesh(tmp_path: Path) -> None:
    rrd: Path = tmp_path / "live.rrd"
    _log_run(rrd, group=2)
    rows = _rows(rrd)
    assert not any(static for _, static, _, _ in rows), "nothing is static, so scrubbing shows the growth"
    block_rows = [(path, frames, cols) for path, _, frames, cols in rows if path.startswith(f"/world/mesh/{BLOCKS}/")]
    # Blocks 0 and 1 share super-block 0_0_0; block 2 is super-block 1_0_0, cleared when its surface vanished.
    assert {path for path, _, _ in block_rows} == {f"/world/mesh/{BLOCKS}/0_0_0", f"/world/mesh/{BLOCKS}/1_0_0"}
    assert any(path.endswith("1_0_0") and frames == [2] and any("Clear" in c for c in cols) for path, frames, cols in block_rows)
    # The end: a recursive clear of the blocks and the full mesh, both at the last time point.
    assert any(path == f"/world/mesh/{BLOCKS}" and frames == [3] and any("Clear" in c for c in cols) for path, _, frames, cols in rows)
    assert any(path == f"/world/mesh/{FULL}" and frames == [3] and any("Mesh3D:vertex_positions" in c for c in cols) for path, _, frames, cols in rows)


def test_group_one_logs_every_block(tmp_path: Path) -> None:
    rrd: Path = tmp_path / "live.rrd"
    _log_run(rrd, group=1)
    paths = {path for path, _, _, _ in _rows(rrd) if path.startswith(f"/world/mesh/{BLOCKS}/")}
    assert paths == {f"/world/mesh/{BLOCKS}/{x}_0_0" for x in range(3)}


def test_a_finished_log_refuses_reuse() -> None:
    rec = rr.RecordingStream("live-mesh-test", recording_id="closed")
    rec.memory_recording()
    log = LiveMeshLog("world/mesh", rec)
    log.finish(_triangle(0.0))
    for call in (lambda: log.log_changed([((0, 0, 0), _triangle(1.0))]), log.restart, lambda: log.finish(_triangle(1.0))):
        with pytest.raises(RuntimeError, match="finished"):
            call()


def test_a_super_block_whose_surface_vanished_is_forgotten() -> None:
    rec = rr.RecordingStream("live-mesh-test", recording_id="forget")
    rec.memory_recording()
    log = LiveMeshLog("world/mesh", rec, group=1)
    log.log_changed([((0, 0, 0), _triangle(0.0)), ((1, 0, 0), _triangle(1.0))])
    log.log_changed([((1, 0, 0), EMPTY)])
    assert list(log._groups) == [(0, 0, 0)]
