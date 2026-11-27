"""The column log writes the rows ``rr.log`` would have written, one chunk per entity per flush."""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.chunk as rrc
from fixture_types import RowsReader

from slam_rs.columns import ColumnLog


def test_held_rows_reach_the_recording_at_their_own_stamps_in_one_chunk_per_entity(tmp_path: Path, read_rows: RowsReader) -> None:
    output: Path = tmp_path / "columns.rrd"
    with rr.RecordingStream("slam-rs-columns-test", recording_id="columns") as recording:
        recording.save(output)
        log: ColumnLog = ColumnLog(recording=recording)
        for step, count in enumerate((3, 0, 2)):
            t_ns: int = 1_000 * (step + 1)
            log.add("/rig", rr.Transform3D.columns, t_ns, translation=np.array([step, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]))
            log.add("/stats/count", rr.Scalars.columns, t_ns, scalars=float(count))
            log.add_batch("/points", rr.Points3D.columns, t_ns, positions=np.full((count, 3), step, dtype=np.float32), colors=np.full(count, 0xFF0000FF, dtype=np.uint32))
        log.add_batch("/strips", rr.LineStrips3D.columns, 4_000, strips=[np.zeros((2, 3)), np.ones((5, 3))])
        log.flush()

    rows = read_rows(output)
    assert [row.t_ns for row in rows["/rig"]] == [1_000, 2_000, 3_000]
    assert [row.values["Transform3D:translation"] for row in rows["/rig"]] == [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]
    assert [row.values["Scalars:scalars"] for row in rows["/stats/count"]] == [[3.0], [0.0], [2.0]]
    assert [len(row.values["Points3D:positions"]) for row in rows["/points"]] == [3, 0, 2]
    assert rows["/points"][2].values["Points3D:positions"] == [[2.0, 2.0, 2.0]] * 2
    assert [len(strip) for strip in rows["/strips"][0].values["LineStrips3D:strips"]] == [2, 5]
    chunks: dict[str, int] = {}
    for chunk in rrc.RrdReader(output).stream():
        chunks[chunk.entity_path] = chunks.get(chunk.entity_path, 0) + 1
    assert {entity: chunks[entity] for entity in ("/rig", "/stats/count", "/points", "/strips")} == {"/rig": 1, "/stats/count": 1, "/points": 1, "/strips": 1}


def test_a_flush_forgets_what_it_sent(tmp_path: Path, read_rows: RowsReader) -> None:
    output: Path = tmp_path / "columns.rrd"
    with rr.RecordingStream("slam-rs-columns-test", recording_id="columns") as recording:
        recording.save(output)
        log: ColumnLog = ColumnLog(recording=recording)
        log.add("/stats/a", rr.Scalars.columns, 1_000, scalars=1.0)
        log.flush()
        log.add("/stats/a", rr.Scalars.columns, 2_000, scalars=2.0)
        log.flush()
        log.flush()
    assert [(row.t_ns, row.values["Scalars:scalars"]) for row in read_rows(output)["/stats/a"]] == [(1_000, [1.0]), (2_000, [2.0])]
