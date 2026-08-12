"""Ground-truth logging regression tests.

Box entity paths must be per-file slot indices shared across sequences — never
per-object uids. Unique paths become dataset schema columns in the catalog and
make registration cost grow ~cubically with file count (docs/full-run-runbook.md §8).
The uid must survive as a component on the slot entity instead.
"""

import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh
from rerun.experimental import RrdReader

from arkitscenes_download.ingest.gt import ArkitMeshSummary, log_arkit_mesh, log_gt_boxes


def write_sequence(sequence_dir: Path, video_id: str, uids: list[str]) -> None:
    boxes = [
        {
            "label": "chair",
            "uid": uid,
            "segments": {
                "obbAligned": {
                    "centroid": [0.0, 0.0, 0.0],
                    "axesLengths": [1.0, 1.0, 1.0],
                    "normalizedAxes": list(np.eye(3).ravel()),
                }
            },
        }
        for uid in uids
    ]
    (sequence_dir / f"{video_id}_3dod_annotation.json").write_text(json.dumps({"data": boxes}))
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], vertex_colors=[[255, 0, 0, 255]] * 3)
    mesh.export(sequence_dir / f"{video_id}_3dod_mesh.ply")


def logged_boxes(tmp_path: Path, video_id: str, uids: list[str]) -> tuple[set[str], dict[str, str]]:
    """Log a synthetic sequence's ground truth and read back (box paths, path -> uid)."""
    sequence_dir = tmp_path / video_id
    sequence_dir.mkdir()
    write_sequence(sequence_dir, video_id, uids)
    rrd_path = tmp_path / f"{video_id}.rrd"
    recording = rr.RecordingStream("gt_test", recording_id=video_id)
    recording.save(str(rrd_path))
    log_gt_boxes(sequence_dir, video_id, recording)
    recording.flush(timeout_sec=10.0)
    reader = RrdReader(rrd_path)
    chunks = list(reader.stream(store=reader.recordings()[0]).to_chunks())
    box_paths = {str(chunk.entity_path) for chunk in chunks if "/gt_boxes/" in str(chunk.entity_path)}
    path_to_uid: dict[str, str] = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        for name in batch.schema.names:
            if "uid" in name.lower():
                path_to_uid[str(chunk.entity_path)] = batch.column(name).to_pylist()[0][0]
    return box_paths, path_to_uid


def test_box_paths_are_slot_indexed_and_shared_across_sequences(tmp_path: Path) -> None:
    paths_a, uids_a = logged_boxes(tmp_path, "0001", ["AAAA1111", "BBBB2222"])
    paths_b, uids_b = logged_boxes(tmp_path, "0002", ["CCCC3333", "DDDD4444"])
    assert paths_a == {"/world/gt_boxes/box_00", "/world/gt_boxes/box_01"}
    # Different physical objects, identical entity paths — the schema-width guard.
    assert paths_a == paths_b
    # No uid ever leaks into a path name.
    assert all("box_" in path.rsplit("/", 1)[-1] for path in paths_a | paths_b)
    # The uid survives as a component on the slot entity.
    assert uids_a["/world/gt_boxes/box_00"] == "AAAA1111"
    assert uids_b["/world/gt_boxes/box_01"] == "DDDD4444"


def test_arkit_mesh_logger_returns_framing_geometry(tmp_path: Path) -> None:
    """The ARKit mesh recording carries only its renamed mesh entity and returns framing data."""
    video_id: str = "0001"
    sequence_dir: Path = tmp_path / video_id
    sequence_dir.mkdir()
    write_sequence(sequence_dir, video_id, [])
    rrd_path: Path = tmp_path / "arkit_mesh.rrd"
    recording = rr.RecordingStream("mesh_test", recording_id=video_id)
    recording.save(rrd_path)
    summary: ArkitMeshSummary = log_arkit_mesh(sequence_dir, video_id, recording)
    recording.flush(timeout_sec=10.0)

    reader = RrdReader(rrd_path)
    chunks = list(reader.stream(store=reader.recordings()[0]).to_chunks())
    assert "/world/arkit_mesh" in {str(chunk.entity_path) for chunk in chunks}
    np.testing.assert_allclose(summary.mesh_center_xyz, [0.5, 0.5, 0.0])
    assert summary.bounding_radius_m > 0.0


def test_zero_box_capture_still_publishes_a_registerable_layer(tmp_path: Path) -> None:
    """An empty annotation list must not produce a chunkless (unregisterable) RRD."""
    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    write_sequence(sequence_dir, "vid0", uids=[])
    rrd_path = tmp_path / "gt_boxes.rrd"
    with rr.RecordingStream(application_id="arkitscenes", recording_id="vid0", send_properties=False) as recording:
        recording.save(rrd_path)
        box_count = log_gt_boxes(sequence_dir, "vid0", recording)

    assert box_count == 0
    chunks = list(RrdReader(rrd_path).stream())
    assert chunks, "zero-box gt_boxes.rrd must carry the box_count marker chunk"
    assert {str(chunk.entity_path) for chunk in chunks} == {"/world/gt_boxes"}
