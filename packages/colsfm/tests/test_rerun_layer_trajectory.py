"""The colsfm layer's path (a trail in the walk gradient, the whole path at the end, its endpoints) and its viewing rig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64, Int64
from mapping_helpers import SyntheticRig, build_synthetic_rig  # pyrefly: ignore[missing-import]
from numpy import ndarray
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation

from colsfm.frame_manifest import FrameSourceManifest, StreamKeyframes
from colsfm.frame_poses import ViewRigSource, view_rig_poses
from colsfm.frames_meta import KeyframeMeta
from colsfm.rerun_layer import log_rig_trajectory, log_view_rig, walk_colors

FRAME_NS: int = 33_366_667
"""One frame at 29.97 fps."""


def wiggly_walk(t_ns: Int64[ndarray, " n"]) -> Float64[ndarray, "n 4 4"]:
    """A hand-held walk: forward with a sway and a nod, so the motion between two keyframes is not a slerp."""
    seconds: Float64[ndarray, " n"] = t_ns * 1e-9
    poses: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (len(t_ns), 1, 1))
    poses[:, :3, 3] = np.stack([0.8 * seconds, 0.05 * np.sin(9.0 * seconds), 0.03 * np.sin(13.0 * seconds)], axis=1)
    poses[:, :3, :3] = Rotation.from_euler("zyx", np.stack([0.3 * seconds, 0.1 * np.sin(7.0 * seconds), 0.08 * np.sin(11.0 * seconds)], axis=1)).as_matrix()
    return poses


def rows(rrd: Path) -> tuple[dict[str, list[tuple[int, dict[str, Any]]]], dict[str, dict[str, Any]]]:
    """Temporal rows by entity (time, values) and static values by entity."""
    temporal: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    static: dict[str, dict[str, Any]] = {}
    for chunk in RrdReader(rrd).store().stream():
        batch: pa.RecordBatch = chunk.to_record_batch()
        entity: str = str(chunk.entity_path)
        names: list[str] = [name for name in batch.schema.names if ":" in name and not name.startswith("rerun")]
        if chunk.is_static:
            static.setdefault(entity, {}).update({name: batch.column(name).to_pylist()[0] for name in names})
            continue
        times: list[int] = batch.column("video_time").cast(pa.int64()).to_pylist()
        for row, t in enumerate(times):
            temporal.setdefault(entity, []).append((t, {name: batch.column(name)[row].as_py() for name in names}))
    for entity in temporal:
        temporal[entity].sort(key=lambda item: item[0])
    return temporal, static


def test_the_walk_gradient_is_slam_rs_s() -> None:
    """The colours ``slam_rs.vio_log`` pins for the same fractions (its walk gradient is a copy: no shared package)."""
    t_ns: Int64[ndarray, " n"] = np.array([0, 25, 50, 75, 100], dtype=np.int64)
    np.testing.assert_array_equal(walk_colors(t_ns, 0, 100), [[38, 255, 38], [147, 255, 38], [255, 255, 38], [255, 147, 38], [255, 38, 38]])


def test_the_path_is_a_gradient_trail_with_the_whole_path_and_its_end_from_the_last_rig_frame(tmp_path: Path) -> None:
    times: Int64[ndarray, " n"] = np.arange(5, dtype=np.int64) * 10 * FRAME_NS
    poses: Float64[ndarray, "n 4 4"] = wiggly_walk(times)
    rrd: Path = tmp_path / "path.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    log_rig_trajectory(recording, "/world/runs/colsfm", times, poses, "video_time")
    recording.flush()
    del recording
    temporal, static = rows(rrd)
    colors = walk_colors(times, int(times[0]), int(times[-1]))
    packed: list[int] = [(int(r) << 24) | (int(g) << 16) | (int(b) << 8) | 255 for r, g, b in colors]
    trail = temporal["/world/runs/colsfm/trail"]
    assert [t for t, _ in trail] == times[1:].tolist()
    assert [values["LineStrips3D:colors"] for _, values in trail] == [[c] for c in packed[1:]]
    np.testing.assert_allclose(np.asarray(trail[0][1]["LineStrips3D:strips"][0]), poses[:2, :3, 3], atol=1e-6)
    full = temporal["/world/runs/colsfm/trajectory"]
    assert [t for t, _ in full] == [int(times[-1])] and len(full[0][1]["LineStrips3D:strips"]) == 4
    assert [t for t, _ in temporal["/world/runs/colsfm/endpoints/end"]] == [int(times[-1])]
    assert static["/world/runs/colsfm/endpoints/start"]["Points3D:labels"] == ["start"]
    assert "LineStrips3D:strips" not in static.get("/world/runs/colsfm/trajectory", {}), "no whole path stands over the walk's cursors"


def test_the_view_rig_is_the_run_at_its_rig_frames_and_moves_as_the_prior_between_them() -> None:
    """With one world-frame correction of the prior (the run's world a little off the prior's), the viewing rig is that
    correction times the prior at every viewing frame: exactly the run at its rig frames, and the prior's sway between
    them (the every-frame export's rule, ``colsfm.frame_poses.reanchor``)."""
    frames: Int64[ndarray, " n"] = np.arange(120, dtype=np.int64) * FRAME_NS
    prior: Float64[ndarray, "n 4 4"] = wiggly_walk(frames)
    offset: Float64[ndarray, "4 4"] = np.eye(4)
    offset[:3, :3] = Rotation.from_euler("xyz", [0.02, -0.01, 0.03]).as_matrix()
    offset[:3, 3] = [0.01, -0.02, 0.005]
    rig_t_ns: Int64[ndarray, " k"] = frames[5:115:10]
    world_T_rig: Float64[ndarray, "k 4 4"] = offset @ prior[5:115:10]
    source = ViewRigSource(prior_t_ns=frames, world_T_prior=prior, view_t_ns=frames, view_entities={})
    view_t_ns, world_T_view = view_rig_poses(rig_t_ns, world_T_rig, source)
    np.testing.assert_array_equal(view_t_ns, frames[5:106])
    np.testing.assert_allclose(world_T_view, offset @ prior[5:106], atol=1e-9)
    np.testing.assert_allclose(world_T_view[::10], world_T_rig, atol=1e-9)


def test_each_view_rig_camera_references_its_viewing_face_at_every_viewing_frame(tmp_path: Path) -> None:
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=30)
    meta = rig.frames_meta
    streams: list[StreamKeyframes] = []
    for camera in meta.cameras.values():
        own: list[KeyframeMeta] = [keyframe for keyframe in meta.keyframes if keyframe.camera_params_id == camera.camera_params_id]
        streams.append(
            StreamKeyframes(
                entity=f"world/rig_00/cam_{camera.camera_params_id:02d}/pinhole/video",
                names=[keyframe.image_name for keyframe in own],
                t_ns=[keyframe.timestamp_microseconds * 1000 for keyframe in own],
            )
        )
    manifest = FrameSourceManifest(catalog_url="rerun+http://catalog", dataset="synthetic", segment="segment", timeline="video_time", streams=streams)
    rig_t_ns: Int64[ndarray, " k"] = np.asarray([rig_frame.timestamp_microseconds * 1000 for rig_frame in meta.rig_frames()], dtype=np.int64)
    span: Int64[ndarray, " n"] = np.arange(int(rig_t_ns[0]), int(rig_t_ns[-1]) + 1, FRAME_NS // 4, dtype=np.int64)
    walk: Float64[ndarray, "n 4 4"] = wiggly_walk(span)
    views: dict[str, str] = {f"/{stream.entity}": f"/{stream.entity.removesuffix('/video')}/video_view" for stream in streams}
    source = ViewRigSource(prior_t_ns=span, world_T_prior=walk, view_t_ns=span, view_entities=views)
    rrd: Path = tmp_path / "view.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    log_view_rig(recording, "/world/runs/colsfm", meta, manifest, rig_t_ns, wiggly_walk(rig_t_ns), source)
    recording.flush()
    del recording
    temporal, static = rows(rrd)
    shown: list[int] = span[span <= rig_t_ns[rig_t_ns <= span[-1]][-1]].tolist()  # up to the last rig frame the prior covers
    assert [t for t, _ in temporal["/world/runs/colsfm/rig_view"]] == shown
    for camera in meta.cameras.values():
        entity: str = f"/world/runs/colsfm/rig_view/{camera.sensor_name}/pinhole/video"
        assert static[entity]["VideoFrameReference:video_reference"] == [f"/world/rig_00/cam_{camera.camera_params_id:02d}/pinhole/video_view"]
        assert [(t, values["VideoFrameReference:timestamp"]) for t, values in temporal[entity]] == [(t, [t]) for t in shown]
        assert "Pinhole:image_from_camera" in static[f"/world/runs/colsfm/rig_view/{camera.sensor_name}/pinhole"]


def test_a_comparison_source_keeps_one_colour(tmp_path: Path) -> None:
    """``color`` replaces the walk gradient, so a comparison source stays apart from a colsfm run beside it."""
    times: Int64[ndarray, " n"] = np.arange(4, dtype=np.int64) * FRAME_NS
    rrd: Path = tmp_path / "flat.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    log_rig_trajectory(recording, "/world/runs/slam_rs", times, wiggly_walk(times), "video_time", color=(60, 160, 255))
    recording.flush()
    del recording
    temporal, _ = rows(rrd)
    flat: int = (60 << 24) | (160 << 16) | (255 << 8) | 255
    assert {values["LineStrips3D:colors"][0] for _, values in temporal["/world/runs/slam_rs/trail"]} == {flat}
