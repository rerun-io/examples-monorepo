"""The colsfm layer's pictures: each rig camera's frustum references its own keyframe in the segment's video stream."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import rerun as rr
from mapping_helpers import SyntheticRig, build_synthetic_rig  # pyrefly: ignore[missing-import]
from rerun.chunk import RrdReader

from colsfm.frame_manifest import FrameSourceManifest, StreamKeyframes
from colsfm.frames_meta import KeyframeMeta
from colsfm.rerun_layer import log_frame_references

SKEW_NS: int = 4_000
"""How far each synthetic camera's sample stamp sits from its rig frame's stamp: the two must not be confused."""


def test_each_camera_references_its_keyframe_of_every_rig_frame(tmp_path: Path) -> None:
    """Under `<root>/rig/<camera>/pinhole/video`, at each rig frame's time, a `VideoFrameReference` to that camera's
    sample of the rig frame, in the stream the run read it from. No stream is copied or encoded again."""
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=30)
    meta = rig.frames_meta
    streams: list[StreamKeyframes] = []
    for camera in meta.cameras.values():
        own: list[KeyframeMeta] = [keyframe for keyframe in meta.keyframes if keyframe.camera_params_id == camera.camera_params_id]
        streams.append(
            StreamKeyframes(
                entity=f"world/rig_00/cam_{camera.camera_params_id:02d}/pinhole/video",
                names=[keyframe.image_name for keyframe in own],
                t_ns=[keyframe.timestamp_microseconds * 1000 + SKEW_NS * (1 + camera.camera_params_id) for keyframe in own],
            )
        )
    manifest = FrameSourceManifest(catalog_url="rerun+http://catalog", dataset="synthetic", segment="segment", timeline="video_time", streams=streams)
    rrd: Path = tmp_path / "layer.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("colsfm-test", recording_id="segment")
    recording.save(rrd)
    log_frame_references(recording, "/world/runs/colsfm", meta, manifest)
    recording.flush()
    del recording

    rows: dict[str, dict[int, int]] = {}
    targets: dict[str, str] = {}
    for chunk in RrdReader(rrd).store().stream():
        batch: pa.RecordBatch = chunk.to_record_batch()
        entity: str = str(chunk.entity_path)
        if chunk.is_static and "VideoFrameReference:video_reference" in batch.schema.names:
            targets[entity] = batch.column("VideoFrameReference:video_reference").to_pylist()[0][0]
        elif "VideoFrameReference:timestamp" in batch.schema.names:
            times: list[int] = batch.column("video_time").cast(pa.int64()).to_pylist()
            rows.setdefault(entity, {}).update({t: values[0] for t, values in zip(times, batch.column("VideoFrameReference:timestamp").to_pylist(), strict=True)})
    rig_frame_ns: list[int] = [rig_frame.timestamp_microseconds * 1000 for rig_frame in meta.rig_frames()]
    assert len(rig_frame_ns) == 4
    for camera in meta.cameras.values():
        entity = f"/world/runs/colsfm/rig/{camera.sensor_name}/pinhole/video"
        assert targets[entity] == f"/world/rig_00/cam_{camera.camera_params_id:02d}/pinhole/video"
        own_ns: dict[int, int] = {
            keyframe.synced_sample_id: keyframe.timestamp_microseconds * 1000 + SKEW_NS * (1 + camera.camera_params_id)
            for keyframe in meta.keyframes
            if keyframe.camera_params_id == camera.camera_params_id
        }
        assert rows[entity] == {rig_frame.timestamp_microseconds * 1000: own_ns[rig_frame.synced_sample_id] for rig_frame in meta.rig_frames()}
