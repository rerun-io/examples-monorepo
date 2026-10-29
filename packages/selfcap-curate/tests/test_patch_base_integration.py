"""Read-only golden inputs, with both outputs written below pytest's tmp dir."""

import os
from collections.abc import Iterable
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
from dataforge import schema
from dataforge.datasets.selfcap import SelfcapConfig, build_episode_plan, episode_is_readable
from dataforge.identity import SequenceIdentity
from dataforge.schema import CAPTURE_TIMELINE
from dataforge.video_encoding import read_video_start_ns
from jaxtyping import Int64
from numpy import ndarray
from rerun.chunk import Chunk, RrdReader
from rrd_fixtures import entity_times

from selfcap_curate.patch_base import patch
from selfcap_curate.records import read_sync

pytestmark = pytest.mark.integration
KEYS = ("2025-11-25__45003c99__ep001", "2025-11-24__027d8215__ep002")
RAW_ROOT = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")
SERVED_ROOT = Path("/mnt/nas/datasets/dataforge-selfcap-rrd/base")
SYNC_ROOT = Path("/home/pablo/exoego-data/dataforge-selfcap-aligned/v2/sync")


@pytest.mark.parametrize("key", KEYS)
def test_served_patch_agrees_with_fresh_ingest_then_alignment(key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sync_path = SYNC_ROOT / f"{key}.json"
    served = SERVED_ROOT / f"selfcap__{key}.rrd"
    for asset in (sync_path, served):
        if not os.access(asset, os.R_OK):
            pytest.skip(f"golden SelfCap asset absent or unreadable: {asset}")
    sync = read_sync(sync_path)
    raw = RAW_ROOT / sync.date / sync.session / "episodes" / f"episode-{sync.episode.removeprefix('ep')}"
    if not episode_is_readable(raw):
        pytest.skip(f"golden raw episode absent or unreadable: {raw}")
    for side in ("left", "right"):
        if not os.access(raw / "quest" / f"{side}_hand_poses.csv", os.R_OK):
            pytest.skip(f"golden raw hands absent: {raw / 'quest' / (side + '_hand_poses.csv')}")
    plan = build_episode_plan(raw)
    # Cut AV1 files take Mp4Reader's packet passthrough; never request a GPU
    # transcode for this read-only integration comparison.
    for rig in plan.rigs:
        for camera in rig.cameras:
            with av.open(str(camera.video_path)) as container:
                assert not container.streams.video[0].codec_context.has_b_frames, "GPU-free reference conversion requires cut videos without B frames"
    patched = patch(served, raw, sync, output_path=tmp_path / "patched.rrd", mode="migrate-keep-clock")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "fresh"))
    fresh = SelfcapConfig().setup().convert(SequenceIdentity(dataset="selfcap", parts=(sync.date, sync.session[:8], sync.episode)), raw, force=False)
    offsets: dict[str, float] = {}
    starts: dict[str, int] = {}
    rgb_entity = schema.video_path(plan.ego_rig, 0)
    for rig in plan.rigs:
        for camera in rig.cameras:
            entity = schema.video_path(camera.rig, camera.cam)
            starts[entity] = read_video_start_ns(camera.video_path)
            if rig.kind == "exo":
                offset = sync.offsets_ms[f"exo/{rig.name}"]
            elif rig.kind == "quest":
                offset = sync.offsets_ms["quest_video"]
            else:
                offset = 0.0
            offsets[entity] = 0.0 if offset is None else offset
    for entity in (schema.gyro_path(plan.ego_rig, 0), schema.accel_path(plan.ego_rig, 0)):
        assert sync.offsets_ms["oak_imu"] is not None
        offsets[entity] = sync.offsets_ms["oak_imu"]
    for entity in (schema.rig_path(plan.quest_rig), schema.coco133_xyz_path()):
        assert sync.offsets_ms["quest_pose"] is not None
        offsets[entity] = sync.offsets_ms["quest_pose"] - (48.0 if entity == schema.coco133_xyz_path() else 0.0)

    def align_fresh(chunk: Chunk) -> Iterable[Chunk]:
        if chunk.is_static:
            return [chunk]
        batch = chunk.to_record_batch()
        entity = str(chunk.entity_path)
        capture: Int64[ndarray, "n"] = np.asarray(batch.column(CAPTURE_TIMELINE).cast(pa.int64()))
        if entity == rgb_entity:
            # The served RGB model was fitted on rebased Mp4Reader time.
            model = sync.oak_rgb_model
            shift = np.rint((model.a_ms + model.b_ms_per_s * (model.t_start_session_s + (capture - starts[entity]) / 1e9)) * 1e6).astype(np.int64)
        else:
            shift = round(offsets[entity] * 1e6)
        index = batch.schema.get_field_index("video_time")
        batch = batch.set_column(index, batch.schema.field(index), pa.array(capture + shift, type=batch.schema.field(index).type))
        return Chunk.from_record_batch(batch)

    expected = tmp_path / "fresh-aligned.rrd"
    reader = RrdReader(fresh)
    store = reader.recordings()[0]
    reader.stream(store=store).flat_map(align_fresh).write_rrd(expected, application_id=store.application_id, recording_id=store.recording_id)
    migrated_times = entity_times(patched)
    fresh_times = entity_times(expected)
    assert migrated_times.keys() == fresh_times.keys()
    for entity, actual in migrated_times.items():
        reference = fresh_times[entity]
        assert actual.samples == reference.samples
        assert len(actual.capture_ns) == len(reference.capture_ns)
        capture_delta = int(np.max(np.abs(actual.capture_ns - reference.capture_ns)))
        video_delta = int(np.max(np.abs(actual.video_ns - reference.video_ns)))
        print("BASE_COMPARISON", key, entity, len(actual.capture_ns), actual.samples, capture_delta, video_delta, flush=True)
        assert capture_delta == 0, entity
        assert video_delta <= (1000 if entity == rgb_entity else 0), entity
