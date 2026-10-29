"""Full-frame OAK GPU parity and served session calibration, private catalogs."""

import json
import os
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from dataforge.schema import CAPTURE_TIMELINE, video_path
from frozen_v2 import GOLDEN_REFS, RAW, ROOT, SERVED, load_oak_detections, load_oak_reference, oak_median_reference
from serde import from_dict
from simplecv.catalog_video import read_catalog_videos

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_factories
from selfcap_curate.device_tables import read_rotations
from selfcap_curate.factory import oak_centre
from selfcap_curate.hands_gpu import detect_oak_hands_gpu, pick_oak_frames, pose_models
from selfcap_curate.oak_mount import fixed_rotation, gyro_mount, solve_oak
from selfcap_curate.quest_fit import load_quest_hands, read_session_episode
from selfcap_curate.records import OakFit, oak_quest_offset_ms, read_frozen_sync

pytestmark = pytest.mark.golden
SESSIONS = ("2025-11-24__027d8215", "2025-12-17__e0385cf5", "2025-12-30__65d00748")


@pytest.fixture(scope="module")
def pose():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for full-frame OAK hand golden pass")
    return pose_models()[1]


@pytest.mark.parametrize("session", SESSIONS)
def test_oak_gpu_matches_frozen_frames_and_served_mount(session, pose, tmp_path, private_catalog):
    started = time.monotonic()
    sources = sorted((ROOT / "v2/sync").glob(f"{session}__ep*.json"))
    reference = ROOT / "calib/sessions" / f"{session}.json"
    if not sources or not os.access(reference, os.R_OK):
        pytest.skip(f"OAK served session reference absent: {reference}")
    expected = from_dict(OakFit, json.loads(reference.read_text())["oak"])
    syncs = [read_frozen_sync(path) for path in sources]

    for sync in syncs:
        raw = RAW / sync.date / sync.session / "episodes" / f"episode-{sync.episode[2:]}"
        base = SERVED / "base" / f"selfcap__{sync.key}.rrd"
        layer = SERVED / "calib-v2" / f"selfcap__{sync.key}.rrd"
        for asset in (
            base,
            layer,
            raw / "ego/calibration.json",
            raw / "quest/left_hand_poses.csv",
            GOLDEN_REFS / "oak-sapiens2" / sync.episode_key.rrd_name,
        ):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"OAK golden asset absent or unreadable: {asset}")

    rotations = read_rotations()
    with private_catalog(syncs, tmp_path) as server:
        dataset = server.client().get_dataset(DATASET_NAME)

        segments = {sync.key: open_segment(dataset, sync.episode_key.segment) for sync in syncs}
        factories = read_factories(segments[syncs[0].key])
        factory = factories["ego"]
        mounts = [read_session_episode(segments[sync.key], sync) for sync in syncs]
        gyro = gyro_mount(session, mounts)
        fixed = fixed_rotation(gyro, factory.device_id, rotations)
        intrinsic = oak_centre(factory)
        generated, frozen = [], []
        failures = []
        for mount in mounts:
            key = mount.sync.key
            source = GOLDEN_REFS / "oak-sapiens2" / f"selfcap__{key}.rrd"
            video = read_catalog_videos(dataset, f"selfcap__{key}", [video_path(len(mount.sync.exo_fit), 0)], CAPTURE_TIMELINE)[0]
            indices = pick_oak_frames(load_quest_hands(mount.hands), len(video.samples), 1e9 / float(np.median(np.diff(video.t_ns))), mount.sync)
            if oak_quest_offset_ms(mount.sync) is None:
                assert indices.size == 0
                actual = detect_oak_hands_gpu(dataset, mount.sync, load_quest_hands(mount.hands), mount.layout, pose)
                assert actual.video_time.size == 0 and actual.xy.shape == (0, 133, 2)
                generated.append(replace(mount, oak=actual))
                frozen.append(replace(mount, oak=load_oak_detections(source)))
                print("OAK_HANDS_EXCLUDED", key, flush=True)
                continue
            fixture = load_oak_reference(source)
            assert indices.tolist() == fixture.indices.tolist(), key
            capture = video.t_ns[indices].astype(np.float64) * 1e-9
            time_error = float(np.max(np.abs(capture - fixture.times_s))) if len(indices) else 0.0
            aligned_error = (
                float(np.max(np.abs(mount.sync.oak_rgb_model.video_time_s(capture) - fixture.detections.video_time))) if len(indices) else 0.0
            )
            before = time.monotonic()
            actual = detect_oak_hands_gpu(dataset, mount.sync, load_quest_hands(mount.hands), mount.layout, pose)
            old = load_oak_detections(source)
            np.testing.assert_allclose(actual.video_time, old.video_time, rtol=0.0, atol=1e-6)
            valid = old.scores >= 0.3
            assert np.isfinite(actual.xy[valid]).all()
            delta = np.abs(actual.xy[valid] - old.xy[valid])
            median = float(np.median(delta)) if delta.size else 0.0
            maximum = float(np.max(delta)) if delta.size else 0.0
            print("OAK_HANDS", key, len(indices), time_error, aligned_error, median, maximum, time.monotonic() - before, flush=True)
            if time_error > 1e-6 or aligned_error > 1e-6 or median > 1.0:
                failures.append(f"{key}: times={time_error}/{aligned_error}, median={median}, max={maximum}")
            generated.append(replace(mount, oak=actual))
            frozen.append(replace(mount, oak=old))
        cv2.setRNGSeed(0)
        baseline = solve_oak(session, frozen, intrinsic, factory.device_id, fixed)
        cv2.setRNGSeed(0)
        result = solve_oak(session, generated, intrinsic, factory.device_id, fixed)
    print("OAK_HANDS_MOUNT", session, expected.median_px, baseline.median_px, result.median_px, flush=True)
    print("OAK_HANDS_TIME", session, time.monotonic() - started, flush=True)
    assert result.ok == expected.ok
    assert result.rotation_source == expected.rotation_source
    assert result.median_px == pytest.approx(oak_median_reference(syncs, expected.median_px), abs=0.5)
    assert not failures, "\n".join(failures)
