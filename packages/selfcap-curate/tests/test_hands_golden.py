"""GPU parity on isolated patched catalogs; raw assets only feed migration."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from frozen_v2 import RAW, SERVED, load_frozen_links

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_catalog_hands, read_phone_detections, read_phone_videos
from selfcap_curate.hands_gpu import detect_episode, pick_frames, pose_models
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import read_frozen_sync
from selfcap_curate.session_fit import FitEpisode
from selfcap_curate.sync_policy import fit_session

pytestmark = pytest.mark.golden
V2 = Path("/home/pablo/exoego-data/dataforge-selfcap-aligned/v2")


@pytest.fixture(scope="module")
def models():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for catalog hand golden pass")
    return pose_models()


@pytest.mark.parametrize("target", ["2025-12-30__65d00748__ep001", "2025-11-24__027d8215__ep002"])
def test_catalog_hands_match_frozen_and_session_offsets(target, models, tmp_path, private_catalog):
    started = time.monotonic()
    paths = sorted((V2 / "sync").glob(f"{target.rsplit('__', 1)[0]}__ep*.json"))
    if not paths:
        pytest.skip(f"golden sync records absent: {target}")
    references = {p.stem: read_frozen_sync(p) for p in paths}

    for key, record in references.items():
        raw = RAW / record.date / record.session / "episodes" / f"episode-{record.episode[2:]}"
        base = SERVED / "base" / f"selfcap__{key}.rrd"
        calib = SERVED / "calib-v2" / f"selfcap__{key}.rrd"
        for asset in (base, calib, raw / "quest/left_hand_poses.csv", SERVED / "sapiens2" / f"selfcap__{key}.rrd", V2 / "links" / f"{key}.json"):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"golden asset absent or unreadable: {asset}")

    with private_catalog(references.values(), tmp_path) as server:
        dataset = server.client().get_dataset(DATASET_NAME)

        inputs = []
        for key, record in references.items():
            phones = tuple(k[4:] for k in record.offsets_ms if k.startswith("exo/"))
            handle = open_segment(dataset, f"selfcap__{key}")
            frozen = read_phone_detections(handle)
            quest = load_quest_hands(read_catalog_hands(handle))
            detections = frozen
            if key == target:
                videos = read_phone_videos(handle)
                first = next(iter(videos.values()))
                indices = pick_frames(quest, len(first.samples), 1e9 / float(np.median(np.diff(first.t_ns))))
                for phone, video in videos.items():
                    np.testing.assert_array_equal(video.t_ns[indices], np.rint(frozen[phone].times_s * 1e9).astype(np.int64))
                detections = detect_episode(handle, quest, *models)
                assert detections.keys() == frozen.keys()
                for phone, actual in detections.items():
                    expected = frozen[phone]
                    time_error = float(np.max(np.abs(actual.times_s - expected.times_s)))
                    valid = expected.scores >= 0.3
                    assert valid.any()
                    assert np.isfinite(actual.xy[expected.scores >= 0.3]).all()
                    delta = np.abs(actual.xy[valid] - expected.xy[valid])
                    median, maximum = float(np.median(delta)), float(np.max(delta))
                    print("HAND_PARITY", key, phone, len(indices), time_error, median, maximum, flush=True)
                    assert time_error <= 1e-6
                    assert median <= 1.0
            inputs.append(
                FitEpisode(
                    record.session, load_frozen_links(V2 / "links" / f"{key}.json", V2 / "links_rgb" / f"{key}.json"), phones, detections, quest
                )
            )
        outputs = fit_session(inputs)
    for key, result in outputs.items():
        reference = references[key]
        for stream, expected in reference.offsets_ms.items():
            actual = result.offsets_ms[stream]
            delta = None if expected is None or actual is None else abs(expected - actual)
            print("HAND_FIT", key, stream, expected, actual, delta, flush=True)
            assert actual is None if expected is None else actual is not None and delta <= 2.0
        episode = next(ep for ep in inputs if ep.key == key)
        mid = episode.observation.duration_s / 2.0
        expected = reference.oak_rgb_model
        actual = result.oak_rgb_model
        delta = abs(
            actual.a_ms
            + actual.b_ms_per_s * (actual.t_start_session_s + mid)
            - expected.a_ms
            - expected.b_ms_per_s * (expected.t_start_session_s + mid)
        )
        print("HAND_FIT", key, "oak_rgb_midpoint", delta, flush=True)
        assert delta <= 2.0
    print("HAND_RUNTIME", target, time.monotonic() - started, flush=True)
