"""Nine golden episodes, isolated from the shared catalog and served files."""

import os
import time
from pathlib import Path

import pytest
from frozen_v2 import load_frozen_links

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_catalog_hands, read_phone_detections
from selfcap_curate.patch_base import patch
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import read_frozen_sync, read_sync, write_sync
from selfcap_curate.session_fit import FitEpisode
from selfcap_curate.sync_policy import fit_session

pytestmark = pytest.mark.golden
SESSIONS = ("2025-11-24__027d8215", "2025-12-17__e0385cf5", "2025-12-30__65d00748")
V2 = Path("/home/pablo/exoego-data/dataforge-selfcap-aligned/v2")
SERVED = Path("/mnt/nas/datasets/dataforge-selfcap-rrd")
RAW = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")


@pytest.mark.parametrize("session", SESSIONS)
def test_session_offsets_match_frozen(session: str, tmp_path: Path, private_server) -> None:
    paths = sorted((V2 / "sync").glob(f"{session}__ep*.json"))
    if not paths:
        pytest.skip(f"golden session sync records absent: {V2 / 'sync' / session}")
    references = {path.stem: read_frozen_sync(path) for path in paths}
    files: list[Path] = []
    calibrations: list[Path] = []
    t0 = time.monotonic()
    for key, reference in references.items():
        raw = RAW / reference.date / reference.session / "episodes" / f"episode-{reference.episode.removeprefix('ep')}"
        base = SERVED / "base" / f"selfcap__{key}.rrd"
        calib = SERVED / "calib-v2" / f"selfcap__{key}.rrd"
        for asset in (
            raw / "ego/left.mp4",
            raw / "quest/left_hand_poses.csv",
            raw / "quest/right_hand_poses.csv",
            SERVED / "sapiens2" / f"selfcap__{key}.rrd",
            V2 / "links" / f"{key}.json",
            base,
            calib,
        ):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"golden SelfCap asset absent or unreadable: {asset}")
        assert reference.offsets_ms is not None
        phones = tuple(name.removeprefix("exo/") for name in reference.offsets_ms if name.startswith("exo/"))
        patch_start = time.monotonic()
        files.append(patch(base, raw, reference, output_path=tmp_path / f"selfcap__{key}.rrd", mode="migrate-keep-clock"))
        print("EPISODE_TIME", key, "patch", time.monotonic() - patch_start, flush=True)
        calibrations.append(calib)
    print("SESSION_TIME", session, "patch", time.monotonic() - t0, flush=True)
    with private_server({DATASET_NAME: files}) as server:
        assert ":51235" not in server.url()
        dataset = server.client().get_dataset(DATASET_NAME)
        dataset.register([path.resolve().as_uri() for path in calibrations], layer_name="calib-v2").wait()
        dataset.register([(SERVED / "sapiens2" / f"selfcap__{key}.rrd").resolve().as_uri() for key in references], layer_name="sapiens2").wait()
        inputs: list[FitEpisode] = []
        for key, reference in references.items():
            assert reference.offsets_ms is not None
            phones = tuple(name.removeprefix("exo/") for name in reference.offsets_ms if name.startswith("exo/"))
            input_start = time.monotonic()
            handle = open_segment(dataset, f"selfcap__{key}")
            detections = read_phone_detections(handle)
            observations = load_frozen_links(V2 / "links" / f"{key}.json", V2 / "links_rgb" / f"{key}.json")
            quest = load_quest_hands(read_catalog_hands(handle))
            inputs.append(FitEpisode(reference.session, observations, phones, detections, quest))
            print("EPISODE_TIME", key, "inputs", time.monotonic() - input_start, flush=True)
        start = time.monotonic()
        outputs = fit_session(inputs)
        print("SESSION_TIME", session, "port", time.monotonic() - start, flush=True)
    failures: list[str] = []
    for episode in inputs:
        key = episode.key
        frozen = references[key]
        result = outputs[key]
        target = tmp_path / "fit" / f"{key}.json"
        target.parent.mkdir(exist_ok=True)
        write_sync(target, result)
        assert read_sync(target) == result
        assert "NaN" not in target.read_text()
        assert result.offsets_ms is not None and frozen.offsets_ms is not None
        for stream, expected in frozen.offsets_ms.items():
            measured = result.offsets_ms[stream]
            delta = None if measured is None or expected is None else abs(measured - expected)
            print("FIT_OFFSET", key, stream, expected, measured, delta, flush=True)
            if expected is None:
                assert measured is None
            elif measured is None or abs(measured - expected) > 2.0:
                failures.append(f"{key} {stream}: frozen={expected}, port={measured}")
        midpoint = episode.observation.duration_s / 2.0
        models = (frozen.oak_rgb_model, result.oak_rgb_model)
        values = [model.a_ms + model.b_ms_per_s * (model.t_start_session_s + midpoint) for model in models]
        print("FIT_OFFSET", key, "oak_rgb_midpoint", *values, abs(values[1] - values[0]), flush=True)
        if abs(values[1] - values[0]) > 2.0:
            failures.append(f"{key} RGB midpoint: frozen/port={values}")
    print("SESSION_TIME", session, "total", time.monotonic() - t0, flush=True)
    assert not failures, "\n".join(failures)
