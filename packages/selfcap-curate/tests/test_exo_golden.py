"""Three session calibration goldens; only ephemeral catalogs receive registrations."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from frozen_v2 import RAW, ROOT, SERVED, session_episode
from scipy.spatial.transform import Rotation
from serde.json import from_json, to_json

from selfcap_curate.apis.pool import Config, PoolConfig, main, pool_main
from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_catalog_hands, read_phone_detections
from selfcap_curate.device_tables import FocalProfile, pool, read_table
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import read_frozen_sync

pytestmark = pytest.mark.golden
SESSIONS = ("2025-11-24__027d8215", "2025-12-17__e0385cf5", "2025-12-30__65d00748")


@pytest.mark.parametrize("session", SESSIONS)
def test_exo_session_matches_frozen_medians_and_episode_flags(session: str, tmp_path: Path, private_catalog) -> None:
    records = sorted((ROOT / "v2/sync").glob(f"{session}__ep*.json"))
    frozen_path = ROOT / "calib/sessions" / f"{session}.json"
    if not records or not os.access(frozen_path, os.R_OK):
        pytest.skip(f"golden SelfCap calibration/sync assets absent: {frozen_path}")
    syncs = [read_frozen_sync(path) for path in records]
    for sync in syncs:
        key = sync.key
        raw = RAW / sync.date / sync.session / "episodes" / f"episode-{sync.episode.removeprefix('ep')}"
        base = SERVED / "base" / f"selfcap__{key}.rrd"
        layer = SERVED / "calib-v2" / f"selfcap__{key}.rrd"
        for asset in (
            base,
            layer,
            SERVED / "sapiens2" / f"selfcap__{key}.rrd",
            raw / "ego/left.mp4",
            raw / "quest/left_hand_poses.csv",
            raw / "quest/right_hand_poses.csv",
        ):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"golden SelfCap asset absent or unreadable: {asset}")

    with private_catalog(syncs, tmp_path) as server:
        dataset = server.client().get_dataset(DATASET_NAME)

        episodes = []
        for sync in syncs:
            segment = open_segment(dataset, sync.episode_key.segment)
            episodes.append(
                session_episode(
                    sync,
                    detections=read_phone_detections(segment),
                    quest=load_quest_hands(read_catalog_hands(segment)),
                )
            )
        focals = {p: v.f for p, v in read_table().devices.items()}
        result = calibrate(session, episodes, focals)
        (tmp_path / "port.json").write_text(to_json(result))
        if session == SESSIONS[0]:
            main(Config(tmp_path / "cli.json", tuple(f"selfcap__{sync.key}" for sync in syncs), catalog_url=server.url()))
            profile = from_json(FocalProfile, (tmp_path / "cli.json").read_text())
            reference = from_json(FocalProfile, (ROOT / "calib/f_profile" / f"{session}.json").read_text())
            assert set(profile.curves) == set(reference.curves)
            actual_focals = pool([profile], "2026-10-26").focals
            frozen_focals = pool([reference], "2026-10-26").focals
            for phone, curve in profile.curves.items():
                relative = float(np.max(np.abs(np.asarray(curve) - reference.curves[phone]) / np.abs(reference.curves[phone])))
                focal_delta = abs(actual_focals[phone] - frozen_focals[phone])
                print("PROFILE_CATALOG", session, phone, relative, frozen_focals[phone], actual_focals[phone], focal_delta, flush=True)
                # 2%: Titanium-B00700F9 (minimum on the grid edge) measures 0.80% with default threads, 1.33% with OMP_NUM_THREADS=1.
                assert relative <= 0.02
                assert focal_delta <= 2.0

    frozen = json.loads(frozen_path.read_text())["exo"]
    failures = []
    assert set(result.phones) == set(frozen["phones"])
    for phone, expected in frozen["phones"].items():
        actual = result.phones[phone]
        assert actual.camera is not None and actual.median_px is not None
        old_transform = np.asarray(expected["world_T_cam"])
        transform = actual.camera.world_T_cam
        degrees = float(Rotation.from_matrix(old_transform[:3, :3].T @ transform[:3, :3]).magnitude() * 180.0 / np.pi)
        centimetres = float(np.linalg.norm(old_transform[:3, 3] - transform[:3, 3]) * 100.0)
        print(
            "EXO_COMPARE",
            session,
            phone,
            expected["median_px"],
            actual.median_px,
            degrees,
            centimetres,
            flush=True,
        )
        if abs(actual.median_px - expected["median_px"]) > 0.5:
            failures.append(f"{phone}: frozen={expected['median_px']}, port={actual.median_px}")
        for ep, reference in expected["episodes"].items():
            measured = actual.episodes[ep]
            print("EXO_FLAG", session, phone, ep, reference["ok"], measured.ok, measured.scope, flush=True)
            if measured.ok != reference["ok"]:
                failures.append(f"{phone}/{ep}: frozen ok={reference['ok']}, port={measured.ok}")
    print("EXO_OUTPUT", session, tmp_path, flush=True)
    assert not failures, "\n".join(failures)


def test_pool_cli_rebuilds_the_frozen_device_table(tmp_path: Path) -> None:

    profiles = tuple(sorted((ROOT / "calib/f_profile").glob("*.json")))
    if not profiles:
        pytest.skip(f"golden focal-profile assets absent: {ROOT / 'calib/f_profile'}")
    target = tmp_path / "device_intrinsics.json"
    pool_main(PoolConfig(profiles, target, "2026-10-26"))
    rebuilt = read_table(target)
    frozen = read_table()
    assert set(rebuilt.devices) == set(frozen.devices)
    for phone, device in frozen.devices.items():
        assert rebuilt.devices[phone].f == pytest.approx(device.f, abs=1e-9)
        assert rebuilt.devices[phone].source_sessions == device.source_sessions
