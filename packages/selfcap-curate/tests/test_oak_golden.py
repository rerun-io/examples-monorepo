"""OAK/Quest goldens on patched local bases; no shared-catalog registrations."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from frozen_v2 import GOLDEN_REFS, RAW, ROOT, SERVED, assert_quest_block, load_oak_detections, oak_median_reference
from scipy.spatial.transform import Rotation
from serde.json import from_json

from selfcap_curate.apis.pool import RotationConfig as Config
from selfcap_curate.apis.pool import rotation_main as main
from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_factories
from selfcap_curate.device_tables import read_camera50_table, read_rotations
from dataforge.datasets.selfcap_raw import crop_shift_px
from selfcap_curate.factory import intrinsic_matrix, oak_centre, oak_stereo, read_quest_block
from selfcap_curate.oak_mount import fixed_rotation, gyro_mount, solve_oak
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import GyroFit, gyro_head_offsets, oak_quest_offset_ms, read_frozen_sync

pytestmark = pytest.mark.golden
SESSIONS = ("2025-11-24__027d8215", "2025-12-17__e0385cf5", "2025-12-30__65d00748")


@pytest.mark.parametrize("session", SESSIONS)
def test_oak_and_quest_match_served_session(session: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_catalog) -> None:
    records = sorted((ROOT / "v2/sync").glob(f"{session}__ep*.json"))
    reference = ROOT / "calib/sessions" / f"{session}.json"
    if not records or not os.access(reference, os.R_OK):
        pytest.skip(f"golden OAK session assets absent: {reference}")
    syncs = [read_frozen_sync(path) for path in records]

    for sync in syncs:
        raw = RAW / sync.date / sync.session / "episodes" / f"episode-{sync.episode[2:]}"
        base = SERVED / "base" / f"selfcap__{sync.key}.rrd"
        layer = SERVED / "calib-v2" / f"selfcap__{sync.key}.rrd"
        for asset in (
            base,
            layer,
            raw / "ego/calibration.json",
            raw / "quest/calibration.json",
            GOLDEN_REFS / "oak-sapiens2" / sync.episode_key.rrd_name,
        ):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"golden OAK asset absent: {asset}")

    table = read_rotations()
    with private_catalog(syncs, tmp_path) as server:
        dataset = server.client().get_dataset(DATASET_NAME)

        segments = {sync.key: open_segment(dataset, sync.episode_key.segment) for sync in syncs}
        factories = read_factories(segments[syncs[0].key])
        ego = factories["ego"]
        quest = factories["quest"]
        block = read_quest_block(segments[syncs[0].key])
        inputs = [read_session_episode(segments[sync.key], sync) for sync in syncs]
        gyro = gyro_mount(session, inputs)
        fixed = fixed_rotation(gyro, ego.device_id, table)
        oak = solve_oak(
            session,
            [replace(motion, oak=load_oak_detections(GOLDEN_REFS / "oak-sapiens2" / motion.sync.episode_key.rrd_name)) for motion in inputs],
            oak_centre(ego),
            ego.device_id,
            fixed,
        )
    trusted = [m for m in inputs if oak_quest_offset_ms(m.sync) is not None]
    if len(trusted) != len(inputs):
        good_gyro = gyro_mount(session, [m for m in trusted if gyro_head_offsets(m.sync) is not None])
        np.testing.assert_array_equal(gyro.R_head_from_imu, good_gyro.R_head_from_imu)
        assert gyro.max_episode_dev_deg == good_gyro.max_episode_dev_deg
        good_oak = solve_oak(
            session,
            [replace(m, oak=load_oak_detections(GOLDEN_REFS / "oak-sapiens2" / m.sync.episode_key.rrd_name)) for m in trusted],
            oak_centre(ego),
            ego.device_id,
            fixed,
        )
        np.testing.assert_array_equal(oak.questrig_T_rgb, good_oak.questrig_T_rgb)
        assert oak.median_px == good_oak.median_px
        print(
            "MEASURED_CLOCK_FIT",
            session,
            [m.sync.key for m in inputs if oak_quest_offset_ms(m.sync) is None],
            oak.median_px,
            gyro.max_episode_dev_deg,
            flush=True,
        )
    expected = json.loads(reference.read_text())
    old_gyro = from_json(GyroFit, (ROOT / "calib/gyro" / f"{session}.json").read_text())
    assert oak.ok == expected["oak"]["ok"] and gyro.ok == old_gyro.ok
    assert oak.median_px == pytest.approx(oak_median_reference(syncs, expected["oak"]["median_px"]), abs=0.5)
    assert oak.rotation_source == expected["oak"]["rotation_source"]
    assert_quest_block(block, expected["quest"])
    assert bool(oak_stereo(ego)) == expected["oak_factory"]["ok"]
    assert quest.device_id == expected["quest"]["device_id"]
    for name, transform in oak_stereo(ego).items():
        np.testing.assert_allclose(transform, expected["oak_factory"][name], rtol=0.0, atol=1e-14)
    rotation = float(
        np.degrees(Rotation.from_matrix(np.asarray(expected["oak"]["questrig_T_rgb"])[:3, :3].T @ oak.questrig_T_rgb[:3, :3]).magnitude())
    )
    translation = float(np.linalg.norm(np.asarray(expected["oak"]["questrig_T_rgb"])[:3, 3] - oak.questrig_T_rgb[:3, 3]) * 100.0)
    gyro_degrees = float(np.degrees(Rotation.from_matrix(old_gyro.R_head_from_imu.T @ gyro.R_head_from_imu).magnitude()))
    print("OAK_COMPARE", session, expected["oak"]["median_px"], oak.median_px, rotation, translation, gyro_degrees, flush=True)
    print("OAK_FLAGS", session, oak.ok, gyro.ok, block.mode, flush=True)


def test_rebuild_oak_device_rotations(tmp_path: Path) -> None:
    free = tuple((ROOT / "calib/oak_free").glob("*.json"))
    if not free:
        pytest.skip(f"frozen OAK free fits absent: {ROOT / 'calib/oak_free'}")
    target = tmp_path / "oak-device.json"
    main(Config(free, ROOT / "calib/gyro", target, "2026-10-26"))
    rebuilt, frozen = read_rotations(target), read_rotations()
    assert set(rebuilt.devices) == set(frozen.devices)
    for device, expected in frozen.devices.items():
        actual = rebuilt.devices[device]
        np.testing.assert_allclose(actual.rgb_R_imu, expected.rgb_R_imu, rtol=0.0, atol=1e-13)
        assert actual.source_sessions == expected.source_sessions
        assert actual.used_sessions == expected.used_sessions


def test_faulty_left_quest_resolves_from_a_healthy_catalog_segment(tmp_path: Path, private_catalog) -> None:

    session = "2025-12-09__021d2e8b"
    target = f"selfcap__{session}__ep001"
    borrowed = read_camera50_table().sessions[session]
    segments = (target, borrowed.healthy_segment)
    syncs = []

    for segment in segments:
        key = segment.removeprefix("selfcap__")
        sync_path = ROOT / "v2/sync" / f"{key}.json"
        if not sync_path.is_file():
            pytest.skip(f"faulty-left golden sync absent: {sync_path}")
        sync = read_frozen_sync(sync_path)
        syncs.append(sync)
        raw = RAW / sync.date / sync.session / "episodes" / f"episode-{sync.episode[2:]}"
        base, layer = SERVED / "base" / f"{segment}.rrd", SERVED / "calib-v2" / f"{segment}.rrd"
        for asset in (base, layer, raw / "quest/calibration.json"):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"faulty-left golden asset absent: {asset}")

    expected = json.loads((ROOT / "calib/sessions" / f"{session}.json").read_text())["quest"]
    with private_catalog(syncs, tmp_path) as server:
        dataset = server.client().get_dataset(DATASET_NAME)

        actual = read_quest_block(open_segment(dataset, target))
    assert actual.mode == expected["mode"] == "faulty-left-borrowed-cam50"
    assert bool(actual.pinholes) == expected["ok"]
    assert set(actual.pinholes) == set(expected["pinholes"])
    for slot, pinhole in actual.pinholes.items():
        frozen = expected["pinholes"][slot]
        assert pinhole.camera_id == frozen["camera_id"]
        assert crop_shift_px(pinhole) == frozen["cy_shift_px"]
        assert (pinhole.capture_resolution.width, pinhole.capture_resolution.height) == (frozen["width"], frozen["height"])
        np.testing.assert_allclose(
            intrinsic_matrix(pinhole), [[frozen["fx"], 0.0, frozen["cx"]], [0.0, frozen["fy"], frozen["cy"]], [0.0, 0.0, 1.0]], rtol=0.0, atol=1e-13
        )
    assert set(actual.rig_T_cam) == set(expected["rig_T_cam"])
    for slot, transform in actual.rig_T_cam.items():
        np.testing.assert_allclose(transform, expected["rig_T_cam"][slot], rtol=0.0, atol=1e-13)
    print(
        "FAULTY_QUEST",
        session,
        borrowed.healthy_segment,
        actual.mode,
        np.max(np.abs(actual.rig_T_cam["1"] - np.asarray(expected["rig_T_cam"]["1"]))),
        flush=True,
    )
