"""Local calibration owns triangulation even when a catalog has older poses."""

from dataclasses import replace

import numpy as np
import pytest
import rerun as rr
from frozen_v2 import fixture_sync, session_episode

import selfcap_curate.keypoints as keypoints
from selfcap_curate.apis import keypoints as api
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.catalog_io import DATASET_NAME, open_segment
from selfcap_curate.exo_calibration import Camera, PhoneCalibration, SessionCalibration
from selfcap_curate.factory import QuestBlock
from selfcap_curate.keypoints import Keypoints2D, project, read_cameras, triangulate
from selfcap_curate.records import CatalogHands, ExoFit, OakFit, SelfcapLayout


def calibration(path, *, episode="ep001", depth=4.0, missing=()):
    sync = fixture_sync(
        key=f"2025-12-30__65d00748__{episode}",
        date="2025-12-30",
        session="65d00748-0000-0000-0000-000000000000",
        episode=episode,
        offsets_ms={"quest_pose": 0.0, **{f"exo/{p}": 0.0 for p in "ABC"}},
        flags=[],
        exo_fit={p: ExoFit(0.0, "measured") for p in "ABC"},
    )
    hands = CatalogHands(np.empty(0), np.empty((0, 133, 3)), np.array([0.0, 1.0]), np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    phones = {
        p: PhoneCalibration(True, Camera(np.array([0.0, 0.0, 0.0, -float(i), 0.0, depth, 700.0, 960.0, 540.0, 0.0, 0.0])), 2.0)
        for i, p in enumerate("ABC")
        if p not in missing
    }
    state = CalibSession("synthetic", SessionCalibration("synthetic", phones), OakFit(), QuestBlock({}, {}, "unknown-layout"), {})
    write_calib_layer(path, state, replace(session_episode(sync, hands, np.empty(0), np.empty((0, 3))), layout=SelfcapLayout(3, 4, tuple("ABC"))))
    return sync


@pytest.mark.integration
def test_cli_uses_local_poses_instead_of_equal_provenance_catalog(tmp_path, monkeypatch, synthetic_base, private_server):
    local = tmp_path / "local"
    local.mkdir()
    key = "2025-12-30__65d00748__ep001"
    base = tmp_path / "base.rrd"
    synthetic_base(base, segment=f"selfcap__{key}", phones=tuple("ABC"), oak=False, quest=False)
    sync = calibration(tmp_path / "catalog.rrd", depth=4.0)
    calibration(local / f"selfcap__{key}.rrd", depth=9.0)

    class Checked(Exception):
        pass

    def check(observations, cameras, anchors):
        assert set(cameras) == set("ABC")
        assert all(camera.world_T_cam[2, 3] == -9.0 for camera in cameras.values())
        raise Checked

    monkeypatch.setattr(api, "read_time_sync", lambda *_args: sync)
    monkeypatch.setattr(api, "keypoint_models", lambda: (None, None))
    monkeypatch.setattr(api, "detect_keypoints", lambda *_args, **_kwargs: Keypoints2D(np.array([0.0]), {}, 3.0))
    monkeypatch.setattr(api, "read_anchors", lambda *_args, **_kwargs: np.zeros((1, 3, 3)))
    monkeypatch.setattr(api, "triangulate_episode", check)
    with private_server({DATASET_NAME: [base]}) as server:
        server.client().get_dataset(DATASET_NAME).register([(tmp_path / "catalog.rrd").as_uri()], layer_name="calib").wait()
        with pytest.raises(Checked):
            api.main(api.Config(local, tmp_path / "out", rate=3.0, segment_ids=(key,), catalog_url=server.url()))


@pytest.mark.integration
def test_missing_phone_is_omitted_and_remaining_cameras_triangulate(tmp_path, private_server):
    calibration(tmp_path / "complete.rrd")
    sync = calibration(tmp_path / "partial.rrd", episode="ep002", missing=("B",))
    with private_server({DATASET_NAME: [tmp_path / "complete.rrd", tmp_path / "partial.rrd"]}) as server:
        cameras = read_cameras(server.client().get_dataset(DATASET_NAME), f"selfcap__{sync.key}")
    assert set(cameras) == {"A", "C"}
    point = np.array([[0.2, 0.3, 2.0]])
    uv = np.stack([project(camera, point)[:, :2] for camera in cameras.values()])
    result = triangulate(list(cameras.values()), uv, np.ones((2, 1)))
    np.testing.assert_allclose(result.xyz, point, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("flag", ["quest_unsynced_kept_as_cut", "quest_pose_unsynced"])
def test_flagged_quest_pose_keeps_relative_phone_anchors(tmp_path, monkeypatch, flag, private_server, synthetic_base):
    sync = replace(fixture_sync(), offsets_ms={"quest_pose": 0.0, "exo/A": -48.0}, flags=[flag])
    times = np.arange(10, dtype=np.float64) * 0.02
    source = CatalogHands(times, np.ones((10, 133, 3)) + times[:, None, None] * 0.1, times, np.ones((10, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (10, 1)))

    monkeypatch.setattr(keypoints, "read_catalog_hands", lambda *_args, **_kwargs: source)
    base = synthetic_base(tmp_path / "base.rrd", segment=sync.episode_key.segment, phones=("A", "B", "C"))
    with private_server({DATASET_NAME: [base]}) as server:
        anchors = keypoints.read_anchors(
            open_segment(server.client().get_dataset(DATASET_NAME), sync.episode_key.segment), np.array([0.06, 0.08]), sync
        )
    assert np.isfinite(anchors).all()


@pytest.mark.integration
@pytest.mark.parametrize("reason", ["calibrated", "synced"])
@pytest.mark.parametrize("all_skipped", [False, True])
def test_cli_skips_unprocessable_episodes_before_models_and_runs_others(
    tmp_path, monkeypatch, capsys, reason, all_skipped, synthetic_base, private_server
):
    local = tmp_path / "local"
    local.mkdir()
    keys = tuple(f"2025-12-30__65d00748__ep{i:03d}" for i in (1, 2))
    syncs = {}
    for key in keys:
        sync = calibration(
            local / f"selfcap__{key}.rrd", episode=key[-5:], missing=("B", "C") if reason == "calibrated" and (all_skipped or key == keys[1]) else ()
        )
        sync = replace(sync, flags=["quest_unsynced_kept_as_cut"])
        if (all_skipped or key == keys[1]) and reason == "synced":
            sync = replace(sync, offsets_ms={"quest_pose": 0.0, "exo/A": 0.0, "exo/B": None, "exo/C": None})
        syncs[f"selfcap__{key}"] = sync
    base_files = []
    for key in keys:
        base = tmp_path / f"{key}.base.rrd"
        synthetic_base(base, segment=f"selfcap__{key}", phones=tuple("ABC"), oak=False, quest=False)
        base_files.append(base)
    servers = []
    real_server = rr.server.Server

    def session_server(*args, **kwargs):
        servers.append(kwargs["datasets"][DATASET_NAME])
        return real_server(*args, **kwargs)

    monkeypatch.setattr(api, "Server", session_server, raising=False)
    loaded = []

    def models():
        output = capsys.readouterr().out
        assert f"KEYPOINTS_SKIP {keys[1]}" in output
        loaded.append(output)
        return None, None

    monkeypatch.setattr(api, "read_time_sync", lambda dataset, segment: syncs[segment])
    monkeypatch.setattr(api, "keypoint_models", models)
    monkeypatch.setattr(api, "detect_keypoints", lambda *_args, **_kwargs: Keypoints2D(np.array([0.0]), {}, 3.0))
    monkeypatch.setattr(api, "read_anchors", lambda *_args, **_kwargs: np.zeros((1, 3, 3)))
    with private_server({DATASET_NAME: base_files}) as server:
        api.main(api.Config(local, tmp_path / "out", rate=3.0, segment_ids=keys, catalog_url=server.url()))
    assert len(servers) == 1
    assert len(servers[0]) == len(keys)
    assert len(loaded) == (0 if all_skipped else 1)
    assert not (tmp_path / "out" / f"selfcap__{keys[1]}.rrd").exists()
    assert (tmp_path / "out" / f"selfcap__{keys[0]}.rrd").exists() == (not all_skipped)
    output = "\n".join(loaded) + capsys.readouterr().out
    assert "fewer than two calibrated and synced phones" in output


@pytest.mark.integration
def test_local_phone_identity_survives_changed_base_names(tmp_path, private_server):
    sync = calibration(tmp_path / "calib.rrd")
    with private_server({DATASET_NAME: [tmp_path / "calib.rrd"]}) as server:
        cameras = read_cameras(server.client().get_dataset(DATASET_NAME), f"selfcap__{sync.key}")
    base_names = ("A", "C", "D")
    matched = {phone: cameras[phone] for phone in base_names if phone in cameras}
    assert set(matched) == {"A", "C"}
    assert matched["C"].world_T_cam[0, 3] == 2.0
    assert matched["C"].entity == "/world/rig_02/cam_00"


@pytest.mark.integration
def test_cli_skips_calibration_for_different_base_before_gpu(tmp_path, monkeypatch, capsys, synthetic_base, private_server):
    local = tmp_path / "local"
    local.mkdir()
    key = "2025-12-30__65d00748__ep001"
    calibration(local / f"selfcap__{key}.rrd")
    base = tmp_path / "base.rrd"
    synthetic_base(base, segment=f"selfcap__{key}", phones=tuple("ACD"), oak=False, quest=False)

    def forbidden(*_args):
        pytest.fail("mismatched calibration reached GPU or sync reading")

    monkeypatch.setattr(api, "keypoint_models", forbidden)
    monkeypatch.setattr(api, "read_time_sync", forbidden)
    with private_server({DATASET_NAME: [base]}) as server:
        api.main(api.Config(local, tmp_path / "out", rate=2.0, segment_ids=(key,), catalog_url=server.url()))
    assert f"KEYPOINTS_SKIP {key} local calib phones absent from base: B" in capsys.readouterr().out
    assert not (tmp_path / "out" / f"selfcap__{key}.rrd").exists()
