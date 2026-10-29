"""Calibration recovers a camera in Quest world, with held device intrinsics."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from frozen_v2 import fixture_sync, session_episode
from rerun.catalog import DatasetEntry
from scipy.spatial.transform import Rotation
from serde.json import from_json, to_json

import selfcap_curate.exo_calibration as module
from selfcap_curate import apis as api_inputs
from selfcap_curate.apis import pool as api
from selfcap_curate.apis import pool as apis_pool
from selfcap_curate.apis.pool import PoolConfig, pool_main
from selfcap_curate.device_tables import GRID, FocalProfile, pool, read_table
from selfcap_curate.exo_calibration import BodyObservation, BundleResult, Camera, SessionCalibration, bundle, calibrate, profile_session, solve_camera
from selfcap_curate.quest_fit import Correspondences
from selfcap_curate.records import ExoFit, HandTrack, PhoneDetections, write_sync


def test_fixed_intrinsics_recovers_known_camera_and_world_convention() -> None:
    rng = np.random.default_rng(42)
    world = rng.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0], (120, 3))
    camera = Camera(np.array([0.1, -0.2, 0.03, 0.2, -0.1, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    image = camera.project(world)[:, :2]
    result = solve_camera(Correspondences(world, image, np.ones(len(world))), 700.0)
    assert result is not None
    assert np.median(result.residuals(Correspondences(world, image, np.ones(len(world))))) < 1e-5
    assert result.parameters[6:].tolist() == [700.0, 960.0, 540.0, 0.0, 0.0]
    np.testing.assert_allclose(result.world_T_cam[:3, :3], Rotation.from_rotvec(camera.parameters[:3]).as_matrix().T, atol=1e-6)
    np.testing.assert_allclose(result.world_T_cam[:3, 3], -result.world_T_cam[:3, :3] @ camera.parameters[3:6], atol=1e-6)


def test_profile_pooling_uses_normalized_cost_curves_and_keeps_provenance() -> None:

    grid = np.array([450.0, 550.0, 650.0, 750.0, 850.0, 1000.0, 1200.0, 1450.0, 1750.0])
    curve = tuple(float(x) for x in (grid - 700.0) ** 2 + 100.0)
    result = pool([FocalProfile("session-b", {"phone": curve}), FocalProfile("session-a", {"phone": curve})], "2026-10-26")
    assert result.devices["phone"].f == pytest.approx(700.0, abs=1e-6)
    assert result.devices["phone"].source_sessions == ("session-a", "session-b")
    assert result.devices["phone"].date == "2026-10-26"
    with pytest.raises(ValueError, match="nine"):
        FocalProfile("bad", {"phone": (1.0, 2.0)})


def synthetic_episode(index: int, camera: Camera, corrupt: bool = False):

    rng = np.random.default_rng(12)
    times = np.arange(0.0, 4.0, 0.01)
    joints = rng.normal(scale=0.2, size=(21, 3))[None] + np.stack([np.sin(times), np.cos(times * 0.6), np.sin(times * 1.3)], axis=1)[:, None] * 0.3
    tracks = {
        side: HandTrack(times, joints + delta, np.ones(len(times), dtype=bool), np.ones(len(times)))
        for side, delta in (("left", 0.0), ("right", 0.1))
    }
    indices = np.arange(10, 390, 8)
    xy = np.full((len(indices), 133, 2), np.nan)
    scores = np.zeros((len(indices), 133))
    for side, base in (("left", 91), ("right", 112)):
        xy[:, base : base + 21] = camera.project(tracks[side].joints[indices].reshape(-1, 3))[:, :2].reshape(-1, 21, 2)
        scores[:, base : base + 21] = 1.0
    if corrupt:
        xy[:, 91:] += rng.normal(scale=80.0, size=xy[:, 91:].shape)
    sync = fixture_sync()
    sync = replace(sync, key=f"2025-11-24__027d8215__ep{index:03d}", episode=f"ep{index:03d}", exo_fit={"phone": ExoFit(0.0, "synthetic")})
    return session_episode(sync, detections={"phone": PhoneDetections(times[indices], xy, scores)}, quest=tracks)


def test_session_re_solves_a_moved_phone_and_marks_unusable_episode_false() -> None:

    camera = Camera(np.array([0.1, -0.1, 0.02, 0.0, 0.0, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    moved = Camera(camera.parameters + np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    episodes = [synthetic_episode(1, camera), synthetic_episode(2, camera), synthetic_episode(3, moved), synthetic_episode(4, camera, corrupt=True)]
    result = calibrate("synthetic", episodes, {"phone": 700.0})
    checks = result.phones["phone"].episodes
    assert checks["ep001"].ok
    assert checks["ep003"].ok and checks["ep003"].scope == "episode"
    assert checks["ep003"].median_px < 8.0
    assert not checks["ep004"].ok and checks["ep004"].scope == "flagged"


def test_bundle_recovers_two_poses_with_body_points() -> None:

    rng = np.random.default_rng(8)
    world = rng.normal(size=(90, 3)) * 0.3
    a = Camera(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    b = Camera(a.parameters + np.array([0.0, 0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    cameras = {"a": a, "b": b}
    hands = {p: [("ep001", Correspondences(world, c.project(world)[:, :2], np.ones(len(world))))] for p, c in cameras.items()}
    body = [{p: BodyObservation(c.project(world[:13])[:, :2], np.ones(13)) for p, c in cameras.items()}]
    seeded = {p: Camera(c.parameters + np.array([0.01, 0.01, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) for p, c in cameras.items()}
    result = bundle({p: items[0][1] for p, items in hands.items()}, body, seeded)
    assert result.cost < 1e-5
    for phone, camera in result.cameras.items():
        np.testing.assert_allclose(camera.parameters[6:], cameras[phone].parameters[6:], rtol=1e-12)
        np.testing.assert_allclose(camera.world_T_cam, cameras[phone].world_T_cam, atol=1e-5)


def test_pool_cli_round_trips_the_packaged_device_schema(tmp_path) -> None:

    profile: Path = tmp_path / "profile.json"
    profile.write_text(to_json(FocalProfile("session", {"phone": (9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0)})))
    output: Path = tmp_path / "table.json"
    pool_main(PoolConfig((profile,), output, "2026-10-26"))
    result = read_table(output)
    assert result.devices["phone"].f == pytest.approx(750.0)
    assert result.devices["phone"].source_sessions == ("session",)


def test_failed_pass_one_is_explicit_and_successful_result_serializes() -> None:

    camera = Camera(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    episode = synthetic_episode(1, camera)
    detected = episode.detections["phone"]
    sparse = PhoneDetections(detected.times_s[:1], detected.xy[:1], detected.scores[:1])
    failed = calibrate("synthetic", [replace(episode, detections={"phone": sparse})], {"phone": 700.0})
    assert not failed.phones["phone"].ok
    assert failed.phones["phone"].why == "fewer than 60 hand points"
    success = calibrate("synthetic", [episode], {"phone": 700.0})
    restored = from_json(SessionCalibration, to_json(success))
    assert restored.phones["phone"].p90_px < 1e-4
    assert restored.phones["phone"].camera.parameters.dtype == np.float64


def test_profile_reports_insufficient_phones() -> None:
    profile = profile_session("empty", [])
    assert profile.curves == {} and profile.error == "fewer than two phones"


def test_profile_producer_keeps_grid_other_focals_and_iteration_budget(monkeypatch) -> None:

    data = Correspondences(np.zeros((60, 3)), np.zeros((60, 2)), np.ones(60))
    monkeypatch.setattr(module, "gather", lambda _: ({"a": [("ep001", data)], "b": [("ep001", data)]}, []))
    initialized, budgets = [], []

    def solve(data, focal):
        initialized.append(focal)
        return Camera(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, focal, 960.0, 540.0, 0.0, 0.0]))

    def bundle(hands, body, cameras, *, max_nfev):
        budgets.append(max_nfev)
        return BundleResult(cameras, float(sum(c.parameters[6] for c in cameras.values())))

    monkeypatch.setattr(module, "solve_camera", solve)
    monkeypatch.setattr(module, "bundle", bundle)
    result = module.profile_session("session", [], {"a": 800.0, "b": 900.0})
    assert result.curves == {"a": tuple(f + 900.0 for f in GRID), "b": tuple(800.0 + f for f in GRID)}
    assert initialized == [v for f in GRID for v in (f, 900.0)] + [v for f in GRID for v in (800.0, f)]
    assert budgets == [40] * 18


@pytest.mark.parametrize("command", ["profile", "pool", "rotation"])
def test_profile_and_pool_commands_preserve_existing_output(tmp_path, command) -> None:

    target = tmp_path / "existing.json"
    target.write_text("keep")
    function, config = {
        "profile": (apis_pool.main, apis_pool.Config(target, segment_ids=(fixture_sync().key,))),
        "pool": (apis_pool.pool_main, apis_pool.PoolConfig((), target)),
        "rotation": (apis_pool.rotation_main, apis_pool.RotationConfig((), tmp_path / "gyro", target)),
    }[command]
    with pytest.raises(FileExistsError):
        function(config)
    assert target.read_text() == "keep"


def test_focal_profile_uses_selected_catalog_sync_not_local_tau(tmp_path, monkeypatch) -> None:

    embedded = fixture_sync()
    stale = replace(embedded, exo_fit={p: replace(fit, tau_phone_to_pose_ms=123456.0) for p, fit in embedded.exo_fit.items()})
    hands = tmp_path / "hands"
    hands.mkdir()
    write_sync(hands / f"{embedded.key}.json", stale)
    keys = tmp_path / "keys.txt"
    keys.write_text(f"selfcap__{embedded.key}\n")
    selected = []

    def read(_dataset, segment):
        selected.append(segment)
        return embedded

    monkeypatch.setattr(api_inputs, "CatalogClient", lambda *_: SimpleNamespace(get_dataset=lambda _: Mock(spec=DatasetEntry)))
    monkeypatch.setattr(api, "open_segment", lambda *_: None)
    monkeypatch.setattr(api, "read_time_sync", read)
    monkeypatch.setattr(api, "read_session_episode", lambda _dataset, sync: session_episode(sync))
    monkeypatch.setattr(
        api,
        "read_phone_detections",
        lambda *_, **_kw: {p: PhoneDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))) for p in embedded.exo_fit},
    )

    def solve(_session, episodes):
        assert episodes[0].sync is embedded
        assert episodes[0].sync.exo_fit != stale.exo_fit
        raise RuntimeError("canonical tau reached profile")

    monkeypatch.setattr(api, "profile_session", solve)
    with pytest.raises(RuntimeError, match="canonical tau reached profile"):
        api.main(api.Config(tmp_path / "profile.json", key_file=keys))
    assert selected == [f"selfcap__{embedded.key}"]


def test_focal_profile_refuses_missing_sapiens2_layer(tmp_path, monkeypatch):
    sync = fixture_sync()
    monkeypatch.setattr(api_inputs, "CatalogClient", lambda *_: SimpleNamespace(get_dataset=lambda _: Mock(spec=DatasetEntry)))
    monkeypatch.setattr(api, "open_segment", lambda *_: None)
    monkeypatch.setattr(api, "read_time_sync", lambda *_: sync)
    monkeypatch.setattr(api, "read_session_episode", lambda *_: session_episode(sync))
    monkeypatch.setattr(api, "read_phone_detections", lambda *_, **_kw: {})
    with pytest.raises(ValueError, match="Sapiens2 layer missing"):
        api.main(api.Config(tmp_path / "profile.json", segment_ids=(sync.key,)))
    assert not (tmp_path / "profile.json").exists()
