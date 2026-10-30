"""Layer writing owns only calibration nodes and one property group."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from frozen_v2 import fixture_sync, session_episode
from rerun.catalog import DatasetEntry
from rerun.chunk import RrdReader

from selfcap_curate import apis as api_inputs
from selfcap_curate.apis import calibrate as api
from selfcap_curate.apis.calibrate import Config, main
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.exo_calibration import Camera, EpisodeCalibration, PhoneCalibration, SessionCalibration
from selfcap_curate.exo_refine import Refinement
from selfcap_curate.factory import QuestBlock
from selfcap_curate.records import CatalogHands, ExoFit, GyroFit, OakDetections, OakFit, PhoneDetections, SelfcapLayout, write_sync


def test_writer_uses_episode_override_and_aligned_oak_pose_without_hands(tmp_path: Path) -> None:
    sync = fixture_sync()
    sync = replace(sync, offsets_ms={"quest_pose": 1000.0}, exo_fit={"phone": ExoFit(None, "test")})
    raw = CatalogHands(
        np.empty(0), np.empty((0, 133, 3)), np.array([0.0, 1.0]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    )
    motion = session_episode(sync, raw, np.empty(0), np.empty((0, 3)))
    override = np.eye(4)
    override[0, 3] = 0.2
    camera = Camera(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    phone = PhoneCalibration(True, camera, 2.0, episodes={"ep002": EpisodeCalibration(100, 3.0, False, "flagged", override)})
    state = CalibSession(
        "synthetic",
        SessionCalibration("synthetic", {"phone": phone}),
        OakFit("synthetic", True, "oak", 4.0, np.eye(4), 1.0),
        QuestBlock({}, {}, "unknown-layout"),
        {},
    )
    path = tmp_path / "calib.rrd"
    write_calib_layer(path, state, replace(motion, layout=SelfcapLayout(1, 2, ("phone",))))
    chunks = list(RrdReader(path).stream())
    paths = {str(c.entity_path) for c in chunks}
    assert not any(p.startswith(("/world/gt", "/world/pred")) for p in paths)
    assert {p for p in paths if p.startswith("/__properties")} == {"/__properties/calib"}
    transform = next(
        c.to_record_batch()
        for c in chunks
        if str(c.entity_path) == "/world/rig_00/cam_00" and "Transform3D:translation" in c.to_record_batch().schema.names
    )
    np.testing.assert_allclose(transform.column("Transform3D:translation")[0].as_py()[0], [0.2, 0.0, 0.0])
    pose = next(c.to_record_batch() for c in chunks if str(c.entity_path) == "/world/rig_01" and not c.is_static)
    assert pose.column("video_time").cast("int64").to_pylist() == [1_000_000_000, 2_000_000_000]
    prop = next(c.to_record_batch() for c in chunks if str(c.entity_path) == "/__properties/calib")
    assert prop.column("exo_flagged")[0].as_py() == [1]


def test_calibrate_cli_rejects_mixed_sessions_before_catalog_access(tmp_path: Path) -> None:

    config = Config(tmp_path / "output", ("selfcap__2025-11-24__027d8215__ep001", "selfcap__2025-12-30__65d00748__ep001"))
    with pytest.raises(ValueError, match="exactly one session"):
        main(config)
    assert not config.output_dir.exists()


def test_exo_refinement_preserves_quest_and_oak_transform_bytes(tmp_path: Path) -> None:
    phones = ('A', 'B', 'C', 'D')
    sync = replace(fixture_sync(), offsets_ms={"quest_pose": 0.0}, exo_fit={p: ExoFit(None, "test") for p in phones})
    raw = CatalogHands(np.empty(0), np.empty((0, 133, 3)), np.array([0.0, 1.0]), np.array([[0.,0.,0.],[1.,2.,3.]]),
                       np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    motion = replace(session_episode(sync, raw, np.empty(0), np.empty((0, 3))), layout=SelfcapLayout(4, 5, phones))
    camera = Camera(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 900.0, 960.0, 540.0, 0.0, 0.0]))
    right = np.eye(4)
    right[0, 3] = 0.06
    refinement = Refinement((camera,) * 4, np.array([0.01, -0.03, 0.02]), "label-hash", "fixed")
    state = CalibSession("synthetic", SessionCalibration("synthetic", {p: PhoneCalibration(True, camera) for p in phones}),
                         OakFit("synthetic", True, "oak", 4.0, np.eye(4), 1.0), QuestBlock({}, {"1": right}, "healthy"),
                         {"rgb_T_left": np.eye(4), "rgb_T_right": right})
    paths = [tmp_path / "before.rrd", tmp_path / "after.rrd"]
    write_calib_layer(paths[0], state, motion)
    write_calib_layer(paths[1], replace(state, exo_refinements={sync.episode_key.segment: refinement}), motion)

    def transform_bytes(path: Path) -> list[tuple[str, bytes]]:
        result = []
        for chunk in RrdReader(path).stream():
            entity = str(chunk.entity_path)
            if not entity.startswith(("/world/rig_04", "/world/rig_05")):
                continue
            batch = chunk.to_record_batch()
            columns = [name for name in batch.schema.names if name.startswith(("Transform3D:", "Pinhole:")) or name == "video_time"]
            if any(name.startswith(("Transform3D:", "Pinhole:")) for name in columns):
                result.append((entity, batch.select(columns).serialize().to_pybytes()))
        return sorted(result)

    assert transform_bytes(paths[0])
    assert transform_bytes(paths[0]) == transform_bytes(paths[1])


def test_calibrate_cli_refuses_existing_output_before_solving(tmp_path: Path) -> None:

    output = tmp_path / "output"
    output.mkdir()
    existing = output / "2025-11-24__027d8215.json"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        main(Config(output, ("selfcap__2025-11-24__027d8215__ep001",)))
    assert existing.read_text() == "keep"


def test_failed_links_keep_oak_and_quest_on_their_physical_rigs(tmp_path: Path) -> None:
    sync = fixture_sync()
    sync = replace(sync, offsets_ms=None, exo_fit={name: ExoFit(None, "unsynced") for name in ("A", "B", "C")})
    raw = CatalogHands(np.empty(0), np.empty((0, 133, 3)), np.array([0.0, 1.0]), np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    state = CalibSession(
        "failed",
        SessionCalibration("failed", {}),
        OakFit(),
        QuestBlock({}, {"0": np.eye(4)}, "healthy"),
        {"rgb_T_left": np.eye(4), "rgb_T_right": np.eye(4)},
    )
    output = tmp_path / "failed.rrd"
    write_calib_layer(output, state, replace(session_episode(sync, raw, np.empty(0), np.empty((0, 3))), layout=SelfcapLayout(3, 4, ("A", "B", "C"))))
    entities = {str(chunk.entity_path) for chunk in RrdReader(output).stream()}
    assert "/world/rig_03/cam_01" in entities
    assert "/world/rig_03/cam_02" in entities
    assert "/world/rig_04/cam_00" in entities
    assert "/world/rig_00/cam_01" not in entities


@pytest.fixture
def layer_case():
    sync = fixture_sync(flags=[], offsets_ms={"quest_pose": 0.0, "oak_imu": 0.0}, exo_fit={p: ExoFit(None, "test") for p in ("A", "B", "C")})
    hands = CatalogHands(np.empty(0), np.empty((0, 133, 3)), np.array([0.0, 1.0]), np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    camera = Camera(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 700.0, 960.0, 540.0, 0.0, 0.0]))
    phone = PhoneCalibration(True, camera, 3.0)
    state = CalibSession(
        "synthetic",
        SessionCalibration("synthetic", {"A": phone}),
        OakFit("synthetic", True, "oak", 4.0, np.eye(4), 1.0),
        QuestBlock({}, {}, "unknown-layout"),
        {},
    )
    return state, session_episode(sync, hands, np.empty(0), np.empty((0, 3)))


@pytest.mark.parametrize("flag", [None, "quest_unsynced_kept_as_cut", "quest_pose_unsynced"])
def test_unsynced_quest_has_no_oak_track(tmp_path: Path, layer_case, flag) -> None:
    state, motion = layer_case
    motion = replace(motion, sync=replace(motion.sync, flags=[] if flag is None else [flag]))
    path = tmp_path / "layer.rrd"
    write_calib_layer(path, state, replace(motion, layout=SelfcapLayout(3, 4, ("A", "B", "C"))))
    chunks = list(RrdReader(path).stream())
    track = [c for c in chunks if str(c.entity_path) == "/world/rig_03" and not c.is_static]
    prop = next(c.to_record_batch() for c in chunks if str(c.entity_path) == "/__properties/calib")
    assert bool(track) == (flag is None)
    assert prop.column("oak_posed")[0].as_py() == [flag is None]


@pytest.mark.parametrize("episode_median,session_median,expected", [(None, None, None), (None, 3.0, None), (0.0, 3.0, 0.0), (2.0, 3.0, 2.0)])
def test_layer_never_invents_an_unknown_hand_residual(tmp_path, layer_case, episode_median, session_median, expected) -> None:
    state, motion = layer_case
    phone = replace(
        state.exo.phones["A"],
        median_px=session_median,
        episodes={motion.sync.episode: EpisodeCalibration(0, episode_median, False, "session-few-points")},
    )
    state = replace(state, exo=replace(state.exo, phones={"A": phone}))
    path = tmp_path / "layer.rrd"
    write_calib_layer(path, state, replace(motion, layout=SelfcapLayout(3, 4, ("A", "B", "C"))))
    batches = [c.to_record_batch() for c in RrdReader(path).stream() if str(c.entity_path) == "/world/rig_00/cam_00"]
    residuals = [b.column("calib_hand_median_px")[0].as_py() for b in batches if "calib_hand_median_px" in b.schema.names]
    assert residuals == ([] if expected is None else [[expected]])


def test_layer_refuses_a_sync_phone_set_different_from_base(tmp_path, layer_case) -> None:

    state, motion = layer_case
    motion = replace(motion, sync=replace(motion.sync, exo_fit={p: ExoFit(None, "test") for p in ("A", "B", "C", "D")}))
    layout = SelfcapLayout(3, 4, ("A", "B", "C"))
    path = tmp_path / "mismatch.rrd"
    with pytest.raises(ValueError, match="phone"):
        write_calib_layer(path, state, replace(motion, layout=layout))
    assert not path.exists()


def test_calibrate_uses_catalog_tau_despite_stale_local_json(calibration_call, tmp_path, monkeypatch) -> None:

    api, config, _ = calibration_call
    embedded = fixture_sync()
    stale = replace(embedded, exo_fit={p: replace(fit, tau_phone_to_pose_ms=123456.0) for p, fit in embedded.exo_fit.items()})
    write_sync(tmp_path / f"{embedded.key}.json", stale)
    monkeypatch.setattr(api.catalog_io, "read_time_sync", lambda *_args: embedded)

    def solve(_session, episodes, _focals):
        assert episodes[0].sync is embedded
        assert episodes[0].sync.exo_fit != stale.exo_fit
        raise RuntimeError("canonical tau reached solve")

    monkeypatch.setattr(api, "calibrate", solve)
    with pytest.raises(RuntimeError, match="canonical tau reached solve"):
        api.main(config)


@pytest.fixture
def calibration_call(tmp_path, layer_case, monkeypatch):

    state, motion = layer_case
    monkeypatch.setattr(api_inputs, "CatalogClient", lambda *_args: SimpleNamespace(get_dataset=lambda _: Mock(spec=DatasetEntry)))
    monkeypatch.setattr(
        api.catalog_io, "read_time_sync", lambda _dataset, segment: replace(motion.sync, key=segment.removeprefix("selfcap__"), episode=segment[-5:])
    )
    monkeypatch.setattr(
        api.catalog_io, "open_segment", lambda _dataset, segment: SimpleNamespace(id=segment, layout=SelfcapLayout(3, 4, ("A", "B", "C")), videos=None)
    )
    monkeypatch.setattr(api, "read_session_episode", lambda _dataset, sync: replace(motion, sync=sync))
    monkeypatch.setattr(
        api.catalog_io, "read_factories", lambda *_args, **_kwargs: {"ego": SimpleNamespace(device_id="oak", centre=None), "quest": None}
    )
    monkeypatch.setattr(api, "read_quest_block", lambda *_args, **_kwargs: state.quest)
    monkeypatch.setattr(
        api.catalog_io,
        "read_phone_detections",
        lambda *_args, **_kwargs: {p: PhoneDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))) for p in motion.layout.phones},
    )
    monkeypatch.setattr(api, "pose_models", lambda: (None, None))
    monkeypatch.setattr(api, "detect_oak_hands_gpu", lambda *_args, **_kwargs: OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))))
    monkeypatch.setattr(api, "calibrate", lambda *_args: state.exo)
    monkeypatch.setattr(api, "gyro_mount", lambda *_args: GyroFit("test", False))
    monkeypatch.setattr(api, "solve_oak", lambda *_args, **_kwargs: state.oak)
    monkeypatch.setattr(api, "oak_stereo", lambda *_args: state.stereo)
    monkeypatch.setattr(api, "oak_centre", lambda *_args: None)
    config = api.Config(tmp_path / "output", (f"selfcap__{motion.sync.key}",), oak_method="sapiens")
    return api, config, motion.sync.key.rsplit("__", 1)[0]


def test_calibrate_egohand_uses_default_path_and_writes_named_layer(calibration_call, monkeypatch):
    api, config, _ = calibration_call
    config = replace(config, oak_method="egohand")
    calls = []
    monkeypatch.setattr(api, "fixed_rotation", lambda *_args: np.eye(3))
    monkeypatch.setattr(api, "release_cuda_memory", lambda **kwargs: None)
    monkeypatch.setattr(api, "read_ego_cameras", lambda *_args: [])
    monkeypatch.setattr(api, "read_catalog_videos", lambda *_args: [None])
    monkeypatch.setattr(api, "detect_oak_egohand", lambda *args, **kwargs: calls.append("egohand") or
                        OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))))
    monkeypatch.setattr(api, "detect_oak_hands_gpu", lambda *args, **kwargs: pytest.fail("legacy OAK model must not run"))
    monkeypatch.setattr(api, "write_oak_egohand_layer", lambda path, *args: calls.append(path.parent.name))
    result = api.main(config)
    assert result.oak.rotation_source == "gyro"
    assert result.oak.rotation_reason is None
    assert calls == ["egohand", "egohand"]
    assert api.Config(config.output_dir).oak_method == "egohand"


@pytest.mark.parametrize("failure", ["fit", "later_layer", "json"])
def test_failed_calibration_is_unpublished_and_retryable(calibration_call, monkeypatch, failure) -> None:
    api, config, session = calibration_call
    key = config.segment_ids[0]
    config = replace(config, segment_ids=(key, key[:-3] + "003"))
    target = config.output_dir / f"{session}.json"
    failed = False
    written = []

    def layer(path, *_args):
        nonlocal failed
        assert not config.output_dir.exists()
        assert path.parent.parent == config.output_dir.parent
        path.write_bytes(b"complete layer")
        written.append(path)
        if failure == "later_layer" and len(written) == 2 and not failed:
            failed = True
            raise OSError("injected failure")

    original_fit = api.calibrate

    def fit(*args):
        nonlocal failed
        if failure == "fit" and not failed:
            failed = True
            raise OSError("injected failure")
        return original_fit(*args)

    original = Path.write_text

    def json(path, text, **kwargs):
        nonlocal failed
        assert not config.output_dir.exists()
        original(path, text, **kwargs)
        if failure == "json" and not failed:
            failed = True
            raise OSError("injected failure")

    monkeypatch.setattr(api, "write_calib_layer", layer)
    monkeypatch.setattr(api, "calibrate", fit)
    monkeypatch.setattr(Path, "write_text", json)
    with pytest.raises(OSError, match="injected failure"):
        api.main(config)
    assert not config.output_dir.exists()
    assert not list(config.output_dir.parent.glob(".output.*"))
    api.main(config)
    assert target.is_file()
    assert len(list(config.output_dir.glob("*.rrd"))) == 2
    assert not list(config.output_dir.parent.glob(".output.*"))


def test_calibrate_refuses_an_existing_empty_output(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        main(Config(output, (fixture_sync().key,)))
    assert list(output.iterdir()) == []


def test_layer_binds_each_written_camera_to_its_phone(tmp_path, layer_case) -> None:
    state, motion = layer_case
    state = replace(state, exo=replace(state.exo, phones={name: state.exo.phones["A"] for name in ("C", "A", "B")}))
    path = tmp_path / "named.rrd"
    write_calib_layer(path, state, replace(motion, layout=SelfcapLayout(3, 4, ("A", "B", "C"))))
    batches = [(str(c.entity_path), c.to_record_batch()) for c in RrdReader(path).stream()]
    named = {entity: b for entity, b in batches if "calib_phone" in b.schema.names}
    assert set(named) == {"/world/rig_00/cam_00", "/world/rig_01/cam_00", "/world/rig_02/cam_00"}
    for rig, phone in enumerate(("A", "B", "C")):
        batch = named[f"/world/rig_{rig:02d}/cam_00"]
        assert batch.column("calib_phone")[0].as_py() == [phone]
        assert batch.column("calib_status")[0].as_py() == ["ok"]


def test_calibrate_reuses_catalog_phones_without_phone_inference(calibration_call, monkeypatch):
    api, config, _ = calibration_call
    monkeypatch.setattr(api, "detect_episode", lambda *_args, **_kwargs: pytest.fail("stored phones must be reused"))
    api.main(config)


def test_calibrate_missing_layer_falls_back_to_gpu(calibration_call, monkeypatch):
    api, config, _ = calibration_call
    monkeypatch.setattr(api.catalog_io, "read_phone_detections", lambda *_args, **_kwargs: {})
    calls = []

    def detect(segment, *_args):
        calls.append(segment.id)
        return {p: PhoneDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))) for p in segment.layout.phones}

    monkeypatch.setattr(api, "detect_episode", detect)
    api.main(config)
    assert calls == list(config.segment_ids)


def test_egohand_missing_clock_finishes_without_sampling(calibration_call, monkeypatch) -> None:
    api, config, _ = calibration_call
    config = replace(config, oak_method="egohand")
    previous = api.catalog_io.read_time_sync
    monkeypatch.setattr(api.catalog_io, "read_time_sync", lambda *args: replace(previous(*args), flags=["quest_unsynced_kept_as_cut"]))
    monkeypatch.setattr(api, "fixed_rotation", lambda *_args: None)
    monkeypatch.setattr(api, "detect_oak_egohand", lambda *args, **kwargs: pytest.fail("must not sample without a clock"))
    monkeypatch.setattr(api, "solve_oak", lambda *args, **kwargs: pytest.fail("must not fit without a clock"))
    state = api.main(config)
    assert not state.oak.ok
    assert "missing Quest/OAK clock" in state.oak.rotation_source
    assert (config.output_dir / "egohand" / f"{config.segment_ids[0]}.rrd").is_file()
    path = config.output_dir / f"{config.segment_ids[0]}.rrd"
    prop = next(c.to_record_batch() for c in RrdReader(path).stream() if str(c.entity_path) == "/__properties/calib")
    assert prop.column("oak_posed")[0].as_py() == [False]
    assert prop.column("oak_unposed_reason")[0].as_py() == ["missing Quest/OAK clock"]


@pytest.mark.parametrize("reason", ["gyro fit not ok", "episode spread exceeds 1.5 degrees", "device missing from rgb_R_imu table"])
def test_egohand_ineligible_gyro_uses_free_fit_and_records_reason(calibration_call, monkeypatch, reason):
    api, config, _ = calibration_call
    config = replace(config, oak_method="egohand")
    gyro = GyroFit("test", reason != "gyro fit not ok", max_episode_dev_deg=2.0 if "spread" in reason else 1.0,
                   R_head_from_imu=np.eye(3))
    monkeypatch.setattr(api, "gyro_mount", lambda *_args: gyro)
    monkeypatch.setattr(api, "read_rotations", lambda: SimpleNamespace(devices={}))
    monkeypatch.setattr(api, "release_cuda_memory", lambda **kwargs: None)
    monkeypatch.setattr(api, "read_ego_cameras", lambda *_args: [])
    monkeypatch.setattr(api, "read_catalog_videos", lambda *_args: [None])
    old_solve = api.solve_oak

    def detect(*args, **kwargs):
        assert args[3] is None
        return OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133)))

    def solve(*args, **kwargs):
        assert args[4] is None
        return old_solve(*args, **kwargs)

    monkeypatch.setattr(api, "detect_oak_egohand", detect)
    monkeypatch.setattr(api, "solve_oak", solve)
    state = api.main(config)
    assert state.oak.rotation_source == "free"
    assert state.oak.rotation_reason == reason
    path = config.output_dir / f"{config.segment_ids[0]}.rrd"
    prop = next(c.to_record_batch() for c in RrdReader(path).stream() if str(c.entity_path) == "/__properties/calib")
    assert prop.column("oak_rotation_source")[0].as_py() == ["free"]
    assert prop.column("oak_rotation_reason")[0].as_py() == [reason]
