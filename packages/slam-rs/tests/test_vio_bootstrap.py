"""``config.vio_bootstrap`` through the binding, and the replay's mono-proxy option."""

import json
from pathlib import Path

import numpy as np
import pytest
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, CameraFactory, PipelineFactory, RigFactory, TextureFactory, gravity_batch
from jaxtyping import Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.config import SLAM_CONFIG_PATH


def test_the_stereo_start_is_the_default_and_is_not_written_out() -> None:
    text: str = (SLAM_CONFIG_PATH.parent / "configs/msdmi_config.json").read_text()
    config: _core.VioConfig = _core.VioConfig.from_json(text)
    assert config.vio_bootstrap == "stereo"
    assert "vio_bootstrap" not in config.to_json()


@pytest.mark.parametrize("value", ["inertial", "auto", "stereo"])
def test_the_key_round_trips(value: str) -> None:
    config: _core.VioConfig = _core.VioConfig()
    config.vio_bootstrap = value
    assert config.vio_bootstrap == value
    written: dict = json.loads(config.to_json())["value0"]
    assert written.get("config.vio_bootstrap", "stereo") == value
    assert _core.VioConfig.from_json(config.to_json()).vio_bootstrap == value


def test_an_unknown_value_is_refused() -> None:
    config: _core.VioConfig = _core.VioConfig()
    with pytest.raises(ValueError, match="stereo, inertial or auto"):
        config.vio_bootstrap = "mono"


def test_a_stereo_pipeline_has_no_bootstrap_report() -> None:
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    assert _core.Vio(calibration, config, backend="cpu").bootstrap_report() is None
    config.vio_bootstrap = "inertial"
    vio: _core.Vio = _core.Vio(calibration, config, backend="cpu")
    assert json.loads(vio.bootstrap_report() or "") == {"attempts": [], "start_t_ns": None, "accepted_at_t_ns": None, "replayed": 0}


def test_the_report_accessors_match_the_report_text() -> None:
    """Python reads the bootstrap report through typed getters; the JSON text is only written out, never parsed."""
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    for value in ("stereo", "inertial"):
        config.vio_bootstrap = value
        vio: _core.Vio = _core.Vio(calibration, config, backend="cpu")
        assert vio.bootstrap_attempts == []
        assert vio.bootstrap_start_t_ns is None
        assert vio.bootstrap_accepted_at_t_ns is None
        assert vio.bootstrap_replayed == 0


def test_blanked_cameras_track_nothing() -> None:
    """A black image gives the frontend nothing to detect or track."""
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    flow = _core.OpticalFlow(calibration, config)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, size=(960, 960), dtype=np.uint8)
    black = np.zeros((960, 960), dtype=np.uint8)
    for k in range(3):
        frame = flow.process(k * 33_333_333, [np.roll(textured, k, axis=1), black])
    assert len(frame.ids(0)) > 0
    assert len(frame.ids(1)) == 0


@pytest.mark.parametrize(
    "dataset",
    ["tartanair_fisheye_config", "tartanair_pinhole_config"],
)
def test_rigs_that_start_in_motion_bootstrap_inertially(dataset: str) -> None:
    """The Insta360 has no stereo pair and every TartanAir clip starts moving: the stereo start cannot work there."""
    text: str = (SLAM_CONFIG_PATH.parent / f"configs/{dataset}.json").read_text()
    assert _core.VioConfig.from_json(text).vio_bootstrap == "inertial"


def test_the_attempt_accessors_read_what_the_report_text_says(rig: RigFactory, texture: TextureFactory) -> None:
    """A still synthetic rig run until the bootstrap's first 2 s window is solved: each getter holds the report's value."""
    config: _core.VioConfig = _core.VioConfig()
    config.vio_bootstrap = "inertial"
    vio: _core.Vio = _core.Vio(rig(2), config, backend="cpu")
    for step in range(120):
        t_ns: Int64[ndarray, " n"] = np.arange(step * FRAME_PERIOD_NS, (step + 1) * FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        vio.push_imu_batch(t_ns, *gravity_batch(t_ns))
        vio.track(step * FRAME_PERIOD_NS, [texture(step, 0), texture(step - 2, 0)])
        if vio.bootstrap_attempts:
            break
    report: dict = json.loads(vio.bootstrap_report() or "")
    assert vio.bootstrap_attempts, "no bootstrap attempt in 120 framesets"
    for attempt, written in zip(vio.bootstrap_attempts, report["attempts"], strict=True):
        assert (attempt.first_t_ns, attempt.last_t_ns, attempt.frames, attempt.tracks) == (
            written["first_t_ns"],
            written["last_t_ns"],
            written["frames"],
            written["tracks"],
        )
        assert (attempt.verdict, attempt.accepted, attempt.residual_rms_rad, attempt.scale_sigma) == (
            written["verdict"],
            written["accepted"],
            written["residual_rms_rad"],
            written["scale_sigma"],
        )
    assert (vio.bootstrap_start_t_ns, vio.bootstrap_accepted_at_t_ns, vio.bootstrap_replayed) == (
        report["start_t_ns"],
        report["accepted_at_t_ns"],
        report["replayed"],
    )


def test_a_report_writer_error_does_not_replace_the_replays_own(
    pipeline: PipelineFactory, camera: CameraFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a failed replay the writer's own error is printed, so the replay's exception is the one that propagates."""
    from slam_rs.apis import replay
    from slam_rs.tracking import Lockstep
    from slam_rs.trajectory import empty_trajectory
    from slam_rs.vio_log import VioLogger, VioStage

    def full_disk(*_args: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(replay, "_write_bootstrap_report", full_disk)
    stage: VioStage = VioStage(
        lockstep=Lockstep(vio=pipeline(2)),
        logger=VioLogger(cameras=(camera(0, 0.0), camera(1, 0.1)), ground_truth=empty_trajectory(), frame_t_ns=np.zeros(1, dtype=np.int64)),
    )
    replay._write_bootstrap_report_after_failure(stage, tmp_path / "slam_rs.csv", 0)
    assert "No space left on device" in capsys.readouterr().err
