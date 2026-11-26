"""``config.vio_bootstrap`` through the binding, and the replay's mono-proxy option."""

import json
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, RigFactory, Rows, RowsReader, TextureFactory, gravity_batch
from jaxtyping import Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.columns import ColumnLog
from slam_rs.config import SLAM_CONFIG_PATH
from slam_rs.vio_log import BOOTSTRAP_ENTITY, BootstrapLog

MSD_CONFIG_KEYS: frozenset[str] = frozenset(
    f"config.{key}"
    for key in ["optical_flow_detection_grid_size", "optical_flow_detection_max_threshold", "optical_flow_detection_min_threshold", "optical_flow_detection_nonoverlap", "optical_flow_detection_num_points_cell", "optical_flow_epipolar_error", "optical_flow_image_safe_radius", "optical_flow_levels", "optical_flow_matching_default_depth", "optical_flow_matching_guess_type", "optical_flow_max_iterations", "optical_flow_max_recovered_dist2", "optical_flow_pattern", "optical_flow_recall_all_cams", "optical_flow_recall_enable", "optical_flow_recall_max_patch_dist", "optical_flow_recall_max_patch_norms", "optical_flow_recall_num_points_cell", "optical_flow_recall_over_tracking", "optical_flow_recall_update_patch_viewpoint", "optical_flow_skip_frames", "optical_flow_type", "vio_debug", "vio_enforce_realtime", "vio_extended_logging", "vio_fix_long_term_keyframes", "vio_init_ba_weight", "vio_init_bg_weight", "vio_init_pose_weight", "vio_kf_marg_criteria", "vio_kf_marg_feature_ratio", "vio_linearization_type", "vio_lm_lambda_initial", "vio_lm_lambda_max", "vio_lm_lambda_min", "vio_marg_lost_landmarks", "vio_max_iterations", "vio_max_kfs", "vio_max_states", "vio_min_frames_after_kf", "vio_min_triangulation_dist", "vio_new_kf_keypoints_thresh", "vio_obs_huber_thresh", "vio_obs_std_dev", "vio_scale_jacobian", "vio_sqrt_marg", "vio_use_lm"]
)
"""Every key the three shipped MSD configs state: the RoboCap and MSD lanes run on exactly these, every other key at its default."""


@pytest.mark.parametrize("name", ["msdmi", "msdmg", "msdmo"])
def test_the_shipped_msd_configs_state_the_pinned_keys(name: str) -> None:
    """The key set of each shipped MSD config is pinned, and loading then writing it keeps every value it states."""
    stated: dict = json.loads((SLAM_CONFIG_PATH.parent / f"configs/{name}_config.json").read_text())["value0"]
    assert set(stated) == MSD_CONFIG_KEYS
    written: dict = json.loads(_core.VioConfig.from_json(json.dumps({"value0": stated})).to_json())["value0"]
    assert {key: written[key] for key in stated} == stated


def test_the_stereo_start_is_the_default_and_is_written_out() -> None:
    """``config.vio_bootstrap`` is written at its default like every other key."""
    text: str = (SLAM_CONFIG_PATH.parent / "configs/msdmi_config.json").read_text()
    config: _core.VioConfig = _core.VioConfig.from_json(text)
    assert config.vio_bootstrap == "stereo"
    assert json.loads(config.to_json())["value0"]["config.vio_bootstrap"] == "stereo"


@pytest.mark.parametrize("value", ["inertial", "stereo"])
def test_the_key_round_trips(value: str) -> None:
    config: _core.VioConfig = _core.VioConfig()
    config.vio_bootstrap = value
    assert config.vio_bootstrap == value
    written: dict = json.loads(config.to_json())["value0"]
    assert written["config.vio_bootstrap"] == value
    assert _core.VioConfig.from_json(config.to_json()).vio_bootstrap == value


def test_an_unknown_value_is_refused() -> None:
    config: _core.VioConfig = _core.VioConfig()
    with pytest.raises(ValueError, match="stereo or inertial"):
        config.vio_bootstrap = "mono"


def test_a_stereo_pipeline_has_no_bootstrap_report() -> None:
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    assert _core.Vio(calibration, config, backend="cpu").bootstrap_report() is None
    config.vio_bootstrap = "inertial"
    report: _core.BootstrapReport | None = _core.Vio(calibration, config, backend="cpu").bootstrap_report()
    assert report is not None
    assert (report.attempts, report.start_t_ns, report.accepted_at_t_ns, report.replayed) == ([], None, None, 0)


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
    ["insta360_native_config", "tartanair_fisheye_config", "tartanair_pinhole_config"],
)
def test_rigs_that_start_in_motion_bootstrap_inertially(dataset: str) -> None:
    """The Insta360 has no stereo pair and every TartanAir clip starts moving: the stereo start cannot work there."""
    text: str = (SLAM_CONFIG_PATH.parent / f"configs/{dataset}.json").read_text()
    assert _core.VioConfig.from_json(text).vio_bootstrap == "inertial"


def still_rig_until_first_attempt(rig: RigFactory, texture: TextureFactory) -> _core.Vio:
    """A still synthetic rig on the inertial bootstrap, run until its first 2 s window is solved."""
    config: _core.VioConfig = _core.VioConfig()
    config.vio_bootstrap = "inertial"
    vio: _core.Vio = _core.Vio(rig(2), config, backend="cpu")
    for step in range(120):
        t_ns: Int64[ndarray, " n"] = np.arange(step * FRAME_PERIOD_NS, (step + 1) * FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        vio.push_imu_batch(t_ns, *gravity_batch(t_ns))
        vio.track(step * FRAME_PERIOD_NS, [texture(step, 0), texture(step - 2, 0)])
        report: _core.BootstrapReport | None = vio.bootstrap_report()
        if report is not None and report.attempts:
            return vio
    raise AssertionError("no bootstrap attempt in 120 framesets")


def test_a_still_rigs_first_attempt_is_a_typed_low_excitation_rejection(rig: RigFactory, texture: TextureFactory) -> None:
    """The first attempt of a rig at rest: a 2 s window rejected for low excitation, read through typed getters."""
    from slam_rs.apis.insv_calibrate import first_attempt

    vio: _core.Vio = still_rig_until_first_attempt(rig, texture)
    report: _core.BootstrapReport | None = vio.bootstrap_report()
    assert report is not None
    (attempt,) = report.attempts
    assert (attempt.verdict, attempt.accepted) == ("LowExcitation", False)
    assert attempt.first_t_ns == 0
    assert attempt.last_t_ns - attempt.first_t_ns >= 2_000_000_000
    assert attempt.frames == attempt.last_t_ns // FRAME_PERIOD_NS + 1
    assert attempt.excitation_mps2 < 0.25
    assert (report.start_t_ns, report.accepted_at_t_ns, report.replayed) == (None, None, 0)
    first: _core.BootstrapAttempt | None = first_attempt(vio)
    assert first is not None and first.last_t_ns == attempt.last_t_ns


def test_each_attempt_goes_into_the_recording_once_at_its_own_time(rig: RigFactory, texture: TextureFactory, tmp_path: Path, read_rows: RowsReader) -> None:
    """The layer logs an attempt when it appears, at its last frameset's time, and states the status as a property."""
    output: Path = tmp_path / "bootstrap.rrd"
    vio: _core.Vio = still_rig_until_first_attempt(rig, texture)
    with rr.RecordingStream("slam-rs-bootstrap-test", recording_id="bootstrap") as recording:
        recording.save(output)
        bootstrap: BootstrapLog = BootstrapLog(cam_time_offset_ns=0, columns=ColumnLog(recording=recording))
        bootstrap.log_new(vio)
        bootstrap.log_new(vio)
        bootstrap.columns.flush()
    report: _core.BootstrapReport | None = vio.bootstrap_report()
    assert report is not None
    attempt_t_ns: int = report.attempts[0].last_t_ns
    rows: Rows = read_rows(output)
    assert [row.t_ns for row in rows[BOOTSTRAP_ENTITY]] == [attempt_t_ns]
    assert {row.t_ns for entity, entity_rows in rows.items() if entity.startswith(BOOTSTRAP_ENTITY) for row in entity_rows} == {attempt_t_ns}
    properties: set[str] = {chunk.entity_path for chunk in rrc.RrdReader(output).stream().collect().stream() if chunk.entity_path.startswith("/__properties")}
    assert "/__properties/bootstrap" in properties
    assert BootstrapLog.summary(vio, 0) == "inertial bootstrap: 1 attempts ['LowExcitation']; no window accepted"
