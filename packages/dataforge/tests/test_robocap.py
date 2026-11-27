"""RoboCap discovery and IMU parsing against synthetic fixtures (no corpus needed)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64
from numpy import ndarray

pytest.importorskip("rerun.catalog", reason="catalog metadata tests need the Rerun catalog dependencies")

from rerun.catalog import CatalogClient, DatasetEntry  # noqa: E402
from robocap_factory import CAP_B_SLAM_SHIFTS_S, write_factory
from simplecv.imu_calibration import ImuCalibration  # noqa: E402

from dataforge import blueprints, schema  # noqa: E402
from dataforge.datasets.robocap import (  # noqa: E402
    ACCEL_SCALE,
    CAMERA_TO_IMU_OFFSET_NS,
    CAPS,
    FACTORY_TIME_SHIFT_SOURCE,
    GYRO_SCALE,
    LEGACY_TIME_SHIFT_SOURCE,
    TIME_SHIFT_CAMERAS,
    FactoryCalibration,
    RobocapConfig,
    RobocapDataset,
    RobocapSource,
    build_blueprint,
    build_table_blueprint,
    read_imu_database,
)
from dataforge.identity import SequenceIdentity  # noqa: E402
from dataforge.logging_toolkit import ImuChannel  # noqa: E402
from dataforge.robocap_kalibr import CAMERA_DISPLAY_ORDER  # noqa: E402
from dataforge.writing import blueprint_views

DEVICE: str = "f408193e6447b3b0"
"""Device id used by every fake session directory in these tests."""
CAM_NAMES: tuple[str, ...] = ("right-eye", "left-front", "left-eye", "right", "left", "right-front")
"""Video filename suffixes in device order (dev0..dev5)."""


def make_segment(session_dir: Path, session: int, segment: int) -> None:
    """Create the six empty MP4s and three empty IMU dbs of one fake segment."""
    session_dir.mkdir(parents=True, exist_ok=True)
    for device_index, cam_name in enumerate(CAM_NAMES):
        (session_dir / f"video_dev{device_index}_session{session}_segment{segment}_{cam_name}.mp4").touch()
    for imu_index in range(3):
        (session_dir / f"IMUWriter_dev{imu_index}_session{session}_segment{segment}.db").touch()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A fake RoboCap root: two video sessions, one empty session, and one ``-old`` decoy."""
    (tmp_path / f"0factory-calibration-{DEVICE}").mkdir()
    make_segment(tmp_path / f"{DEVICE}_session_1", session=1, segment=1)
    make_segment(tmp_path / f"{DEVICE}_session_1", session=1, segment=2)
    make_segment(tmp_path / f"{DEVICE}_session_10", session=10, segment=1)
    (tmp_path / f"{DEVICE}_session_20").mkdir()
    make_segment(tmp_path / f"{DEVICE}_session_10-old", session=10, segment=9)
    return tmp_path


def test_discover_pairs_every_session_with_its_segments_and_skips_old_dirs(corpus: Path) -> None:
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=corpus))
    discovered: list[tuple[SequenceIdentity, RobocapSource]] = dataset.discover()
    assert [identity.sequence_key for identity, _ in discovered] == [
        f"{DEVICE}/s00000001",
        f"{DEVICE}/s00000010",
    ]
    assert discovered[0][0].recording_id == f"robocap__{DEVICE}__s00000001"
    # The session is the sequence unit; its file-roll segments merge at convert.
    assert discovered[0][1] == RobocapSource(session_dir=corpus / f"{DEVICE}_session_1", device=DEVICE, session=1, segments=(1, 2))
    assert discovered[1][1].segments == (1,)
    # sequences() is derived from discover(), so the two can never disagree.
    assert dataset.sequences() == [identity for identity, _ in discovered]


def test_download_verifies_local_corpus_and_reports_video_less_sessions(corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    RobocapConfig(root=corpus).setup().download()
    output: str = capsys.readouterr().out
    assert "3 session dirs, 2 with videos" in output
    assert f"warning: session directories contain no videos: {DEVICE}_session_20" in output
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        RobocapConfig(root=empty_root).setup().download()


def test_blueprints_serialize_with_canonical_camera_order(tmp_path: Path) -> None:
    camera_names: list[str] = list(CAMERA_DISPLAY_ORDER)
    segment_blueprint: rrb.Blueprint = build_blueprint(camera_names)
    table_blueprint: rrb.Blueprint = build_table_blueprint(camera_names)
    segment_path: Path = tmp_path / "robocap.rbl"
    table_path: Path = tmp_path / "robocap-table.rbl"

    segment_blueprint.save("robocap", str(segment_path))
    table_blueprint.save("robocap", str(table_path))

    assert segment_path.stat().st_size > 0
    assert table_path.stat().st_size > 0
    assert schema.trail_path("basalt") == "/world/runs/basalt/trail"



@pytest.mark.parametrize("source", ["basalt", "slam_rs", "slam_rs_vio"])
def test_robocaps_follow_view_dims_every_run_path_like_every_rig_layout(source: str) -> None:
    """The path/trail pair is one shared skeleton, so every RoboCap run gets the same treatment.

    ``blueprints.rig_blueprint`` builds both datasets' layouts and takes the run
    sources as arguments, so a change to how the follow view separates the path
    from the trail lands on RoboCap's ``basalt`` paths and on both slam-rs runs
    (final and raw VIO) too. That is intended, and asserted here rather than left
    to be noticed on the next viewer session.
    """
    views: list[rrb.View] = blueprint_views(build_blueprint(list(CAMERA_DISPLAY_ORDER)))

    follow: rrb.View = next(view for view in views if view.name == "Follow")
    dimmed: object = follow.visualizer_overrides[schema.trajectory_path(source)]
    assert isinstance(dimmed, rr.LineStrips3D), "the Follow view styles the path rather than hiding it"
    assert dimmed.radii is not None
    assert dimmed.radii.as_arrow_array().to_pylist() == [-blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS]
    assert follow.visualizer_overrides[schema.trail_path(source)] == blueprints.trail_window_range()

    rig: rrb.View = next(view for view in views if view.name == "Rig")
    assert rig.visualizer_overrides[schema.trail_path(source)] == rrb.EntityBehavior(visible=False)


def write_imu_db(db_path: Path) -> None:
    """Write a two-sample synthetic IMU db in the RoboCap writer's schema."""
    with sqlite3.connect(db_path) as database:
        for table in ("gyro_data", "acc_data"):
            database.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, imuid_ INTEGER, x INTEGER, y INTEGER, z INTEGER, timestamp INTEGER)")
        database.execute("INSERT INTO gyro_data VALUES (1, 0, 10, -20, 30, 1000000000)")
        database.execute("INSERT INTO gyro_data VALUES (2, 0, 11, -21, 31, 1002000000)")
        database.execute("INSERT INTO acc_data VALUES (1, 0, 100, 200, -300, 1001000000)")


def test_imu_parsing_scales_values_and_shifts_onto_camera_clock(tmp_path: Path) -> None:
    db_path: Path = tmp_path / "IMUWriter_dev0_session1_segment1.db"
    write_imu_db(db_path)
    offset_ns: int = 1_217_000
    channels: tuple[ImuChannel, ImuChannel] | None = read_imu_database(db_path, offset_ns)
    assert channels is not None
    gyro: ImuChannel = channels[0]
    accel: ImuChannel = channels[1]

    expected_gyro_times: Int64[ndarray, "2"] = np.array([1000000000, 1002000000], dtype=np.int64) - offset_ns
    np.testing.assert_array_equal(gyro.times_ns, expected_gyro_times)
    expected_gyro_xyz: Float64[ndarray, "2 3"] = np.array([[10.0, -20.0, 30.0], [11.0, -21.0, 31.0]]) * GYRO_SCALE
    np.testing.assert_allclose(gyro.values_xyz, expected_gyro_xyz)

    np.testing.assert_array_equal(accel.times_ns, np.array([1001000000 - offset_ns], dtype=np.int64))
    np.testing.assert_allclose(accel.values_xyz, np.array([[100.0, 200.0, -300.0]]) * ACCEL_SCALE)


def test_malformed_imu_db_is_skipped_not_fatal(tmp_path: Path) -> None:
    db_path: Path = tmp_path / "IMUWriter_dev2_session1_segment1.db"
    db_path.write_bytes(b"SQLite format 3\x00 this is not a database")
    assert read_imu_database(db_path, CAMERA_TO_IMU_OFFSET_NS) is None


def test_imu_time_shift_is_the_mean_factory_shift_of_the_slam_cameras(tmp_path: Path) -> None:
    write_factory(tmp_path, DEVICE)
    factory: FactoryCalibration = RobocapDataset(RobocapConfig(root=tmp_path)).factory_calibration(DEVICE)
    assert TIME_SHIFT_CAMERAS == ("left_front", "right_front", "left", "right")
    # The eye pair (2 ms) is not a SLAM camera and does not count.
    assert factory.time_shifts_ns == {"left_front": 866_129, "right_front": 976_658, "left": 1_591_904, "right": 1_449_320}
    assert factory.mean_time_shift_ns == round(sum(CAP_B_SLAM_SHIFTS_S) / 4 * 1e9) == 1_221_003
    assert factory.time_shift_spread_ns == 725_775
    assert factory.cap == CAPS[DEVICE]
    assert set(factory.fisheye62()) == {"left_front", "right_front", "left_eye", "right_eye", "left", "right"}


def test_a_slam_camera_without_a_factory_time_shift_is_an_error(tmp_path: Path) -> None:
    write_factory(tmp_path, DEVICE, (0.012, 0.013, None, 0.016))
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    with pytest.raises(ValueError, match=f"Cap A \\({DEVICE}\\): no factory timeshift_cam_imu for left in"):
        dataset.factory_calibration(DEVICE)


def test_a_non_finite_factory_time_shift_is_an_error(tmp_path: Path) -> None:
    write_factory(tmp_path, DEVICE, (0.012, float("nan"), 0.019, 0.016))
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    with pytest.raises(ValueError, match="front-camchain-imucam.yaml#cam1: timeshift_cam_imu must be finite"):
        dataset.factory_calibration(DEVICE)


def test_an_unknown_serial_is_refused_with_a_message_to_register_it(tmp_path: Path) -> None:
    serial: str = "0123456789abcdef"
    write_factory(tmp_path, serial)
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    with pytest.raises(ValueError, match=f"unknown RoboCap serial {serial}.*add it to CAPS"):
        dataset.factory_calibration(serial)


def test_a_known_cap_without_its_calibration_folder_is_an_error(tmp_path: Path) -> None:
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    with pytest.raises(FileNotFoundError, match=f"Cap A \\({DEVICE}\\).*{tmp_path / CAPS[DEVICE].calibration_dir}"):
        dataset.factory_calibration(DEVICE)


def test_the_registry_names_both_calibrated_caps() -> None:
    assert {serial: cap.name for serial, cap in CAPS.items()} == {"f408193e6447b3b0": "Cap A", "fe6fede545c972fa": "Cap B"}
    assert all(cap.calibration_dir == f"0factory-calibration-{serial}" for serial, cap in CAPS.items())


@pytest.mark.integration
@pytest.mark.parametrize(
    ("serial", "shifts_ns", "mean_ns", "spread_ns"),
    [
        ("f408193e6447b3b0", {"left_front": 12_219_303, "right_front": 13_052_872, "left": 18_961_111, "right": 16_751_992}, 15_246_320, 6_741_808),
        ("fe6fede545c972fa", {"left_front": 866_129, "right_front": 976_658, "left": 1_591_904, "right": 1_449_320}, 1_221_003, 725_775),
    ],
)
def test_factory_time_shifts_of_both_caps(serial: str, shifts_ns: dict[str, int], mean_ns: int, spread_ns: int) -> None:
    dataset: RobocapDataset = RobocapDataset(RobocapConfig())
    if not (dataset.config.root / CAPS[serial].calibration_dir).is_dir():
        pytest.skip(f"factory calibration {CAPS[serial].calibration_dir} not under {dataset.config.root}")
    factory: FactoryCalibration = dataset.factory_calibration(serial)
    assert factory.time_shifts_ns == shifts_ns
    assert factory.mean_time_shift_ns == mean_ns
    assert factory.time_shift_spread_ns == spread_ns


def read_metadata(dataset: RobocapDataset, path: Path, applied_shift_ns: int, time_shift_source: str) -> pa.Table:
    """Log one device's sensor metadata (left camera at ``cam_02``) to ``path`` and read it back as a catalog table."""
    with rr.RecordingStream("metadata", recording_id="test") as recording:
        recording.save(path)
        dataset.log_sensor_metadata(
            recording, dataset.factory_calibration(DEVICE), {"left": schema.cam_path(0, 2)}, applied_shift_ns=applied_shift_ns, time_shift_source=time_shift_source
        )
    with rr.server.Server(datasets={"test": [path]}) as server:
        client: CatalogClient = server.client()
        entry: DatasetEntry = client.get_dataset("test")
        return entry.reader(index=None).to_arrow_table()


@pytest.mark.integration
def test_factory_metadata_states_the_cap_the_applied_mean_shift_and_the_per_camera_spread(tmp_path: Path) -> None:
    write_factory(tmp_path, DEVICE)
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    table: pa.Table = read_metadata(dataset, tmp_path / "metadata.rrd", 1_221_003, FACTORY_TIME_SHIFT_SOURCE)
    imu: ImuCalibration = ImuCalibration.from_catalog(table, "/world/rig_00/imu_00")
    assert imu.gyro_noise_density == 0.0007
    assert imu.rate_hz == 200.0
    assert imu.source == f"0factory-calibration-{DEVICE}/imus_intrinsic/imu_mid_0.yaml"
    assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[-1_221_003]]
    assert table["/world/rig_00/imu_00:time_shift_spread_ns"].to_pylist() == [[725_775]]
    assert table["/world/rig_00/imu_00:time_shift_source"].to_pylist() == [[FACTORY_TIME_SHIFT_SOURCE]]
    assert table["/world/rig_00/cam_02:camera_imu_time_offset_ns"].to_pylist() == [[1_591_904]]
    assert table["/world/rig_00/cam_02:time_offset_reference"].to_pylist() == [["/world/rig_00/imu_00"]]
    assert table["/world/rig_00:cap_name"].to_pylist() == [["Cap A"]]
    assert table["/world/rig_00:cap_serial"].to_pylist() == [[DEVICE]]
    assert table["/world/rig_00:calibration_source"].to_pylist() == [[f"0factory-calibration-{DEVICE}"]]
    assert table["/world/rig_00:calibration_date"].to_pylist() == [["2026-03-02"]]


@pytest.mark.integration
def test_backfill_metadata_states_the_legacy_shift_old_recordings_carry(tmp_path: Path) -> None:
    write_factory(tmp_path, DEVICE)
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    table: pa.Table = read_metadata(dataset, tmp_path / "metadata.rrd", CAMERA_TO_IMU_OFFSET_NS, LEGACY_TIME_SHIFT_SOURCE)
    assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[-14_902_432]]
    assert table["/world/rig_00/imu_00:time_shift_source"].to_pylist() == [[LEGACY_TIME_SHIFT_SOURCE]]
    assert table["/world/rig_00:cap_name"].to_pylist() == [["Cap A"]]
