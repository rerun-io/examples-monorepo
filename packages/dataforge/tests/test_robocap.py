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
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.data.ego.robocap_ego import CAMERA_DISPLAY_ORDER
from simplecv.imu_calibration import ImuCalibration

from dataforge import blueprints, schema
from dataforge.datasets.robocap import (
    ACCEL_SCALE,
    CAMERA_TO_IMU_OFFSET_NS,
    GYRO_SCALE,
    RobocapConfig,
    RobocapDataset,
    RobocapSource,
    build_blueprint,
    build_table_blueprint,
    read_imu_database,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel

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



def blueprint_views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Every view in a blueprint, depth-first, whatever containers nest them."""
    found: list[rrb.View] = []

    def walk(node: rrb.View | rrb.Container) -> None:
        if isinstance(node, rrb.View):
            found.append(node)
            return
        for child in node.contents or ():
            walk(child)

    walk(blueprint.root_container)
    return found


def test_robocaps_follow_view_dims_its_basalt_path_like_every_rig_layout() -> None:
    """The path/trail pair is one shared skeleton, so RoboCap gets the same treatment.

    ``blueprints.rig_blueprint`` builds both datasets' layouts and takes the run
    source as an argument, so a change to how the follow view separates the path
    from the trail lands on RoboCap's ``basalt`` paths too. That is intended, and
    asserted here rather than left to be noticed on the next viewer session.
    """
    views: list[rrb.View] = blueprint_views(build_blueprint(list(CAMERA_DISPLAY_ORDER)))

    follow: rrb.View = next(view for view in views if view.name == "Follow")
    dimmed: object = follow.visualizer_overrides[schema.trajectory_path("basalt")]
    assert isinstance(dimmed, rr.LineStrips3D), "the Follow view styles the path rather than hiding it"
    assert dimmed.radii is not None
    assert dimmed.radii.as_arrow_array().to_pylist() == [-blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS]

    rig: rrb.View = next(view for view in views if view.name == "Rig")
    assert rig.visualizer_overrides[schema.trail_path("basalt")] == rrb.EntityBehavior(visible=False)


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
    channels: tuple[ImuChannel, ImuChannel] | None = read_imu_database(db_path)
    assert channels is not None
    gyro: ImuChannel = channels[0]
    accel: ImuChannel = channels[1]

    expected_gyro_times: Int64[ndarray, "2"] = np.array([1000000000, 1002000000], dtype=np.int64) - CAMERA_TO_IMU_OFFSET_NS
    np.testing.assert_array_equal(gyro.times_ns, expected_gyro_times)
    expected_gyro_xyz: Float64[ndarray, "2 3"] = np.array([[10.0, -20.0, 30.0], [11.0, -21.0, 31.0]]) * GYRO_SCALE
    np.testing.assert_allclose(gyro.values_xyz, expected_gyro_xyz)

    np.testing.assert_array_equal(accel.times_ns, np.array([1001000000 - CAMERA_TO_IMU_OFFSET_NS], dtype=np.int64))
    np.testing.assert_allclose(accel.values_xyz, np.array([[100.0, 200.0, -300.0]]) * ACCEL_SCALE)


def test_malformed_imu_db_is_skipped_not_fatal(tmp_path: Path) -> None:
    db_path: Path = tmp_path / "IMUWriter_dev2_session1_segment1.db"
    db_path.write_bytes(b"SQLite format 3\x00 this is not a database")
    assert read_imu_database(db_path) is None


def test_factory_metadata_preserves_offsets_and_applied_correction(tmp_path: Path) -> None:
    factory: Path = tmp_path / f"0factory-calibration-{DEVICE}"
    noise: Path = factory / "imus_intrinsic/imu_mid_0.yaml"
    noise.parent.mkdir(parents=True)
    noise.write_text("gyroscope_noise_density: 0.0007\naccelerometer_noise_density: 0.006\n"
                     "gyroscope_random_walk: 0.00003\naccelerometer_random_walk: 0.0002\nupdate_rate: 200.0\n")
    camera: Path = factory / "imus_cam_l_extrinsic/left-camchain-imucam.yaml"
    camera.parent.mkdir()
    camera.write_text(
        "cam0:\n  camera_model: pinhole\n  distortion_model: equidistant\n  intrinsics: [300.0, 300.0, 320.0, 240.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n  resolution: [640, 480]\n"
        "  T_cam_imu: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]\n"
        "  timeshift_cam_imu: 0.018961111236788484\n"
    )
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    path: Path = tmp_path / "metadata.rrd"
    with rr.RecordingStream("metadata", recording_id="test") as recording:
        recording.save(path)
        dataset.log_sensor_metadata(recording, DEVICE, {"left": schema.cam_path(0, 2)})
    with rr.server.Server(datasets={"test": [path]}) as server:
        client: CatalogClient = server.client()
        entry: DatasetEntry = client.get_dataset("test")
        table: pa.Table = entry.reader(index=None).to_arrow_table()
        imu: ImuCalibration = ImuCalibration.from_catalog(table, "/world/rig_00/imu_00")
        assert imu.gyro_noise_density == 0.0007
        assert imu.rate_hz == 200.0
        assert imu.source == f"0factory-calibration-{DEVICE}/imus_intrinsic/imu_mid_0.yaml"
        assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[-14_902_432]]
        assert table["/world/rig_00/cam_02:camera_imu_time_offset_ns"].to_pylist() == [[18_961_111]]
        assert table["/world/rig_00/cam_02:time_offset_reference"].to_pylist() == [["/world/rig_00/imu_00"]]


def test_missing_device_calibration_is_not_replaced_with_another_caps(tmp_path: Path) -> None:
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=tmp_path))
    path: Path = tmp_path / "unknown.rrd"
    with rr.RecordingStream("metadata", recording_id="unknown") as recording:
        recording.save(path)
        dataset.log_sensor_metadata(recording, "unknown-device", {})
    with rr.server.Server(datasets={"test": [path]}) as server:
        client: CatalogClient = server.client()
        entry: DatasetEntry = client.get_dataset("test")
        table: pa.Table = entry.reader(index=None).to_arrow_table()
        assert ImuCalibration.from_catalog(table, "/world/rig_00/imu_00") == ImuCalibration()
