"""RoboCap discovery and IMU parsing against synthetic fixtures (no corpus needed)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray

from dataforge.datasets.robocap import (
    ACCEL_SCALE,
    CAMERA_TO_IMU_OFFSET_NS,
    GYRO_SCALE,
    RobocapConfig,
    RobocapDataset,
    RobocapSource,
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
    """A fake RoboCap root: two sessions, three segments, one ``-old`` decoy."""
    (tmp_path / f"0factory-calibration-{DEVICE}").mkdir()
    make_segment(tmp_path / f"{DEVICE}_session_1", session=1, segment=1)
    make_segment(tmp_path / f"{DEVICE}_session_1", session=1, segment=2)
    make_segment(tmp_path / f"{DEVICE}_session_10", session=10, segment=1)
    make_segment(tmp_path / f"{DEVICE}_session_10-old", session=10, segment=9)
    return tmp_path


def test_discover_pairs_every_session_with_its_segments_and_skips_old_dirs(corpus: Path) -> None:
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=corpus))
    discovered: list[tuple[SequenceIdentity, RobocapSource]] = dataset.discover()
    assert [identity.sequence_key for identity, _ in discovered] == [
        f"{DEVICE}/s1",
        f"{DEVICE}/s10",
    ]
    assert discovered[0][0].recording_id == f"robocap__{DEVICE}__s1"
    # The session is the sequence unit; its file-roll segments merge at convert.
    assert discovered[0][1] == RobocapSource(session_dir=corpus / f"{DEVICE}_session_1", device=DEVICE, session=1, segments=(1, 2))
    assert discovered[1][1].segments == (1,)
    # sequences() is derived from discover(), so the two can never disagree.
    assert dataset.sequences() == [identity for identity, _ in discovered]


def test_download_verifies_local_corpus(corpus: Path, tmp_path: Path) -> None:
    RobocapConfig(root=corpus).setup().download()
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        RobocapConfig(root=empty_root).setup().download()


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
