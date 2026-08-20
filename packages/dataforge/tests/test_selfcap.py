"""SelfCap discovery and raw-csv parsing against synthetic fixtures (no corpus needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from dataforge.datasets.selfcap import (
    CameraCalibration,
    HeadPoses,
    ImuSamples,
    SelfcapConfig,
    SelfcapDataset,
    exo_devices,
    read_calibration,
    read_frame_times_ns,
    read_head_poses,
    read_imu_csv,
    read_session_task,
)
from dataforge.identity import SequenceIdentity

SESSION_A: str = "25aeb66a-cf5a-44a3-845f-62606cefabe7"
"""Session uuid whose first eight characters become the identity's middle part."""
SESSION_B: str = "1dbe8c49-5f09-4b58-875e-b0a3315d9d4a"
"""Second session, on a different capture day."""
EXO_DEVICES: tuple[str, ...] = ("Dragon-413A7CC1", "Tiger-FE57138F", "Titanium-B00700F9", "Wolf-387E89E1")
"""Exo device directory names, which also form the video stems."""
UNREADABLE: int = 0o000
"""The mode the NAS sync leaves on much of the corpus."""


def make_episode(episode_dir: Path, *, unreadable: str | None = None) -> None:
    """Create every file the converter requires, optionally crippling one of them.

    Args:
        episode_dir: Target ``episodes/episode-<NNN>`` directory.
        unreadable: Path relative to ``episode_dir`` to chmod 000, mimicking the NAS sync.
    """
    (episode_dir / "ego").mkdir(parents=True)
    for stream in ("rgb", "left", "right"):
        (episode_dir / "ego" / f"{stream}.mp4").touch()
        (episode_dir / "ego" / f"{stream}.csv").write_text("ts_ns,frame_idx\n0,10\n")
    (episode_dir / "ego" / "imu.csv").touch()
    (episode_dir / "ego" / "calibration.json").touch()
    (episode_dir / "quest").mkdir()
    for stream in ("left", "right"):
        (episode_dir / "quest" / f"{stream}.mp4").touch()
    (episode_dir / "quest" / "head_pose.csv").touch()
    (episode_dir / "quest" / "calibration.json").touch()
    for device in EXO_DEVICES:
        (episode_dir / "exo" / device).mkdir(parents=True)
        (episode_dir / "exo" / device / f"{device}.mp4").touch()
        (episode_dir / "exo" / device / "calibration.json").touch()
    # The prior-art recording that dataforge deliberately rebuilds from raw.
    (episode_dir / f"{episode_dir.name}.rrd").touch()
    if unreadable is not None:
        (episode_dir / unreadable).chmod(UNREADABLE)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A fake SelfCap root: two days, one session each, one crippled episode."""
    session_a: Path = tmp_path / "cut" / "2025-12-20" / SESSION_A
    make_episode(session_a / "episodes" / "episode-001")
    make_episode(session_a / "episodes" / "episode-009")
    (session_a / "metadata.json").write_text(json.dumps({"task": "hotdog_in_ricecooker", "collector_name": "Adil", "location": {}}))

    session_b: Path = tmp_path / "cut" / "2025-12-21" / SESSION_B
    make_episode(session_b / "episodes" / "episode-002")
    make_episode(session_b / "episodes" / "episode-003", unreadable="quest/head_pose.csv")
    (session_b / "metadata.json").write_text("{}")

    # Loose files beside the day directories (the corpus keeps manifests there).
    (tmp_path / "cut" / "manifest.json").write_text("{}")
    return tmp_path


def test_sequences_skips_episodes_with_unreadable_files(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset: SelfcapDataset = SelfcapDataset(SelfcapConfig(root=corpus))
    identities: list[SequenceIdentity] = dataset.sequences()
    assert [identity.sequence_key for identity in identities] == [
        f"2025-12-20/{SESSION_A[:8]}/ep001",
        f"2025-12-20/{SESSION_A[:8]}/ep009",
        f"2025-12-21/{SESSION_B[:8]}/ep002",
    ]
    assert "skipped 1 episode" in capsys.readouterr().out


def test_identity_maps_to_recording_id_and_back_to_the_episode_directory(corpus: Path) -> None:
    dataset: SelfcapDataset = SelfcapDataset(SelfcapConfig(root=corpus))
    identity: SequenceIdentity = dataset.sequences()[1]
    assert identity.recording_id == f"selfcap__2025-12-20__{SESSION_A[:8]}__ep009"
    assert dataset._locate(identity) == corpus / "cut" / "2025-12-20" / SESSION_A / "episodes" / "episode-009"


def test_download_verifies_the_local_corpus(corpus: Path, tmp_path: Path) -> None:
    SelfcapConfig(root=corpus).setup().download()
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        SelfcapConfig(root=empty_root).setup().download()


def test_exo_devices_requires_video_and_calibration_in_one_directory(tmp_path: Path) -> None:
    make_episode(tmp_path / "episode-001")
    split_brain: Path = tmp_path / "episode-001" / "exo" / "multiCam-Ghost-1._multicam._tcp.local."
    split_brain.mkdir()
    (split_brain / f"{split_brain.name}.mp4").touch()  # no calibration.json beside it
    (tmp_path / "episode-001" / "exo" / EXO_DEVICES[0] / "calibration.json").chmod(UNREADABLE)
    assert exo_devices(tmp_path / "episode-001") == list(EXO_DEVICES[1:])


def test_session_task_degrades_to_empty_when_metadata_is_unreadable(corpus: Path) -> None:
    metadata_path: Path = corpus / "cut" / "2025-12-20" / SESSION_A / "metadata.json"
    assert read_session_task(metadata_path) == {"task": "hotdog_in_ricecooker", "collector": "Adil"}
    metadata_path.chmod(UNREADABLE)
    assert read_session_task(metadata_path) == {}
    assert read_session_task(corpus / "cut" / "2025-12-21" / SESSION_B / "metadata.json") == {}


def test_frame_times_come_from_ts_ns_in_row_order(tmp_path: Path) -> None:
    # frame_idx is a session-global counter, so it must not be used as a row index.
    csv_path: Path = tmp_path / "rgb.csv"
    csv_path.write_text("ts_ns,frame_idx\n0,2019\n41653000,2020\n83306000,2021\n")
    times_ns: Int64[ndarray, "3"] = read_frame_times_ns(csv_path)
    np.testing.assert_array_equal(times_ns, np.array([0, 41653000, 83306000], dtype=np.int64))
    assert times_ns.dtype == np.int64


def test_frame_times_of_a_header_only_csv_are_empty(tmp_path: Path) -> None:
    csv_path: Path = tmp_path / "rgb.csv"
    csv_path.write_text("ts_ns,frame_idx\n")
    assert read_frame_times_ns(csv_path).size == 0


def test_imu_csv_keeps_accel_and_gyro_and_drops_mag_and_rot(tmp_path: Path) -> None:
    csv_path: Path = tmp_path / "imu.csv"
    csv_path.write_text(
        "ts_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,rot_i,rot_j,rot_k\n"
        "0,0.71875,-4.95703125,-8.796875,-0.13671875,-0.15625,-0.07421875,19.25,10.1875,73.0625,-0.85,-0.45,0.068\n"
        "10035000,0.76953125,-5.01171875,-8.88671875,-0.111328125,-0.154296875,-0.0703125,20.3125,4.8125,69.75,-0.851,-0.455,0.068\n"
    )
    samples: ImuSamples = read_imu_csv(csv_path)
    np.testing.assert_array_equal(samples.times_ns, np.array([0, 10035000], dtype=np.int64))
    expected_accel: Float64[ndarray, "2 3"] = np.array([[0.71875, -4.95703125, -8.796875], [0.76953125, -5.01171875, -8.88671875]])
    np.testing.assert_allclose(samples.accel_xyz, expected_accel)
    expected_gyro: Float64[ndarray, "2 3"] = np.array([[-0.13671875, -0.15625, -0.07421875], [-0.111328125, -0.154296875, -0.0703125]])
    np.testing.assert_allclose(samples.gyro_xyz, expected_gyro)


def test_head_poses_take_the_left_eye_track_as_xyzw_quaternions(tmp_path: Path) -> None:
    csv_path: Path = tmp_path / "head_pose.csv"
    csv_path.write_text(
        "ts_ns,left_pos_x,left_pos_y,left_pos_z,left_quat_x,left_quat_y,left_quat_z,left_quat_w,"
        "right_pos_x,right_pos_y,right_pos_z,right_quat_x,right_quat_y,right_quat_z,right_quat_w\n"
        "0,-0.224171,1.953459,0.502586,0.448053,0.358166,-0.794444,0.199559,-0.257063,1.953081,0.448476,0.446841,0.35412,-0.793884,0.211378\n"
        "17181303,-0.224091,1.953531,0.502348,0.448075,0.358003,-0.794497,0.199592,-0.256979,1.95314,0.448235,0.44686,0.353958,-0.793937,0.211411\n"
    )
    poses: HeadPoses = read_head_poses(csv_path)
    np.testing.assert_array_equal(poses.times_ns, np.array([0, 17181303], dtype=np.int64))
    expected_translations: Float32[ndarray, "2 3"] = np.array([[-0.224171, 1.953459, 0.502586], [-0.224091, 1.953531, 0.502348]], dtype=np.float32)
    np.testing.assert_allclose(poses.translations_xyz, expected_translations)
    # xyzw, not the wxyz the csv header might suggest to a careless reader.
    np.testing.assert_allclose(poses.quaternions_xyzw[0], np.array([0.448053, 0.358166, -0.794444, 0.199559], dtype=np.float32))
    assert np.isclose(np.linalg.norm(poses.quaternions_xyzw, axis=-1), 1.0, atol=1e-3).all()


def test_calibration_keys_on_positional_layout_and_maps_14_coefficients(tmp_path: Path) -> None:
    coefficients: list[float] = [5.53, 1.66, 8.6e-05, 1.9e-04, 0.0259, 5.9, 3.36, 0.226, 0.0, 0.0, 0.0, 0.0, -0.00158, 0.0067]
    calibration_path: Path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "intrinsics": [
                    {
                        "camera_id": "CAM_C",
                        "positional_layout": "left",
                        "lens_intrinsics": {"focal_length_x": 572.2, "focal_length_y": 572.1, "principal_point_x": 648.4, "principal_point_y": 356.7},
                        "distortion": {"coefficients": coefficients, "available": True},
                        "capture_resolution": {"width": 1280, "height": 720},
                    },
                    {
                        "camera_id": "avcapturedevice:5",
                        "positional_layout": "center",
                        "lens_intrinsics": {"focal_length_x": 700.0, "focal_length_y": 700.0, "principal_point_x": 960.0, "principal_point_y": 540.0},
                        "distortion": {"available": False},
                        "capture_resolution": {"width": 1920, "height": 1080},
                    },
                    {"camera_id": "broken", "positional_layout": "right", "lens_intrinsics": {"available": False}},
                ],
                "extrinsics": [],
            }
        )
    )
    cameras: dict[str, CameraCalibration] = read_calibration(calibration_path)
    assert sorted(cameras) == ["center", "left"]  # the intrinsics-free entry is dropped
    left: CameraCalibration = cameras["left"]
    assert (left.width, left.height) == (1280, 720)
    np.testing.assert_allclose(left.focal_length_xy, np.array([572.2, 572.1]))
    np.testing.assert_allclose(left.principal_point_xy, np.array([648.4, 356.7]))
    assert left.distortion_coefficients is not None
    np.testing.assert_allclose(left.distortion_coefficients, np.asarray(coefficients))
    assert cameras["center"].distortion_coefficients is None


def test_default_blueprint_covers_nine_cameras() -> None:
    blueprint = SelfcapConfig().setup().default_blueprint()
    assert blueprint is not None
