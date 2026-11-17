"""tartanair: the download of one trajectory's zip members, and the NED -> RDF conversions of the converter.

The unit tests use in-memory zips on the fake Hub and hand-made poses and IMU samples. The integration test converts the first
2 s of a downloaded trajectory and reads the layers back the way a consumer does; it skips naming the trajectory
when the raw data is not on this machine.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import zip_bytes  # pyrefly: ignore[missing-import]
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from dataforge.datasets.tartanair import CameraSet


def test_download_lands_only_the_trajectory_members_from_the_pinned_zips_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from hub_fake import HubStore  # pyrefly: ignore[missing-import]

    from dataforge.datasets import tartanair
    from dataforge.datasets.tartanair import CAMERA_SETS, HF_REPO, HF_REVISION, TartanAirConfig

    hub = HubStore({HF_REPO: HF_REVISION})
    archives: list[str] = ["imu", *(f"image_{camera.name}" for cameras in CAMERA_SETS.values() for camera in cameras)]
    for archive in archives:
        members: dict[str, bytes] = {f"Env/Data_easy/{number}/{archive}/{number}.txt": number.encode() * 100 for number in ("P000", "P001")}
        hub.add(HF_REPO, f"Env/Data_easy/{archive}.zip", zip_bytes(members), lfs=True)
    hub.install(monkeypatch, tartanair)
    config = TartanAirConfig(root=tmp_path, trajectories=("Env/easy/P000",))
    config.setup().download()
    landed: list[str] = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.txt"))
    assert landed == sorted(f"Env/Data_easy/P000/{archive}/P000.txt" for archive in archives)
    assert [call.paths[0] for call in hub.calls] == [f"Env/Data_easy/{archive}.zip" for archive in archives]
    assert capsys.readouterr().out.rstrip().endswith("0 already complete")
    config.setup().download()
    assert capsys.readouterr().out.rstrip().endswith(f"; {len(archives)} already complete")
    with pytest.raises(FileNotFoundError, match="no member under Env/Data_easy/P002/"):
        TartanAirConfig(root=tmp_path, trajectories=("Env/easy/P002",)).setup().download()


def test_rig_pose_of_a_north_east_down_pose_in_the_z_up_world() -> None:
    from dataforge.datasets.tartanair import rig_poses

    # Facing east (yaw +90 deg about NED down), 1 m north, 2 m east, 3 m below the origin.
    quaternion_xyzw: Float64[ndarray, "4"] = Rotation.from_euler("z", 90.0, degrees=True).as_quat()
    translations, quaternions = rig_poses(np.array([[1.0, 2.0, 3.0, *quaternion_xyzw]]))
    np.testing.assert_allclose(translations, [[1.0, -2.0, -3.0]])
    world_R_rig: Float64[ndarray, "3 3"] = Rotation.from_quat(quaternions[0]).as_matrix()
    np.testing.assert_allclose(world_R_rig @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], atol=1e-12)  # rig forward (RDF +z) = east = world -y
    np.testing.assert_allclose(world_R_rig @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)  # rig down (RDF +y) = world -z


def test_imu_at_rest_points_up_and_a_right_turn_is_about_rig_down() -> None:
    from dataforge.datasets.tartanair import imu_in_rig, rig_poses

    accel_rig: Float64[ndarray, "1 3"] = imu_in_rig(np.array([[0.0, 0.0, -9.8]]))  # NED specific force of a level body at rest
    np.testing.assert_allclose(accel_rig, [[0.0, -9.8, 0.0]])  # rig up is RDF -y
    _, quaternions = rig_poses(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(Rotation.from_quat(quaternions[0]).as_matrix() @ accel_rig[0], [0.0, 0.0, 9.8], atol=1e-12)
    np.testing.assert_allclose(imu_in_rig(np.array([[0.0, 0.0, 1.0]])), [[0.0, 1.0, 0.0]])  # yaw right = about NED down = RDF +y


@pytest.mark.parametrize(
    ("camera_set", "names", "size", "focal", "centre", "model", "coefficients"),
    [
        ("pinhole", ("lcam_front", "rcam_front"), (640, 640), (320.0, 320.0), (319.5, 319.5), "brown_conrady", (0.0,) * 14),
        ("fisheye", ("lcam_fish", "rcam_fish"), (1000, 1000), (1000 / math.radians(195),) * 2, (499.5, 499.5), "kannala_brandt", (0.0,) * 8),
        ("equirect", ("lcam_equirect",), (2048, 1024), (1024 / math.pi, 1024 / math.pi), (1023.5, 511.5), "equirectangular", (math.pi,)),
    ],
)
def test_camera_sets(
    camera_set: CameraSet,
    names: tuple[str, ...],
    size: tuple[int, int],
    focal: tuple[float, float],
    centre: tuple[float, float],
    model: str,
    coefficients: tuple[float, ...],
) -> None:
    from dataforge.datasets.tartanair import CAMERA_SETS

    cameras = CAMERA_SETS[camera_set]
    assert tuple(camera.name for camera in cameras) == names
    for index, camera in enumerate(cameras):
        assert (camera.width, camera.height, camera.model, camera.coefficients) == (*size, model, coefficients)
        np.testing.assert_allclose((camera.fx, camera.fy, camera.cx, camera.cy), (*focal, *centre))
        np.testing.assert_allclose(camera.rig_t_cam, [0.25 * index, 0.0, 0.0])  # the right camera sits 0.25 m to the right


def test_equirect_image_centre_looks_left_and_front_is_three_quarters_across() -> None:
    from dataforge.datasets.tartanair import CAMERA_SETS

    camera = CAMERA_SETS["equirect"][0]
    np.testing.assert_allclose(camera.rig_R_cam @ [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], atol=1e-12)  # optical axis = rig left
    front_in_cam: Float64[ndarray, "3"] = camera.rig_R_cam.T @ [0.0, 0.0, 1.0]
    longitude: float = math.atan2(front_in_cam[0], front_in_cam[2])
    assert camera.cx + camera.fx * longitude == pytest.approx(3 * 2048 / 4 - 0.5)


@pytest.mark.parametrize(
    ("num_frames", "num_imu", "frame_limit", "expected_frames", "expected_imu"),
    [
        (91, 900, 20, np.arange(1, 21), (10, 202)),  # the review's case: 2.00 s is the last frame, 2.01 s the last sample
        (91, 900, None, np.arange(1, 89), (10, 890)),  # 8.89 s is the last sample, after the 8.8 s frame
        (91, 901, None, np.arange(1, 89), (10, 891)),  # the last trimmed sample lands on 8.9 s: that frame goes
        (91, 900, 1, np.array([1]), (10, 12)),  # one frame: 0.10 s, with 0.11 s after it
        (3, 31, None, np.array([1]), (10, 21)),  # the shortest source that keeps a frame: samples 0.10 .. 0.20 s
    ],
)
def test_every_kept_frame_has_an_imu_sample_strictly_after_it(
    num_frames: int, num_imu: int, frame_limit: int | None, expected_frames: Int64[ndarray, "k"], expected_imu: tuple[int, int]
) -> None:
    from dataforge.datasets.tartanair import FRAME_PERIOD_NS, IMU_PERIOD_NS, IMU_TRIM, aligned_window

    frames, imu = aligned_window(num_frames=num_frames, num_imu=num_imu, frame_limit=frame_limit)
    np.testing.assert_array_equal(frames, expected_frames)
    assert (imu.start, imu.stop) == expected_imu
    assert imu.start >= IMU_TRIM and imu.stop <= num_imu - IMU_TRIM  # inside the trimmed source
    assert imu.start * IMU_PERIOD_NS <= frames[0] * FRAME_PERIOD_NS  # a sample at or before the first frame
    assert (imu.stop - 1) * IMU_PERIOD_NS > frames[-1] * FRAME_PERIOD_NS  # and one strictly after the last


@pytest.mark.parametrize(
    ("num_frames", "num_imu", "frame_limit", "message"),
    [
        (0, 900, None, "no frame of pose_lcam_front.txt"),
        (91, 0, None, "no frame of pose_lcam_front.txt"),
        (91, 15, None, "no frame of pose_lcam_front.txt"),  # 15 samples keep none after the 10-sample trims
        (91, 21, None, "no frame of pose_lcam_front.txt"),  # one kept sample, on frame 1: no sample after it
        (91, 900, 0, "--frame-limit 0"),
    ],
)
def test_an_empty_or_too_short_trajectory_is_refused_with_a_sentence(num_frames: int, num_imu: int, frame_limit: int | None, message: str) -> None:
    from dataforge.datasets.tartanair import aligned_window
    with pytest.raises(ValueError, match=message):
        aligned_window(num_frames=num_frames, num_imu=num_imu, frame_limit=frame_limit)


def _imu_files(directory: Path, samples: int, gyro_rows: int, acc_rows: int) -> Path:
    directory.mkdir()
    np.savetxt(directory / "imu_time.txt", np.arange(samples) * 0.01)
    np.savetxt(directory / "gyro.txt", np.zeros((gyro_rows, 3)))
    np.savetxt(directory / "acc.txt", np.zeros((acc_rows, 3)))
    return directory


def test_the_imu_files_must_hold_one_row_per_sample(tmp_path: Path) -> None:
    from dataforge.datasets.tartanair import read_imu

    gyro, accel = read_imu(_imu_files(tmp_path / "whole", 30, 30, 30))
    assert gyro.shape == accel.shape == (30, 3)
    with pytest.raises(ValueError, match=r"gyro\.txt holds an array of shape \(29, 3\)"):
        read_imu(_imu_files(tmp_path / "short_gyro", 30, 29, 30))
    with pytest.raises(ValueError, match=r"acc\.txt holds an array of shape \(31, 3\)"):
        read_imu(_imu_files(tmp_path / "long_acc", 30, 30, 31))


@pytest.mark.integration
@pytest.mark.parametrize("camera_set", ["pinhole", "fisheye", "equirect"])
def test_two_seconds_of_a_real_trajectory_agree_across_gt_gyro_and_frames(
    camera_set: CameraSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    import rerun.chunk as rrc
    from conftest import column_rows, raw_asset, read_back, recording_properties, static_values  # pyrefly: ignore[missing-import]

    from dataforge import paths, schema
    from dataforge.datasets.tartanair import CAMERA_SETS, TartanAirConfig, TartanAirDataset

    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    root: Path = TartanAirConfig().root
    raw_asset("TartanAir V2 ArchVizTinyHouseDay easy P006 (dataforge-download tartanair)", root / "ArchVizTinyHouseDay/Data_easy/P006/pose_lcam_front.txt")
    config = TartanAirConfig(root=root, camera_set=camera_set, trajectories=("ArchVizTinyHouseDay/easy/P006",), frame_limit=20)
    dataset: TartanAirDataset = config.setup()  # pyrefly: ignore[bad-assignment]
    ((identity, source),) = dataset.discover()
    base: rrc.ChunkStore = read_back(dataset.convert(identity, source, force=False))
    gt: rrc.ChunkStore = read_back(dataset.targets(identity)[paths.GT_LAYER])

    rig: dict[str, object] = static_values(base, schema.rig_path(0))
    assert rig["reference"] == "imu_00"
    imu_node: dict[str, object] = static_values(base, schema.imu_path(0, 0))
    assert imu_node["applied_time_shift_ns"] == 0
    for name in ("gyro_noise_density", "accel_noise_density", "gyro_bias_random_walk", "accel_bias_random_walk", "rate_hz"):
        assert imu_node[name] > 0.0  # pyrefly: ignore[unsupported-operation]
    assert recording_properties(base, "capture")["start_time_ns"] == 0
    frame_ns: list[int] = []
    for index, camera in enumerate(CAMERA_SETS[camera_set]):
        pinhole: dict[str, object] = static_values(base, schema.pinhole_path(0, index))
        assert pinhole["simplecv.components.DistortionModel"] == camera.model
        assert pinhole["resolution"] == [camera.width, camera.height]
        samples = column_rows(base, f"{schema.video_path(0, index)}:VideoStream:sample")
        frame_ns: list[int] = [stamp.value for stamp in samples[schema.TIMELINE]]
        assert frame_ns == [100_000_000 * k for k in range(1, 21)]  # the IMU's first kept sample is at 0.10 s

    poses = column_rows(gt, f"{schema.rig_path(0)}:Transform3D:quaternion")
    assert [stamp.value for stamp in poses[schema.TIMELINE]] == frame_ns
    world_R_rig: Rotation = Rotation.from_quat(np.array(poses[1].to_pylist()).reshape(-1, 4))
    gyro = column_rows(base, f"{schema.gyro_path(0, 0)}:Scalars:scalars")
    gyro_ns: list[int] = [stamp.value for stamp in gyro[schema.TIMELINE]]
    rates: Float64[ndarray, "n 3"] = np.array(gyro[1].to_pylist()).reshape(-1, 3)
    assert gyro_ns[0] == frame_ns[0] and gyro_ns[-1] == frame_ns[-1] + 10_000_000  # one sample strictly after the last frame
    # Body rates in the rig frame integrate (right-multiplied, trapezoid over 0.01 s steps) to the GT relative rotation.
    for first, last in ((0, 10), (10, 19), (0, 19)):
        integrated: Rotation = Rotation.identity()
        for j in range(gyro_ns.index(frame_ns[first]), gyro_ns.index(frame_ns[last])):
            integrated = integrated * Rotation.from_rotvec(0.5 * (rates[j] + rates[j + 1]) * 0.01)
        relative: Rotation = world_R_rig[first].inv() * world_R_rig[last]
        assert np.degrees((integrated.inv() * relative).magnitude()) < 0.5
