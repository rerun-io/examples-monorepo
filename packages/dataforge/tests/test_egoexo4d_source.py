"""Ego-Exo4D raw readers against the public format samples (Project Aria's fixtures and the Ego-Exo4D docs)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge import aria
from dataforge.datasets.egoexo4d_source import (
    Take,
    localized,
    read_gopro_calibs,
    read_take_clock,
    read_takes,
    sample_frames,
    stored_size,
)

GOPRO_HEADER: str = (
    "cam_uid,graph_uid,tx_world_cam,ty_world_cam,tz_world_cam,qx_world_cam,qy_world_cam,qz_world_cam,qw_world_cam,image_width,image_height,"
    "intrinsics_type,intrinsics_0,intrinsics_1,intrinsics_2,intrinsics_3,intrinsics_4,intrinsics_5,intrinsics_6,intrinsics_7,"
    "start_frame_idx,end_frame_idx,quality"
)
# The row of projectaria_tools core/mps/test/TestStaticCameraCalibration.cpp (leading space kept), then a
# rejected camera and a portrait one built from it.
GOPRO_ROWS: list[str] = [
    " cam01,40084458-f219-16ca-0c3c-8bd94ee06e6c,1.691723,0.545741,-0.116511,-0.437056,-0.661141,0.501553,0.346871,3840,2160,"
    "KANNALABRANDTK3,1761.275757,1761.275757,1920.000000,1080.000000,0.036902,0.057131,-0.057154,0.018458,-1,-1,1",
    "cam02,40084458-f219-16ca-0c3c-8bd94ee06e6c,0,0,0,0,0,0,1,3840,2160,KANNALABRANDTK3,1700,1700,1920,1080,0.03,0.05,-0.05,0.01,-1,-1,0",
    "cam03,40084458-f219-16ca-0c3c-8bd94ee06e6c,0,0,2,0,0,0,1,2160,3840,KANNALABRANDTK3,1700,1700,1080,1920,0.03,0.05,-0.05,0.01,-1,-1,1",
]
# First rows of projectaria_tools data/gen1/mps_sample/trajectory/closed_loop_trajectory.csv, base columns.
TRAJECTORY: str = """graph_uid,tracking_timestamp_us,utc_timestamp_ns,tx_world_device,ty_world_device,tz_world_device,qx_world_device,qy_world_device,qz_world_device,qw_world_device,device_linear_velocity_x_device,device_linear_velocity_y_device,device_linear_velocity_z_device,angular_velocity_x_device,angular_velocity_y_device,angular_velocity_z_device,gravity_x_world,gravity_y_world,gravity_z_world,quality_score
ea7288b8-b6a0-012d-8156-257aba6bd796,149202610,1686767222002923694,0.000292,-0.006405,0.000467,0.500638545,0.496952726,-0.477400195,0.523916109,0.038913,-0.322935,0.242351,-0.054750,0.052271,-0.064212,0.000000,0.000000,-9.810000,0.5
ea7288b8-b6a0-012d-8156-257aba6bd796,149203459,1686767222002923694,0.000030,-0.006619,0.000427,0.500623062,0.496990395,-0.477393233,0.523901516,0.039788,-0.324063,0.243213,-0.050773,0.054919,-0.068246,0.000000,0.000000,-9.810000,0.5
"""


def take_entry(**overrides: object) -> dict[str, object]:
    """The docs' takes.json example, completed with the fields the converter reads."""
    entry: dict[str, object] = {
        "root_dir": "takes/cmu_bike01_2",
        "take_name": "cmu_bike01_2",
        "take_uid": "13f01c79-5bfd-42f5-90ce-ee350aa1c3ad",
        "capture_uid": "d37b73eb-fa42-43a6-8115-56832996ebd7",
        "timesync_start_idx": 2,
        "timesync_end_idx": 6,
        "duration_sec": 21.366666666666667,
        "task_name": "Remove a Wheel",
        "parent_task_name": "Bike Repair",
        "university_name": "cmu",
        "capture": {"capture_name": "cmu_bike01", "cameras": [{"cam_id": "cam01", "is_ego": False}, {"cam_id": "aria01", "is_ego": True}]},
        "frame_aligned_videos": {
            "cam01": {"0": {"relative_path": "frame_aligned_videos/cam01.mp4", "readable_stream_id": "0", "clip_uid": "x"}},
            "aria01": {"rgb": {"relative_path": "frame_aligned_videos/aria01_214-1.mp4", "readable_stream_id": "rgb"}},
            # The release's placeholder for the best exo view, a collage with no file (1,583 of 5,035 takes in v2).
            "best_exo": {"0": {"clip_uid": None, "stream_id": "0", "readable_stream_id": "0", "is_collage": True, "relative_path": None}},
        },
    }
    entry.update(overrides)
    return entry


def test_takes_json(tmp_path: Path) -> None:
    path: Path = tmp_path / "takes.json"
    path.write_text(json.dumps([take_entry()]))
    take: Take = read_takes(path)["cmu_bike01_2"]
    assert take.aria == "aria01"
    assert take.video("aria01", "rgb") == "takes/cmu_bike01_2/frame_aligned_videos/aria01_214-1.mp4"
    with pytest.raises(ValueError, match="slam-left"):
        take.video("aria01", "slam-left")
    with pytest.raises(ValueError, match="best_exo/0 ships no file"):
        take.video("best_exo", "0")


def test_exo_cameras_follow_the_capture_not_the_name(tmp_path: Path) -> None:
    # UPenn's layout: GoPros are gp01.., the head-mounted gp05 is ego, the phone ships no frame-aligned video.
    gopro: dict[str, object] = {"relative_path": "frame_aligned_videos/gp01.mp4", "readable_stream_id": "0"}
    cameras: list[dict[str, object]] = [
        {"cam_id": cam_id, "is_ego": cam_id in ("aria01", "gp05")} for cam_id in ("aria01", "gp05", "gp06", "gp01", "mobile")
    ]
    videos: dict[str, object] = {
        "aria01": {"rgb": {"relative_path": "frame_aligned_videos/aria01_214-1.mp4", "readable_stream_id": "rgb"}},
        **{cam_id: {"0": gopro} for cam_id in ("gp01", "gp05", "gp06")},
        "collage": {"0": {"relative_path": "ego_preview.mp4", "readable_stream_id": "0"}},
        "best_exo": {"0": {"relative_path": None, "readable_stream_id": "0"}},
    }
    take: Take = read_takes_entry(tmp_path, take_entry(capture={"capture_name": "upenn_0701", "cameras": cameras}, frame_aligned_videos=videos))
    assert take.exo_cameras == ("gp01", "gp06")
    assert read_takes_entry(tmp_path, take_entry()).exo_cameras == ("cam01",)


def test_sample_frames_finds_the_real_frame_of_each_three() -> None:
    # cmu_bike02_4's eye video: black, image, black, ... (luma peaks 0 and 239-255).
    assert sample_frames(np.array([0, 250, 0, 0, 239, 0, 0, 255], dtype=np.int64), 3, "et.mp4") == range(1, 8, 3)
    assert sample_frames(np.array([250, 0, 0, 250, 0], dtype=np.int64), 3, "et.mp4") == range(0, 4, 3)
    assert sample_frames(np.array([250, 240, 255], dtype=np.int64), 1, "rgb.mp4") == range(0, 3, 1)


def test_sample_frames_allows_a_late_start_and_an_early_stop() -> None:
    # uniandes_dance_016_46: eye images in frames 2..1649 of 1835, then the eye camera stops; black outside means no eye data.
    peaks = np.array([0, 0, 0, 0, 250, 0, 0, 250, 0, 0, 0, 0, 0], dtype=np.int64)
    assert sample_frames(peaks, 3, "et.mp4") == range(4, 8, 3)


@pytest.mark.parametrize(
    "peaks",
    [
        [0, 250, 0, 0, 0, 0, 0, 250, 0],  # a sample is missing between two images
        [0, 250, 250, 0, 250, 0],  # a padding frame carries an image
    ],
)
def test_sample_frames_refuses_an_irregular_padding(peaks: list[int]) -> None:
    with pytest.raises(ValueError, match="et.mp4: real frames are not exactly one in every 3"):
        sample_frames(np.array(peaks, dtype=np.int64), 3, "et.mp4")


def test_sample_frames_refuses_a_video_without_an_image() -> None:
    with pytest.raises(ValueError, match="et.mp4: no frame carries an image"):
        sample_frames(np.zeros(6, dtype=np.int64), 3, "et.mp4")


def test_gopro_calibs_keep_localized_cameras_in_file_order(tmp_path: Path) -> None:
    path: Path = tmp_path / "gopro_calibs.csv"
    path.write_text("\n".join([GOPRO_HEADER, *GOPRO_ROWS]) + "\n")
    calibs = localized(read_gopro_calibs(path))
    assert [calib.cam_uid for calib in calibs] == ["cam01", "cam03"]
    assert [stored_size(calib) for calib in calibs] == [(1920, 1080), (1080, 1920)]
    camera = calibs[0].camera(1920, 1080)
    assert camera.intrinsics.fl_x == pytest.approx(1761.275757 / 2)
    assert camera.intrinsics.cx == pytest.approx(960.0)
    assert camera.extrinsics.world_t_cam is not None
    np.testing.assert_allclose(camera.extrinsics.world_t_cam, [1.691723, 0.545741, -0.116511])


def test_kb4_projection_matches_opencv_fisheye(tmp_path: Path) -> None:
    """gopro_calibs' KANNALABRANDTK3 is OpenCV's fisheye model; simplecv's Fisheye62 with k5=k6=p=0 must agree."""
    path: Path = tmp_path / "gopro_calibs.csv"
    path.write_text("\n".join([GOPRO_HEADER, GOPRO_ROWS[0]]) + "\n")
    camera = read_gopro_calibs(path)[0].camera(1920, 1080)
    rng = np.random.default_rng(0)
    points: np.ndarray = np.column_stack([rng.uniform(-1.5, 1.5, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(0.5, 4.0, 200)])
    ours = project_fisheye62(points, camera)
    lens = camera.distortion
    assert lens is not None and camera.intrinsics.k_matrix is not None
    reference = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.intrinsics.k_matrix, np.array([lens.k1, lens.k2, lens.k3, lens.k4])
    )[0].reshape(-1, 2)
    inside = np.isfinite(ours).all(axis=1)
    assert inside.sum() > 100
    np.testing.assert_allclose(ours[inside], reference[inside], atol=1e-6)


def test_take_clock_reads_rows_start_to_end_and_fills_gaps(tmp_path: Path) -> None:
    path: Path = tmp_path / "timesync.csv"
    stamps = ["100", "133", "166", "", "233", "266", "300"]
    path.write_text("cam01_pts,aria01_214-1_capture_timestamp_ns\n" + "\n".join(f"{i},{stamp}" for i, stamp in enumerate(stamps)) + "\n")
    take = read_takes_entry(tmp_path, take_entry(timesync_start_idx=2, timesync_end_idx=6))
    clock = read_take_clock(path, take)
    np.testing.assert_array_equal(clock.times_ns, [166, 166, 233, 266])
    assert clock.filled == 1
    late = read_takes_entry(tmp_path, take_entry(timesync_start_idx=3, timesync_end_idx=5))
    with pytest.raises(ValueError, match="first frame"):
        read_take_clock(path, late)
    beyond = read_takes_entry(tmp_path, take_entry(timesync_start_idx=5, timesync_end_idx=9))
    with pytest.raises(ValueError, match="has 7"):
        read_take_clock(path, beyond)


def read_takes_entry(tmp_path: Path, entry: dict[str, object]) -> Take:
    path: Path = tmp_path / "takes.json"
    path.write_text(json.dumps([entry]))
    return read_takes(path)["cmu_bike01_2"]


def test_trajectory_reads_the_mps_sample(tmp_path: Path) -> None:
    """The shared MPS reader (``aria.read_trajectory``) on Project Aria's closed-loop sample; poses interpolate inside 2 ms."""
    path: Path = tmp_path / "closed_loop_trajectory.csv"
    path.write_text(TRAJECTORY)
    trajectory = aria.read_trajectory(path)
    np.testing.assert_array_equal(trajectory.times_ns, [149202610000, 149203459000])
    np.testing.assert_allclose(trajectory.poses[0, :3, 3], [0.000292, -0.006405, 0.000467])
    poses = trajectory.at(np.array([149202610000, 149203000000, 149203459000 + 6_000_000], dtype=np.int64))
    np.testing.assert_array_equal(poses[0], trajectory.poses[0])
    assert np.isfinite(poses[1]).all() and np.isnan(poses[2]).all()
