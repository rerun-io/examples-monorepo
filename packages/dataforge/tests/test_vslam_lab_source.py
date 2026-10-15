"""VSLAM-LAB's per-sequence layout: calibration.yaml, rgb.csv, groundtruth.csv and imu_0.csv, read through pyserde.

The literals below are cut from files VSLAM-LAB itself wrote (RGBDTUM freiburg1_xyz and EUROC MH_01_easy), so the
headers, the flow-mapping YAML and its trailing comments are the real ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets.vslam_lab_source import (
    Calibration,
    FrameIndex,
    ImuStream,
    StreamName,
    read_calibration,
    read_euroc_gt,
    read_frame_index,
    read_imu,
    read_poses,
)
from dataforge.euroc import GtTrajectory

TUM_CALIBRATION: str = """%YAML 1.2
---
cameras:
  - {cam_name: rgb_0,
     cam_type: rgb+depth,
     depth_name: depth_0,
     cam_model: pinhole,
     distortion_type: radtan5,
     focal_length: [517.306408, 516.469215],
     principal_point: [318.64304, 255.313989],
     distortion_coefficients: [0.262383, -0.953104, -0.005358, 0.002628, 1.163314],
     depth_factor: 5000.0,
     image_dimension: [640, 480],
     fps: 30.0,
     T_BS: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] # Sensor extrinsics wrt. the body-frame.
    }
"""

EUROC_CALIBRATION: str = """%YAML 1.2
---
cameras:
  - {cam_name: rgb_0,
     cam_type: gray,
     cam_model: pinhole,
     distortion_type: radtan4,
     focal_length: [458.654, 457.296],
     principal_point: [367.215, 248.375],
     distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
     image_dimension: [752, 480],
     fps: 20,
     T_BS: [0.0148655429818, -0.9998809296980, 0.0041402967942, -0.0216401454975, 0.9995572490080, 0.0149672133247, 0.0257155299480, -0.0646769867680, -0.0257744366974, 0.0037561883580, 0.9996607271780, 0.0098107305895, 0.0, 0.0, 0.0, 1.0] # Sensor extrinsics wrt. the body-frame.
    }

  - {cam_name: rgb_1,
     cam_type: gray,
     cam_model: pinhole,
     distortion_type: radtan4,
     focal_length: [457.587, 456.134],
     principal_point: [379.999, 255.238],
     distortion_coefficients: [-0.28368365, 0.07451284, -0.00010473, -3.555907e-05],
     image_dimension: [752, 480],
     fps: 20,
     T_BS: [0.0125552670891, -0.9997550997230, 0.0182237714554, -0.0198435579556, 0.9995987811510, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918, 0.0179005838253, 0.9995173470780, 0.0078621244704, 0.0, 0.0, 0.0, 1.0] # Sensor extrinsics wrt. the body-frame.
    }


imus:
  - {imu_name: imu_0,
     a_max: 176.0, # acceleration saturation [m/s^2]
     sigma_g_c: 2.000000e-03, # gyro noise density ( gyro "white noise" ) [rad/s/sqrt(Hz)]
     sigma_gw_c: 2.000000e-04, # gyro drift noise density ( gyro bias diffusion ) [rad/s^2/sqrt(Hz)]
     sigma_a_c: 2.000000e-02, # accelerometer noise density ( accel "white noise" ) [m/s^2/sqrt(Hz)]
     sigma_aw_c: 2.000000e-02, # accelerometer drift noise density ( accel bias diffusion ) [m/s^3/sqrt(Hz)]
     a0: [-0.05, 0.09, 0.01], # initial accelerometer bias [m/s^2]
     fps: 200.0,
     T_BS: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Sensor extrinsics wrt. the body-frame.
    }
"""

TUM_RGB_CSV: str = """ts_rgb_0 (ns),path_rgb_0,ts_depth_0 (ns),path_depth_0
1305031102175303936,rgb_0/1305031102.175304.png,1305031102175303936,depth_0/1305031102.160407.png
1305031102211214080,rgb_0/1305031102.211214.png,1305031102211214080,depth_0/1305031102.226738.png
"""

IMU_HEADER: str = "ts (ns),wx (rad s^-1),wy (rad s^-1),wz (rad s^-1),ax (m s^-2),ay (m s^-2),az (m s^-2)"
"""imu_0.csv's header, as VSLAM-LAB writes it."""

EUROC_RGB_CSV: str = """ts_rgb_0 (ns),path_rgb_0,ts_rgb_1 (ns),path_rgb_1
1403636579763555584,rgb_0/1403636579763555584.png,1403636579763555584,rgb_1/1403636579763555584.png
1403636579813555457,rgb_0/1403636579813555456.png,1403636579813555457,rgb_1/1403636579813555456.png
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path: Path = tmp_path / name
    path.write_text(text)
    return path


def test_a_tum_camera_carries_its_depth_stream_and_meter_scale(tmp_path: Path) -> None:
    calibration: Calibration = read_calibration(write(tmp_path, "calibration.yaml", TUM_CALIBRATION))
    assert [camera.cam_name for camera in calibration.cameras] == ["rgb_0"]
    camera = calibration.cameras[0]
    assert camera.depth_name == "depth_0"
    assert camera.depth_factor == 5000.0
    assert camera.image_dimension == (640, 480)
    assert camera.distortion_coefficients == (0.262383, -0.953104, -0.005358, 0.002628, 1.163314)
    np.testing.assert_array_equal(camera.body_T_sensor, np.eye(4))
    assert calibration.imus == ()


def test_an_euroc_rig_reads_both_cameras_extrinsics_and_the_imu_noise_terms(tmp_path: Path) -> None:
    calibration: Calibration = read_calibration(write(tmp_path, "calibration.yaml", EUROC_CALIBRATION))
    assert [camera.cam_name for camera in calibration.cameras] == ["rgb_0", "rgb_1"]
    # cam0's optical axis is the body's +z and its image-up the body's +x: the MAV's IMU frame.
    np.testing.assert_allclose(calibration.cameras[0].body_T_sensor[:3, 2], [0.0041402967942, 0.0257155299480, 0.9996607271780])
    np.testing.assert_allclose(calibration.cameras[1].body_T_sensor[:3, 3], [-0.0198435579556, 0.0453689425024, 0.0078621244704])
    imu = calibration.imus[0]
    assert (imu.imu_name, imu.fps, imu.sigma_g_c, imu.sigma_a_c, imu.sigma_gw_c, imu.sigma_aw_c) == ("imu_0", 200.0, 2e-3, 2e-2, 2e-4, 2e-2)


def test_a_t_bs_that_is_not_sixteen_numbers_names_the_file(tmp_path: Path) -> None:
    path: Path = write(tmp_path, "calibration.yaml", TUM_CALIBRATION.replace("0.0, 0.0, 0.0, 1.0] #", "0.0, 0.0, 1.0] #"))
    with pytest.raises(ValueError, match="calibration.yaml"):
        read_calibration(path)


def test_tum_depth_rides_the_rgb_stamp_vslam_lab_associated_it_with(tmp_path: Path) -> None:
    streams: dict[StreamName, FrameIndex] = read_frame_index(write(tmp_path, "rgb.csv", TUM_RGB_CSV))
    assert sorted(streams) == ["depth_0", "rgb_0"]
    np.testing.assert_array_equal(streams["rgb_0"].times_ns, [1305031102175303936, 1305031102211214080])
    np.testing.assert_array_equal(streams["depth_0"].times_ns, streams["rgb_0"].times_ns)
    assert streams["depth_0"].paths == ("depth_0/1305031102.160407.png", "depth_0/1305031102.226738.png")


def test_euroc_stereo_lists_both_cameras_on_their_own_clocks(tmp_path: Path) -> None:
    streams: dict[StreamName, FrameIndex] = read_frame_index(write(tmp_path, "rgb.csv", EUROC_RGB_CSV))
    assert sorted(streams) == ["rgb_0", "rgb_1"]
    assert streams["rgb_1"].paths[1] == "rgb_1/1403636579813555456.png"
    # Every stamp survives exactly: 1.4e18 is beyond float64's integer range.
    assert int(streams["rgb_0"].times_ns[1]) == 1403636579813555457


@pytest.mark.parametrize("last_row", ["1403636579863555584,rgb_0/c.png", "1403636579863555584,rgb_0/c.png,1403636579863555584,"])
def test_a_stream_the_header_lists_but_a_row_lacks_names_rgb_csv_and_the_stream(tmp_path: Path, last_row: str) -> None:
    with pytest.raises(ValueError, match="rgb.csv: rgb_1"):
        read_frame_index(write(tmp_path, "rgb.csv", f"{EUROC_RGB_CSV}{last_row}\n"))


def test_a_stamp_written_as_a_float_literal_is_read_exactly(tmp_path: Path) -> None:
    streams: dict[StreamName, FrameIndex] = read_frame_index(write(tmp_path, "rgb.csv", "ts_rgb_0 (ns),path_rgb_0\n1403636579813555457.0,rgb_0/a.png\n"))
    assert streams["rgb_0"].times_ns.tolist() == [1403636579813555457]


def test_ground_truth_comes_back_world_t_body_scalar_last(tmp_path: Path) -> None:
    poses: GtTrajectory = read_poses(
        write(
            tmp_path,
            "groundtruth.csv",
            "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n"
            "1403636580838555648,4.688319,-1.786938,0.783338,-0.153029,-0.827383,-0.082152,0.534108\n"
            "1403636580843555328,4.688177,-1.786770,0.787350,-0.152990,-0.826976,-0.082863,0.534640\n",
        )
    )
    assert poses.times_ns.tolist() == [1403636580838555648, 1403636580843555328]
    np.testing.assert_array_equal(poses.translations_xyz[0], [4.688319, -1.786938, 0.783338])
    np.testing.assert_array_equal(poses.quaternions_xyzw[1], [-0.152990, -0.826976, -0.082863, 0.534640])


def test_euroc_s_own_ground_truth_comes_back_world_t_imu0_scalar_last(tmp_path: Path) -> None:
    poses: GtTrajectory = read_euroc_gt(
        write(
            tmp_path,
            "state_groundtruth_estimate0.csv",
            "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1]\n"
            "1403636580838555648,4.688319,-1.786938,0.783338,0.534108,-0.153029,-0.827383,-0.082152,-0.027876\n",
        )
    )
    assert poses.times_ns.tolist() == [1403636580838555648]
    np.testing.assert_array_equal(poses.translations_xyz, [[4.688319, -1.786938, 0.783338]])
    np.testing.assert_array_equal(poses.quaternions_xyzw, [[-0.153029, -0.827383, -0.082152, 0.534108]])


def test_an_empty_euroc_ground_truth_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="state_groundtruth_estimate0.csv"):
        read_euroc_gt(write(tmp_path, "state_groundtruth_estimate0.csv", "#timestamp, p_RS_R_x [m]\n"))


def test_an_imu_csv_splits_into_gyro_and_accel(tmp_path: Path) -> None:
    imu: ImuStream = read_imu(
        write(
            tmp_path,
            "imu_0.csv",
            f"{IMU_HEADER}\n1403636579758555392,-0.0991,0.1473,0.0272,8.1476,-0.3759,-2.4026\n",
        )
    )
    assert imu.gyro.times_ns.tolist() == imu.accel.times_ns.tolist() == [1403636579758555392]
    np.testing.assert_array_equal(imu.gyro.values_xyz, [[-0.0991, 0.1473, 0.0272]])
    np.testing.assert_array_equal(imu.accel.values_xyz, [[8.1476, -0.3759, -2.4026]])


def test_imu_and_gt_stamps_written_as_float_literals_are_read_exactly(tmp_path: Path) -> None:
    imu: ImuStream = read_imu(write(tmp_path, "imu_0.csv", f"{IMU_HEADER}\n1403636579758555393.0,0,0,0,0,0,9.8\n"))
    poses: GtTrajectory = read_poses(write(tmp_path, "groundtruth.csv", "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n1403636580838555649.0,1,2,3,0,0,0,1\n"))
    assert imu.gyro.times_ns.tolist() == [1403636579758555393]
    assert poses.times_ns.tolist() == [1403636580838555649]
    np.testing.assert_array_equal(poses.translations_xyz, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("body", ["ts (ns),wx (rad s^-1)\n1,0.1\n", f"{IMU_HEADER}\n1,0.1,,0.3,9.8,0,0\n"])
def test_an_imu_csv_with_a_missing_column_or_cell_names_the_file(tmp_path: Path, body: str) -> None:
    with pytest.raises(ValueError, match="imu_0.csv"):
        read_imu(write(tmp_path, "imu_0.csv", body))
