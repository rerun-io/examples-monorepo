"""Synthetic RoboCap factory calibration folders for the robocap test modules.

Not a test module: ``test_robocap`` and ``test_robocap_calibration_backfill`` both write a cap's Kalibr camchain files
and IMU noise model with it.
"""

from __future__ import annotations

from pathlib import Path


def write_camchain(path: Path, timeshifts_s: list[float | None]) -> None:
    """One Kalibr camchain-imucam file with a camera per shift (``None`` leaves ``timeshift_cam_imu`` out)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for index, shift in enumerate(timeshifts_s):
        entries.append(
            f"cam{index}:\n  camera_model: pinhole\n  distortion_model: equidistant\n  intrinsics: [300.0, 300.0, 320.0, 240.0]\n"
            "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n  resolution: [640, 480]\n"
            "  T_cam_imu: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]\n"
            + ("" if shift is None else f"  timeshift_cam_imu: {shift!r}\n")
        )
    path.write_text("".join(entries))


CAP_B_SLAM_SHIFTS_S: tuple[float, float, float, float] = (0.0008661293958066206, 0.0009766584965254315, 0.001591904372869934, 0.0014493197885760412)
"""Cap B's factory ``timeshift_cam_imu`` for left_front, right_front, left, right (fe6fede545c972fa)."""


def write_factory(root: Path, device: str, slam_shifts_s: tuple[float | None, ...] = CAP_B_SLAM_SHIFTS_S) -> None:
    """A factory calibration folder with the IMU noise model and the four SLAM cameras (plus an eye pair at 2 ms)."""
    factory: Path = root / f"0factory-calibration-{device}"
    left_front, right_front, left, right = slam_shifts_s
    write_camchain(factory / "imus_cam_lr_front_extrinsic/front-camchain-imucam.yaml", [left_front, right_front])
    write_camchain(factory / "imus_cam_lr_eye_extrinsic/eye-camchain-imucam.yaml", [0.002, 0.002])
    write_camchain(factory / "imus_cam_l_extrinsic/l-camchain-imucam.yaml", [left])
    write_camchain(factory / "imus_cam_r_extrinsic/r-camchain-imucam.yaml", [right])
    noise: Path = factory / "imus_intrinsic/imu_mid_0.yaml"
    noise.parent.mkdir(parents=True)
    noise.write_text("gyroscope_noise_density: 0.0007\naccelerometer_noise_density: 0.006\n"
                     "gyroscope_random_walk: 0.00003\naccelerometer_random_walk: 0.0002\nupdate_rate: 200.0\n")
