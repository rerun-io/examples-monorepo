"""basalt's calibration.json: the two camera models, the extrinsics, and the follow frame.

The literals below are trimmed from real MSD files, so what is asserted is what
upstream actually ships rather than what a fixture happens to make convenient.
"""

from __future__ import annotations

import numpy as np
import pytest
import serde.json
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Fisheye62Parameters,
    KannalaBrandtDistortion,
    PinholeParameters,
)

from dataforge.basalt import BasaltCalibration, BasaltPose, FollowFrame, camera_parameters, follow_frame

KB4_CALIBRATION: str = """
{"value0": {
  "T_imu_cam": [
    {"px": 0.069, "py": 0.121, "pz": 0.013, "qx": -0.0274, "qy": -0.0269, "qz": -0.7011, "qw": 0.7120},
    {"px": 0.069, "py": -0.013, "pz": 0.013, "qx": 0.0207, "qy": 0.0331, "qz": -0.6987, "qw": 0.7144}
  ],
  "intrinsics": [
    {"camera_type": "kb4", "intrinsics": {"fx": 420.5, "fy": 420.7, "cx": 469.5, "cy": 479.1,
      "k1": 0.193, "k2": 0.042, "k3": -0.233, "k4": 0.095}},
    {"camera_type": "kb4", "intrinsics": {"fx": 421.2, "fy": 421.5, "cx": 467.7, "cy": 484.5,
      "k1": 0.190, "k2": 0.055, "k3": -0.249, "k4": 0.103}}
  ],
  "resolution": [[960, 960], [960, 960]],
  "imu_update_rate": 1000.0,
  "cam_time_offset_ns": 0,
  "vignette": []
}}
"""
"""A two-camera kb4 calibration in basalt's format, trimmed from the Valve Index file."""

RADTAN8_CALIBRATION: str = """
{"value0": {
  "comment": "trimmed Reverb G2 calibration",
  "T_imu_cam": [
    {"px": -0.052, "py": 0.013, "pz": 0.006, "qx": -0.155, "qy": 0.034, "qz": 0.692, "qw": 0.704}
  ],
  "intrinsics": [
    {"camera_type": "pinhole-radtan8", "intrinsics": {"fx": 269.7, "fy": 269.8, "cx": 322.6, "cy": 228.9,
      "k1": 0.302, "k2": -0.021, "p1": -0.00025, "p2": 6.1e-05, "k3": 0.0158, "k4": 0.575, "k5": -0.063, "k6": 0.034,
      "rpmax": 2.7276}}
  ],
  "resolution": [[640, 480]]
}}
"""
"""A one-camera radtan8 calibration; ``rpmax`` and ``comment`` must be ignored, not rejected."""


def test_kb4_becomes_a_fisheye_with_its_four_radial_terms() -> None:
    calibration: BasaltCalibration = serde.json.from_json(BasaltCalibration, KB4_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 1, name="cam1")
    assert isinstance(camera, Fisheye62Parameters)
    assert isinstance(camera.distortion, KannalaBrandtDistortion)
    assert (camera.distortion.k1, camera.distortion.k2, camera.distortion.k3, camera.distortion.k4) == (0.190, 0.055, -0.249, 0.103)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (960, 960)
    assert camera.intrinsics.fl_x == 421.2


def test_radtan8_becomes_a_pinhole_with_brown_conrady_and_no_rpmax() -> None:
    calibration: BasaltCalibration = serde.json.from_json(BasaltCalibration, RADTAN8_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 0, name="cam0")
    assert isinstance(camera, PinholeParameters)
    assert isinstance(camera.distortion, BrownConradyDistortion)
    assert (camera.distortion.k1, camera.distortion.k2, camera.distortion.p1, camera.distortion.p2) == (0.302, -0.021, -0.00025, 6.1e-05)
    assert (camera.distortion.k3, camera.distortion.k4, camera.distortion.k5, camera.distortion.k6) == (0.0158, 0.575, -0.063, 0.034)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (640, 480)


def test_rpmax_is_read_off_a_radtan8_block_and_is_absent_from_a_kb4_one() -> None:
    """Only the rational model has a validity radius, so only radtan8 carries one."""
    radtan8: BasaltCalibration = serde.json.from_json(BasaltCalibration, RADTAN8_CALIBRATION)
    kb4: BasaltCalibration = serde.json.from_json(BasaltCalibration, KB4_CALIBRATION)
    assert radtan8.value0.intrinsics[0].intrinsics.rpmax == 2.7276
    assert kb4.value0.intrinsics[1].intrinsics.rpmax is None


def test_extrinsics_are_the_camera_pose_in_the_rig_frame() -> None:
    """``T_imu_cam`` is ``rig_T_cam``; the rig frame is the IMU frame."""
    calibration: BasaltCalibration = serde.json.from_json(BasaltCalibration, KB4_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 0, name="cam0")
    pose: BasaltPose = calibration.value0.T_imu_cam[0]
    expected_rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_R_cam: Float64[ndarray, "3 3"] = np.asarray(camera.extrinsics.world_R_cam, dtype=np.float64)
    rig_t_cam: Float64[ndarray, "3"] = np.asarray(camera.extrinsics.world_t_cam, dtype=np.float64)
    np.testing.assert_allclose(rig_R_cam, expected_rig_R_cam, atol=1e-12)
    np.testing.assert_allclose(rig_t_cam, [pose.px, pose.py, pose.pz], atol=1e-12)


# ── follow frame ──────────────────────────────────────────────────────────

UPRIGHT_PAIR_CALIBRATION: str = """
{"value0": {
  "T_imu_cam": [
    {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
    {"px": 1.0, "py": 0.0, "pz": 0.5, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
  ],
  "intrinsics": [
    {"camera_type": "kb4", "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5}},
    {"camera_type": "kb4", "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5}}
  ],
  "resolution": [[1, 1], [1, 1]]
}}
"""
"""Two unrotated cameras whose baseline is deliberately *not* perpendicular to them.

Unrotated means RDF as it comes: the optical axis is rig +z and the sensor's
right is rig +x. The 0.5 m of ``pz`` between the two is the point — a baseline
with a component along the optical axis, which only an orthogonalized frame
survives.
"""


def test_the_index_pair_looks_along_rig_z_and_calls_rig_minus_x_up() -> None:
    """Both answers are known outside this file, from the headset itself.

    The Index's cameras face where the wearer faces, and its 13.4 cm stereo
    baseline runs along rig y — so y is the lateral axis and the optical axis is
    rig +z. Up is then ±x, and MIO09's raw accelerometer mean (an accelerometer
    at rest reads *up*) picks -x, which is the measurement ``MSD_DEVICES``
    already records for this device.
    """
    calibration: BasaltCalibration = serde.json.from_json(BasaltCalibration, KB4_CALIBRATION)
    frame: FollowFrame = follow_frame(calibration)

    np.testing.assert_allclose(frame.forward, [0.0, 0.0, 1.0], atol=0.05)
    np.testing.assert_allclose(frame.up, [-1.0, 0.0, 0.0], atol=0.05)


def test_a_baseline_tilted_out_of_the_image_plane_still_yields_an_orthonormal_frame() -> None:
    """Gram-Schmidt: the baseline is a hint about the lateral axis, not the axis itself."""
    calibration: BasaltCalibration = serde.json.from_json(BasaltCalibration, UPRIGHT_PAIR_CALIBRATION)
    frame: FollowFrame = follow_frame(calibration)

    np.testing.assert_allclose(frame.forward, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(frame.up, [0.0, -1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("calibration_text", [KB4_CALIBRATION, UPRIGHT_PAIR_CALIBRATION])
def test_a_follow_frame_is_two_orthogonal_unit_vectors(calibration_text: str) -> None:
    frame: FollowFrame = follow_frame(serde.json.from_json(BasaltCalibration, calibration_text))
    assert np.linalg.norm(frame.forward) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(frame.up) == pytest.approx(1.0, abs=1e-9)
    assert float(np.dot(frame.forward, frame.up)) == pytest.approx(0.0, abs=1e-9)
