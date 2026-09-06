"""LaMAria's file readers: the official calibration JSON, the pGT, the control points.

Every fixture under ``reference_data/lamaria/`` is a verbatim (or trimmed)
excerpt of a published file, so these run without the 897 MB VRS. The readers
that need a real VRS are exercised in ``test_aria_vrs.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from dataforge import aria

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "lamaria"

# ``imuR_T_device @ device_T_cam`` for camera-slam-left, read out of R_01_easy's VRS
# device calibration — an independent derivation of the same transform the
# calibration JSON publishes as ``cam0.T_b_s``.
CAM0_RIG_R_CAM: Float64[ndarray, "3 3"] = np.array(
    [
        [0.980111788, 0.197635468, 0.017919401],
        [0.067107635, -0.245107143, -0.967170644],
        [-0.186755050, 0.949137877, -0.253495249],
    ]
)

# The pGT's first pose, with its quaternion read as x, y, z, w. Reading the same
# four numbers as w, x, y, z yields a rotation that differs in every entry, so
# this literal pins the file's convention rather than the reader's arithmetic.
PGT_WORLD_R_CAM0: Float64[ndarray, "3 3"] = np.array(
    [
        [0.150924979, 0.055771294, 0.986970726],
        [-0.069472690, 0.996537067, -0.045688283],
        [-0.986101007, -0.061672008, 0.154276918],
    ]
)


# ── the official aria_calibrations JSON ───────────────────────────────────


def test_read_calibration_json_reads_both_slam_cameras() -> None:
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(REFERENCE_DIR / "R_01_easy.calibration.json")
    # ``imu0`` is deliberately absent: it is the body frame, so its T_b_s is the
    # identity by definition and its entry carries noise densities, not a camera.
    assert sorted(published) == ["cam0", "cam1"]
    cam0: aria.PublishedCamera = published["cam0"]
    assert cam0.model == "RAD_TAN_THIN_PRISM_FISHEYE"
    assert (cam0.resolution.width, cam0.resolution.height) == (640, 480)
    assert len(cam0.params) == 16, "the published layout is [fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]"
    assert cam0.params[:4] == pytest.approx([241.6048925567284, 241.6048925567284, 322.895555771704, 240.4445599560633])


def test_published_rig_T_cam_reads_the_quaternion_as_xyzw() -> None:
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(REFERENCE_DIR / "R_01_easy.calibration.json")
    rig_T_cam: Float64[ndarray, "4 4"] = published["cam0"].rig_T_cam.to_matrix()
    assert rig_T_cam[:3, 3] == pytest.approx([0.01678296927310889, -0.10892921838506506, 0.07605583988186104], abs=1e-12)
    assert rig_T_cam[:3, :3] == pytest.approx(CAM0_RIG_R_CAM, abs=1e-6)
    assert rig_T_cam[3].tolist() == [0.0, 0.0, 0.0, 1.0]


# ── pseudo ground truth ──────────────────────────────────────────────────


def test_read_pseudo_gt_keeps_every_row_in_file_order() -> None:
    pgt: aria.PseudoGt = aria.read_pseudo_gt(REFERENCE_DIR / "R_01_easy.pseudo_gt.txt")
    # The fixture is R_01_easy's first three rows and its last two.
    assert pgt.times_ns.tolist() == [1389350666375, 1389400666375, 1389450666375, 1534150666375, 1534200666375]
    assert pgt.world_T_cam0.shape == (5, 4, 4)
    # Rows 0-2 are the device sitting still, and the corpus repeats the pose
    # verbatim rather than dropping it; a reader must not deduplicate.
    assert np.array_equal(pgt.world_T_cam0[0], pgt.world_T_cam0[2])
    assert pgt.world_T_cam0[0, :3, 3] == pytest.approx([0.027444266686277308, -0.19185417745945016, -0.2505208413120663], abs=1e-12)
    assert pgt.world_T_cam0[0, :3, :3] == pytest.approx(PGT_WORLD_R_CAM0, abs=1e-6)
    assert np.array_equal(pgt.world_T_cam0[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)))


# ── control points ───────────────────────────────────────────────────────


def test_read_control_points_translates_measurements_to_the_custom_origin() -> None:
    points: dict[str, aria.ControlPoint] = {
        point.name: point for point in aria.read_control_points(REFERENCE_DIR / "R_11_5cp.control_points.json").points
    }
    assert sorted(points) == ["OB1878", "OB1881"]
    surveyed: aria.ControlPoint = points["OB1878"]
    assert surveyed.has_height
    # LV95/LN02 minus CUSTOM_ORIGIN_XYZ, so the scene sits near the origin.
    assert surveyed.position_xyz_m == pytest.approx([351.07500000018626, 348.1920000030659, 44.89299999999997], abs=1e-6)
    assert surveyed.uncertainty_xyz_m == pytest.approx([0.01, 0.01, 0.029926112705422614], abs=1e-12)


def test_read_control_points_marks_an_unknown_height() -> None:
    points: dict[str, aria.ControlPoint] = {
        point.name: point for point in aria.read_control_points(REFERENCE_DIR / "R_11_5cp.control_points.json").points
    }
    unlevelled: aria.ControlPoint = points["OB1881"]
    assert not unlevelled.has_height
    assert unlevelled.position_xyz_m[:2] == pytest.approx([347.59700000006706, 337.6929999976419], abs=1e-6)
    assert unlevelled.position_xyz_m[2] == 0.0, "an unknown height is drawn at the origin's level, not guessed"
    assert np.isnan(unlevelled.uncertainty_xyz_m[2]), "unknown is not zero uncertainty"


def test_read_control_points_reads_detections_per_stream() -> None:
    control_points: aria.ControlPointSet = aria.read_control_points(REFERENCE_DIR / "R_11_5cp.control_points.json")
    assert [detection.stream_id for detection in control_points.detections] == ["1201-1", "1201-1", "1201-2"]
    first: aria.ControlPointDetection = control_points.detections[0]
    assert first.control_point == "OB1881"
    assert first.timestamp_ns == 928783591225
    assert first.uv_px == pytest.approx([439.0790695728934, 105.161001582523], abs=1e-12)
    # Image names start with the stream id, which is what the map is keyed by —
    # the file itself keys this map by stream *label*.
    assert set(control_points.timestamps) == {"1201-1", "1201-2"}
    assert control_points.timestamps["1201-1"][928783591225] == "1201-1-00468-928.784.jpg"
    assert control_points.timestamps["1201-2"][1010383591250] == "1201-2-02100-1010.384.jpg"
