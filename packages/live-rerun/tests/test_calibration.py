"""Device-free tests for the OAK calibration -> generic rig conversion.

These guard the unit-prone bits: cm->m on extrinsic translation, the K matrix
flowing through at the encoded resolution, the reference camera (left) getting
the identity pose, index/name/kind propagation, and distortion handling. No OAK
device required.
"""

from __future__ import annotations

import numpy as np
import pytest

from live_rerun.calibration import OakCameraCalib, oak_calibration_to_rig
from live_rerun.rig import RigCalibration, entity_id


def _k(fx: float = 800.0, fy: float = 800.0, cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _calib(
    label: str,
    kind: str,
    ref_T_cam_cm: np.ndarray | None,
    *,
    k_matrix: np.ndarray | None = None,
    width: int = 1280,
    height: int = 720,
    distortion: list[float] | None = None,
) -> OakCameraCalib:
    return OakCameraCalib(
        label=label,
        width=width,
        height=height,
        k_matrix=_k() if k_matrix is None else k_matrix,
        distortion=distortion or [],
        kind=kind,  # type: ignore[arg-type]
        ref_T_cam_cm=ref_T_cam_cm,
    )


def _ordered_rig() -> RigCalibration:
    """A realistic left(reference)/rgb/right rig as the source yields it."""
    rgb_ext = np.eye(4)
    rgb_ext[:3, 3] = [7.5, -2.0, 0.0]  # centimetres, as DepthAI reports
    right_ext = np.eye(4)
    right_ext[:3, 3] = [-7.5, 0.0, 0.0]
    return oak_calibration_to_rig(
        [
            _calib("left", "grayscale", None),
            _calib("rgb", "rgb", rgb_ext),
            _calib("right", "grayscale", right_ext),
        ]
    )


def test_entity_id_is_zero_padded_and_sorts_lexicographically() -> None:
    assert entity_id("cam", 0) == "cam_00"
    assert entity_id("rig", 0) == "rig_00"
    assert entity_id("imu", 7) == "imu_07"
    assert entity_id("cam", 12) == "cam_12"
    # The reason for padding: lexicographic order matches numeric order.
    ids = [entity_id("cam", i) for i in (10, 2, 0, 12)]
    assert sorted(ids) == ["cam_00", "cam_02", "cam_10", "cam_12"]


def test_left_is_reference_with_identity_pose() -> None:
    rig = _ordered_rig()

    assert rig.reference_index == 0
    cam0 = rig.cameras[0]
    assert (cam0.index, cam0.name, cam0.kind) == (0, "left", "grayscale")
    pose = cam0.pinhole.extrinsics
    assert pose.cam_t_world is not None
    np.testing.assert_allclose(pose.world_T_cam, np.eye(4))
    np.testing.assert_allclose(pose.cam_t_world, np.zeros(3))
    assert cam0.pinhole.name == "cam_00"


def test_indices_names_kinds_follow_input_order() -> None:
    rig = _ordered_rig()

    assert [(c.index, c.name, c.kind) for c in rig.cameras] == [
        (0, "left", "grayscale"),
        (1, "rgb", "rgb"),
        (2, "right", "grayscale"),
    ]


def test_non_reference_translation_converted_cm_to_m() -> None:
    rig = _ordered_rig()

    rgb_pose = rig.cameras[1].pinhole.extrinsics
    assert rgb_pose.world_t_cam is not None
    np.testing.assert_allclose(rgb_pose.world_t_cam, [0.075, -0.02, 0.0])


def test_requires_exactly_one_reference_camera() -> None:
    offset = np.eye(4)
    offset[:3, 3] = [1.0, 0.0, 0.0]
    # zero reference cameras (none at identity)
    with pytest.raises(ValueError, match="exactly one reference"):
        oak_calibration_to_rig([_calib("left", "grayscale", offset), _calib("rgb", "rgb", offset)])
    # multiple reference cameras (two at identity)
    with pytest.raises(ValueError, match="exactly one reference"):
        oak_calibration_to_rig([_calib("left", "grayscale", None), _calib("rgb", "rgb", None)])


def test_intrinsics_use_encoded_resolution_and_k() -> None:
    k = _k(fx=1234.0, fy=1240.0, cx=960.0, cy=540.0)
    rig = oak_calibration_to_rig([_calib("left", "grayscale", None, k_matrix=k, width=1920, height=1080)])
    intr = rig.cameras[0].pinhole.intrinsics

    assert (intr.width, intr.height) == (1920, 1080)
    assert intr.fl_x == 1234.0 and intr.fl_y == 1240.0
    assert intr.k_matrix is not None
    np.testing.assert_allclose(intr.k_matrix, k)


def test_distortion_built_from_coeffs_and_skipped_when_too_few() -> None:
    coeffs = [0.1, -0.2, 0.001, -0.002, 0.05] + [0.0] * 9  # 14 DepthAI coeffs
    with_dist = oak_calibration_to_rig([_calib("left", "grayscale", None, distortion=coeffs)])
    dist = with_dist.cameras[0].pinhole.distortion
    assert dist is not None
    assert (dist.k1, dist.k2, dist.p1, dist.p2, dist.k3) == (0.1, -0.2, 0.001, -0.002, 0.05)

    no_dist = oak_calibration_to_rig([_calib("left", "grayscale", None, distortion=[0.1, 0.2])])
    assert no_dist.cameras[0].pinhole.distortion is None
