"""basalt's ``calibration.json``, as code: camera models, extrinsics, and the follow frame.

The Monado SLAM Datasets ship one ``extras/calibration.json`` per headset in the
format `basalt <https://gitlab.com/VladyslavUsenko/basalt>`_ writes through
cereal: a ``value0`` wrapper holding one ``T_imu_cam`` pose, one intrinsics block
and one resolution per camera. Two things come out of it — each camera's simplecv
parameters, and the device's forward/up pair — and both are pure functions of the
file, so they live here rather than in a dataset that happens to read one.

**Frames.** ``T_imu_cam`` is the camera's pose *in the IMU frame*. A rig whose
reference sensor is its IMU therefore has ``rig_T_cam = T_imu_cam`` with no
inversion; simplecv's ``Extrinsics`` calls that parent frame "world", so the rig
goes in as ``world_R_cam`` / ``world_t_cam``. This is the same convention as
simplecv's RoboCap loader, one inversion away (Kalibr states ``T_cam_imu``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import serde
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)

CameraModel: TypeAlias = Literal["kb4", "pinhole-radtan8"]
"""``camera_type`` values basalt writes in ``calibration.json``; MSD uses no others."""
DEGENERATE_DIRECTION_NORM: float = 1e-9
"""Shortest vector ``follow_frame`` still accepts as a direction; below it the inputs cancelled."""


@dataclass(frozen=True, slots=True)
class FollowFrame:
    """Where a headset looks and which way is up, as unit vectors in the rig frame.

    The rig frame is the IMU frame, and every headset mounts its IMU differently
    — the Index's up is rig -x, the G2's and the Odyssey+'s rig -y — so a follow
    camera cannot be placed by hand-picked numbers that happen to suit one
    device. ``follow_frame`` derives this pair from the calibration instead.
    """

    forward: tuple[float, float, float]
    """Where the front cameras look, unit length."""
    up: tuple[float, float, float]
    """The wearer's up, unit length and orthogonal to ``forward``."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltPose:
    """One ``T_imu_cam`` entry: the camera's pose in the IMU frame, quaternion xyzw."""

    px: float
    """Translation x, metres."""
    py: float
    """Translation y, metres."""
    pz: float
    """Translation z, metres."""
    qx: float
    """Quaternion x."""
    qy: float
    """Quaternion y."""
    qz: float
    """Quaternion z."""
    qw: float
    """Quaternion w (basalt writes xyzw fields, scalar last)."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltIntrinsics:
    """Projection and distortion terms of one camera, in basalt's flat layout.

    Both models share the ``fx fy cx cy`` head. ``kb4`` fills ``k1..k4`` only;
    ``pinhole-radtan8`` fills all eight plus a ``rpmax`` validity radius.
    """

    fx: float
    """Focal length in x, pixels."""
    fy: float
    """Focal length in y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    k1: float = 0.0
    """First radial term (both models)."""
    k2: float = 0.0
    """Second radial term (both models)."""
    k3: float = 0.0
    """Third radial term: kb4's fourth-order coefficient, radtan8's third."""
    k4: float = 0.0
    """Fourth radial term."""
    k5: float = 0.0
    """Fifth radial term (radtan8 only)."""
    k6: float = 0.0
    """Sixth radial term (radtan8 only)."""
    p1: float = 0.0
    """First tangential term (radtan8 only)."""
    p2: float = 0.0
    """Second tangential term (radtan8 only)."""
    rpmax: float | None = None
    """radtan8's validity radius in normalized image coordinates; ``None`` for kb4, which has no such limit."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltCamera:
    """One camera's model tag and its coefficients."""

    camera_type: CameraModel
    """``kb4`` (Kannala-Brandt fisheye) or ``pinhole-radtan8`` (Brown-Conrady)."""
    intrinsics: BasaltIntrinsics
    """The coefficients themselves."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltCalibrationValue:
    """The one member of basalt's ``value0`` wrapper that dataforge reads."""

    T_imu_cam: list[BasaltPose]  # noqa: N815 — basalt's own key; renaming it would need a serde alias for no gain
    """Camera poses in the IMU frame, one per camera, in camera order."""
    intrinsics: list[BasaltCamera]
    """Camera models, one per camera, in the same order."""
    resolution: list[list[int]]
    """``[width, height]`` per camera, in the same order."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltCalibration:
    """One device's whole ``calibration.json``, as basalt writes it."""

    value0: BasaltCalibrationValue
    """cereal's single-root wrapper; everything lives under it."""


def camera_parameters(calibration: BasaltCalibration, index: int, *, name: str) -> PinholeParameters | Fisheye62Parameters:
    """Build one camera's simplecv parameters from the device calibration.

    The extrinsics are the camera's pose **in the rig frame**, because MSD's rig
    frame is the IMU frame (``RIG_REFERENCE``) and ``T_imu_cam`` is exactly that
    pose. simplecv's ``Extrinsics`` calls the parent frame "world", so the rig
    goes in as ``world_R_cam`` / ``world_t_cam`` — the same convention simplecv's
    RoboCap loader uses, where Kalibr's inverse ``T_cam_imu`` goes in as
    ``cam_R_world`` / ``cam_t_world``.

    Args:
        calibration: Parsed ``calibration.json`` of the device.
        index: Camera index, matching the ``cam<index>`` directory in a sequence.
        name: Stream label carried into the parameters.

    Returns:
        A ``Fisheye62Parameters`` for a ``kb4`` camera, a ``PinholeParameters``
        for a ``pinhole-radtan8`` one.
    """
    value: BasaltCalibrationValue = calibration.value0
    pose: BasaltPose = value.T_imu_cam[index]
    camera: BasaltCamera = value.intrinsics[index]
    terms: BasaltIntrinsics = camera.intrinsics
    width: int = value.resolution[index][0]
    height: int = value.resolution[index][1]

    rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_t_cam: Float64[ndarray, "3"] = np.array([pose.px, pose.py, pose.pz], dtype=np.float64)
    extrinsics: Extrinsics = Extrinsics(world_R_cam=rig_R_cam, world_t_cam=rig_t_cam)
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF", fl_x=terms.fx, fl_y=terms.fy, cx=terms.cx, cy=terms.cy, height=height, width=width
    )
    if camera.camera_type == "kb4":
        return Fisheye62Parameters(
            name=name,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=KannalaBrandtDistortion(k1=terms.k1, k2=terms.k2, k3=terms.k3, k4=terms.k4),
        )
    return PinholeParameters(
        name=name,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(
            k1=terms.k1, k2=terms.k2, p1=terms.p1, p2=terms.p2, k3=terms.k3, k4=terms.k4, k5=terms.k5, k6=terms.k6
        ),
    )


def follow_frame(calibration: BasaltCalibration, camera_indices: Sequence[int] = (0, 1)) -> FollowFrame:
    """Derive a headset's forward and up in the rig frame from its front stereo pair.

    Forward is the mean optical axis (camera +z in RDF) of the listed cameras;
    averaging the pair cancels the slight outward yaw each one carries. Up comes
    from the pair's *baseline*, not from image-up: the baseline runs along the
    wearer's lateral axis, so ``right x forward`` is the wearer's up, and the
    cameras' own roll about the optical axis cannot tilt it. That distinction is
    not academic — the G2's four cameras are all mounted rolled 90 degrees, so
    its image-up is rig +x while its up is rig -y (which is also where its
    accelerometer reads gravity at rest, and what puts its front pair 15 degrees
    *below* the horizon, where tracking cameras are aimed). Deriving up from
    image-up matches the baseline within about a degree on the Index and the
    Odyssey+ and is 90 degrees out on the G2.

    ``camera_indices`` names the front pair, left camera first; the corpus lists
    every device that way, and on the G2 the two side cameras are ``cam2``/``cam3``.

    Args:
        calibration: Parsed ``calibration.json`` of the device.
        camera_indices: Front cameras, left first; the baseline runs from the
            first to the last, and every listed camera contributes to forward.

    Returns:
        The orthonormal forward/up pair, in the rig (IMU) frame.

    Raises:
        ValueError: Fewer than two cameras were named, the mean optical axis
            cancels out, or the baseline is parallel to it.
    """
    if len(camera_indices) < 2:
        raise ValueError(f"a follow frame needs a stereo pair to place its up axis, got {len(camera_indices)} camera(s)")
    poses: list[BasaltPose] = calibration.value0.T_imu_cam
    forward_sum_xyz: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    for index in camera_indices:
        pose: BasaltPose = poses[index]
        rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        forward_sum_xyz += rig_R_cam @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_norm: float = float(np.linalg.norm(forward_sum_xyz))
    if forward_norm < DEGENERATE_DIRECTION_NORM:
        raise ValueError(f"cameras {tuple(camera_indices)} look in opposing directions; their mean optical axis is degenerate")
    forward_xyz: Float64[ndarray, "3"] = forward_sum_xyz / forward_norm

    left_camera: BasaltPose = poses[camera_indices[0]]
    right_camera: BasaltPose = poses[camera_indices[-1]]
    baseline_xyz: Float64[ndarray, "3"] = np.array(
        [right_camera.px - left_camera.px, right_camera.py - left_camera.py, right_camera.pz - left_camera.pz], dtype=np.float64
    )
    # Gram-Schmidt: a real baseline is only nearly perpendicular to the optical axis
    # (a few tenths of a degree off on all three headsets), so it is a hint about the
    # lateral axis rather than the axis itself.
    right_xyz: Float64[ndarray, "3"] = baseline_xyz - float(np.dot(baseline_xyz, forward_xyz)) * forward_xyz
    right_norm: float = float(np.linalg.norm(right_xyz))
    if right_norm < DEGENERATE_DIRECTION_NORM:
        raise ValueError(f"the baseline of cameras {tuple(camera_indices)} is parallel to their optical axis; it fixes no lateral axis")
    up_xyz: Float64[ndarray, "3"] = np.cross(right_xyz / right_norm, forward_xyz)
    return FollowFrame(
        forward=(float(forward_xyz[0]), float(forward_xyz[1]), float(forward_xyz[2])),
        up=(float(up_xyz[0]), float(up_xyz[1]), float(up_xyz[2])),
    )
