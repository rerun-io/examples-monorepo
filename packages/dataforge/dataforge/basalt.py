"""basalt's ``calibration.json``, as validated records: camera models, extrinsics, and the follow frame.

The Monado SLAM Datasets ship one ``extras/calibration.json`` per headset in the
format `basalt <https://gitlab.com/VladyslavUsenko/basalt>`_ writes through
cereal: a ``value0`` wrapper holding three parallel lists — one ``T_imu_cam``
pose, one flat intrinsics block and one ``[width, height]`` per camera.

**Parallel lists are checked once, at the boundary.** ``load_calibration`` reads
that file and returns one ``CalibratedCamera`` per camera, so nothing downstream
can index the three lists out of step, read a coefficient the file never held, or
disagree about how many cameras a device has. The flat intrinsics block becomes
whichever of ``Kb4Intrinsics`` / ``Radtan8Intrinsics`` its ``camera_type`` names,
and every coefficient that model needs is required to be **present** — basalt
writes a real value for each, so a zero default would silently turn a truncated
file into an undistorted camera.

Two things then come out of those records — each camera's simplecv parameters and
the device's forward/up pair — and both are pure functions of them.

**Frames.** ``T_imu_cam`` is the camera's pose *in the IMU frame*. A rig whose
reference sensor is its IMU therefore has ``rig_T_cam = T_imu_cam`` with no
inversion; simplecv's ``Extrinsics`` calls that parent frame "world", so the rig
goes in as ``world_R_cam`` / ``world_t_cam``. This is the same convention as
simplecv's RoboCap loader, one inversion away (Kalibr states ``T_cam_imu``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import serde
import serde.json
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
KB4_COEFFICIENTS: tuple[str, ...] = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
"""Every key a ``kb4`` block must hold; a missing one is a truncated file, not a zero."""
RADTAN8_COEFFICIENTS: tuple[str, ...] = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "rpmax")
"""Every key a ``pinhole-radtan8`` block must hold, ``rpmax`` (its validity radius) included."""
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
class _RawIntrinsics:
    """The file's flat intrinsics block, **raw**: every coefficient optional because
    which ones a camera holds depends on its ``camera_type``.

    ``None`` means *the key was absent*, which is why nothing here defaults to
    ``0.0``: a zero radial term is a legitimate value basalt does write, and
    conflating the two would turn a truncated block into an undistorted camera.
    ``load_calibration`` is the only reader, and it requires exactly the keys the
    named model needs.
    """

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None
    k4: float | None = None
    k5: float | None = None
    k6: float | None = None
    p1: float | None = None
    p2: float | None = None
    rpmax: float | None = None


@serde.serde
@dataclass(frozen=True, slots=True)
class _RawCamera:
    """One camera's model tag and its raw coefficients, as the file pairs them."""

    camera_type: str
    """``kb4`` or ``pinhole-radtan8``; a plain ``str`` so an unknown tag is a
    validated ``ValueError`` naming it rather than a deserialization failure."""
    intrinsics: _RawIntrinsics
    """The flat coefficient block."""


@serde.serde
@dataclass(frozen=True, slots=True)
class _RawCalibrationValue:
    """The three parallel lists dataforge reads out of basalt's ``value0`` wrapper."""

    T_imu_cam: list[BasaltPose]  # noqa: N815 — basalt's own key; renaming it would need a serde alias for no gain
    """Camera poses in the IMU frame, one per camera, in camera order."""
    intrinsics: list[_RawCamera]
    """Camera models, one per camera, in the same order."""
    resolution: list[list[int]]
    """``[width, height]`` per camera, in the same order."""


@serde.serde
@dataclass(frozen=True, slots=True)
class _RawCalibration:
    """One device's whole ``calibration.json``, as basalt writes it."""

    value0: _RawCalibrationValue
    """cereal's single-root wrapper; everything lives under it."""


@dataclass(frozen=True, slots=True)
class Kb4Intrinsics:
    """A Kannala-Brandt fisheye camera: the shared projection head plus four radial terms.

    kb4's model is valid over the whole fisheye, so unlike radtan8 it declares no
    validity radius.
    """

    fx: float
    """Focal length in x, pixels."""
    fy: float
    """Focal length in y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    k1: float
    """First radial term."""
    k2: float
    """Second radial term."""
    k3: float
    """Third radial term."""
    k4: float
    """Fourth radial term."""


@dataclass(frozen=True, slots=True)
class Radtan8Intrinsics:
    """A Brown-Conrady rational camera: the projection head, six radial and two tangential terms."""

    fx: float
    """Focal length in x, pixels."""
    fy: float
    """Focal length in y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    k1: float
    """First radial term."""
    k2: float
    """Second radial term."""
    p1: float
    """First tangential term."""
    p2: float
    """Second tangential term."""
    k3: float
    """Third radial term."""
    k4: float
    """Fourth radial term (numerator of the rational model's second half)."""
    k5: float
    """Fifth radial term."""
    k6: float
    """Sixth radial term."""
    rpmax: float
    """Validity radius in normalized image coordinates: past it the rational model
    stops holding, so a consumer needs it as much as the coefficients."""


@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    """One camera of a device, with its three parallel-list entries already joined and checked."""

    index: int
    """Camera index, matching the ``cam<index>`` directory in a sequence."""
    rig_pose: BasaltPose
    """``T_imu_cam``: this camera's pose in the rig (IMU) frame."""
    resolution: tuple[int, int]
    """``(width, height)`` in pixels, both positive."""
    model: Kb4Intrinsics | Radtan8Intrinsics
    """Whichever projection the file's ``camera_type`` named, with every coefficient present."""

    @property
    def camera_model(self) -> CameraModel:
        """The model tag a camera node records, so a consumer needs no ``isinstance``."""
        return "kb4" if isinstance(self.model, Kb4Intrinsics) else "pinhole-radtan8"

    @property
    def distortion_valid_radius(self) -> float | None:
        """radtan8's ``rpmax``; ``None`` on kb4, which is valid over the whole fisheye."""
        return None if isinstance(self.model, Kb4Intrinsics) else self.model.rpmax


def _required_coefficients(raw: _RawIntrinsics, *, keys: Sequence[str], index: int, camera_type: str) -> dict[str, float]:
    """Pull the named keys off a raw block, refusing one the file never held.

    Args:
        raw: The camera's raw coefficient block.
        keys: Keys the named model needs, all of them.
        index: Camera index, for the error message.
        camera_type: Model tag, for the error message.

    Returns:
        Every named key's value, ready to splat into the model dataclass.

    Raises:
        ValueError: One of ``keys`` is absent from the json.
    """
    present: dict[str, float] = {}
    for key in keys:
        value: float | None = getattr(raw, key)
        if value is None:
            raise ValueError(f"cam{index} is a {camera_type} camera but its intrinsics hold no {key!r}; the calibration is incomplete")
        present[key] = value
    return present


def load_calibration(path: Path, *, expected_cameras: int | None = None) -> tuple[CalibratedCamera, ...]:
    """Read one device's ``calibration.json`` into validated per-camera records.

    Everything a later reader would otherwise have to assume is settled here: the
    three lists are the same length, each resolution is two positive ints, the
    ``camera_type`` is one this package knows, and the named model's coefficients
    are all present rather than defaulted to zero.

    Args:
        path: The device's ``extras/calibration.json``.
        expected_cameras: Cameras the caller's device table says this headset has;
            ``None`` accepts whatever the file holds. Given, a mismatch is an
            error — a device whose calibration lists a different number of cameras
            is either the wrong file or a corpus change, and both need a human.

    Returns:
        One record per camera, in camera order.

    Raises:
        ValueError: Any of the above does not hold.
    """
    raw: _RawCalibration = serde.json.from_json(_RawCalibration, path.read_text())
    value: _RawCalibrationValue = raw.value0
    counts: tuple[int, int, int] = (len(value.T_imu_cam), len(value.intrinsics), len(value.resolution))
    if len(set(counts)) != 1:
        raise ValueError(
            f"{path} lists {counts[0]} T_imu_cam pose(s), {counts[1]} intrinsics block(s) and {counts[2]} resolution(s); "
            "the three are per-camera and must agree"
        )
    if expected_cameras is not None and counts[0] != expected_cameras:
        raise ValueError(f"{path} describes {counts[0]} camera(s) but this device has {expected_cameras}")

    cameras: list[CalibratedCamera] = []
    for index, (pose, camera, resolution) in enumerate(zip(value.T_imu_cam, value.intrinsics, value.resolution, strict=True)):
        if len(resolution) != 2 or any(side <= 0 for side in resolution):
            raise ValueError(f"{path} gives cam{index} the resolution {resolution}; it must be one positive [width, height] pair")
        if camera.camera_type == "kb4":
            model: Kb4Intrinsics | Radtan8Intrinsics = Kb4Intrinsics(
                **_required_coefficients(camera.intrinsics, keys=KB4_COEFFICIENTS, index=index, camera_type=camera.camera_type)
            )
        elif camera.camera_type == "pinhole-radtan8":
            model = Radtan8Intrinsics(
                **_required_coefficients(camera.intrinsics, keys=RADTAN8_COEFFICIENTS, index=index, camera_type=camera.camera_type)
            )
        else:
            raise ValueError(f"{path} gives cam{index} the camera_type {camera.camera_type!r}; only 'kb4' and 'pinhole-radtan8' are known")
        cameras.append(CalibratedCamera(index=index, rig_pose=pose, resolution=(resolution[0], resolution[1]), model=model))
    return tuple(cameras)


def camera_parameters(camera: CalibratedCamera, *, name: str) -> PinholeParameters | Fisheye62Parameters:
    """Build one camera's simplecv parameters from its validated record.

    The extrinsics are the camera's pose **in the rig frame**, because MSD's rig
    frame is the IMU frame (``RIG_REFERENCE``) and ``T_imu_cam`` is exactly that
    pose. simplecv's ``Extrinsics`` calls the parent frame "world", so the rig
    goes in as ``world_R_cam`` / ``world_t_cam`` — the same convention simplecv's
    RoboCap loader uses, where Kalibr's inverse ``T_cam_imu`` goes in as
    ``cam_R_world`` / ``cam_t_world``.

    Args:
        camera: One camera out of ``load_calibration``.
        name: Stream label carried into the parameters.

    Returns:
        A ``Fisheye62Parameters`` for a ``kb4`` camera, a ``PinholeParameters``
        for a ``pinhole-radtan8`` one.
    """
    pose: BasaltPose = camera.rig_pose
    rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_t_cam: Float64[ndarray, "3"] = np.array([pose.px, pose.py, pose.pz], dtype=np.float64)
    extrinsics: Extrinsics = Extrinsics(world_R_cam=rig_R_cam, world_t_cam=rig_t_cam)
    model: Kb4Intrinsics | Radtan8Intrinsics = camera.model
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=model.fx,
        fl_y=model.fy,
        cx=model.cx,
        cy=model.cy,
        height=camera.resolution[1],
        width=camera.resolution[0],
    )
    if isinstance(model, Kb4Intrinsics):
        return Fisheye62Parameters(
            name=name,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=KannalaBrandtDistortion(k1=model.k1, k2=model.k2, k3=model.k3, k4=model.k4),
        )
    return PinholeParameters(
        name=name,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(
            k1=model.k1, k2=model.k2, p1=model.p1, p2=model.p2, k3=model.k3, k4=model.k4, k5=model.k5, k6=model.k6
        ),
    )


def follow_frame(cameras: Sequence[CalibratedCamera], camera_indices: Sequence[int] = (0, 1)) -> FollowFrame:
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
        cameras: The device's cameras, from ``load_calibration``.
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
    forward_sum_xyz: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    for index in camera_indices:
        pose: BasaltPose = cameras[index].rig_pose
        rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        forward_sum_xyz += rig_R_cam @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_norm: float = float(np.linalg.norm(forward_sum_xyz))
    if forward_norm < DEGENERATE_DIRECTION_NORM:
        raise ValueError(f"cameras {tuple(camera_indices)} look in opposing directions; their mean optical axis is degenerate")
    forward_xyz: Float64[ndarray, "3"] = forward_sum_xyz / forward_norm

    left_camera: BasaltPose = cameras[camera_indices[0]].rig_pose
    right_camera: BasaltPose = cameras[camera_indices[-1]].rig_pose
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
