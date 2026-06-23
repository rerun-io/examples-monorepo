"""Device-free conversion of an OAK/DepthAI calibration into a generic rig.

The DepthAI-touching extraction lives in :mod:`live_rerun.sources.depthai`; this
module takes the already-extracted raw values (a 3x3 K matrix at the *encoded*
resolution, the Brown-Conrady distortion coefficients, and the 4x4 transform
relative to the **reference** camera, with translation in **centimetres** as
DepthAI reports it) and builds the system-agnostic :class:`live_rerun.rig`
descriptors (:class:`~live_rerun.rig.CameraSensor` /
:class:`~live_rerun.rig.RigCalibration`) the core logger consumes. Keeping it
pure makes the unit-prone bits (cm->m, K at encoded resolution, extrinsics
direction, reference identity) testable without a camera attached.

The reference sensor is the OAK's **left** mono camera (CAM_B): it gets the
identity ``rig_T_cam`` pose (the rig origin), and ``rgb``/``right`` are expressed
relative to it. SLAM is run around the left camera, so the rig moves rigidly
with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Extrinsics,
    Intrinsics,
    PinholeParameters,
)

from live_rerun.rig import CameraSensor, RigCalibration, SensorKind, entity_id

# DepthAI reports extrinsic translations in centimetres; simplecv / Rerun log in
# metres. See getCameraExtrinsics docs and the calibration_reader example.
_CM_TO_M: float = 0.01


@dataclass
class OakCameraCalib:
    """Raw per-camera calibration extracted from a DepthAI device.

    ``k_matrix`` must already be scaled to ``(width, height)`` of the encoded
    stream (``getCameraIntrinsics(socket, width, height)`` does this). ``distortion``
    is the DepthAI coefficient list, whose order matches
    :class:`~simplecv.camera_parameters.BrownConradyDistortion`. ``kind`` is the
    image content (``"rgb"`` / ``"grayscale"``). ``ref_T_cam_cm`` is the 4x4
    reference->camera transform with translation in centimetres, or ``None`` for
    the reference camera itself (identity pose, the rig origin).
    """

    label: str
    width: int
    height: int
    k_matrix: Float[ndarray, "3 3"]
    distortion: list[float]
    kind: SensorKind
    ref_T_cam_cm: Float[ndarray, "4 4"] | None = None


def _extrinsics_from_ref(ref_T_cam_cm: Float[ndarray, "4 4"] | None) -> Extrinsics:
    """Build rig->camera extrinsics, converting the translation from cm to m.

    The reference camera (``None``) is the rig origin (identity pose). For the
    others, DepthAI's ``getCameraExtrinsics(reference, socket)`` returns the 4x4
    that maps a point **from the reference frame into ``socket``'s frame** — that
    is ``cam_T_world`` (world->camera, "world" == rig), the matrix used directly in
    the projection ``K @ cam_T_world``. So it fills ``cam_R_world`` / ``cam_t_world``;
    the camera's *pose in the rig* (``world_T_cam``) is its inverse.

    This is the subtle bit: feeding the transform into ``world_R_cam`` /
    ``world_t_cam`` instead **inverts every non-reference camera** — frusta land on
    the wrong side and triangulated points fall *behind* the cameras (verified on
    an OAK-D-W: the person triangulated to negative depth until this was fixed).
    """
    if ref_T_cam_cm is None:
        return Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))

    transform: Float[ndarray, "4 4"] = np.asarray(ref_T_cam_cm, dtype=float)
    cam_R_world: Float[ndarray, "3 3"] = transform[:3, :3].copy()
    cam_t_world: Float[ndarray, "3"] = transform[:3, 3].copy() * _CM_TO_M
    return Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world)


def _distortion_from_coeffs(coeffs: list[float]) -> BrownConradyDistortion | None:
    """Build a Brown-Conrady model from DepthAI coefficients (needs >=5)."""
    if len(coeffs) < 5:
        return None
    values: list[float] = [float(c) for c in coeffs[:14]]
    return BrownConradyDistortion(*values)


def oak_calibration_to_rig(calibs: list[OakCameraCalib]) -> RigCalibration:
    """Convert extracted OAK calibrations into a generic :class:`RigCalibration`.

    ``calibs`` is ordered (reference first); each camera's list position becomes
    its ``cam_<index>`` entity index. Intrinsics use the OpenCV ``RDF`` convention
    (DepthAI/OpenCV native); the K matrix must already correspond to the encoded
    resolution so the projected video fills the camera frustum. Exactly one
    camera must be the reference (``ref_T_cam_cm is None``, the rig origin); it
    defines ``reference_index``. A ``ValueError`` is raised otherwise, so a
    malformed backend can't silently mislabel a non-identity camera as the origin.
    """
    reference_indices: list[int] = [index for index, calib in enumerate(calibs) if calib.ref_T_cam_cm is None]
    if len(reference_indices) != 1:
        raise ValueError(
            f"a rig needs exactly one reference camera (the rig origin, with ref_T_cam_cm=None); "
            f"got {len(reference_indices)} of {len(calibs)} cameras"
        )
    reference_index: int = reference_indices[0]
    cameras: list[CameraSensor] = []
    for index, calib in enumerate(calibs):
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=np.asarray(calib.k_matrix, dtype=float),
            height=calib.height,
            width=calib.width,
        )
        pinhole = PinholeParameters(
            name=entity_id("cam", index),
            extrinsics=_extrinsics_from_ref(calib.ref_T_cam_cm),
            intrinsics=intrinsics,
            distortion=_distortion_from_coeffs(calib.distortion),
        )
        cameras.append(CameraSensor(index=index, name=calib.label, kind=calib.kind, pinhole=pinhole))
    return RigCalibration(cameras=cameras, reference_index=reference_index)
