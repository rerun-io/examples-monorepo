"""Device-free conversion of an OAK/DepthAI calibration into simplecv pinholes.

The DepthAI-touching extraction lives in :mod:`live_rerun.sources.depthai`; this
module takes the already-extracted raw values (a 3x3 K matrix at the *encoded*
resolution, the Brown-Conrady distortion coefficients, and the 4x4 transform from
the reference camera, with translation in **centimetres** as DepthAI reports it)
and builds :class:`simplecv.camera_parameters.PinholeParameters`. Keeping it pure
makes the unit-prone bits (cm->m, K at encoded resolution, extrinsics direction)
testable without a camera attached.
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

# DepthAI reports extrinsic translations in centimetres; simplecv / Rerun log in
# metres. See getCameraExtrinsics docs and the calibration_reader example.
_CM_TO_M: float = 0.01


@dataclass
class OakCameraCalib:
    """Raw per-camera calibration extracted from a DepthAI device.

    ``k_matrix`` must already be scaled to ``(width, height)`` of the encoded
    stream (``getCameraIntrinsics(socket, width, height)`` does this). ``distortion``
    is the DepthAI coefficient list, whose order matches
    :class:`~simplecv.camera_parameters.BrownConradyDistortion`. ``ref_T_cam_cm`` is
    the 4x4 world->camera transform from the reference camera with translation in
    centimetres, or ``None`` for the reference camera itself (identity pose).
    """

    label: str
    width: int
    height: int
    k_matrix: Float[ndarray, "3 3"]
    distortion: list[float]
    ref_T_cam_cm: Float[ndarray, "4 4"] | None = None


def _extrinsics_from_ref(ref_T_cam_cm: Float[ndarray, "4 4"] | None) -> Extrinsics:
    """Build world->camera extrinsics, converting the translation from cm to m.

    The reference camera (``None``) is the world origin (identity pose). For the
    others, DepthAI's ``getCameraExtrinsics(reference, socket)`` returns the 4x4
    that maps a point from the reference frame into ``socket``'s frame, i.e. the
    world->camera transform, so the rotation/translation map straight onto
    ``world_R_cam`` / ``world_t_cam``.
    """
    if ref_T_cam_cm is None:
        return Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3))

    transform: Float[ndarray, "4 4"] = np.asarray(ref_T_cam_cm, dtype=float)
    world_R_cam: Float[ndarray, "3 3"] = transform[:3, :3].copy()
    world_t_cam: Float[ndarray, "3"] = transform[:3, 3].copy() * _CM_TO_M
    return Extrinsics(world_R_cam=world_R_cam, world_t_cam=world_t_cam)


def _distortion_from_coeffs(coeffs: list[float]) -> BrownConradyDistortion | None:
    """Build a Brown-Conrady model from DepthAI coefficients (needs >=5)."""
    if len(coeffs) < 5:
        return None
    values: list[float] = [float(c) for c in coeffs[:14]]
    return BrownConradyDistortion(*values)


def oak_calibration_to_pinholes(calibs: list[OakCameraCalib]) -> dict[str, PinholeParameters]:
    """Convert extracted OAK calibrations into simplecv pinholes keyed by label.

    Intrinsics use the OpenCV ``RDF`` convention (DepthAI/OpenCV native). The K
    matrix must already correspond to the encoded resolution so the projected
    video fills the camera frustum.
    """
    pinholes: dict[str, PinholeParameters] = {}
    for calib in calibs:
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=np.asarray(calib.k_matrix, dtype=float),
            height=calib.height,
            width=calib.width,
        )
        pinholes[calib.label] = PinholeParameters(
            name=f"oak_{calib.label}",
            extrinsics=_extrinsics_from_ref(calib.ref_T_cam_cm),
            intrinsics=intrinsics,
            distortion=_distortion_from_coeffs(calib.distortion),
        )
    return pinholes
