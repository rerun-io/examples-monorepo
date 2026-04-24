"""Static calibration logging via simplecv's PinholeWithDistortion.

VSLAM-LAB's ``calibration.yaml`` uses three distortion types: ``radtan4``
(Brown–Conrady, [k1, k2, p1, p2]), ``radtan5`` (Brown–Conrady, [k1, k2, p1,
p2, k3]), and ``equid4`` (Kannala–Brandt fisheye, [k1, k2, k3, k4]). These
all round-trip through simplecv's existing ``BrownConradyDistortion`` /
``KannalaBrandtDistortion`` → ``PinholeWithDistortion`` pipeline, the same
code path ``packages/robocap-slam/robocap_slam/visualization.py:88`` uses
via ``simplecv.rerun_log_utils.log_pinhole``.

VSLAM-LAB's ``T_BS`` is body-from-sensor (body = T_BS @ sensor), which maps
directly to simplecv's ``Extrinsics(world_R_cam=R, world_t_cam=t)`` with
``world`` = body frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_log_utils import log_pinhole

from slam_evals.data.types import Calibration, CameraIntrinsics


def _t_bs_to_matrix(t_bs: tuple[float, ...]) -> Float64[ndarray, "4 4"]:
    return np.asarray(t_bs, dtype=np.float64).reshape(4, 4)


def _build_distortion(cam: CameraIntrinsics) -> BrownConradyDistortion | KannalaBrandtDistortion | None:
    """Map VSLAM-LAB ``distortion_type`` + coefficients to a simplecv distortion dataclass."""
    dt = (cam.distortion_type or "").lower()
    coeffs = cam.distortion_coefficients
    if not dt or not coeffs:
        return None
    if dt.startswith("radtan"):
        # radtan4: [k1, k2, p1, p2]; radtan5 adds k3. Extras beyond 5 ignored.
        k1, k2, p1, p2 = (list(coeffs) + [0.0, 0.0, 0.0, 0.0])[:4]
        k3 = coeffs[4] if len(coeffs) >= 5 else 0.0
        return BrownConradyDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
    if dt.startswith("equid") or dt.startswith("kannala"):
        # equid4: [k1, k2, k3, k4]
        padded = list(coeffs) + [0.0] * 4
        return KannalaBrandtDistortion(k1=padded[0], k2=padded[1], k3=padded[2], k4=padded[3])
    return None


def _to_simplecv_camera(cam: CameraIntrinsics) -> PinholeParameters | Fisheye62Parameters:
    intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=cam.fx,
        fl_y=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
        height=cam.height,
        width=cam.width,
    )
    t_bs = _t_bs_to_matrix(cam.t_bs)
    # T_BS is body(world)-from-sensor(cam); feed as world_R_cam / world_t_cam.
    extrinsics = Extrinsics(
        world_R_cam=t_bs[:3, :3].astype(np.float64),
        world_t_cam=t_bs[:3, 3].astype(np.float64),
    )
    distortion = _build_distortion(cam)
    if isinstance(distortion, KannalaBrandtDistortion):
        return Fisheye62Parameters(
            name=cam.cam_name,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=distortion,
        )
    return PinholeParameters(
        name=cam.cam_name,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortion=distortion,
    )


def log_camera_static(
    cam: CameraIntrinsics,
    *,
    entity_path: str,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log one camera's intrinsics + body-from-sensor extrinsics at ``entity_path``.

    Delegates to ``simplecv.rerun_log_utils.log_pinhole`` so distortion
    coefficients are preserved alongside the Pinhole archetype.
    """
    # log_pinhole logs the extrinsic Transform3D at `entity_path` and the
    # PinholeWithDistortion at `entity_path/pinhole`.
    log_pinhole(
        _to_simplecv_camera(cam),
        cam_log_path=Path(entity_path),
        image_plane_distance=0.5,
        static=True,
        recording=recording,
    )


def log_calibration_static(
    calib: Calibration | None,
    *,
    cam_entity_paths: dict[str, str],
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log every camera in ``calib`` under its mapped entity path.

    Cameras whose ``cam_name`` isn't in ``cam_entity_paths`` are silently
    skipped — VSLAM-LAB's ``depth_0`` /``depth_1`` are deliberately omitted
    because they're pre-registered to ``rgb_0`` / ``rgb_1`` and share those
    cameras' pinholes.
    """
    if calib is None:
        return
    for cam in calib.cameras:
        path = cam_entity_paths.get(cam.cam_name)
        if path is None:
            continue
        log_camera_static(cam, entity_path=path, recording=recording)
