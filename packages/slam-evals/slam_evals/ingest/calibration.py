"""Static calibration logging: pinhole intrinsics + body→sensor extrinsics."""

from __future__ import annotations

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray

from slam_evals.data.types import Calibration, CameraIntrinsics


def _t_bs_to_matrix(t_bs: tuple[float, ...]) -> Float64[ndarray, "4 4"]:
    return np.asarray(t_bs, dtype=np.float64).reshape(4, 4)


def log_camera_static(
    cam: CameraIntrinsics,
    *,
    entity_path: str,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log pinhole + body-from-sensor transform as static data at ``entity_path``.

    Hierarchy: ``entity_path`` holds the ``Transform3D`` (body -> camera), and
    ``entity_path + "/pinhole"`` holds the ``rr.Pinhole`` intrinsics archetype.
    """
    t_bs = _t_bs_to_matrix(cam.t_bs)
    rr.log(
        entity_path,
        rr.Transform3D(translation=t_bs[:3, 3], mat3x3=t_bs[:3, :3]),
        static=True,
        recording=recording,
    )
    if cam.fx > 0 and cam.fy > 0 and cam.width > 0 and cam.height > 0:
        rr.log(
            f"{entity_path}/pinhole",
            rr.Pinhole(
                focal_length=[cam.fx, cam.fy],
                principal_point=[cam.cx, cam.cy],
                width=cam.width,
                height=cam.height,
                image_plane_distance=0.5,
            ),
            static=True,
            recording=recording,
        )


def log_calibration_static(
    calib: Calibration | None,
    *,
    cam_entity_paths: dict[str, str],
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log every camera in ``calib`` under its mapped entity path (if present)."""
    if calib is None:
        return
    for cam in calib.cameras:
        path = cam_entity_paths.get(cam.cam_name)
        if path is None:
            continue
        log_camera_static(cam, entity_path=path, recording=recording)
