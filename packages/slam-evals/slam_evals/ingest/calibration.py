"""Static calibration logging via simplecv's ``log_pinhole``.

VSLAM-LAB's ``calibration.yaml`` is parsed in
``slam_evals.data.parse._to_camera_spec`` straight into simplecv's
``PinholeParameters`` (Brown-Conrady distortion) or
``Fisheye62Parameters`` (Kannala-Brandt fisheye), wrapped in
``CameraSpec`` together with the two extra YAML fields (fps,
depth_factor) that simplecv doesn't carry. This module is now a thin
adapter that hands the simplecv parameters straight to ``log_pinhole``
and adds the wireframe-colour override for cam_0.
"""

from __future__ import annotations

from pathlib import Path

import rerun as rr
from simplecv.rerun_log_utils import log_pinhole

from slam_evals.data.types import CameraSpec


def log_camera_static(
    cam: CameraSpec,
    *,
    entity_path: str,
    color: tuple[int, int, int, int] | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log one camera's intrinsics + rig-from-camera extrinsics at ``entity_path``.

    Delegates to ``simplecv.rerun_log_utils.log_pinhole`` so distortion
    coefficients are preserved alongside the Pinhole archetype. When
    ``color`` is provided, additionally log it as a partial update on the
    pinhole entity so the viewer renders the frustum wireframe in that
    colour — used to highlight the rig's reference sensor (cam_0) in
    multi-camera setups.
    """
    # log_pinhole logs the extrinsic Transform3D at `entity_path` and the
    # PinholeWithDistortion at `entity_path/pinhole`.
    log_pinhole(
        cam.parameters,
        cam_log_path=Path(entity_path),
        image_plane_distance=0.5,
        static=True,
        recording=recording,
    )
    if color is not None:
        rr.log(
            f"{entity_path}/pinhole",
            rr.Pinhole.from_fields(image_plane_distance=0.5, color=color),
            static=True,
            recording=recording,
        )
