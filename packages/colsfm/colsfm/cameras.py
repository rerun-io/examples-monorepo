"""cuSFM camera parameters to `pycolmap.Camera`.

The mapping is `kpmap_to_colmap`'s, read off the decompiled
`ColmapManager::WriteCamerasFile` and reproduced byte for byte against the
shipped models (export.md §3.3, keypoints_mapper_main.md §9.4):

| cuSFM `CameraProjectionModelType` | COLMAP model | Source matrix | PARAMS |
|---|---|---|---|
| `PINHOLE` | `PINHOLE` | `projection_matrix` (3x4) | `fx fy cx cy` |
| `DISTORTED_PINHOLE` | `FULL_OPENCV` | `camera_matrix` (3x3) | `fx fy cx cy` + 8 coefficients |
| `FTHETA_WINDSHIELD` | `PINHOLE` | `projection_matrix` (3x4) | `fx fy cx cy`, lossy |
| `OPENCV_FISHEYE` | `OPENCV_FISHEYE` | `camera_matrix` (3x3) | `fx fy cx cy k1 k2 k3 k4` |

Note the asymmetry: the two undistorted models read the rectified 3x4
`projection_matrix`, the two distorted ones read the raw 3x3 `camera_matrix`.
`FTHETA_WINDSHIELD` is exported as a rectified pinhole rather than rejected —
`docs/camera.md`'s summary table calls it unsupported, but the binary emits a
`PINHOLE` line and only warns. That is the one genuinely lossy conversion here.

**Pixel-centre caveat** (export.md §3.6): cuSFM copies both keypoints and
`cx, cy` in the OpenCV convention, where the top-left pixel centre is `(0, 0)`,
while COLMAP places it at `(0.5, 0.5)`. The exported model is self-consistent —
reprojection errors are unaffected — but it sits half a pixel off a COLMAP-native
model. Do not mix these keypoints with COLMAP-detected features.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray

from colsfm.frames_meta import CameraParams, CameraProjectionModel, FramesMeta

ColmapCameraModel: TypeAlias = Literal["PINHOLE", "FULL_OPENCV", "OPENCV_FISHEYE"]
"""The three COLMAP camera models cuSFM's four projection models map onto."""

CameraParameters: TypeAlias = Float64[ndarray, "num_params"]
"""COLMAP camera parameters in the model's own order."""

COLMAP_MODEL_BY_PROJECTION_MODEL: Final[dict[CameraProjectionModel, ColmapCameraModel]] = {
    "PINHOLE": "PINHOLE",
    "DISTORTED_PINHOLE": "FULL_OPENCV",
    "FTHETA_WINDSHIELD": "PINHOLE",
    "OPENCV_FISHEYE": "OPENCV_FISHEYE",
}
"""cuSFM projection model to COLMAP camera model, per export.md §3.3."""

DISTORTION_COUNT_BY_PROJECTION_MODEL: Final[dict[CameraProjectionModel, int]] = {
    "DISTORTED_PINHOLE": 8,
    "OPENCV_FISHEYE": 4,
}
"""How many distortion coefficients each distorted model consumes."""

PINHOLE_FAMILY_MODELS: Final[frozenset[CameraProjectionModel]] = frozenset({"PINHOLE", "FTHETA_WINDSHIELD"})
"""The two models exported from the rectified 3x4 `projection_matrix` rather than the
raw 3x3 `camera_matrix` (export.md §3.3). `colsfm.rerun_log` picks its intrinsics the
same way, so the membership test lives here rather than in two hardcoded tuples."""


def _pinhole_parameters_from_projection_matrix(camera: CameraParams) -> CameraParameters:
    """Read `fx fy cx cy` out of the rectified 3x4 projection matrix.

    Args:
        camera: A camera whose projection model reads `projection_matrix`.

    Returns:
        Float64 `[fx, fy, cx, cy]` with shape `[4]`.

    Raises:
        ValueError: When the calibration carries no projection matrix, which is a
            `LOG(FATAL)` in cuSFM.
    """
    if camera.projection_matrix is None:
        raise ValueError(f"Projection matrix is not set for camera parameter {camera.camera_params_id}")
    flat: Float64[ndarray, "12"] = camera.projection_matrix.reshape(-1)
    return np.array([flat[0], flat[5], flat[2], flat[6]])


def _pinhole_parameters_from_camera_matrix(camera: CameraParams) -> CameraParameters:
    """Read `fx fy cx cy` out of the raw 3x3 intrinsic matrix.

    Args:
        camera: A camera whose projection model reads `camera_matrix`.

    Returns:
        Float64 `[fx, fy, cx, cy]` with shape `[4]`.

    Raises:
        ValueError: When the calibration carries no camera matrix.
    """
    if camera.camera_matrix is None:
        raise ValueError(f"Camera matrix is not set for camera parameter {camera.camera_params_id}")
    flat: Float64[ndarray, "9"] = camera.camera_matrix.reshape(-1)
    return np.array([flat[0], flat[4], flat[2], flat[5]])


def fit_distortion_coefficients(camera: CameraParams, wanted: int) -> CameraParameters:
    """Truncate or zero-pad the distortion coefficients to the model's arity.

    cuSFM warns and carries on in all three off-nominal cases rather than
    failing, so this does too (export.md §3.3).

    Args:
        camera: The camera whose coefficients to fit.
        wanted: Number of coefficients the COLMAP model takes.

    Returns:
        Float64 coefficients with shape `[wanted]`.
    """
    coefficients: CameraParameters = camera.distortion_coefficients
    if coefficients.size > wanted:
        print(f"Too many distortion coefficients for camera {camera.camera_params_id}. Only writing the first {wanted}")
        return coefficients[:wanted].astype(np.float64)
    if coefficients.size < wanted:
        # An empty vector is the `size < wanted` case, and cuSFM logs it the same way.
        print(f"Too few distortion coefficients for camera {camera.camera_params_id}. Will pad with zeros.")
        return np.concatenate([coefficients.astype(np.float64), np.zeros(wanted - coefficients.size)])
    return coefficients.astype(np.float64)


def colmap_camera_parameters(camera: CameraParams) -> CameraParameters:
    """Build the COLMAP parameter vector for one cuSFM camera.

    Args:
        camera: The cuSFM camera parameters.

    Returns:
        Float64 parameters in the COLMAP model's order.

    Raises:
        ValueError: When the projection model is unknown or its source matrix is missing.
    """
    if camera.projection_model not in COLMAP_MODEL_BY_PROJECTION_MODEL:
        raise ValueError(f"Unsupported camera projection model type: {camera.projection_model}")
    if camera.projection_model in PINHOLE_FAMILY_MODELS:
        return _pinhole_parameters_from_projection_matrix(camera)
    intrinsics: CameraParameters = _pinhole_parameters_from_camera_matrix(camera)
    wanted: int = DISTORTION_COUNT_BY_PROJECTION_MODEL[camera.projection_model]
    return np.concatenate([intrinsics, fit_distortion_coefficients(camera, wanted)])


def colmap_camera(camera: CameraParams) -> pycolmap.Camera:
    """Convert one cuSFM camera into a `pycolmap.Camera`.

    Args:
        camera: The cuSFM camera parameters.

    Returns:
        A camera whose `camera_id` is the cuSFM `camera_params_id`, verbatim, and
        whose `has_prior_focal_length` is True. The calibration came from the
        metadata, and COLMAP 4.2 reads that flag to choose the calibrated
        two-view-geometry path: leaving it False silently degrades a PINHOLE pair
        to `UNCALIBRATED` and an `OPENCV_FISHEYE` pair to `DEGENERATE` with zero
        inliers (pycolmap-capabilities.md §5). Setting it here is what lets the
        database and reconstruction builders stop stamping it after the fact.

    Raises:
        ValueError: When the projection model is unknown or its source matrix is missing.
    """
    if camera.projection_model == "FTHETA_WINDSHIELD":
        print(
            f"Camera parameter {camera.camera_params_id} is FTHETA_WINDSHIELD, "
            "thus the output in rectified model may be inaccurate."
        )
    colmap: pycolmap.Camera = pycolmap.Camera(
        camera_id=camera.camera_params_id,
        model=COLMAP_MODEL_BY_PROJECTION_MODEL[camera.projection_model],
        width=camera.image_width,
        height=camera.image_height,
        params=colmap_camera_parameters(camera),
    )
    colmap.has_prior_focal_length = True
    return colmap


def colmap_cameras(frames_meta: FramesMeta) -> dict[int, pycolmap.Camera]:
    """Convert every camera of a metadata collection.

    One COLMAP camera per `camera_params_id`, not per image, and all of them are
    emitted regardless of whether any keyframe used them (export.md §3.3).

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Cameras keyed by `camera_params_id`.
    """
    return {camera_params_id: colmap_camera(camera) for camera_params_id, camera in sorted(frames_meta.cameras.items())}
