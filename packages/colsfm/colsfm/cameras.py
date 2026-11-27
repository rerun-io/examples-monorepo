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

from collections.abc import Sequence
from dataclasses import replace
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


def colmap_model_and_parameters(
    projection: CameraProjectionModel, k_matrix: Float64[ndarray, "3 3"], distortion: Sequence[float]
) -> tuple[ColmapCameraModel, CameraParameters]:
    """The COLMAP model and parameters of a camera given as a projection, its 3x3 intrinsics and its distortion.

    `COLMAP_MODEL_BY_PROJECTION_MODEL` picks the model; the parameters are `fx fy cx cy` and then the distortion in the
    model's order, which is the order a catalog camera (`colsfm.catalog.CatalogCameraModel`) keeps it in.

    Raises:
        ValueError: When the distortion does not have the model's number of coefficients.
    """
    wanted: int = DISTORTION_COUNT_BY_PROJECTION_MODEL.get(projection, 0)
    if len(distortion) != wanted:
        raise ValueError(f"{projection} takes {wanted} distortion coefficients, got {len(distortion)}")
    pinhole: list[float] = [float(k_matrix[0, 0]), float(k_matrix[1, 1]), float(k_matrix[0, 2]), float(k_matrix[1, 2])]
    return COLMAP_MODEL_BY_PROJECTION_MODEL[projection], np.asarray([*pinhole, *distortion], dtype=np.float64)


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


def _calibration_matrix(fx: float, fy: float, cx: float, cy: float) -> Float64[ndarray, "3 3"]:
    """The zero-skew 3x3 intrinsic matrix of four pinhole parameters.

    Args:
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.

    Returns:
        The matrix.
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_params_with_colmap_intrinsics(camera: CameraParams, colmap: pycolmap.Camera) -> CameraParams:
    """Write a COLMAP camera's parameters back into the cuSFM fields `colmap_camera_parameters` reads them from.

    The inverse of `colmap_camera_parameters`. The pinhole family rewrites the rectified `projection_matrix` as
    `K_new @ inv(K_old) @ P`, so the 3x3 block becomes the new intrinsics and the 4th column (a stereo camera's baseline
    term) keeps its geometry; the distorted models rewrite `camera_matrix` (skew kept) and the first coefficients
    the COLMAP model consumes. A camera whose parameters did not move is returned as it is, byte for byte.

    Args:
        camera: The calibration the solve started from.
        colmap: The same camera after the solve.

    Returns:
        The calibration carrying the solve's parameters.

    Raises:
        ValueError: When the projection model is unknown or its source matrix is missing.
    """
    nominal: CameraParameters = colmap_camera_parameters(camera)
    params: CameraParameters = np.asarray(colmap.params, dtype=np.float64)
    if np.array_equal(nominal, params):
        return camera
    k_new: Float64[ndarray, "3 3"] = _calibration_matrix(float(params[0]), float(params[1]), float(params[2]), float(params[3]))
    if camera.projection_model in PINHOLE_FAMILY_MODELS:
        assert camera.projection_matrix is not None, "colmap_camera_parameters has checked it"
        k_old: Float64[ndarray, "3 3"] = _calibration_matrix(float(nominal[0]), float(nominal[1]), float(nominal[2]), float(nominal[3]))
        projection: Float64[ndarray, "3 4"] = k_new @ np.linalg.solve(k_old, camera.projection_matrix)
        return replace(camera, projection_matrix=projection)
    assert camera.camera_matrix is not None, "colmap_camera_parameters has checked it"
    camera_matrix: Float64[ndarray, "3 3"] = camera.camera_matrix.astype(np.float64)
    camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]] = k_new[[0, 1, 0, 1], [0, 1, 2, 2]]
    wanted: int = DISTORTION_COUNT_BY_PROJECTION_MODEL[camera.projection_model]
    distortion: CameraParameters = np.concatenate([params[4 : 4 + wanted], camera.distortion_coefficients[wanted:].astype(np.float64)])
    return replace(camera, camera_matrix=camera_matrix, distortion_coefficients=distortion)


def with_colmap_intrinsics(frames_meta: FramesMeta, reconstruction: pycolmap.Reconstruction) -> FramesMeta:
    """The collection with every camera's calibration taken from a solved reconstruction.

    What makes `--refine-camera-model` reach `frames_meta.json` and the Rerun layer, not only `sparse/cameras.txt`.

    Args:
        frames_meta: The collection the reconstruction was built from (`camera_id` is `camera_params_id`).
        reconstruction: The solved model.

    Returns:
        The collection carrying the reconstruction's intrinsics.
    """
    return frames_meta.with_intrinsics(
        {
            camera_id: camera_params_with_colmap_intrinsics(frames_meta.cameras[camera_id], colmap)
            for camera_id, colmap in reconstruction.cameras.items()
            if camera_id in frames_meta.cameras
        }
    )
