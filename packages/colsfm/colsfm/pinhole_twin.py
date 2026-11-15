"""Map a distorted-pinhole rig on CASPAR by solving on its undistorted PINHOLE twin.

CASPAR (COLMAP's GPU bundle adjuster) has adapters for PINHOLE, SIMPLE_RADIAL and, in `recipes/colmap-src`,
OPENCV_FISHEYE. A cuSFM `DISTORTED_PINHOLE` camera is COLMAP FULL_OPENCV (eight OpenCV coefficients), which it does not
project, so a run on such a rig used to fall back to Ceres for the whole of stage 7.

The intrinsics are held fixed in colsfm's bundle adjustments, so nothing is lost by solving on the ideal camera the
distortion maps to instead: `make_pinhole_twin` writes a copy of the database whose keypoints are the exact (iterative)
undistortion of the originals into a PINHOLE camera with the same `fx fy cx cy`, and whose camera rows are that
PINHOLE camera. Keypoints the model cannot invert (NaN; the rational model folds over near some image corners) are
removed from the verified two-view geometries, so no track uses them, and are counted. Features, matching and loop
poses stay on the raw camera; only stages 7 and 7b run on the twin.

`restore_original_cameras` puts the original cameras and the raw keypoint positions back into the final model before
export, so the exported model and its colours stay consistent with the raw images. Point and keypoint indices are the
same in both databases, which is what makes the swap exact.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.cameras import colmap_camera
from colsfm.frames_meta import CameraParams, CameraProjectionModel, FramesMeta

TWINNED_PROJECTION_MODELS: Final[frozenset[CameraProjectionModel]] = frozenset({"DISTORTED_PINHOLE"})
"""cuSFM projection models a twin replaces: distorted pinholes CASPAR cannot project, which undistort into a pinhole."""


@dataclass(frozen=True)
class PinholeTwin:
    """The collection and database stages 7 and 7b run on, and what the undistortion dropped."""

    frames_meta: FramesMeta
    """The collection with every twinned camera replaced by its PINHOLE twin."""
    database_path: Path
    """The copy of the database with undistorted keypoints and PINHOLE camera rows."""
    twinned_camera_ids: tuple[int, ...]
    """`camera_params_id`s that were replaced."""
    num_dropped_keypoints: int
    """Keypoints that do not undistort (NaN) and were removed from the verified matches."""
    num_dropped_matches: int
    """Verified matches removed with them."""


def twin_camera(camera: CameraParams) -> CameraParams:
    """The PINHOLE camera with the same focal lengths and principal point.

    Args:
        camera: A twinned (distorted-pinhole) camera.

    Returns:
        A copy with `projection_model = "PINHOLE"`, the projection matrix `[K | 0]` and no distortion.

    Raises:
        ValueError: When the camera carries no camera matrix.
    """
    if camera.camera_matrix is None:
        raise ValueError(f"camera {camera.camera_params_id} has no camera_matrix to twin")
    k: Float64[ndarray, "3 3"] = np.asarray(camera.camera_matrix, dtype=np.float64).reshape(3, 3)
    return dataclasses.replace(
        camera,
        projection_model="PINHOLE",
        projection_matrix=np.hstack([k, np.zeros((3, 1))]),
        camera_matrix=None,
        distortion_coefficients=np.zeros(0, dtype=np.float64),
    )


def undistort_pixels(raw: pycolmap.Camera, twin: pycolmap.Camera, xy: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
    """Undistort pixel coordinates of `raw` into pixels of the ideal camera `twin`.

    Args:
        raw: The distorted camera.
        twin: Its pinhole twin.
        xy: Raw pixel coordinates.

    Returns:
        Twin pixel coordinates; NaN rows where the distortion does not invert.
    """
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    rays: Float64[ndarray, "n 2"] = np.asarray(raw.cam_from_img(np.ascontiguousarray(xy, dtype=np.float64)), dtype=np.float64)
    finite: Bool[ndarray, " n"] = np.isfinite(rays).all(axis=1)
    out: Float64[ndarray, "n 2"] = np.full_like(rays, np.nan)
    if finite.any():
        out[finite] = np.asarray(twin.img_from_cam(np.hstack([rays[finite], np.ones((int(finite.sum()), 1))])), dtype=np.float64)
    return out


def make_pinhole_twin(frames_meta: FramesMeta, database_path: Path, twin_database_path: Path) -> PinholeTwin | None:
    """Write the twin database and collection, or return None when no camera needs one.

    Args:
        frames_meta: The collection stage 7 would map.
        database_path: The run's database (keypoints, matches, verified two-view geometries).
        twin_database_path: Where the copy goes; replaced if present.

    Returns:
        The twin, or None when every camera already has a CASPAR adapter.
    """
    twinned: dict[int, CameraParams] = {
        camera_id: camera for camera_id, camera in frames_meta.cameras.items() if camera.projection_model in TWINNED_PROJECTION_MODELS
    }
    if not twinned:
        return None
    twins: dict[int, CameraParams] = {camera_id: twin_camera(camera) for camera_id, camera in twinned.items()}
    raw_cameras: dict[int, pycolmap.Camera] = {camera_id: colmap_camera(camera) for camera_id, camera in twinned.items()}
    twin_cameras: dict[int, pycolmap.Camera] = {camera_id: colmap_camera(camera) for camera_id, camera in twins.items()}

    shutil.copyfile(database_path, twin_database_path)
    dropped: dict[int, set[int]] = {}
    num_dropped_matches: int = 0
    with pycolmap.Database.open(twin_database_path) as database:
        for camera in twin_cameras.values():
            database.update_camera(camera)
        for image in database.read_all_images():
            if image.camera_id not in twin_cameras:
                continue
            keypoints = np.asarray(database.read_keypoints(image.image_id), dtype=np.float32)
            undistorted: Float64[ndarray, "n 2"] = undistort_pixels(
                raw_cameras[image.camera_id], twin_cameras[image.camera_id], keypoints[:, :2].astype(np.float64)
            )
            bad: Bool[ndarray, " n"] = ~np.isfinite(undistorted).all(axis=1)
            if bad.any():
                dropped[image.image_id] = set(np.flatnonzero(bad).tolist())
                # Far outside the image: never a valid projection, and the matches that use them go below.
                undistorted[bad] = -1.0e6
            updated = keypoints.copy()
            updated[:, :2] = undistorted.astype(np.float32)
            database.update_keypoints(image.image_id, updated)
        if dropped:
            pair_ids, geometries = database.read_two_view_geometries()
            for pair_id, geometry in zip(pair_ids, geometries, strict=True):
                image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
                bad1: set[int] = dropped.get(image_id1, set())
                bad2: set[int] = dropped.get(image_id2, set())
                if not bad1 and not bad2:
                    continue
                inliers = np.asarray(geometry.inlier_matches)
                if len(inliers) == 0:
                    continue
                keep: Bool[ndarray, " m"] = np.array([int(a) not in bad1 and int(b) not in bad2 for a, b in inliers], dtype=bool)
                if keep.all():
                    continue
                num_dropped_matches += int((~keep).sum())
                geometry.inlier_matches = inliers[keep]
                database.update_two_view_geometry(image_id1, image_id2, geometry)
    twin_meta: FramesMeta = dataclasses.replace(frames_meta, cameras={**frames_meta.cameras, **twins})
    num_dropped_keypoints: int = sum(len(indices) for indices in dropped.values())
    print(
        f"[colsfm] pinhole twin for CASPAR: cameras {sorted(twins)} undistorted into PINHOLE (same K); "
        f"{num_dropped_keypoints} keypoints do not undistort, {num_dropped_matches} verified matches dropped with them"
    )
    return PinholeTwin(
        frames_meta=twin_meta,
        database_path=twin_database_path,
        twinned_camera_ids=tuple(sorted(twins)),
        num_dropped_keypoints=num_dropped_keypoints,
        num_dropped_matches=num_dropped_matches,
    )


def restore_original_cameras(reconstruction: pycolmap.Reconstruction, original: FramesMeta, database_path: Path, twin: PinholeTwin) -> None:
    """Put the original cameras and raw keypoint positions back into a model mapped on the twin.

    Args:
        reconstruction: The final model (modified in place).
        original: The collection with the original cameras.
        database_path: The run's original database (raw keypoints).
        twin: The twin the model was mapped on.
    """
    with pycolmap.Database.open(database_path) as database:
        for camera_id in twin.twinned_camera_ids:
            if camera_id in reconstruction.cameras:
                reconstruction.cameras[camera_id] = colmap_camera(original.cameras[camera_id])
        for image_id, image in reconstruction.images.items():
            if image.camera_id not in twin.twinned_camera_ids:
                continue
            raw = np.asarray(database.read_keypoints(image_id), dtype=np.float64)
            for index, point2D in enumerate(image.points2D):
                point2D.xy = raw[index, :2]
