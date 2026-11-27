"""Check a calibration from `colsfm calibrate` against another calibration of the same rig (e.g. the factory's).

The check Pablo asked for (2026-11-27), per camera:

- **reprojection**: the calibration's own rig model adjusted twice, each time with one calibration held (its intrinsics,
  lens model and rig extrinsics; the frame poses and the points move), and scored on one set of observations taken from
  that model: RMS reprojection error per camera. Ours not worse than the other is the pass.
- **differences**, information only: rotation and translation direction of each camera relative to the first one,
  focal length and principal point.

The estimate's rig frame is its first camera and its scale is the reconstruction's own, so holding the other
calibration's extrinsics needs them in that scale: one scalar, fitted by least squares over the first-camera-relative
translations, does that here and only here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.calibrate import CalibrationResult
from colsfm.cameras import colmap_model_and_parameters
from colsfm.catalog import CatalogCameraModel
from colsfm.reconstruction import camera_sensor_id


@dataclass(frozen=True, slots=True)
class CameraCheck:
    """One camera of a calibration against the same camera of a reference calibration."""

    name: str
    """The camera."""
    registered_images: int
    """Images of this camera in the rig model."""
    observations: int
    """Observations both arms score (the same ones)."""
    dropped_observations: int
    """Observations behind the camera in either arm, scored in neither."""
    reprojection_px: float
    """RMS reprojection error with this calibration held."""
    reference_held_px: float
    """RMS reprojection error with the reference calibration held, on the same observations."""
    rotation_deg: float
    """Angle between the two first-camera-relative rotations (information)."""
    direction_deg: float
    """Angle between the two first-camera-relative translations; 0 for the first camera (information)."""
    focal_change_percent: float
    """Mean of fx and fy relative to the reference's, in percent (information)."""
    principal_point_offset_px: float
    """Distance between the two principal points (information)."""

    @property
    def not_worse(self) -> bool:
        """Whether this calibration reprojects no worse than the reference held."""
        return self.reprojection_px <= self.reference_held_px


def first_T_cams(cameras: Sequence[CatalogCameraModel]) -> list[Float64[ndarray, "4 4"]]:
    """Each camera's pose relative to the first camera."""
    first_T_rig: Float64[ndarray, "4 4"] = np.linalg.inv(cameras[0].rig_T_cam)
    return [first_T_rig @ camera.rig_T_cam for camera in cameras]


@dataclass(frozen=True, slots=True)
class Observations:
    """Every observation of a triangulated point in a reconstruction, grouped by image."""

    by_image: dict[int, tuple[Float64[ndarray, "n 2"], list[int]]]
    """Image id -> its observed pixels and the ids of the points they observe."""


def observations(model: pycolmap.Reconstruction) -> Observations:
    """The observations a held comparison scores, taken once from the model both arms start from."""
    by_image: dict[int, tuple[Float64[ndarray, "n 2"], list[int]]] = {}
    for image_id, image in model.images.items():
        if image.has_pose:
            seen = [(point2D.xy, point2D.point3D_id) for point2D in image.points2D if point2D.has_point3D()]
            if seen:
                by_image[image_id] = (np.asarray([xy for xy, _ in seen], dtype=np.float64), [point for _, point in seen])
    return Observations(by_image=by_image)


def squared_errors(model: pycolmap.Reconstruction, observed: Observations) -> dict[int, Float64[ndarray, " n"]]:
    """Per image, each observation's squared reprojection error in `model`; NaN where the point is behind the camera."""
    errors: dict[int, Float64[ndarray, " n"]] = {}
    for image_id, (pixels, point_ids) in observed.by_image.items():
        image: pycolmap.Image = model.images[image_id]
        cam_from_world: Float64[ndarray, "3 4"] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        xyz: Float64[ndarray, "n 3"] = np.asarray([model.points3D[point_id].xyz for point_id in point_ids], dtype=np.float64)
        in_camera: Float64[ndarray, "n 3"] = xyz @ cam_from_world[:, :3].T + cam_from_world[:, 3]
        squared: Float64[ndarray, " n"] = np.full(len(point_ids), np.nan)
        ahead = in_camera[:, 2] > 0.0
        squared[ahead] = np.sum((np.asarray(model.cameras[image.camera_id].img_from_cam(in_camera[ahead])) - pixels[ahead]) ** 2, axis=1)
        errors[image_id] = squared
    return errors


def held(reconstruction: pycolmap.Reconstruction, cameras: Sequence[CatalogCameraModel], scale: float) -> pycolmap.Reconstruction:
    """A copy of the rig model with `cameras` installed and held (lens model, intrinsics, rig extrinsics), adjusted.

    Args:
        reconstruction: The rig model; camera `i` of `cameras` is its COLMAP camera `i + 1`, the first the rig origin.
        cameras: The calibration to hold, in any rig frame.
        scale: The calibration's translation units per model unit.

    Returns:
        The adjusted copy: frame poses and points moved, the calibration not.
    """
    model: pycolmap.Reconstruction = pycolmap.Reconstruction(reconstruction)
    rig: pycolmap.Rig = next(iter(model.rigs.values()))
    for camera_id, (camera, first_T_cam) in enumerate(zip(cameras, first_T_cams(cameras), strict=True), start=1):
        colmap: pycolmap.Camera = model.cameras[camera_id]
        name, params = colmap_model_and_parameters(camera.projection, camera.k_matrix, camera.distortion)
        colmap.model = getattr(pycolmap.CameraModelId, name)
        colmap.params = params.tolist()
        if camera_id > 1:
            cam_T_first: Float64[ndarray, "4 4"] = np.linalg.inv(first_T_cam)
            rig.set_sensor_from_rig(
                camera_sensor_id(camera_id), pycolmap.Rigid3d(pycolmap.Rotation3d(cam_T_first[:3, :3]), cam_T_first[:3, 3] / scale)
            )
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    pycolmap.bundle_adjustment(model, options)
    return model


def check_calibration(result: CalibrationResult, reference: Sequence[CatalogCameraModel]) -> tuple[CameraCheck, ...]:
    """Per camera, how `result` reprojects against `reference` on the same observations, each held, and how they differ.

    Args:
        result: A calibration, with its rig model; camera `i` is COLMAP camera `i + 1`.
        reference: The same cameras, in the same order, in any rig frame and units (each keeps its own lens model).

    Returns:
        One check per camera, in order.
    """
    estimated: tuple[CatalogCameraModel, ...] = result.cameras
    ours_rel, reference_rel = first_T_cams(estimated), first_T_cams(reference)
    scale: float = float(sum(r[:3, 3] @ e[:3, 3] for e, r in zip(ours_rel, reference_rel, strict=True)) / sum(e[:3, 3] @ e[:3, 3] for e in ours_rel))
    observed: Observations = observations(result.reconstruction)
    ours: dict[int, Float64[ndarray, " n"]] = squared_errors(held(result.reconstruction, estimated, 1.0), observed)
    theirs: dict[int, Float64[ndarray, " n"]] = squared_errors(held(result.reconstruction, reference, scale), observed)
    camera_of: dict[int, int] = {image_id: result.reconstruction.images[image_id].camera_id for image_id in observed.by_image}
    checks: list[CameraCheck] = []
    for camera_id, (camera, other, e, r) in enumerate(zip(estimated, reference, ours_rel, reference_rel, strict=True), start=1):
        mine: Float64[ndarray, " n"] = np.concatenate([ours[image_id] for image_id, owner in camera_of.items() if owner == camera_id])
        other_px: Float64[ndarray, " n"] = np.concatenate([theirs[image_id] for image_id, owner in camera_of.items() if owner == camera_id])
        both = np.isfinite(mine) & np.isfinite(other_px)  # one support for both: an observation lost in either arm counts in neither
        norms: float = float(np.linalg.norm(e[:3, 3]) * np.linalg.norm(r[:3, 3]))
        checks.append(
            CameraCheck(
                name=camera.name,
                registered_images=result.registered_images[camera.name],
                observations=int(both.sum()),
                dropped_observations=int((~both).sum()),
                reprojection_px=float(np.sqrt(np.mean(mine[both]))),
                reference_held_px=float(np.sqrt(np.mean(other_px[both]))),
                rotation_deg=float(np.degrees(Rotation.from_matrix(r[:3, :3].T @ e[:3, :3]).magnitude())),
                direction_deg=0.0 if norms == 0.0 else float(np.degrees(np.arccos(np.clip(e[:3, 3] @ r[:3, 3] / norms, -1.0, 1.0)))),
                focal_change_percent=float(
                    100.0 * ((camera.k_matrix[0, 0] + camera.k_matrix[1, 1]) / (other.k_matrix[0, 0] + other.k_matrix[1, 1]) - 1.0)
                ),
                principal_point_offset_px=float(np.linalg.norm(camera.k_matrix[:2, 2] - other.k_matrix[:2, 2])),
            )
        )
    return tuple(checks)


def check_table(checks: Sequence[CameraCheck]) -> str:
    """The checks as a Markdown table."""
    rows: list[str] = [
        "| camera | registered | observations (dropped) | ours px | reference held px | not worse | rotation deg | direction deg | focal % | principal point px |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows += [
        f"| {c.name} | {c.registered_images} | {c.observations} ({c.dropped_observations}) | {c.reprojection_px:.3f} | {c.reference_held_px:.3f} | {'yes' if c.not_worse else 'NO'} | "
        f"{c.rotation_deg:.2f} | {c.direction_deg:.2f} | {c.focal_change_percent:+.2f} | {c.principal_point_offset_px:.1f} |"
        for c in checks
    ]
    return "\n".join(rows)
