# SPDX-License-Identifier: Apache-2.0
"""Calibrated rig geometry, as COLMAP's stock rig support wants it described.

`colsfm.reconstruction` builds a `pycolmap.Rig` directly, which is what the
colsfm pipeline needs. The audit tools drive *stock* COLMAP instead, through
`pycolmap.apply_rig_config`, and that path wants the same geometry expressed as
a `RigConfig` of image prefixes. Both baseline tools used to carry their own
copy of that translation, of the `cam_from_rig` composition behind it, and of
the extrinsic comparison that scores what the mapper did to it.

The contract in one line: `cam_from_rig` is `cam_i_T_cam_ref`, and the metadata
gives `vehicle_T_cam`, so `cam_i_T_cam_ref = cam_i_T_vehicle * vehicle_T_cam_ref`
— exactly what `colsfm.reconstruction.build_rig` writes into a cuSFM
reconstruction. The reference sensor is the camera owning the lowest keyframe
id, the rule `FramesMeta.rig_frames` uses.

Dataset selection and experiment settings stay in the tools; nothing here reads
a path or decides which sequence to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from serde import serde

from colsfm.frames_meta import CameraParams, FramesMeta
from colsfm.geometry import MILLIMETRES_PER_METRE, relative_rotation_degrees
from colsfm.reconstruction import RigReference, cam_from_rig_by_camera_params_id


@serde
@dataclass(frozen=True)
class ExtrinsicChange:
    """How far one camera's rig extrinsic moved, whatever moved it.

    Two producers, one record. `extrinsic_deltas` here measures a mapper against the
    calibration — `sensor_from_rig` as COLMAP left it against
    `calibrated_cam_from_rig` — and `colsfm.run_report.extrinsic_changes` measures the
    regularised refinement against its input — `vehicle_T_cam` after the alternation
    against the `frames_meta.json` it started from. The transforms are inverses of one
    another and the pair of numbers is the same pair: millimetres between the origins,
    degrees between the rotations. They were two `@serde` records with four fields
    each under two spellings.
    """

    camera_params_id: int
    """The camera, by its `frames_meta.json` id."""
    sensor_name: str
    """The camera folder name, for reading the table."""
    translation_change_mm: float
    """Distance between the two extrinsics' origins, in millimetres."""
    rotation_change_deg: float
    """Geodesic angle between the two extrinsics' rotations, in degrees."""
    is_reference: bool = False
    """Whether this is the rig origin and fixed camera, whose change is exactly zero.

    `extrinsic_deltas` never emits one — the reference camera's `cam_from_rig` is the
    identity by construction, so there is nothing to compare — and the refinement
    reports it so that a reader can see the gauge was held."""


def calibrated_cam_from_rig(frames_meta: FramesMeta, reference: RigReference) -> dict[int, pycolmap.Rigid3d]:
    """The calibration's `cam_from_rig` per camera, the reference at identity.

    `colsfm.reconstruction.cam_from_rig_by_camera_params_id` is the algebra, and
    `build_rig` writes the very same transforms into a cuSFM reconstruction — which
    is the point: this function's answer has to be what the mapper was given.

    Args:
        frames_meta: The parsed metadata.
        reference: The rig origin and its `vehicle_T_cam_ref` bridge.

    Returns:
        `cam_from_rig` per `camera_params_id`.
    """
    return cam_from_rig_by_camera_params_id(frames_meta, reference)


def rig_config(frames_meta: FramesMeta, reference: RigReference) -> pycolmap.RigConfig:
    """Describe the rig the way `pycolmap.apply_rig_config` wants it.

    Args:
        frames_meta: The parsed metadata.
        reference: The rig origin and its bridge.

    Returns:
        A config listing every camera by `image_prefix`, one of them the
        reference sensor and the rest carrying `cam_from_rig`.
    """
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(frames_meta, reference)
    # Reference first: `Rig::AddSensor` refuses to run before a reference sensor exists
    # (`rig.cc:42`), and `apply_rig_config` walks this list in order.
    ordered: list[int] = sorted(frames_meta.cameras, key=lambda item: (item != reference.camera_params_id, item))
    cameras: list[pycolmap.RigConfigCamera] = []
    for camera_params_id in ordered:
        camera: CameraParams = frames_meta.cameras[camera_params_id]
        entry: pycolmap.RigConfigCamera = pycolmap.RigConfigCamera()
        entry.image_prefix = f"{camera.sensor_name}/"
        entry.ref_sensor = camera_params_id == reference.camera_params_id
        if not entry.ref_sensor:
            entry.cam_from_rig = calibrated[camera_params_id]
        cameras.append(entry)
    config: pycolmap.RigConfig = pycolmap.RigConfig()
    config.cameras = cameras
    return config


def rotation_degrees(delta: pycolmap.Rigid3d) -> float:
    """Geodesic angle of a rigid transform's rotation.

    The angle against the identity rotation, through `colsfm.geometry`'s own
    comparison rather than a second copy of the `acos((trace - 1) / 2)` form the
    two baseline tools each carried. The two agree to 1e-11 degrees on random
    rotations; the quaternion form is the better conditioned of them near zero,
    which is exactly where these deltas sit.

    Args:
        delta: The transform.

    Returns:
        The angle in degrees, in `[0, 180]`.
    """
    return relative_rotation_degrees(pycolmap.Rigid3d(), delta)


def camera_params_id_by_camera_id(reconstruction: pycolmap.Reconstruction, frames_meta: FramesMeta) -> dict[int, int]:
    """Map COLMAP's own camera ids back onto `camera_params_id`, through the folders.

    Args:
        reconstruction: A model whose image names are the dataset's `image_name`s.
        frames_meta: The parsed metadata, for the folder-to-camera map.

    Returns:
        `camera_params_id` per COLMAP `camera_id`, for the cameras the model kept.
    """
    camera_params_id_by_folder: dict[str, int] = {
        camera.sensor_name: camera_params_id for camera_params_id, camera in frames_meta.cameras.items()
    }
    return {image.camera_id: camera_params_id_by_folder[image.name.split("/")[0]] for image in reconstruction.images.values()}


def extrinsic_deltas(
    reconstruction: pycolmap.Reconstruction, frames_meta: FramesMeta, reference: RigReference
) -> tuple[ExtrinsicChange, ...]:
    """Compare a model's recovered rig extrinsics against the calibration.

    `IncrementalPipelineOptions.ba_refine_sensor_from_rig` defaults to on, so a
    stock COLMAP mapper is free to move these; how far it moved them is a quality
    number in its own right. Every entry is zero when the switch was held off,
    which is the CASPAR path.

    Args:
        reconstruction: The model, already renamed to dataset image names.
        frames_meta: The parsed metadata.
        reference: The rig origin used to build the config.

    Returns:
        One change per non-reference camera the model kept, ordered by `camera_params_id`.
    """
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(frames_meta, reference)
    params_id_by_camera_id: dict[int, int] = camera_params_id_by_camera_id(reconstruction, frames_meta)
    rig: pycolmap.Rig = next(iter(reconstruction.rigs.values()))
    deltas: list[ExtrinsicChange] = []
    for camera_id, camera_params_id in sorted(params_id_by_camera_id.items(), key=lambda item: item[1]):
        if camera_params_id == reference.camera_params_id:
            continue
        sensor_id: pycolmap.sensor_t = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
        if not rig.has_sensor(sensor_id):
            continue
        difference: pycolmap.Rigid3d = rig.sensor_from_rig(sensor_id) * calibrated[camera_params_id].inverse()
        deltas.append(
            ExtrinsicChange(
                camera_params_id=camera_params_id,
                sensor_name=frames_meta.cameras[camera_params_id].sensor_name,
                translation_change_mm=MILLIMETRES_PER_METRE * float(np.linalg.norm(np.asarray(difference.translation, dtype=np.float64))),
                rotation_change_deg=rotation_degrees(difference),
            )
        )
    return tuple(deltas)
