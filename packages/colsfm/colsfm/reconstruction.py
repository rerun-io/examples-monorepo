"""A `pycolmap.Reconstruction` built from cuSFM's `frames_meta.json`.

This is the data-model half of `keypoints_mapper_main`: the rig, the frames and
the images the triangulator and the bundle adjuster then work on
(keypoints_mapper_main.md §9.1). No triangulation happens here — see
`colsfm.mapping`.

## Two reference sensors, and which one each job needs

cuSFM's `VEHICLE_RIG` mode carries one pose per rig instant (`synced_sample_id`)
in the **vehicle** FLU frame, plus one shared `sensor_to_vehicle_transform` per
camera (§5.2, §6.2). COLMAP expresses exactly that with a `Rig` whose
`sensor_from_rig` per camera is `cam_T_vehicle` — the **inverse** of the
metadata's `vehicle_T_cam` — and a `Frame` per rig instant whose
`rig_from_world` is `vehicle_T_world`.

COLMAP forces the rig's **reference** sensor to identity
(`rig.h:200 Check failed: sensor_id != ref_sensor_id_`), so no camera can be the
reference without moving the rig origin onto that camera. With
`reference_camera_params_id=None` the vehicle body is therefore registered as a
non-camera reference sensor, `sensor_t(SensorType.IMU, 0)`, which owns no data.
Measured: triangulation, `ObservationManager` and
`create_default_bundle_adjuster` all accept a rig whose reference sensor has no
images, and `frame.rig_from_world` is then literally `vehicle_T_world`, which
keeps every comparison against cuSFM's vehicle poses a direct one.

**That rig cannot have its extrinsics refined.** COLMAP's
`ParameterizeRigsAndFrames` (`estimators/bundle_adjustment_ceres.cc`) ends with
"Set the rig poses as constant, if the reference sensor is not part of the
problem. Otherwise, the relative pose between the sensors is not well
constrained": for every rig whose reference sensor is absent from the
parameterised sensors it calls `SetParameterBlockConstant` on **every**
non-reference `sensor_from_rig`. A data-less IMU reference is never
parameterised, so `refine_sensor_from_rig = True` silently does nothing — a
20 mm perturbation of one camera survives bundle adjustment untouched
(NOTES.md's colsfm gotchas). COLMAP's own rig documentation
(<https://colmap.github.io/rigs.html>) describes the intended shape: "one camera
would be defined as the reference sensor and have an identity `sensor_from_rig`
pose, whereas the second camera would be posed relative to the reference
camera", refined through `--Mapper.ba_refine_sensor_from_rig`.

Passing a `reference_camera_params_id` therefore builds exactly that rig:

```
sensor_from_rig(cam_i) = cam_i_T_cam_ref = cam_i_T_vehicle * vehicle_T_cam_ref
frame.rig_from_world   = cam_ref_T_world = cam_ref_T_vehicle * vehicle_T_world
```

Every `cam_T_world` is unchanged — it is the same problem under a fixed change
of basis — but the reference camera is now a real, parameterised sensor, so the
other seven extrinsics move. `RigReference` owns the inverse bookkeeping: given
the adjusted reconstruction and the input `vehicle_T_cam_ref` (which bundle
adjustment never sees, and which the mapper's fixed camera holds still), it
returns `vehicle_T_cam` per camera and `world_T_vehicle` per frame.

## Identifiers are cuSFM's own, verbatim

| cuSFM | COLMAP |
|---|---|
| `KeyframeMetaData.id` | `image_id` |
| `camera_params_id` | `camera_id` |
| `synced_sample_id` | `frame_id` |

All three are zero-based in RoboCap and COLMAP accepts 0 for each, so nothing is
renumbered. That is what lets a database written by the feature and matching
stages address the same images by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pycolmap

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame

RIG_ID: Final[int] = 1
"""The single rig every reconstruction carries; cuSFM has no multi-rig concept."""

VEHICLE_SENSOR_ID: Final[int] = 0
"""Sensor id of the vehicle body, the rig's reference sensor."""


def vehicle_sensor_id() -> pycolmap.sensor_t:
    """The rig's reference sensor: the vehicle body, which owns no images.

    Returns:
        A non-camera `sensor_t` standing for the vehicle FLU frame.
    """
    return pycolmap.sensor_t(pycolmap.SensorType.IMU, VEHICLE_SENSOR_ID)


def camera_sensor_id(camera_params_id: int) -> pycolmap.sensor_t:
    """The rig sensor of one cuSFM camera.

    Args:
        camera_params_id: cuSFM's `camera_params_id`, used verbatim as the COLMAP camera id.

    Returns:
        The camera's `sensor_t`.
    """
    return pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_params_id)


@dataclass(frozen=True, slots=True)
class RigReference:
    """Which sensor the rig origin sits on, and the fixed bridge back to the vehicle.

    Built by `rig_reference`; the inverse of what `build_rig` and
    `build_reconstruction` do, and the only place that inversion is written.
    """

    camera_params_id: int | None
    """Reference camera, or None when the vehicle body is the reference sensor."""
    vehicle_T_reference: pycolmap.Rigid3d
    """`vehicle_T_cam_ref` as the input metadata declared it; identity for the vehicle
    reference. Bundle adjustment never holds a parameter block for it — the reference
    sensor's `sensor_from_rig` is identity by construction — so it stays exactly as the
    input had it and is what turns adjusted rig quantities back into vehicle-frame ones."""

    def sensor_id(self) -> pycolmap.sensor_t:
        """The rig's reference `sensor_t`.

        Returns:
            The reference camera's sensor id, or the vehicle body's.
        """
        return vehicle_sensor_id() if self.camera_params_id is None else camera_sensor_id(self.camera_params_id)

    def sensor_from_rig(self, camera_params_id: int, reconstruction: pycolmap.Reconstruction) -> pycolmap.Rigid3d:
        """One camera's `sensor_from_rig`, identity for the reference camera.

        `Rig.sensor_from_rig` raises on the reference sensor rather than returning
        the identity it is fixed to (`rig.h:200`), so that case is answered here.

        Args:
            camera_params_id: The camera to read.
            reconstruction: The model holding the rig.

        Returns:
            `cam_T_rig` for that camera.
        """
        if camera_params_id == self.camera_params_id:
            return pycolmap.Rigid3d()
        return reconstruction.rig(RIG_ID).sensor_from_rig(camera_sensor_id(camera_params_id))

    def vehicle_T_cam_by_camera_params_id(self, reconstruction: pycolmap.Reconstruction) -> dict[int, pycolmap.Rigid3d]:
        """Read every camera's `sensor_to_vehicle_transform` back out of the rig.

        `vehicle_T_cam_i = vehicle_T_cam_ref * cam_ref_T_cam_i`, which for the
        vehicle reference degenerates to `sensor_from_rig(cam_i)^-1`.

        Args:
            reconstruction: The model, adjusted or not.

        Returns:
            `vehicle_T_cam` per `camera_params_id`.
        """
        return {
            camera_params_id: self.vehicle_T_reference * self.sensor_from_rig(camera_params_id, reconstruction).inverse()
            for camera_params_id in sorted(reconstruction.cameras)
        }

    def world_T_vehicle_by_frame_id(self, reconstruction: pycolmap.Reconstruction) -> dict[int, pycolmap.Rigid3d]:
        """Read every rig frame's vehicle pose back out of the frames.

        `world_T_vehicle = world_T_cam_ref * cam_ref_T_vehicle`, which for the
        vehicle reference degenerates to `rig_from_world^-1`.

        Args:
            reconstruction: The model, adjusted or not.

        Returns:
            `world_T_vehicle` per `frame_id`, for every frame carrying a pose.
        """
        reference_T_vehicle: pycolmap.Rigid3d = self.vehicle_T_reference.inverse()
        return {
            frame_id: frame.rig_from_world.inverse() * reference_T_vehicle
            for frame_id, frame in reconstruction.frames.items()
            if frame.has_pose
        }


def rig_reference(frames_meta: FramesMeta, reference_camera_params_id: int | None = None) -> RigReference:
    """Describe the rig origin a reconstruction built with the same argument uses.

    Args:
        frames_meta: The parsed metadata, for the reference camera's extrinsic.
        reference_camera_params_id: Camera to put the rig origin on, or None for
            the vehicle body.

    Returns:
        The reference and the fixed `vehicle_T_cam_ref` bridge.

    Raises:
        KeyError: When the metadata carries no such `camera_params_id`.
    """
    if reference_camera_params_id is None:
        return RigReference(camera_params_id=None, vehicle_T_reference=pycolmap.Rigid3d())
    return RigReference(
        camera_params_id=reference_camera_params_id,
        vehicle_T_reference=frames_meta.cameras[reference_camera_params_id].vehicle_T_cam,
    )


def cam_from_rig_by_camera_params_id(frames_meta: FramesMeta, reference: RigReference) -> dict[int, pycolmap.Rigid3d]:
    """`sensor_from_rig` per camera against a chosen rig origin.

    `cam_i_T_cam_ref = cam_i_T_vehicle * vehicle_T_cam_ref`, which is what `build_rig`
    writes into the reconstruction and what `colsfm.rig_calibration` compares a
    mapper's answer against. One expression, so a comparison against the calibration
    cannot be measuring a different calibration.

    Args:
        frames_meta: The parsed metadata.
        reference: The rig origin and its `vehicle_T_cam_ref` bridge.

    Returns:
        `cam_from_rig` per `camera_params_id`, ascending; the reference camera's is
        the identity by construction.
    """
    return {
        camera_params_id: camera.vehicle_T_cam.inverse() * reference.vehicle_T_reference
        for camera_params_id, camera in sorted(frames_meta.cameras.items())
    }


def build_rig(frames_meta: FramesMeta, reference_camera_params_id: int | None = None) -> pycolmap.Rig:
    """Build the rig: one sensor per `camera_params_id`, posed against the reference.

    Args:
        frames_meta: The parsed metadata.
        reference_camera_params_id: Camera to put the rig origin on, or None to
            use the vehicle body (the module docstring says when each is wanted).

    Returns:
        A rig whose camera sensors each carry `sensor_from_rig = cam_T_cam_ref`,
        which is `cam_T_vehicle` for the vehicle reference.

    Raises:
        KeyError: When the metadata carries no such `camera_params_id`.
    """
    reference: RigReference = rig_reference(frames_meta, reference_camera_params_id)
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(reference.sensor_id())
    for camera_params_id, cam_from_rig in cam_from_rig_by_camera_params_id(frames_meta, reference).items():
        if camera_params_id == reference_camera_params_id:
            continue
        rig.add_sensor(camera_sensor_id(camera_params_id), cam_from_rig)
    return rig


@dataclass(frozen=True, slots=True)
class PosedModel:
    """A reconstruction and the rig reference it was built with, which belong together.

    Every inverse-bookkeeping question — what is this camera's
    `sensor_to_vehicle_transform`, where is this rig frame's vehicle pose — needs both
    halves, and pairing them by hand at each call site is how a vehicle-referenced model
    ends up read as a camera-referenced one. `build_reconstruction` returns the pair, so
    nothing downstream has to re-derive it.
    """

    reconstruction: pycolmap.Reconstruction
    """The model: one rig, one frame per `synced_sample_id`, one image per keyframe."""
    reference: RigReference
    """Where the rig origin sits, and the fixed `vehicle_T_cam_ref` bridge back to the vehicle."""


def build_reconstruction(
    frames_meta: FramesMeta,
    cameras: dict[int, pycolmap.Camera] | None = None,
    reference_camera_params_id: int | None = None,
) -> PosedModel:
    """Build a registered, pose-carrying reconstruction with no 3D points yet.

    Every rig frame is registered, so `IncrementalTriangulator` will triangulate
    against the supplied trajectory rather than re-estimating it. Every camera
    carries `has_prior_focal_length = True` — the calibration comes from the
    metadata, and leaving the flag false silently degrades every downstream
    two-view geometry (pycolmap-capabilities.md §5). That flag is set where the
    cameras are made, in `colsfm.cameras.colmap_camera`, so a caller-supplied
    `cameras` map from `colmap_cameras` already carries it.

    Args:
        frames_meta: The parsed `frames_meta.json`.
        cameras: COLMAP cameras keyed by `camera_params_id`; derived from the
            metadata by `colsfm.cameras.colmap_cameras` when None.
        reference_camera_params_id: Camera to put the rig origin on, or None to
            use the vehicle body. Extrinsic refinement needs a camera; every
            other job wants the vehicle. See the module docstring.

    Returns:
        The model and the `RigReference` it was built with. Every `cam_T_world` is
        the same whichever reference was asked for.

    Raises:
        KeyError: When a keyframe names a `camera_params_id` the cameras lack,
            or when the metadata carries no `reference_camera_params_id`.
    """
    reference: RigReference = rig_reference(frames_meta, reference_camera_params_id)
    reference_T_vehicle: pycolmap.Rigid3d = reference.vehicle_T_reference.inverse()
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    resolved_cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta) if cameras is None else cameras
    for camera_params_id in sorted(resolved_cameras):
        reconstruction.add_camera(resolved_cameras[camera_params_id])
    reconstruction.add_rig(build_rig(frames_meta, reference_camera_params_id))

    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    for rig_frame in frames_meta.rig_frames():
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = rig_frame.synced_sample_id
        frame.rig_id = RIG_ID
        frame.rig_from_world = reference_T_vehicle * rig_frame.world_T_vehicle.inverse()
        for keyframe_id in rig_frame.keyframe_ids:
            frame.add_data_id(
                pycolmap.data_t(camera_sensor_id(keyframe_by_id[keyframe_id].camera_params_id), keyframe_id)
            )
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        # The images go in after their frame is registered, so one pass over the rig
        # frames builds the whole model.
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe: KeyframeMeta = keyframe_by_id[keyframe_id]
            image: pycolmap.Image = pycolmap.Image(
                name=keyframe.image_name, camera_id=keyframe.camera_params_id, image_id=keyframe_id
            )
            image.frame_id = rig_frame.synced_sample_id
            reconstruction.add_image(image)
    return PosedModel(reconstruction=reconstruction, reference=reference)


def registered_image_names(reconstruction: pycolmap.Reconstruction) -> set[str]:
    """Names of the images that observe at least one 3D point.

    **This is not `colsfm.mapping.registered_image_ids`.** That one means "belongs to a
    registered frame", which every image of a `build_reconstruction` model does from the
    start, whether or not it ever sees a point. *This* one is the sense `colsfm.benchmark`,
    `colsfm.export` and the audits use — a COLMAP text export writes exactly these images
    with a non-empty track list, and the blob's own `sparse/` holds 28 of Galileo's 32
    keyframes for the same reason.

    Args:
        reconstruction: The model.

    Returns:
        The image names with at least one observation.
    """
    return {image.name for image in reconstruction.images.values() if image.num_points3D > 0}


def num_registered_images(reconstruction: pycolmap.Reconstruction) -> int:
    """Count the images that observe at least one 3D point.

    **This is not `colsfm.mapping.registered_image_ids`.** See
    `registered_image_names`, whose cardinality this is: the count a COLMAP export
    writes, and the one `colsfm.benchmark`, `colsfm.export` and the audits compare
    against the blob.

    Args:
        reconstruction: The model.

    Returns:
        The number of images with at least one observation.

    Note:
        `Image.num_points2D` is a method while `Image.num_points3D` is a
        property — one of pycolmap 4.2's several method/property splits.
    """
    return sum(1 for image in reconstruction.images.values() if image.num_points3D > 0)


def gauge_camera_params_id(frames_meta: FramesMeta) -> int:
    """The camera cuSFM's constant keyframe belongs to.

    cuSFM fixes one keyframe — in Galileo the lowest id in the map
    (keypoints_mapper_main.md §6.3) — and when extrinsics are refined
    `--fixed_camera_name` pins one camera's extrinsic as well. Using the same
    camera for both, and for the rig origin, is what makes `vehicle_T_cam_ref`
    an input constant the solve never touches, so `RigReference` can invert the
    bookkeeping exactly.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The `camera_params_id` of the lowest keyframe id.

    Raises:
        ValueError: When the collection holds no keyframes.
    """
    if not frames_meta.keyframes:
        raise ValueError("Cannot choose a gauge camera: the metadata holds no keyframes")
    return min(frames_meta.keyframes, key=lambda keyframe: keyframe.keyframe_id).camera_params_id


def gauge_rig_frame(frames_meta: FramesMeta) -> RigFrame:
    """The rig frame whose pose bundle adjustment holds constant.

    cuSFM fixes a single keyframe — in Galileo the lowest id in the map — and
    through the shared rig that fixes its whole rig frame
    (keypoints_mapper_main.md §6.3), which is why rig frame 1 moves 0.0000 mm
    while later frames move millimetres.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The rig frame containing the lowest keyframe id.

    Raises:
        ValueError: When the collection holds no keyframes.
    """
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    if not rig_frames:
        raise ValueError("Cannot choose a gauge rig frame: the metadata holds no keyframes")
    lowest_keyframe_id: int = min(keyframe.keyframe_id for keyframe in frames_meta.keyframes)
    return next(rig_frame for rig_frame in rig_frames if lowest_keyframe_id in rig_frame.keyframe_ids)
