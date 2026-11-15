"""Typed access to cuSFM's `frames_meta.json` (`KeyframesMetadataCollection`).

The file is proto3 JSON with `preserve_proto_field_names`, which means **fields
equal to their default are omitted**: 29 of Galileo's 226 keyframes carry no
`camera_params_id` because theirs is 0 (NOTES.md gotcha 8). Parsing therefore
goes through protobuf's own `json_format` against the schema in `colsfm.schema`,
so defaults are supplied by protobuf rather than by hand.

`FramesMeta` keeps the parsed message alongside the typed view, but the **typed
fields are the authority**: `FramesMeta.current_message` copies the message as a
template and writes the typed fields over it, so an edit made with
`dataclasses.replace` reaches serialisation, filtering and both update methods
alike. Only differing fields are written, so fields the pipeline never models —
`pose_covariance`, `car_mask_polygon`, `chroma`, `raw_image_cropping` — survive a
round trip untouched, exactly as `update_keyframe_pose_main` preserves them
(export.md §5.2), and an unedited collection re-serialises byte for byte.

`message` stays a public field on purpose. Hiding it behind a facade would rename
one attribute and delete nothing: the overlay in `current_message` is what makes
the typed fields authoritative, and it is already the only path serialisation,
filtering, `with_camera_to_world` and `with_extrinsics` take. The template is
readable data, not a second source of truth.

Conventions (export.md §2, keypoints_mapper_main.md §4.3):

- `camera_to_world` **is** `world_T_cam`; axis-angle in degrees, translation in metres.
- `sensor_meta_data.sensor_to_vehicle_transform` **is** `vehicle_T_cam`.
- `synced_sample_id` groups the cameras of one rig instant.
- `world_T_vehicle(sample) = world_T_cam(ref) * vehicle_T_cam(ref)^-1`, where `ref`
  is the **lowest keyframe id** in the sample (pose_graph_main.md §4).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from google.protobuf import json_format
from google.protobuf.message import Message
from jaxtyping import Float64
from numpy import ndarray

from colsfm.geometry import AxisAngleDegrees, axis_angle_degrees_from_rigid3d, rigid3d_from_axis_angle_degrees
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, CusfmSchema, load_schema

CameraProjectionModel: TypeAlias = Literal["PINHOLE", "DISTORTED_PINHOLE", "FTHETA_WINDSHIELD", "OPENCV_FISHEYE"]
"""cuSFM's four camera projection models (`CameraProjectionModelType`)."""

InitialPoseType: TypeAlias = Literal["ALIGNMENT", "EGO_MOTION", "SESSION_EGO_MOTION", "GPS_IMU", "MAP_POSE"]
"""Where the poses in the collection came from; the mapper flips `EGO_MOTION` to `ALIGNMENT`."""

ProjectionMatrix: TypeAlias = Float64[ndarray, "3 4"]
"""Rectified 3x4 projection matrix, row-major, as `calibration_parameters.projection_matrix`."""

CameraMatrix: TypeAlias = Float64[ndarray, "3 3"]
"""Intrinsic 3x3 matrix, row-major, as `calibration_parameters.camera_matrix`."""

DistortionCoefficients: TypeAlias = Float64[ndarray, "num_coefficients"]
"""Distortion coefficients in OpenCV order, however many the calibration carries."""

ScalarField: TypeAlias = bool | int | float | str
"""A protobuf scalar this module writes back; `int` stays distinct from `float` for beartype."""

JSON_INDENT: Final[int] = 2
"""Indent cuSFM's own writer uses, so our output diffs cleanly against the blob's."""

FRAMES_META_NAME: Final[str] = "frames_meta.json"
"""The metadata file name, in an input directory and in every output directory."""


@dataclass(frozen=True, slots=True)
class KeyframeMeta:
    """One camera keyframe: identity, timing and pose."""

    keyframe_id: int
    """`id`; the COLMAP `IMAGE_ID`. Sparse and not 1-based."""
    camera_params_id: int
    """Sensor id; the COLMAP `CAMERA_ID`. Omitted from JSON when 0."""
    timestamp_microseconds: int
    """Capture time in microseconds since the Unix epoch."""
    image_name: str
    """`<camera_name>/<timestamp_nanoseconds>.jpeg`, relative to the raw data directory."""
    world_T_cam: pycolmap.Rigid3d
    """`camera_to_world`: the camera's pose in the world frame."""
    synced_sample_id: int
    """Rig frame id; every camera of one synchronised capture shares it."""
    track_id: int
    """Session segment id."""


@dataclass(frozen=True, slots=True)
class CameraParams:
    """One entry of `camera_params_id_to_camera_params`: a `CameraSensor`."""

    camera_params_id: int
    """Key in the map; the COLMAP `CAMERA_ID`."""
    sensor_name: str
    """Camera folder name, e.g. `front_stereo_camera_left`."""
    model_name: str
    """Hardware model string; empty in both shipped datasets."""
    sensor_id: int
    """Sensor id inside `sensor_meta_data`; not the map key."""
    frequency_hz: float
    """Capture rate in hertz."""
    vehicle_T_cam: pycolmap.Rigid3d
    """`sensor_to_vehicle_transform`: the rig extrinsic, camera to vehicle (FLU)."""
    image_width: int
    """Image width in pixels."""
    image_height: int
    """Image height in pixels."""
    projection_matrix: ProjectionMatrix | None
    """Rectified 3x4 projection matrix, or None when the calibration omits it."""
    camera_matrix: CameraMatrix | None
    """Unrectified 3x3 intrinsic matrix, or None when the calibration omits it."""
    distortion_coefficients: DistortionCoefficients
    """Distortion coefficients; empty when the calibration omits them."""
    projection_model: CameraProjectionModel
    """Which of cuSFM's four projection models this camera uses."""
    rolling_shutter_delay_microseconds: int
    """Row-to-row readout delay; 0 when global shutter or unset."""


@dataclass(frozen=True, slots=True)
class StereoPair:
    """A declared rectified stereo pair."""

    left_camera_params_id: int
    """`camera_params_id` of the left camera. Omitted from JSON when 0."""
    right_camera_params_id: int
    """`camera_params_id` of the right camera."""
    baseline_meters: float
    """Baseline in metres, despite what the surrounding units suggest."""


@dataclass(frozen=True, slots=True)
class RigFrame:
    """One rig instant: the cameras that fired together and their shared pose."""

    synced_sample_id: int
    """The rig frame id all member keyframes carry."""
    keyframe_ids: tuple[int, ...]
    """Member keyframe ids, ascending."""
    reference_keyframe_id: int
    """Lowest member keyframe id; the one the rig pose is derived from."""
    world_T_vehicle: pycolmap.Rigid3d
    """Rig pose in the world frame, in the vehicle FLU frame."""
    timestamp_microseconds: int
    """Timestamp of the reference keyframe."""


@dataclass(frozen=True, slots=True)
class FramesMeta:
    """A parsed `frames_meta.json`, with the source message kept for lossless writes."""

    keyframes: tuple[KeyframeMeta, ...]
    """Camera keyframes in file order."""
    cameras: dict[int, CameraParams]
    """Calibration and rig extrinsic per `camera_params_id`."""
    stereo_pairs: tuple[StereoPair, ...]
    """Declared stereo pairs; empty when the metadata declares none."""
    initial_pose_type: InitialPoseType
    """Provenance of `camera_to_world` for the whole collection."""
    detector_type: str
    """Keypoint detector the keyframes were produced with, e.g. `ALIKED_DETECTOR`."""
    descriptor_type: str
    """Descriptor type, e.g. `ALIKED_DESCRIPTOR`."""
    session_name_by_camera_params_id: dict[int, str]
    """`camera_params_id_to_session_name`; the TUM output subdirectory per camera."""
    track_name_by_track_id: dict[int, str]
    """`track_id_to_track_name`; empty in both shipped datasets."""
    bag_name_by_track_id: dict[int, str]
    """`track_id_to_bag_name`; empty in both shipped datasets."""
    keypoint_feature_has_depth: bool
    """Whether keypoints carry a depth channel (RGBD or stereo-as-RGBD modes)."""
    local_sector_key: str
    """Map sector key; empty in both shipped datasets."""
    message: Message
    """The parsed `KeyframesMetadataCollection`, kept as the template writers copy.

    It carries the fields above as they were read plus the ones this module does not
    model; the typed fields win wherever the two disagree (see `current_message`)."""

    def keyframe_by_id(self) -> dict[int, KeyframeMeta]:
        """Index the keyframes by their `id`.

        Returns:
            Keyframes keyed by `keyframe_id`.
        """
        return {keyframe.keyframe_id: keyframe for keyframe in self.keyframes}

    def world_T_vehicle(self, keyframe: KeyframeMeta) -> pycolmap.Rigid3d:
        """Derive the vehicle pose implied by one keyframe.

        `world_T_vehicle = world_T_cam * cam_T_vehicle`, the composition
        `extract_pose_from_map_main` writes to its TUM files (export.md §4.2).

        Args:
            keyframe: A keyframe of this collection.

        Returns:
            The rig pose in the world frame.

        Raises:
            KeyError: When the keyframe's `camera_params_id` has no calibration.
        """
        vehicle_T_cam: pycolmap.Rigid3d = self.cameras[keyframe.camera_params_id].vehicle_T_cam
        return keyframe.world_T_cam * vehicle_T_cam.inverse()

    def rig_frames(self) -> tuple[RigFrame, ...]:
        """Group the keyframes into rig frames by `synced_sample_id`.

        The rig pose comes from the **lowest keyframe id** of the sample, not from
        an average and not from the lowest `camera_params_id`: those two rules
        disagree on 28 of Galileo's 29 samples, and only the lowest id reproduces
        cuSFM's own `vehicle_frames_meta.json` (to 1.4e-16 m, against 1.4e-5 m for
        the other cameras of the same sample). See pose_graph_main.md §4.

        Returns:
            Rig frames ordered by `synced_sample_id`, which is also time order.
        """
        members: dict[int, list[KeyframeMeta]] = {}
        for keyframe in self.keyframes:
            members.setdefault(keyframe.synced_sample_id, []).append(keyframe)
        rig_frames: list[RigFrame] = []
        for synced_sample_id in sorted(members):
            group: list[KeyframeMeta] = sorted(members[synced_sample_id], key=lambda keyframe: keyframe.keyframe_id)
            reference: KeyframeMeta = group[0]
            rig_frames.append(
                RigFrame(
                    synced_sample_id=synced_sample_id,
                    keyframe_ids=tuple(keyframe.keyframe_id for keyframe in group),
                    reference_keyframe_id=reference.keyframe_id,
                    world_T_vehicle=self.world_T_vehicle(reference),
                    timestamp_microseconds=reference.timestamp_microseconds,
                )
            )
        return tuple(rig_frames)

    def current_message(self) -> Message:
        """The source message with the typed fields written over it.

        The typed fields are the authority: `dataclasses.replace` on this frozen
        collection (or on one of its cameras or keyframes) has to reach every
        consumer. `message` stays the template, so the fields the pipeline never
        models — `pose_covariance`, `car_mask_polygon`, `chroma`,
        `raw_image_cropping` — survive untouched.

        Each field is written only when it differs from what the template already
        holds, which keeps an unedited collection byte-identical through
        `to_json`: no write, no protobuf presence change, no axis-angle rounding.

        Returns:
            A fresh copy of `message`, never `message` itself.
        """
        message: Message = copy.deepcopy(self.message)
        _apply_keyframes(message, self.keyframes)
        _apply_cameras(message, self.cameras)
        _apply_stereo_pairs(message, self.stereo_pairs)
        _apply_collection_fields(message, self)
        return message

    def to_json(self) -> str:
        """Serialise back to proto3 JSON in cuSFM's own layout.

        Serialisation goes through `current_message`, so an edited typed field is
        what lands in the file. Field names stay snake_case and default-valued
        fields stay omitted, so the output diffs against a blob-written file down
        to map entry order (protobuf maps have none) and the enum values cuSFM
        prints even at their default.

        Returns:
            The JSON text, without a trailing newline.
        """
        return json_format.MessageToJson(self.current_message(), indent=JSON_INDENT, preserving_proto_field_name=True)

    def with_camera_to_world(
        self,
        world_T_cam_by_keyframe_id: Mapping[int, pycolmap.Rigid3d],
        initial_pose_type: InitialPoseType | None = None,
    ) -> FramesMeta:
        """Replace keyframe poses, keeping every other field byte-identical.

        This is what `keypoints_mapper_main` writes after bundle adjustment: the
        input collection with `camera_to_world` swapped and `initial_pose_type`
        moved from `EGO_MOTION` to `ALIGNMENT` (keypoints_mapper_main.md §4.4).
        Keyframes missing from the mapping keep their original pose.

        Args:
            world_T_cam_by_keyframe_id: New camera poses keyed by keyframe id.
            initial_pose_type: New provenance, or None to keep the current one.

        Returns:
            A new `FramesMeta`.
        """
        message: Message = self.current_message()
        for entry in message.keyframes_metadata:
            pose: pycolmap.Rigid3d | None = world_T_cam_by_keyframe_id.get(int(entry.id))
            if pose is None:
                continue
            write_rigid_transform(entry.camera_to_world, pose)
        if initial_pose_type is not None:
            message.initial_pose_type = message.DESCRIPTOR.fields_by_name["initial_pose_type"].enum_type.values_by_name[
                initial_pose_type
            ].number
        return parse_message(message)

    def with_extrinsics(self, vehicle_T_cam_by_camera_params_id: Mapping[int, pycolmap.Rigid3d]) -> FramesMeta:
        """Replace rig extrinsics, keeping every other field byte-identical.

        This is what `keypoints_mapper_main --optimize_extrinsics=True` writes back:
        the same collection with each camera's
        `sensor_meta_data.sensor_to_vehicle_transform` swapped
        (keypoints_mapper_main.md §6.3). Cameras missing from the mapping keep
        their original extrinsic; `camera_to_world` is **not** re-derived here, so a
        caller refining extrinsics passes the matching poses to
        `with_camera_to_world` as well.

        Args:
            vehicle_T_cam_by_camera_params_id: New `vehicle_T_cam` per `camera_params_id`.

        Returns:
            A new `FramesMeta`.
        """
        message: Message = self.current_message()
        for camera_params_id, camera_sensor in message.camera_params_id_to_camera_params.items():
            pose: pycolmap.Rigid3d | None = vehicle_T_cam_by_camera_params_id.get(int(camera_params_id))
            if pose is None:
                continue
            write_rigid_transform(camera_sensor.sensor_meta_data.sensor_to_vehicle_transform, pose)
        return parse_message(message)

    def filtered(
        self,
        keyframe_ids: Sequence[int] | set[int],
        synced_sample_id_by_keyframe_id: Mapping[int, int] | None = None,
    ) -> FramesMeta:
        """Keep a subset of keyframes, optionally renumbering their rig frame ids.

        `feature_extractor_main` writes exactly this: the surviving keyframes with
        `synced_sample_id` rewritten to a dense 1-based rank in time order
        (feature_extractor_main.md §4 and §8.1).

        Args:
            keyframe_ids: Keyframe ids to keep.
            synced_sample_id_by_keyframe_id: New `synced_sample_id` per kept keyframe
                id, or None to keep the existing ids.

        Returns:
            A new `FramesMeta` holding only the kept keyframes.
        """
        kept: set[int] = set(keyframe_ids)
        message: Message = self.current_message()
        if synced_sample_id_by_keyframe_id is not None:
            for entry in message.keyframes_metadata:
                if int(entry.id) in kept:
                    entry.synced_sample_id = synced_sample_id_by_keyframe_id[int(entry.id)]
        surviving: list[bytes] = [entry.SerializeToString() for entry in message.keyframes_metadata if int(entry.id) in kept]
        del message.keyframes_metadata[:]
        for encoded in surviving:
            message.keyframes_metadata.add().ParseFromString(encoded)
        return parse_message(message)


def _rigid_transform(transform: Message) -> pycolmap.Rigid3d:
    """Convert a `RigidTransform3d` message into a `Rigid3d`.

    Args:
        transform: A `protos.common.geometry.RigidTransform3d`.

    Returns:
        The equivalent rigid transform; identity when the message is empty.
    """
    return rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([transform.axis_angle.x, transform.axis_angle.y, transform.axis_angle.z]),
        angle_degrees=float(transform.axis_angle.angle_degrees),
        translation_xyz=np.array([transform.translation.x, transform.translation.y, transform.translation.z]),
    )


def write_rigid_transform(transform: Message, pose: pycolmap.Rigid3d) -> None:
    """Overwrite a `RigidTransform3d` message in place from a `Rigid3d`.

    Args:
        transform: A `protos.common.geometry.RigidTransform3d` to mutate.
        pose: The pose to store, encoded as axis-angle in degrees plus metres.
    """
    axis_angle: AxisAngleDegrees = axis_angle_degrees_from_rigid3d(pose)
    transform.axis_angle.x = float(axis_angle.axis_xyz[0])
    transform.axis_angle.y = float(axis_angle.axis_xyz[1])
    transform.axis_angle.z = float(axis_angle.axis_xyz[2])
    transform.axis_angle.angle_degrees = axis_angle.angle_degrees
    transform.translation.x = float(axis_angle.translation_xyz[0])
    transform.translation.y = float(axis_angle.translation_xyz[1])
    transform.translation.z = float(axis_angle.translation_xyz[2])


def _set_scalar(message: Message, field_name: str, value: ScalarField) -> None:
    """Write a scalar field only when it differs from what the message holds.

    Args:
        message: The message to update.
        field_name: Scalar field to write.
        value: The typed model's value for that field.
    """
    if getattr(message, field_name) != value:
        setattr(message, field_name, value)


def _set_enum(message: Message, field_name: str, value_name: str) -> None:
    """Write an enum field from its symbolic name, only when it differs.

    Args:
        message: The message to update.
        field_name: Enum field to write.
        value_name: Symbolic value name, e.g. `PINHOLE`.

    Raises:
        KeyError: When the enum has no value of that name.
    """
    number: int = message.DESCRIPTOR.fields_by_name[field_name].enum_type.values_by_name[value_name].number
    if getattr(message, field_name) != number:
        setattr(message, field_name, number)


def _set_rigid_transform(transform: Message, pose: pycolmap.Rigid3d) -> None:
    """Write a `RigidTransform3d` only when it differs from the pose it holds.

    Skipping the equal case matters: axis-angle to `Rigid3d` and back does not
    round trip bit for bit, so rewriting an unchanged pose would move the digits.

    Args:
        transform: A `protos.common.geometry.RigidTransform3d` to update.
        pose: The typed model's pose for that field.
    """
    if not np.array_equal(_rigid_transform(transform).matrix(), pose.matrix()):
        write_rigid_transform(transform, pose)


def _set_matrix(matrix_message: Message, matrix: Float64[ndarray, "rows cols"] | None, row_count: int, column_count: int) -> None:
    """Write a `MatrixD` of known shape only when it differs from what it holds.

    A None typed value clears the field, but only when the message currently
    carries a matrix of the expected shape: anything else is a calibration this
    module does not model, and clearing it would lose it.

    Args:
        matrix_message: A `protos.common.geometry.MatrixD` to update.
        matrix: The typed model's matrix, or None when the calibration omits it.
        row_count: Rows the field is expected to carry.
        column_count: Columns the field is expected to carry.
    """
    current: Float64[ndarray, "rows cols"] | None = _matrix(matrix_message, row_count, column_count)
    if matrix is None:
        if current is not None:
            matrix_message.Clear()
        return
    if current is not None and np.array_equal(current, matrix):
        return
    del matrix_message.data[:]
    matrix_message.data.extend(float(value) for value in matrix.reshape(-1))
    matrix_message.row_count = row_count
    matrix_message.column_count = column_count


def _set_distortion(matrix_message: Message, coefficients: DistortionCoefficients) -> None:
    """Write the distortion coefficients only when they differ.

    Their count is the calibration's own, so `row_count`/`column_count` stay as
    the template wrote them.

    Args:
        matrix_message: The `distortion_coefficients` `MatrixD` to update.
        coefficients: The typed model's coefficients; empty when there are none.
    """
    if np.array_equal(np.array(list(matrix_message.data), dtype=np.float64), coefficients):
        return
    del matrix_message.data[:]
    matrix_message.data.extend(float(value) for value in coefficients)


def _set_string_map(message: Message, field_name: str, values: Mapping[int, str]) -> None:
    """Rewrite an int-to-string map field only when it differs.

    Args:
        message: The message holding the map.
        field_name: Name of the map field.
        values: The typed model's mapping.
    """
    entries: MutableMapping[int, str] = getattr(message, field_name)
    if {int(key): value for key, value in entries.items()} == dict(values):
        return
    entries.clear()
    for key, value in values.items():
        entries[key] = value


def _apply_keyframes(message: Message, keyframes: Sequence[KeyframeMeta]) -> None:
    """Write the typed keyframes over `keyframes_metadata`.

    Membership and order come from the typed tuple; each surviving entry keeps
    the template's unmodelled fields, matched by keyframe id.

    Args:
        message: The collection message to update.
        keyframes: The typed keyframes, in the order they should appear.
    """
    template_by_id: dict[int, bytes] = {int(entry.id): entry.SerializeToString() for entry in message.keyframes_metadata}
    if [int(entry.id) for entry in message.keyframes_metadata] != [keyframe.keyframe_id for keyframe in keyframes]:
        del message.keyframes_metadata[:]
        for keyframe in keyframes:
            entry: Message = message.keyframes_metadata.add()
            encoded: bytes | None = template_by_id.get(keyframe.keyframe_id)
            if encoded is not None:
                entry.ParseFromString(encoded)
    for entry, keyframe in zip(message.keyframes_metadata, keyframes, strict=True):
        _set_scalar(entry, "id", keyframe.keyframe_id)
        _set_scalar(entry, "camera_params_id", keyframe.camera_params_id)
        _set_scalar(entry, "timestamp_microseconds", keyframe.timestamp_microseconds)
        _set_scalar(entry, "image_name", keyframe.image_name)
        _set_scalar(entry, "synced_sample_id", keyframe.synced_sample_id)
        _set_scalar(entry, "track_id", keyframe.track_id)
        _set_rigid_transform(entry.camera_to_world, keyframe.world_T_cam)


def _apply_camera(camera_sensor: Message, camera: CameraParams) -> None:
    """Write one typed `CameraParams` over its `CameraSensor` message.

    Args:
        camera_sensor: A `protos.common.sensor.CameraSensor` to update.
        camera: The typed camera parameters for it.
    """
    sensor_meta_data: Message = camera_sensor.sensor_meta_data
    _set_scalar(sensor_meta_data, "sensor_name", camera.sensor_name)
    _set_scalar(sensor_meta_data, "model_name", camera.model_name)
    _set_scalar(sensor_meta_data, "sensor_id", camera.sensor_id)
    _set_scalar(sensor_meta_data, "frequency", camera.frequency_hz)
    _set_rigid_transform(sensor_meta_data.sensor_to_vehicle_transform, camera.vehicle_T_cam)
    calibration: Message = camera_sensor.calibration_parameters
    _set_scalar(calibration, "image_width", camera.image_width)
    _set_scalar(calibration, "image_height", camera.image_height)
    _set_scalar(calibration, "rolling_shutter_delay_microseconds", camera.rolling_shutter_delay_microseconds)
    _set_matrix(calibration.projection_matrix, camera.projection_matrix, 3, 4)
    _set_matrix(calibration.camera_matrix, camera.camera_matrix, 3, 3)
    _set_distortion(calibration.distortion_coefficients, camera.distortion_coefficients)
    _set_enum(camera_sensor, "camera_projection_model_type", camera.projection_model)


def _apply_cameras(message: Message, cameras: Mapping[int, CameraParams]) -> None:
    """Write the typed cameras over `camera_params_id_to_camera_params`.

    Args:
        message: The collection message to update.
        cameras: The typed cameras, keyed by `camera_params_id`.
    """
    sensors: MutableMapping[int, Message] = message.camera_params_id_to_camera_params
    for camera_params_id in [int(key) for key in sensors if int(key) not in cameras]:
        del sensors[camera_params_id]
    for camera_params_id, camera in cameras.items():
        _apply_camera(sensors[camera_params_id], camera)


def _apply_stereo_pairs(message: Message, stereo_pairs: Sequence[StereoPair]) -> None:
    """Write the typed stereo pairs over `stereo_pair`, only when they differ.

    Args:
        message: The collection message to update.
        stereo_pairs: The typed stereo pairs, in the order they should appear.
    """
    current: tuple[tuple[int, int, float], ...] = tuple(
        (int(pair.left_camera_param_id), int(pair.right_camera_param_id), float(pair.baseline_meters))
        for pair in message.stereo_pair
    )
    wanted: tuple[tuple[int, int, float], ...] = tuple(
        (pair.left_camera_params_id, pair.right_camera_params_id, pair.baseline_meters) for pair in stereo_pairs
    )
    if current == wanted:
        return
    del message.stereo_pair[:]
    for pair in stereo_pairs:
        entry: Message = message.stereo_pair.add()
        entry.left_camera_param_id = pair.left_camera_params_id
        entry.right_camera_param_id = pair.right_camera_params_id
        entry.baseline_meters = pair.baseline_meters


def _apply_collection_fields(message: Message, frames_meta: FramesMeta) -> None:
    """Write the typed collection-level fields over the message.

    Args:
        message: The collection message to update.
        frames_meta: The typed collection those fields belong to.
    """
    _set_enum(message, "initial_pose_type", frames_meta.initial_pose_type)
    _set_enum(message, "detector_type", frames_meta.detector_type)
    _set_enum(message, "descriptor_type", frames_meta.descriptor_type)
    _set_scalar(message, "keypoint_feature_has_depth", frames_meta.keypoint_feature_has_depth)
    _set_scalar(message, "local_sector_key", frames_meta.local_sector_key)
    _set_string_map(message, "camera_params_id_to_session_name", frames_meta.session_name_by_camera_params_id)
    _set_string_map(message, "track_id_to_track_name", frames_meta.track_name_by_track_id)
    _set_string_map(message, "track_id_to_bag_name", frames_meta.bag_name_by_track_id)


def _matrix(matrix_message: Message, row_count: int, column_count: int) -> Float64[ndarray, "rows cols"] | None:
    """Reshape a `MatrixD` into a numpy array when it holds the expected size.

    Args:
        matrix_message: A `protos.common.geometry.MatrixD`.
        row_count: Expected number of rows.
        column_count: Expected number of columns.

    Returns:
        The row-major matrix, or None when the field is absent or the wrong size.
    """
    data: list[float] = list(matrix_message.data)
    if len(data) != row_count * column_count:
        return None
    return np.array(data, dtype=np.float64).reshape(row_count, column_count)


def _camera_params(camera_params_id: int, camera_sensor: Message, schema: CusfmSchema) -> CameraParams:
    """Convert one `CameraSensor` message into a typed `CameraParams`.

    Args:
        camera_params_id: Map key the sensor was stored under.
        camera_sensor: A `protos.common.sensor.CameraSensor`.
        schema: Loaded cuSFM schema, used to name the projection-model enum value.

    Returns:
        The typed camera parameters.
    """
    calibration: Message = camera_sensor.calibration_parameters
    projection_model: str = schema.enum_value_name(
        "protos.common.sensor.CameraProjectionModelType", camera_sensor.camera_projection_model_type
    )
    return CameraParams(
        camera_params_id=camera_params_id,
        sensor_name=camera_sensor.sensor_meta_data.sensor_name,
        model_name=camera_sensor.sensor_meta_data.model_name,
        sensor_id=int(camera_sensor.sensor_meta_data.sensor_id),
        frequency_hz=float(camera_sensor.sensor_meta_data.frequency),
        vehicle_T_cam=_rigid_transform(camera_sensor.sensor_meta_data.sensor_to_vehicle_transform),
        image_width=int(calibration.image_width),
        image_height=int(calibration.image_height),
        projection_matrix=_matrix(calibration.projection_matrix, 3, 4),
        camera_matrix=_matrix(calibration.camera_matrix, 3, 3),
        distortion_coefficients=np.array(list(calibration.distortion_coefficients.data), dtype=np.float64),
        projection_model=projection_model,
        rolling_shutter_delay_microseconds=int(calibration.rolling_shutter_delay_microseconds),
    )


def parse_message(message: Message, schema: CusfmSchema | None = None) -> FramesMeta:
    """Build the typed view over an already-parsed `KeyframesMetadataCollection`.

    Args:
        message: The parsed collection.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The typed metadata, holding `message` for lossless re-serialisation.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    keyframes: tuple[KeyframeMeta, ...] = tuple(
        KeyframeMeta(
            keyframe_id=int(entry.id),
            camera_params_id=int(entry.camera_params_id),
            timestamp_microseconds=int(entry.timestamp_microseconds),
            image_name=entry.image_name,
            world_T_cam=_rigid_transform(entry.camera_to_world),
            synced_sample_id=int(entry.synced_sample_id),
            track_id=int(entry.track_id),
        )
        for entry in message.keyframes_metadata
    )
    cameras: dict[int, CameraParams] = {
        int(camera_params_id): _camera_params(int(camera_params_id), camera_sensor, resolved_schema)
        for camera_params_id, camera_sensor in message.camera_params_id_to_camera_params.items()
    }
    stereo_pairs: tuple[StereoPair, ...] = tuple(
        StereoPair(
            left_camera_params_id=int(pair.left_camera_param_id),
            right_camera_params_id=int(pair.right_camera_param_id),
            baseline_meters=float(pair.baseline_meters),
        )
        for pair in message.stereo_pair
    )
    initial_pose_type: str = resolved_schema.enum_value_name("protos.visual.general.InitialPoseType", message.initial_pose_type)
    return FramesMeta(
        keyframes=keyframes,
        cameras=cameras,
        stereo_pairs=stereo_pairs,
        initial_pose_type=initial_pose_type,
        detector_type=resolved_schema.enum_value_name("protos.common.image.DetectorType", message.detector_type),
        descriptor_type=resolved_schema.enum_value_name("protos.common.image.DescriptorType", message.descriptor_type),
        session_name_by_camera_params_id={int(key): value for key, value in message.camera_params_id_to_session_name.items()},
        track_name_by_track_id={int(key): value for key, value in message.track_id_to_track_name.items()},
        bag_name_by_track_id={int(key): value for key, value in message.track_id_to_bag_name.items()},
        keypoint_feature_has_depth=bool(message.keypoint_feature_has_depth),
        local_sector_key=message.local_sector_key,
        message=message,
    )


def read_frames_meta(path: Path, schema: CusfmSchema | None = None) -> FramesMeta:
    """Parse a `frames_meta.json` file.

    Args:
        path: The JSON file.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The typed metadata.

    Raises:
        json_format.ParseError: When the file has fields the schema does not know.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    collection_class = resolved_schema.message_class(KEYFRAMES_METADATA_COLLECTION)
    return parse_message(json_format.Parse(path.read_text(), collection_class()), resolved_schema)


def write_frames_meta(path: Path, frames_meta: FramesMeta) -> None:
    """Write a `frames_meta.json` file.

    Args:
        path: Destination; parent directories are created.
        frames_meta: The metadata to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frames_meta.to_json() + "\n")
