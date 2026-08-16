"""Decode camera calibration records from ARKit NSKeyedArchiver metadata."""

import plistlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from rich.console import Console
from scipy.spatial.transform import Rotation

from arkitscenes_download.ingest.mov import MetadataPacket, demux_metadata
from arkitscenes_download.ingest.rig import Trajectory, resample_trajectory

POSE_FIT_MAX_ROTATION_RAD: float = float(np.deg2rad(3.0))
POSE_FIT_MAX_TRANSLATION_M: float = 0.1
CONSOLE: Console = Console(markup=False)


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Rigid wide-camera-from-ultrawide-camera calibration."""

    translation_m: Float64[np.ndarray, "3"]
    """Ultrawide camera origin expressed in the wide frame, in metres."""
    quaternion_xyzw: Float64[np.ndarray, "4"]
    """Wide-from-ultrawide rotation in xyzw order."""


@dataclass(frozen=True, slots=True)
class ImagePoseResult:
    """Stream-4 camera poses aligned into the extracted trajectory world."""

    trajectory: Trajectory
    """Aligned 60 Hz world-from-wide-camera poses."""
    rotation_rms_rad: float
    """Overlap rotation fit RMS residual."""
    translation_rms_m: float
    """Overlap translation fit RMS residual."""


@dataclass(frozen=True, slots=True)
class PoseSelection:
    """Selected camera trajectory and stream-fit provenance."""

    trajectory: Trajectory
    """Trajectory selected for image pose resampling."""
    source: str
    """Stable source identifier."""
    rotation_rms_rad: float
    """Stream fit rotation RMS, or NaN after decode failure."""
    translation_rms_m: float
    """Stream fit translation RMS, or NaN after decode failure."""


@dataclass(frozen=True, slots=True)
class CameraDistortion:
    """First stream-10 polynomial calibration for one physical camera."""

    camera_name: str
    """Stable ingest camera name (``wide`` or ``ultrawide``)."""
    coefficients: np.ndarray
    """Eight Apple distorted-to-rectified radial polynomial coefficients as float32."""
    inverse_coefficients: np.ndarray
    """Eight Apple rectified-to-distorted radial polynomial coefficients as float32."""
    center_xy: Float64[np.ndarray, "2"]
    """Distortion center in reference-image pixels."""
    reference_dimensions_wh: Float64[np.ndarray, "2"]
    """Calibration reference dimensions in pixels."""
    timestamp: float
    """Timestamp of the retained first sample."""
    temporal_sample_count: int
    """Number of stream entries observed for this camera."""


def _archive_root(payload: bytes) -> tuple[dict[str, object], list[object]]:
    archive: dict[str, object] = plistlib.loads(payload)
    objects: list[object] = archive["$objects"]  # type: ignore[assignment]
    root: dict[str, object] = objects[1]  # type: ignore[assignment]
    return root, objects


def _decode_dictionary(dictionary: dict[str, object], objects: list[object]) -> dict[str, object]:
    """Resolve one NSKeyedArchiver NSDictionary into plain key/value pairs."""
    keys: list[plistlib.UID] = dictionary["NS.keys"]  # type: ignore[assignment]
    values: list[plistlib.UID] = dictionary["NS.objects"]  # type: ignore[assignment]
    return {str(objects[key.data]): objects[value.data] for key, value in zip(keys, values, strict=True)}


def _decode_size(value: dict[str, object], objects: list[object]) -> Float64[np.ndarray, "2"]:
    """Resolve an archived Width/Height or X/Y dictionary."""
    resolved: dict[str, object] = _decode_dictionary(value, objects)
    first: str = "Width" if "Width" in resolved else "X"
    second: str = "Height" if "Height" in resolved else "Y"
    return np.asarray([resolved[first], resolved[second]], dtype=np.float64)


def decode_camera_distortions_from_packets(packets: list[MetadataPacket]) -> list[CameraDistortion]:
    """Decode the first polynomial calibration per camera and count its temporal samples."""
    first_samples: dict[str, tuple[dict[str, object], list[object], float]] = {}
    counts: dict[str, int] = {}
    for packet in packets:
        for item in packet.items:
            root, objects = _archive_root(item)
            stream_identifier: str = str(objects[root["si"].data])  # type: ignore[union-attr]
            camera_name: str | None = None
            if "UltraWide" in stream_identifier:
                camera_name = "ultrawide"
            elif "WideAngle" in stream_identifier:
                camera_name = "wide"
            if camera_name is None:
                continue
            counts[camera_name] = counts.get(camera_name, 0) + 1
            if camera_name not in first_samples:
                dictionary: dict[str, object] = objects[root["d"].data]  # type: ignore[assignment,union-attr]
                first_samples[camera_name] = (dictionary, objects, float(root["t"]))  # pyrefly: ignore  # bad-argument-type

    calibrations: list[CameraDistortion] = []
    for camera_name in ("wide", "ultrawide"):
        if camera_name not in first_samples:
            continue
        dictionary, objects, timestamp = first_samples[camera_name]
        values: dict[str, object] = _decode_dictionary(dictionary, objects)
        coefficients: np.ndarray = np.frombuffer(values["LensDistortionCoefficients"], dtype="<f4").copy()  # type: ignore[arg-type]
        inverse_coefficients: np.ndarray = np.frombuffer(values["InverseLensDistortionCoefficients"], dtype="<f4").copy()  # type: ignore[arg-type]
        center: dict[str, object] = values["LensDistortionCenter"]  # type: ignore[assignment]
        dimensions: dict[str, object] = values["IntrinsicMatrixReferenceDimensions"]  # type: ignore[assignment]
        calibrations.append(
            CameraDistortion(
                camera_name,
                coefficients,
                inverse_coefficients,
                _decode_size(center, objects),
                _decode_size(dimensions, objects),
                timestamp,
                counts[camera_name],
            )
        )
    return calibrations


def decode_camera_distortions(mov_path: Path) -> list[CameraDistortion]:
    """Demux and decode the first polynomial calibration per camera."""
    return decode_camera_distortions_from_packets(demux_metadata(mov_path, 10))


def decode_ultrawide_extrinsics_from_packets(packets: list[MetadataPacket]) -> CameraExtrinsics:
    """Decode and validate the wide-from-ultrawide ARExtrinsicsWrapper matrix."""
    packet = next(packet for packet in packets if packet.items)
    root, objects = _archive_root(packet.items[0])
    mapping = objects[root["extrinsicsMap"].data]  # type: ignore[union-attr]
    keys = mapping["NS.keys"]  # type: ignore[index]
    values = mapping["NS.objects"]  # type: ignore[index]
    index: int = next(i for i, key in enumerate(keys) if objects[key.data] == "AVCaptureDeviceTypeBuiltInWideAngleCamera")
    wrapper = objects[values[index].data]
    floats: Float64[np.ndarray, "16"] = np.asarray(struct.unpack("<16f", wrapper["matrix"]), dtype=np.float64)  # type: ignore[index]
    matrix_44: Float64[np.ndarray, "4 4"] = floats.reshape(4, 4, order="F")
    rotation: Rotation = Rotation.from_matrix(matrix_44[:3, :3])
    translation_m: Float64[np.ndarray, "3"] = matrix_44[:3, 3] / 1000.0
    if not 0.005 <= np.linalg.norm(translation_m) <= 0.1:
        raise ValueError(f"implausible wide/ultrawide baseline: {np.linalg.norm(translation_m):.4f} m")
    if rotation.magnitude() > np.deg2rad(5.0):
        raise ValueError(f"implausible wide/ultrawide rotation: {np.rad2deg(rotation.magnitude()):.2f} degrees")
    return CameraExtrinsics(translation_m, rotation.as_quat())


def decode_wide_image_trajectory_from_packets(packets: list[MetadataPacket], reference: Trajectory) -> ImagePoseResult:
    """Decode stream-4 vision transforms and align their AR-session world to ``reference``."""
    timestamps: list[float] = []
    matrices: list[np.ndarray] = []
    for packet in packets:
        for item in packet.items:
            try:
                root, _ = _archive_root(item)
            except plistlib.InvalidFileException:
                continue
            if "visionTransform" not in root:
                continue
            floats: Float64[np.ndarray, "16"] = np.asarray(struct.unpack("<16f", root["visionTransform"]), dtype=np.float64)  # type: ignore[arg-type]
            matrices.append(floats.reshape(4, 4, order="F"))
            timestamps.append(float(root["timestamp"]))  # pyrefly: ignore  # bad-argument-type
    camera_from_session: Float64[np.ndarray, "n 4 4"] = np.asarray(matrices, dtype=np.float64)
    session_from_camera: Float64[np.ndarray, "n 4 4"] = np.linalg.inv(camera_from_session)
    image_trajectory = Trajectory(
        np.asarray(timestamps, dtype=np.float64),
        session_from_camera[:, :3, 3],
        Rotation.from_matrix(session_from_camera[:, :3, :3]).as_quat(),
    )
    overlap = reference.timestamps[
        (reference.timestamps >= image_trajectory.timestamps[0]) & (reference.timestamps <= image_trajectory.timestamps[-1])
    ]
    image_overlap: Trajectory = resample_trajectory(image_trajectory, overlap)
    reference_overlap: Trajectory = resample_trajectory(reference, overlap)
    image_rotations: Rotation = Rotation.from_quat(image_overlap.quaternion_xyzw)
    reference_rotations: Rotation = Rotation.from_quat(reference_overlap.quaternion_xyzw)
    world_from_session: Rotation = (reference_rotations * image_rotations.inv()).mean()
    translation: Float64[np.ndarray, "3"] = np.mean(reference_overlap.translation - world_from_session.apply(image_overlap.translation), axis=0)
    aligned_rotations: Rotation = world_from_session * Rotation.from_quat(image_trajectory.quaternion_xyzw)
    aligned_translation: Float64[np.ndarray, "n 3"] = world_from_session.apply(image_trajectory.translation) + translation
    predicted_overlap: Trajectory = resample_trajectory(
        Trajectory(image_trajectory.timestamps, aligned_translation, aligned_rotations.as_quat()), overlap
    )
    rotation_errors: Float64[np.ndarray, "k"] = (Rotation.from_quat(predicted_overlap.quaternion_xyzw).inv() * reference_rotations).magnitude()
    translation_errors: Float64[np.ndarray, "k"] = np.linalg.norm(predicted_overlap.translation - reference_overlap.translation, axis=1)
    rotation_rms: float = float(np.sqrt(np.mean(np.square(rotation_errors))))
    translation_rms: float = float(np.sqrt(np.mean(np.square(translation_errors))))
    return ImagePoseResult(Trajectory(image_trajectory.timestamps, aligned_translation, aligned_rotations.as_quat()), rotation_rms, translation_rms)


def decode_wide_image_trajectory(mov_path: Path, reference: Trajectory) -> ImagePoseResult:
    """Demux stream-4 image poses and align them to ``reference``."""
    return decode_wide_image_trajectory_from_packets(demux_metadata(mov_path, 4), reference)


def select_pose_source(mov_path: Path, fallback: Trajectory, packets: list[MetadataPacket] | None = None) -> PoseSelection:
    """Prefer a well-fitting stream-4 trajectory, otherwise use the fallback."""
    try:
        image_poses: ImagePoseResult = (
            decode_wide_image_trajectory(mov_path, fallback) if packets is None else decode_wide_image_trajectory_from_packets(packets, fallback)
        )
    except (ValueError, KeyError) as pose_error:
        CONSOLE.print(f"Pose fallback: stream-4 decode failed ({pose_error}); using 10Hz trajectory")
        return PoseSelection(fallback, "lowres_traj_slerp", float("nan"), float("nan"))
    if image_poses.rotation_rms_rad <= POSE_FIT_MAX_ROTATION_RAD and image_poses.translation_rms_m <= POSE_FIT_MAX_TRANSLATION_M:
        return PoseSelection(
            image_poses.trajectory,
            "mebx_stream_4_vision_transform",
            image_poses.rotation_rms_rad,
            image_poses.translation_rms_m,
        )
    CONSOLE.print(
        f"Pose fallback: stream-4 fit {np.rad2deg(image_poses.rotation_rms_rad):.2f}deg / "
        f"{image_poses.translation_rms_m:.3f}m exceeds gate; using 10Hz trajectory"
    )
    return PoseSelection(fallback, "lowres_traj_slerp", image_poses.rotation_rms_rad, image_poses.translation_rms_m)
