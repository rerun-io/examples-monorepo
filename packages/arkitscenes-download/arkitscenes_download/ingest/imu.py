"""Decode CoreMotion NSKeyedArchiver payloads from mebx tracks."""

import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
from jaxtyping import Float64

from arkitscenes_download.ingest.mov import split_metadata_items


def gyro_degrees_to_radians(gyro_degs: Float64[np.ndarray, "n 3"]) -> Float64[np.ndarray, "n 3"]:
    """Convert CVACMGyroData angular velocities from degrees/s to radians/s."""
    return gyro_degs * (np.pi / 180.0)


@dataclass(frozen=True, slots=True)
class ImuSamples:
    """Timestamped raw inertial samples."""

    accel_timestamps: Float64[np.ndarray, "na"]
    """Accelerometer uptime timestamps."""
    accel_ms2: Float64[np.ndarray, "na 3"]
    """Acceleration in metres per second squared."""
    gyro_timestamps: Float64[np.ndarray, "ng"]
    """Gyroscope uptime timestamps."""
    gyro_rads: Float64[np.ndarray, "ng 3"]
    """Angular velocity in radians per second."""
    motion_timestamps: Float64[np.ndarray, "nm"]
    """Device-motion uptime timestamps."""
    attitude_xyzw: Float64[np.ndarray, "nm 4"]
    """Device-motion attitude quaternion in xyzw order."""


def _root_object(payload: bytes) -> dict[str, Any]:
    archive: dict[str, Any] = plistlib.loads(payload)
    objects: list[Any] = archive["$objects"]
    root_uid: plistlib.UID = archive["$top"]["root"]
    return objects[root_uid.data]


def _nested_roots(payload: bytes) -> list[dict[str, Any]]:
    outer: dict[str, Any] = plistlib.loads(payload)
    roots: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, bytes) and value.startswith(b"bplist00"):
            roots.append(_root_object(value))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(outer)
    return roots


def decode_imu(mov_path: Path) -> ImuSamples:
    """Decode batched accelerometer, gyroscope, and device-motion records."""
    vectors: dict[int, list[tuple[float, float, float, float]]] = {6: [], 7: []}
    motion_rows: list[tuple[float, float, float, float, float]] = []
    with av.open(str(mov_path)) as container:
        streams: list[av.stream.Stream] = [container.streams[index] for index in (6, 7, 8)]
        for packet in container.demux(*streams):
            payload: bytes = bytes(packet)
            if not payload:
                continue
            stream_index: int = packet.stream.index
            for item in split_metadata_items(payload):
                for root in _nested_roots(item):
                    if stream_index in vectors:
                        vectors[stream_index].append((float(root["t"]), float(root["x"]), float(root["y"]), float(root["z"])))
                    else:
                        motion_rows.append(
                            (
                                float(root["kCMLogItemCodingKeyTimestamp"]),
                                float(root["kCMDeviceMotionCodingKeyQuaternionX"]),
                                float(root["kCMDeviceMotionCodingKeyQuaternionY"]),
                                float(root["kCMDeviceMotionCodingKeyQuaternionZ"]),
                                float(root["kCMDeviceMotionCodingKeyQuaternionW"]),
                            )
                        )
    accel: Float64[np.ndarray, "na 4"] = np.asarray(vectors[6], dtype=np.float64)
    gyro: Float64[np.ndarray, "ng 4"] = np.asarray(vectors[7], dtype=np.float64)
    accel_xyz: Float64[np.ndarray, "na 3"] = accel[:, 1:] * 9.80665
    motion: Float64[np.ndarray, "nm 5"] = np.asarray(motion_rows, dtype=np.float64)
    gyro_xyz: Float64[np.ndarray, "ng 3"] = gyro_degrees_to_radians(gyro[:, 1:])
    return ImuSamples(accel[:, 0], accel_xyz, gyro[:, 0], gyro_xyz, motion[:, 0], motion[:, 1:])
