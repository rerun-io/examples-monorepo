"""IMU stream handler: logs accelerometer, gyroscope, magnetometer to Rerun.

Mirrors the C++ IMUPlayer from rerun-io/cpp-example-vrs.
"""

import logging

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

logger: logging.Logger = logging.getLogger(__name__)

# RecordableTypeIds for IMU-related streams
IMU_RECORDABLE_TYPE_IDS: set[int] = {
    1202,  # SlamImuData (accelerometer + gyroscope)
    1203,  # SlamMagnetometerData
    281,  # ImuRecordableClass (generic)
}


def might_contain_imu_data(recordable_type_id: int) -> bool:
    """Check if a stream's RecordableTypeId suggests it carries IMU data."""
    return recordable_type_id in IMU_RECORDABLE_TYPE_IDS


class IMUPlayer:
    """Handles IMU/sensor streams from a VRS file and logs them to Rerun.

    Mirrors the C++ IMUPlayer from rerun-io/cpp-example-vrs.
    """

    def __init__(self, stream_id: str, stream_name: str) -> None:
        self._stream_id: str = stream_id
        self._entity_path: str = stream_name
        self._enabled: bool = True
        self._has_accelerometer: bool = False
        self._has_gyroscope: bool = False
        self._has_magnetometer: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_configuration_record(self, metadata: dict[str, object]) -> None:
        """Log static configuration metadata and detect available sensors."""
        if not self._enabled:
            return

        # Detect sensor availability from configuration fields
        self._has_accelerometer = bool(metadata.get("has_accelerometer", False))
        self._has_gyroscope = bool(metadata.get("has_gyroscope", False))
        self._has_magnetometer = bool(metadata.get("has_magnetometer", False))

        logger.info(
            "Stream %s: accel=%s gyro=%s mag=%s",
            self._stream_id,
            self._has_accelerometer,
            self._has_gyroscope,
            self._has_magnetometer,
        )

        config_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        rr.log(f"{self._entity_path}/configuration", rr.TextDocument(config_str), static=True)

    def on_data_record(self, timestamp_sec: float, metadata: dict[str, object]) -> None:
        """Log IMU sensor data to Rerun.

        Args:
            timestamp_sec: Record timestamp in seconds.
            metadata: Dict containing sensor arrays (accelerometer, gyroscope, magnetometer).
        """
        if not self._enabled:
            return

        rr.set_time("timestamp", duration=timestamp_sec)

        if self._has_accelerometer and "accelerometer" in metadata:
            accel_raw: object = metadata["accelerometer"]
            accel: Float32[ndarray, "3"] = np.asarray(accel_raw, dtype=np.float32).flatten()[:3]
            if accel.shape[0] == 3:
                self._log_accelerometer(accel)

        if self._has_gyroscope and "gyroscope" in metadata:
            gyro_raw: object = metadata["gyroscope"]
            gyro: Float32[ndarray, "3"] = np.asarray(gyro_raw, dtype=np.float32).flatten()[:3]
            if gyro.shape[0] == 3:
                self._log_gyroscope(gyro)

        if self._has_magnetometer and "magnetometer" in metadata:
            mag_raw: object = metadata["magnetometer"]
            mag: Float32[ndarray, "3"] = np.asarray(mag_raw, dtype=np.float32).flatten()[:3]
            if mag.shape[0] == 3:
                self._log_magnetometer(mag)

    def _log_accelerometer(self, accel: Float32[ndarray, "3"]) -> None:
        """Log accelerometer as Arrows3D + Scalars."""
        rr.log(f"{self._entity_path}/accelerometer", rr.Arrows3D(vectors=[accel.tolist()]))
        rr.log(f"{self._entity_path}/accelerometer", rr.Scalars(accel.tolist()))

    def _log_gyroscope(self, gyro: Float32[ndarray, "3"]) -> None:
        """Log gyroscope as Scalars."""
        rr.log(f"{self._entity_path}/gyroscope", rr.Scalars(gyro.tolist()))

    def _log_magnetometer(self, mag: Float32[ndarray, "3"]) -> None:
        """Log magnetometer as Scalars."""
        rr.log(f"{self._entity_path}/magnetometer", rr.Scalars(mag.tolist()))
