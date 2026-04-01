"""IMU stream handler: logs accelerometer, gyroscope, magnetometer to Rerun.

Supports two modes:
  - Row-by-row: on_data_record() logs each sample immediately (sequential path)
  - Batch: accumulate_data_record() collects samples, flush_columns() logs all
    at once via rr.send_columns() (282x faster for 224k IMU records)
"""

import logging

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64
from numpy import ndarray

logger: logging.Logger = logging.getLogger(__name__)

# RecordableTypeIds for IMU-related streams.
# These numeric IDs come from the VRS StreamId spec (vrs/StreamId.h).
# Each VRS stream has a type_id-instance_id string (e.g., "1202-1").
# See: https://github.com/facebookresearch/vrs/blob/main/vrs/StreamId.h
IMU_RECORDABLE_TYPE_IDS: set[int] = {
    1202,  # SlamImuData (accelerometer + gyroscope)
    1203,  # SlamMagnetometerData
    281,  # ImuRecordableClass (generic)
}


def might_contain_imu_data(recordable_type_id: int) -> bool:
    """Check if a stream's RecordableTypeId suggests it carries IMU data."""
    return recordable_type_id in IMU_RECORDABLE_TYPE_IDS


class IMUPlayer:
    """Handles IMU/sensor streams from a VRS file and logs them to Rerun."""

    def __init__(self, stream_id: str, stream_name: str) -> None:
        self._stream_id: str = stream_id
        self._entity_path: str = stream_name
        self._enabled: bool = True
        self._has_accelerometer: bool = False
        self._has_gyroscope: bool = False
        self._has_magnetometer: bool = False
        # Accumulators for batch logging via send_columns
        self._timestamps: list[float] = []
        self._accel_data: list[list[float]] = []
        self._gyro_data: list[list[float]] = []
        self._mag_data: list[list[float]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entity_path(self) -> str:
        return self._entity_path

    def on_configuration_record(self, metadata: dict[str, object]) -> None:
        """Log static configuration metadata and detect available sensors."""
        if not self._enabled:
            return

        self._has_accelerometer = bool(metadata.get("has_accelerometer", False))
        self._has_gyroscope = bool(metadata.get("has_gyroscope", False))
        self._has_magnetometer = bool(metadata.get("has_magnetometer", False))

        logger.info(f"Stream {self._stream_id}: accel={self._has_accelerometer} gyro={self._has_gyroscope} mag={self._has_magnetometer}")

        config_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        rr.log(f"{self._entity_path}/configuration", rr.TextDocument(config_str), static=True)

    # ── Row-by-row path (sequential pipeline) ───────────────────────────

    def on_data_record(self, timestamp_sec: float, metadata: dict[str, object]) -> None:
        """Log IMU sensor data to Rerun immediately (row-by-row)."""
        if not self._enabled:
            return

        rr.set_time("timestamp", duration=timestamp_sec)

        if self._has_accelerometer and "accelerometer" in metadata:
            # pyvrs metadata values are list[float] but typed as object in the generic dict
            accel: Float32[ndarray, "3"] = np.asarray(metadata["accelerometer"], dtype=np.float32).flatten()[:3]
            if accel.shape[0] == 3:
                rr.log(f"{self._entity_path}/accelerometer", rr.Arrows3D(vectors=[accel.tolist()]))
                rr.log(f"{self._entity_path}/accelerometer", rr.Scalars(accel.tolist()))

        if self._has_gyroscope and "gyroscope" in metadata:
            gyro: Float32[ndarray, "3"] = np.asarray(metadata["gyroscope"], dtype=np.float32).flatten()[:3]
            if gyro.shape[0] == 3:
                rr.log(f"{self._entity_path}/gyroscope", rr.Scalars(gyro.tolist()))

        if self._has_magnetometer and "magnetometer" in metadata:
            mag: Float32[ndarray, "3"] = np.asarray(metadata["magnetometer"], dtype=np.float32).flatten()[:3]
            if mag.shape[0] == 3:
                rr.log(f"{self._entity_path}/magnetometer", rr.Scalars(mag.tolist()))

    # ── Batch path (parallel pipeline) ──────────────────────────────────

    def accumulate_data_record(self, timestamp_sec: float, metadata: dict[str, object]) -> None:
        """Accumulate IMU data for later batch logging via flush_columns()."""
        if not self._enabled:
            return

        self._timestamps.append(timestamp_sec)

        if self._has_accelerometer and "accelerometer" in metadata:
            accel: Float32[ndarray, "3"] = np.asarray(metadata["accelerometer"], dtype=np.float32).flatten()[:3]
            self._accel_data.append(accel.tolist())

        if self._has_gyroscope and "gyroscope" in metadata:
            gyro: Float32[ndarray, "3"] = np.asarray(metadata["gyroscope"], dtype=np.float32).flatten()[:3]
            self._gyro_data.append(gyro.tolist())

        if self._has_magnetometer and "magnetometer" in metadata:
            mag: Float32[ndarray, "3"] = np.asarray(metadata["magnetometer"], dtype=np.float32).flatten()[:3]
            self._mag_data.append(mag.tolist())

    def flush_columns(self) -> None:
        """Batch-log all accumulated IMU data via rr.send_columns()."""
        if not self._timestamps:
            return

        timestamps: Float64[np.ndarray, "n"] = np.array(self._timestamps, dtype=np.float64)
        time_column: rr.TimeColumn = rr.TimeColumn("timestamp", duration=timestamps)

        if self._accel_data:
            accel_array: Float32[np.ndarray, "n 3"] = np.array(self._accel_data, dtype=np.float32)
            rr.send_columns(
                f"{self._entity_path}/accelerometer",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=accel_array),
            )

        if self._gyro_data:
            gyro_array: Float32[np.ndarray, "n 3"] = np.array(self._gyro_data, dtype=np.float32)
            rr.send_columns(
                f"{self._entity_path}/gyroscope",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=gyro_array),
            )

        if self._mag_data:
            mag_array: Float32[np.ndarray, "n 3"] = np.array(self._mag_data, dtype=np.float32)
            rr.send_columns(
                f"{self._entity_path}/magnetometer",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=mag_array),
            )

        n_samples: int = len(self._timestamps)
        logger.info(f"Stream {self._stream_id}: batch-logged {n_samples} IMU samples via send_columns")

        # Clear accumulators
        self._timestamps.clear()
        self._accel_data.clear()
        self._gyro_data.clear()
        self._mag_data.clear()
