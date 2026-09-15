"""Static IMU calibration shared by ingestion and catalog consumers."""

from dataclasses import dataclass, fields
from math import isfinite

import pyarrow as pa
import rerun as rr
from serde import from_dict, serde


@serde
@dataclass(frozen=True, slots=True)
class ImuCalibration(rr.AsComponents):
    """Optional, isotropic continuous-time noise model; absent means unknown.

    Log on the IMU entity with ``static=True``. Estimated biases and timestamp
    corrections are not part of this sensor noise model.
    """

    gyro_noise_density: float | None = None
    """Gyroscope white-noise density, rad/s/sqrt(Hz)."""
    accel_noise_density: float | None = None
    """Accelerometer white-noise density, m/s²/sqrt(Hz)."""
    gyro_bias_random_walk: float | None = None
    """Gyroscope bias random walk, rad/s²/sqrt(Hz)."""
    accel_bias_random_walk: float | None = None
    """Accelerometer bias random walk, m/s³/sqrt(Hz)."""
    rate_hz: float | None = None
    """Nominal sampling rate in Hz, not a measured stream rate."""
    source: str | None = None
    """Calibration source, including any assumption or placeholder status."""

    def __post_init__(self) -> None:
        for item in fields(self):
            value: float | str | None = getattr(self, item.name)
            if isinstance(value, float) and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{item.name} must be finite and nonnegative")
        if self.rate_hz == 0.0:
            raise ValueError("rate_hz must be finite and positive")

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        batches: list[rr.DescribedComponentBatch] = []
        for item in fields(self):
            value: float | str | None = getattr(self, item.name)
            if value is None:
                continue
            batch: rr.ComponentBatchMixin = rr.components.TextBatch([value]) if isinstance(value, str) else rr.components.ScalarBatch([value])
            batches.append(batch.described(rr.ComponentDescriptor(
                f"simplecv.ImuCalibration:{item.name}", archetype="simplecv.ImuCalibration",
                component_type="rerun.components.Text" if isinstance(value, str) else "rerun.components.Scalar",
            )))
        return batches

    @classmethod
    def from_catalog(cls, table: pa.Table, entity: str) -> "ImuCalibration":
        """Read one IMU from a static catalog query, preserving absent fields.

        Args:
            table: One segment's ``filter_contents(...).reader(index=None)`` result.
            entity: Full path of the IMU entity.
        """
        values: dict[str, float | str] = {}
        for item in fields(cls):
            column: str = f"{entity}:simplecv.ImuCalibration:{item.name}"
            if column not in table.column_names:
                continue
            cells: list[list[float] | list[str]] = table[column].drop_null().to_pylist()
            if len(cells) != 1 or len(cells[0]) != 1:
                raise ValueError(f"{column}: expected one static calibration value")
            values[item.name] = cells[0][0]
        return from_dict(cls, values)
