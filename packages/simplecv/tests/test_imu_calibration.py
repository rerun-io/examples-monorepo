"""Optional sensor calibration survives a real RRD/catalog boundary."""

from pathlib import Path

import pyarrow as pa
import pytest
import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry

from simplecv.imu_calibration import ImuCalibration


def test_catalog_preserves_known_zero_and_absent_calibration(tmp_path: Path) -> None:
    path: Path = tmp_path / "imu.rrd"
    calibration: ImuCalibration = ImuCalibration(gyro_noise_density=0.0, rate_hz=200.0, source="factory/imu.yaml")
    with rr.RecordingStream("calibration", recording_id="test") as recording:
        recording.save(path)
        recording.log("/rig/imu_00", calibration, static=True)
        recording.log("/rig/imu_01", rr.AnyValues(kind="imu"), static=True)
    with rr.server.Server(datasets={"test": [path]}) as server:
        client: CatalogClient = server.client()
        dataset: DatasetEntry = client.get_dataset("test")
        table: pa.Table = dataset.filter_contents(["/rig/imu_00", "/rig/imu_01"]).reader(index=None).to_arrow_table()
        assert ImuCalibration.from_catalog(table, "/rig/imu_00") == calibration
        assert ImuCalibration.from_catalog(table, "/rig/imu_01") == ImuCalibration()
        assert not any("accel_noise_density" in name for name in table.column_names)


@pytest.mark.parametrize("noise", [-0.1, float("nan"), float("inf")])
def test_invalid_noise_is_rejected(noise: float) -> None:
    with pytest.raises(ValueError, match="gyro_noise_density"):
        ImuCalibration(gyro_noise_density=noise)
