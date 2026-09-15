"""Import a Basalt sensor model without copying its camera geometry."""

from pathlib import Path

import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.imu_calibration import ImuCalibration

from slam_rs.apis.import_imu_calibration import Config, main


def test_imports_original_basalt_noise_as_static_catalog_metadata(tmp_path: Path) -> None:
    base: Path = tmp_path / "base.rrd"
    with rr.RecordingStream("test", recording_id="msd-test") as recording:
        recording.save(base)
        recording.log("/world/rig_00/imu_00", rr.AnyValues(kind="imu"), static=True)
    with rr.server.Server(datasets={"msd": [base]}) as server:
        calibration: Path = Path(__file__).parents[1] / "crates/slam-rs/tests/fixtures/msdmi_calib.json"
        main(Config(catalog=server.url(), dataset="msd", calibration=calibration, output_dir=tmp_path / "metadata", applied_time_shift_ns=0))
        client: CatalogClient = server.client()
        dataset: DatasetEntry = client.get_dataset("msd")
        table = dataset.filter_contents(["/world/rig_00/imu_00"]).reader(index=None).to_arrow_table()
        imu: ImuCalibration = ImuCalibration.from_catalog(table, "/world/rig_00/imu_00")
        assert imu.rate_hz == 1000.0
        assert imu.gyro_noise_density == 0.000282
        assert imu.accel_noise_density == 0.016
        assert imu.gyro_bias_random_walk == 0.0001
        assert imu.accel_bias_random_walk == 0.001
        assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[0]]
