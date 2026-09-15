"""Import a Basalt IMU model into existing catalog recordings."""

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from dataforge.paths import SENSOR_METADATA_LAYER
from dataforge.writing import atomic_recording
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from simplecv.imu_calibration import ImuCalibration

from slam_rs import _core
from slam_rs.config import config_text_sha256


@dataclass
class Config:
    """Explicit calibration backfill for datasets whose importer lacks IMU metadata.

    Select only recordings made with this sensor model. This does not establish
    calibration accuracy or apply a new timestamp correction to any sample.
    """

    catalog: str
    """Catalog containing the existing recordings."""
    dataset: str
    """Dataset to augment."""
    calibration: Path
    """Original Basalt calibration JSON, parsed by the Rust extension."""
    applied_time_shift_ns: int
    """Relative IMU timestamp shift already applied at ingestion; explicitly zero if none."""
    output_dir: Path
    """Metadata output directory, visible to the catalog server when registering."""
    segments: tuple[str, ...] = ()
    """Target segment IDs; empty selects all existing segments in the dataset."""
    register: bool = True
    """Disable to prepare files for transfer to the catalog host before registration."""


def main(config: Config) -> None:
    """Write static calibration only; original geometry and sensor streams stay intact."""
    text: str = config.calibration.read_text()
    calibration: _core.Calibration = _core.Calibration.from_json(text)
    for values in (calibration.gyro_noise_std, calibration.accel_noise_std, calibration.gyro_bias_std, calibration.accel_bias_std):
        if len(values) != 3 or any(value != values[0] for value in values):
            raise ValueError("ImuCalibration currently requires isotropic noise; cannot discard per-axis differences")
    metadata: ImuCalibration = ImuCalibration(
        gyro_noise_density=calibration.gyro_noise_std[0], accel_noise_density=calibration.accel_noise_std[0],
        gyro_bias_random_walk=calibration.gyro_bias_std[0], accel_bias_random_walk=calibration.accel_bias_std[0],
        rate_hz=calibration.imu_update_rate,
        source=f"Basalt calibration {config.calibration.name}; sha256={config_text_sha256(text)}; measurement provenance not established",
    )
    client: CatalogClient = CatalogClient(config.catalog)
    dataset: DatasetEntry = client.get_dataset(config.dataset)
    registered: set[str] = set(dataset.segment_ids())
    selected: list[str] = sorted(config.segments or registered)
    if set(selected) - registered:
        raise ValueError(f"segments absent from {config.dataset}: {sorted(set(selected) - registered)}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for segment in selected:
        if Path(segment).name != segment:
            raise ValueError(f"invalid segment ID: {segment}")
        output: Path = config.output_dir.resolve() / f"{segment}.rrd"
        with atomic_recording(output, application_id="sensor-metadata", recording_id=segment) as recording:
            recording.log("/world/rig_00/imu_00", metadata, rr.AnyValues(
                applied_time_shift_ns=config.applied_time_shift_ns,
                time_shift_source="Existing ingestion correction, explicitly supplied during calibration import",
            ), static=True)
        outputs.append(output.as_uri())
    if config.register and outputs:
        dataset.register(outputs, layer_name=SENSOR_METADATA_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"{'Registered' if config.register else 'Prepared'} IMU metadata for {len(outputs)} segments of {config.dataset}")
