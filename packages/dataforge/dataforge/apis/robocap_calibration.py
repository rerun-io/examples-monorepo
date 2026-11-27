"""Add calibration metadata to legacy RoboCap catalog sessions without remuxing."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry

from dataforge import paths, schema, writing
from dataforge.datasets.robocap import CAMERA_TO_IMU_OFFSET_NS, LEGACY_TIME_SHIFT_SOURCE, FactoryCalibration, RobocapConfig, RobocapDataset
from dataforge.robocap_kalibr import CAMERA_DISPLAY_ORDER


@dataclass
class Config:
    """Backfill legacy DataForge recordings with their own device's calibration."""

    output_dir: Path
    """Persistent metadata-layer directory, visible at the same path to the catalog server."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Existing catalog; its base and result layers remain registered."""
    root: Path = Path("/mnt/nas/ssd-8tb/datasets/robocap")
    """Corpus root containing factory calibration directories."""
    segments: tuple[str, ...] = ()
    """Registered legacy session IDs; empty selects every RoboCap session with factory calibration and no factory time shift."""


def main(config: Config) -> None:
    """Read camera identities from the catalog, then register static-only layers."""
    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.get_dataset("robocap")
    selected: list[str] = writing.select_segments(entry, config.segments)
    if not selected:
        print("No RoboCap sessions registered")
        return
    statics: pa.Table = (
        entry.filter_segments(selected)
        .filter_contents([schema.cam_path(0, index) for index in range(len(CAMERA_DISPLAY_ORDER))] + [schema.imu_path(0, 0), "/__properties/**"])
        .reader(index=None).to_arrow_table()
    )
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=config.root))
    shift_column: str = f"{schema.imu_path(0, 0)}:applied_time_shift_ns"
    factories: dict[str, FactoryCalibration] = {}
    for segment in selected:
        parts: list[str] = segment.split("__")
        if len(parts) != 3 or parts[0] != "robocap":
            raise ValueError(f"not a legacy RoboCap session ID: {segment}")
        # A recording that already states a shift other than the legacy one was converted with its cap's factory shift;
        # legacy metadata would contradict its IMU samples.
        stated: list[object] = (
            statics.filter(pc.field("rerun_segment_id") == segment)[shift_column].drop_null().to_pylist() if shift_column in statics.column_names else []
        )
        try:
            if stated not in ([], [[-CAMERA_TO_IMU_OFFSET_NS]]):
                raise ValueError(f"{segment}: states applied_time_shift_ns {stated}, not the legacy {-CAMERA_TO_IMU_OFFSET_NS}")
            factories[segment] = dataset.factory_calibration(parts[1])
        except (ValueError, FileNotFoundError) as error:
            if config.segments:
                raise
            print(f"skip {segment}: {error}")

    def log(segment: str, recording: rr.RecordingStream) -> None:
        segment_statics: pa.Table = statics.filter(pc.field("rerun_segment_id") == segment)
        schema_column: str = "property:capture:schema"
        if schema_column not in segment_statics.column_names or segment_statics[schema_column].drop_null().to_pylist() != [[schema.DATAFORGE_SCHEMA_VERSION]]:
            raise ValueError(f"{segment}: expected a legacy {schema.DATAFORGE_SCHEMA_VERSION} recording")
        cameras: dict[str, str] = {}
        for column in segment_statics.column_names:
            if column.startswith(f"{schema.rig_path(0)}/cam_") and column.endswith(":name"):
                names: list[list[str]] = segment_statics[column].drop_null().to_pylist()
                if not names:
                    continue
                if len(names) != 1 or len(names[0]) != 1:
                    raise ValueError(f"{segment}: expected one static camera name at {column}")
                cameras[names[0][0]] = column.removesuffix(":name")
        if not cameras:
            raise ValueError(f"{segment}: no named cameras in the catalog")
        dataset.log_sensor_metadata(recording, factories[segment], cameras, applied_shift_ns=CAMERA_TO_IMU_OFFSET_NS, time_shift_source=LEGACY_TIME_SHIFT_SOURCE)

    outputs: list[str] = writing.write_segment_layer(entry, paths.SENSOR_METADATA_LAYER, config.output_dir, list(factories), log)
    print(f"Registered sensor metadata for {len(outputs)} legacy RoboCap sessions")
