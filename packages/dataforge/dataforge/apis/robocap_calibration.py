"""Add calibration metadata to legacy RoboCap catalog sessions without remuxing."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from simplecv.data.ego.robocap_ego import CAMERA_DISPLAY_ORDER

from dataforge import paths, schema, writing
from dataforge.datasets.robocap import RobocapConfig, RobocapDataset


@dataclass
class Config:
    """Backfill legacy DataForge recordings with their own device's calibration."""

    output_dir: Path
    """Persistent metadata-layer directory, visible at the same path to the catalog server."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Existing catalog; its base and result layers remain registered."""
    root: Path = Path("/mnt/nas/datasets/robocap")
    """Corpus root containing factory calibration directories."""
    segments: tuple[str, ...] = ()
    """Registered legacy session IDs; empty selects every RoboCap session with factory calibration."""


def main(config: Config) -> None:
    """Read camera identities from the catalog, then register static-only layers."""
    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.get_dataset("robocap")
    registered: set[str] = set(entry.segment_ids())
    selected: list[str] = sorted(config.segments or registered)
    missing: set[str] = set(selected) - registered
    if missing:
        raise ValueError(f"sessions absent from robocap catalog: {sorted(missing)}")
    if not selected:
        print("No RoboCap sessions registered")
        return
    statics: pa.Table = (
        entry.filter_segments(selected).filter_contents([f"/world/rig_00/cam_{index:02d}" for index in range(len(CAMERA_DISPLAY_ORDER))] + ["/__properties/**"])
        .reader(index=None).to_arrow_table()
    )
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=config.root))
    outputs: list[str] = []
    for segment in selected:
        parts: list[str] = segment.split("__")
        if len(parts) != 3 or parts[0] != "robocap" or Path(segment).name != segment:
            raise ValueError(f"not a legacy RoboCap session ID: {segment}")
        device: str = parts[1]
        if not (config.root / f"0factory-calibration-{device}").is_dir():
            if config.segments:
                raise FileNotFoundError(f"{segment}: no factory calibration for device {device}")
            print(f"skip {segment}: no factory calibration for device {device}")
            continue
        segment_statics: pa.Table = statics.filter(pc.field("rerun_segment_id") == segment)
        schema_column: str = "property:capture:schema"
        if schema_column not in segment_statics.column_names or segment_statics[schema_column].drop_null().to_pylist() != [[schema.DATAFORGE_SCHEMA_VERSION]]:
            raise ValueError(f"{segment}: expected a legacy {schema.DATAFORGE_SCHEMA_VERSION} recording")
        cameras: dict[str, str] = {}
        for column in segment_statics.column_names:
            if column.startswith("/world/rig_00/cam_") and column.endswith(":name"):
                names: list[list[str]] = segment_statics[column].drop_null().to_pylist()
                if not names:
                    continue
                if len(names) != 1 or len(names[0]) != 1:
                    raise ValueError(f"{segment}: expected one static camera name at {column}")
                cameras[names[0][0]] = column.removesuffix(":name")
        if not cameras:
            raise ValueError(f"{segment}: no named cameras in the catalog")
        output: Path = config.output_dir.resolve() / f"{segment}.rrd"
        with writing.atomic_recording(output, application_id="dataforge", recording_id=segment) as recording:
            dataset.log_sensor_metadata(recording, device, cameras)
        outputs.append(output.as_uri())
    if outputs:
        entry.register(outputs, layer_name=paths.SENSOR_METADATA_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Registered sensor metadata for {len(outputs)} legacy RoboCap sessions")
