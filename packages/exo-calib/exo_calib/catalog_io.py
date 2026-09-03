"""Catalog access for the exo-calib pipeline: the shared stage config, connection, segment selection, and rig discovery.

The rig schema itself (``/world/rig_XX/cam_YY``, moving rigs, calibration
presence, rig kinds) is parsed by ``simplecv.catalog_rig_layout``; this module
adds the exo-calib policy of which rigs are the exo cameras to calibrate.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_rig_layout import CatalogRigLayout, catalog_components, parse_rig_layout, read_rig_kinds

from exo_calib.layer_io import PIPELINE_CALIBRATION_MARKER

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:9988"
DEFAULT_DATASET_NAME: str = "assembly101"


@dataclass
class StageConfig:
    """What every exo-calib stage needs to find its segment; stage configs add their own knobs on top."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to process; ``None`` uses the dataset's single segment."""
    exo_rigs: tuple[str, ...] | None = None
    """Explicit exo rig names such as ``rig_00 rig_01``; ``None`` discovers them from catalog metadata."""
    output_dir: Path = Path("data/outputs")
    """Directory for the generated layer RRDs (one subdirectory per segment) and ``eval.json``."""
    register: bool = True
    """Register what the stage writes into the catalog; off for a dry run that only writes files."""


@dataclass(slots=True, frozen=True)
class RigLayout:
    """Discovered exo and ego camera entity names for one catalog segment."""

    exo_camera_names: tuple[str, ...]
    """Exo camera names under ``/world``, in rig/camera index order."""
    ego_camera_names: tuple[str, ...]
    """Ego camera names under ``/world``, in rig/camera index order."""
    calibrated_camera_names: tuple[str, ...]
    """Cameras whose base layer already carries a static ``Transform3D`` and a
    ``Pinhole`` (dataset ground truth). Cameras missing from this tuple have no
    calibration on their base node, so the pipeline may write its own there."""


@dataclass(slots=True, frozen=True)
class StageContext:
    """A stage's resolved target: the connected dataset, the one segment, its rig layout, and its output directory."""

    dataset: DatasetEntry
    segment_id: str
    layout: RigLayout
    segment_dir: Path


def select_rig_layout(layout: CatalogRigLayout, *, exo_rigs: tuple[str, ...] | None = None) -> RigLayout:
    """Split a catalog rig layout into the exo cameras to calibrate and the ego cameras.

    Exo rigs are those the writer tagged ``kind == "exo"``, plus any untagged static
    rig with dataset calibration (a base layer that predates rig kinds). Tagged
    ``ego`` and ``quest`` rigs are worn devices and never exo. Calibration the
    pipeline wrote itself (marked ``exocalib_written``) does not count as dataset
    ground truth.

    Args:
        layout: Every rig camera with a video stream.
        exo_rigs: Explicit exo rig names such as ``("rig_00", "rig_01")`` — the
            escape hatch for recordings without rig kinds or calibration.

    Returns:
        Exo, ego, and ground-truth-calibrated camera names.
    """
    if not layout.cameras:
        raise ValueError("no rig camera videos were found in the catalog schema")
    calibrated: tuple[str, ...] = tuple(
        camera.name for camera in layout.cameras if camera.has_calibration and PIPELINE_CALIBRATION_MARKER not in camera.camera_node_components
    )
    if exo_rigs is not None:
        selected_exo_rigs: set[str] = {rig.removeprefix("/world/") for rig in exo_rigs}
        unknown_rigs: set[str] = selected_exo_rigs - set(layout.rigs)
        if unknown_rigs:
            raise ValueError(f"exo-rigs override names rigs without video: {sorted(unknown_rigs)}")
    else:
        selected_exo_rigs = {camera.rig for camera in layout.cameras if camera.rig_kind == "exo"}
        selected_exo_rigs.update(
            camera.rig for camera in layout.cameras if camera.name in calibrated and not camera.rig_is_moving and camera.rig_kind is None
        )
    if not selected_exo_rigs:
        raise ValueError("could not discover any exo rigs; pass --exo-rigs explicitly")
    return RigLayout(
        exo_camera_names=tuple(camera.name for camera in layout.cameras if camera.rig in selected_exo_rigs),
        ego_camera_names=tuple(camera.name for camera in layout.cameras if camera.rig not in selected_exo_rigs),
        calibrated_camera_names=calibrated,
    )


def discover_rig_layout(dataset: DatasetEntry, segment_id: str, *, exo_rigs: tuple[str, ...] | None = None) -> RigLayout:
    """Discover one segment's exo/ego camera layout from its catalog schema and rig-kind metadata."""
    components = catalog_components(dataset.schema())
    rigs: tuple[str, ...] = parse_rig_layout(components).rigs
    layout: CatalogRigLayout = parse_rig_layout(components, read_rig_kinds(dataset, segment_id, rigs))
    return select_rig_layout(layout, exo_rigs=exo_rigs)


def connect_dataset(catalog_url: str = DEFAULT_CATALOG_URL, dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetEntry:
    """Connect to the running catalog and return the dataset entry."""
    client: CatalogClient = CatalogClient(catalog_url)
    return client.get_dataset(dataset_name)


def only_segment_id(dataset: DatasetEntry) -> str:
    """Return the id of the dataset's single segment, failing on any other count."""
    table: pa.Table = pa.Table.from_batches(dataset.segment_table().collect())
    segment_ids: list[str] = [str(v) for v in table.column("rerun_segment_id").to_pylist()]
    if len(segment_ids) != 1:
        raise ValueError(f"expected exactly one segment, found {segment_ids}")
    return segment_ids[0]


def stage_context(config: StageConfig) -> StageContext:
    """Resolve a stage config: connect, pick the segment (``None`` means the dataset's only one), discover the rig."""
    dataset: DatasetEntry = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    layout: RigLayout = discover_rig_layout(dataset, segment_id, exo_rigs=config.exo_rigs)
    return StageContext(dataset=dataset, segment_id=segment_id, layout=layout, segment_dir=config.output_dir / segment_id)
