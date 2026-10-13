"""Which cameras of a catalog segment are the exo rig to calibrate (the rig schema itself is ``simplecv.catalog_rig_layout``)."""

from dagkit.sources import SegmentRef
from rerun.catalog import DatasetEntry
from simplecv.catalog_rig_layout import CatalogRigLayout, catalog_components, parse_rig_layout, read_rig_kinds

from exo_calib.types import RigLayout


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
        camera.name for camera in layout.cameras if camera.has_calibration and "exocalib_written" not in camera.camera_node_components
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


def discover_rig_layout(segment: SegmentRef, *, exo_rigs: tuple[str, ...] | None = None) -> RigLayout:
    """Discover one segment's exo/ego camera layout from its catalog schema and rig-kind metadata."""
    dataset: DatasetEntry = segment.entry()  # the entry must outlive its queries
    components = catalog_components(dataset.schema())
    rigs: tuple[str, ...] = parse_rig_layout(components).rigs
    layout: CatalogRigLayout = parse_rig_layout(components, read_rig_kinds(dataset, segment.segment, rigs))
    return select_rig_layout(layout, exo_rigs=exo_rigs)
