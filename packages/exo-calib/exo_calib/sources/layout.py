"""Which cameras of a catalog segment are the exo rig to calibrate, and their lenses (the rig schema itself is
``simplecv.catalog_rig_layout``)."""

from collections.abc import Sequence
from dataclasses import replace
from typing import cast, get_args

import numpy as np
import pyarrow as pa
from dagkit.sources import SegmentRef
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout, RigKind, catalog_components, parse_rig_layout, rig_kinds_from_table
from simplecv.rrd_query_utils import first_valid_value

from exo_calib.types import PINHOLE, Lens, LensModel, RigLayout


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


def lenses_from_table(table: pa.Table, cameras: Sequence[str], *, fallback: Lens) -> dict[str, Lens]:
    """Each camera's stored lens (static ``DistortionModel`` + ``DistortionCoefficients`` on its pinhole node), else ``fallback``.

    Raises:
        ValueError: If a camera stores a lens model refine cannot undistort.
    """
    lenses: dict[str, Lens] = {}
    for camera in cameras:
        prefix: str = f"/world/{camera}/pinhole:simplecv.components.Distortion"
        if f"{prefix}Model" not in table.column_names or table.column(f"{prefix}Model").null_count == table.num_rows:
            lenses[camera] = fallback
            continue
        model: str = str(np.asarray(first_valid_value(table.column(f"{prefix}Model"), component_name=f"{prefix}Model"), dtype=str).reshape(-1)[0])
        if model not in get_args(LensModel):
            raise ValueError(f"{camera} stores lens model {model!r}; refine undistorts {get_args(LensModel)}")
        coefficients: Float64[ndarray, " k"] = np.asarray(first_valid_value(table.column(f"{prefix}Coefficients")), dtype=np.float64).reshape(-1)
        lenses[camera] = Lens(cast(LensModel, model), tuple(coefficients.tolist()))
    return lenses


def segment_cameras(union: CatalogRigLayout, statics: pa.Table) -> CatalogRigLayout:
    """The dataset's cameras (``union``, from the schema: every segment's) as one segment stores them in its static table
    (``reader(index=None)`` over the rig, camera, pinhole and video nodes).

    A camera is in the segment when its video has a codec here (the static read has no samples, so the codec stands for the
    video; SelfCap's 3-phone episodes put the OAK where 4-phone ones have a phone). It is calibrated when its camera node
    stores a ``Transform3D`` and its pinhole a ``Pinhole`` here, and its camera-node components are those stored here, so a
    base calibration the pipeline registered on one segment marks only that segment. Rig kinds come from the same table;
    whether a rig moves stays the union's (a temporal transform).
    """
    stored: set[str] = {name for name in statics.column_names if statics.column(name).null_count < statics.num_rows}

    def components(entity: str) -> frozenset[str]:
        return frozenset(name.removeprefix(f"{entity}:") for name in stored if name.startswith(f"{entity}:"))

    present: list[CatalogCamera] = [c for c in union.cameras if f"{c.video_entity}:VideoStream:codec" in stored]
    kinds: dict[str, RigKind] = rig_kinds_from_table(statics, dict.fromkeys(c.rig for c in present))
    cameras: list[CatalogCamera] = []
    for camera in present:
        node: frozenset[str] = components(camera.transform_entity)
        calibrated: bool = any(c.startswith("Transform3D:") for c in node) and any(c.startswith("Pinhole:") for c in components(camera.pinhole_entity))
        cameras.append(replace(camera, rig_kind=kinds.get(camera.rig), has_calibration=calibrated, camera_node_components=node))
    return CatalogRigLayout(tuple(cameras))


def discover_rig_layout(segment: SegmentRef, *, exo_rigs: tuple[str, ...] | None = None, lens: Lens = PINHOLE) -> RigLayout:
    """Discover one segment's exo/ego camera layout (``segment_cameras`` over one static read, then ``select_rig_layout``),
    and each exo camera's lens (the stored one, else ``lens``)."""
    dataset: DatasetEntry = segment.entry()  # the entry must outlive its queries
    union: CatalogRigLayout = parse_rig_layout(catalog_components(dataset.schema()))
    nodes: list[str] = [f"/world/{rig}" for rig in union.rigs] + [e for c in union.cameras for e in (c.transform_entity, c.pinhole_entity, c.video_entity)]
    statics: pa.Table = dataset.filter_segments(segment.segment).filter_contents(nodes).reader(index=None).to_arrow_table()
    layout: RigLayout = select_rig_layout(segment_cameras(union, statics), exo_rigs=exo_rigs)
    return replace(layout, lenses=lenses_from_table(statics, layout.exo_camera_names, fallback=lens))
