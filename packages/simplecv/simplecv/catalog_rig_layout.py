"""Read-side model of the ``exoego:v2`` rig layout as it appears in a Rerun catalog schema.

The writer (``simplecv.rerun_rig_logger`` through ``BaseExoEgoSequence.build_rig_layout``)
puts every camera under ``/world/rig_NN/cam_MM``: the video stream on
``.../pinhole/video``, the intrinsics on ``.../pinhole``, the camera's static
``Transform3D`` on the camera node, and a temporal ``Transform3D`` on the rig node
of a moving (worn) rig. See ``docs/exoego_schema.md``. This module is the one place
that turns a catalog schema — entity paths, component identifiers, static flags —
back into typed cameras; consumers (exo-calib, mv-api) keep only their own
selection policy on top of :class:`CatalogRigLayout`.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, get_args, runtime_checkable

import numpy as np
import pyarrow as pa
from rerun.catalog import DatasetEntry

from simplecv.rrd_query_utils import first_valid_value

RigKind: TypeAlias = Literal["exo", "ego", "quest"]
"""The ``kind`` metadata the writer logs on a rig node (``docs/exoego_schema.md``): a static exocentric rig,
a worn egocentric device, or a Quest headset."""
RIG_KINDS: tuple[RigKind, ...] = get_args(RigKind)
RIG_KIND_BY_NAME: dict[str, RigKind] = {kind: kind for kind in RIG_KINDS}
"""Narrows a stored ``kind`` string back to :data:`RigKind` (``None`` for anything else)."""

_CAMERA_VIDEO_PATH: re.Pattern[str] = re.compile(r"^/world/(rig_\d+)/(cam_\d+)/pinhole/video$")
_CAMERA_PINHOLE_PATH: re.Pattern[str] = re.compile(r"^/world/(rig_\d+)/(cam_\d+)/pinhole$")
_CAMERA_NODE_PATH: re.Pattern[str] = re.compile(r"^/world/(rig_\d+)/(cam_\d+)$")
_RIG_NODE_PATH: re.Pattern[str] = re.compile(r"^/world/(rig_\d+)$")


@runtime_checkable
class ComponentColumnLike(Protocol):
    """The three attributes this module reads from a schema column.

    ``rerun_bindings.ComponentColumnDescriptor`` (what ``Schema.component_columns()``
    yields) satisfies it; tests use plain stand-ins.
    """

    @property
    def entity_path(self) -> str: ...

    @property
    def component(self) -> str: ...

    @property
    def is_static(self) -> bool: ...


@runtime_checkable
class SchemaLike(Protocol):
    """A catalog schema: ``rerun.catalog.Schema`` or anything else that lists its component columns."""

    def component_columns(self) -> Iterable[ComponentColumnLike]: ...


@dataclass(frozen=True, slots=True)
class CatalogComponent:
    """One component column of a catalog schema, with rerun's ``Archetype:field`` identifier parsed."""

    entity_path: str
    """Entity path with a leading slash, e.g. ``/world/rig_00/cam_00``."""
    archetype: str | None
    """Archetype of the component, e.g. ``Transform3D``; ``None`` for a bare ``AnyValues`` field."""
    field: str
    """Field within the archetype (``translation``), or the bare field name."""
    is_static: bool
    """Whether the column is static (logged without a timeline)."""

    @classmethod
    def parse(cls, entity_path: str, component: str, is_static: bool) -> "CatalogComponent":
        """Build from rerun's column identifier, ``Archetype:field`` or a bare field name.

        A namespaced archetype (``rerun.archetypes.Transform3D:mat3x3``, as older bindings
        spelled it) is reduced to its bare name.
        """
        archetype, separator, field = component.partition(":")
        return cls(
            entity_path=f"/{entity_path.lstrip('/')}",
            archetype=archetype.rsplit(".", 1)[-1] if separator else None,
            field=field if separator else archetype,
            is_static=is_static,
        )

    @property
    def component(self) -> str:
        """Rerun's column identifier, as stored in the catalog."""
        return f"{self.archetype}:{self.field}" if self.archetype is not None else self.field


def catalog_components(schema: SchemaLike) -> list[CatalogComponent]:
    """Parse a catalog schema's component columns."""
    return [
        CatalogComponent.parse(str(column.entity_path), str(column.component), bool(column.is_static))
        for column in schema.component_columns()
    ]


@dataclass(frozen=True, slots=True)
class CatalogCamera:
    """One rig camera that has a video stream in the catalog."""

    rig: str
    """Rig entity id, e.g. ``rig_00``."""
    camera: str
    """Camera entity id within the rig, e.g. ``cam_00``."""
    rig_kind: RigKind | None
    """The rig node's static ``kind`` metadata when the writer logged one."""
    rig_is_moving: bool
    """Whether the rig node carries a temporal ``Transform3D`` (a worn device)."""
    has_calibration: bool
    """Whether the camera node has a static ``Transform3D`` and its pinhole node a ``Pinhole``."""
    camera_node_components: frozenset[str]
    """Static component identifiers on the camera node (calibration plus any writer markers)."""

    @property
    def name(self) -> str:
        """Camera name under ``/world``, e.g. ``rig_00/cam_00``."""
        return f"{self.rig}/{self.camera}"

    @property
    def transform_entity(self) -> str:
        """Entity path of the camera node, which carries ``Transform3D``."""
        return f"/world/{self.rig}/{self.camera}"

    @property
    def pinhole_entity(self) -> str:
        """Entity path of the camera's pinhole node."""
        return f"/world/{self.rig}/{self.camera}/pinhole"

    @property
    def video_entity(self) -> str:
        """Entity path of the camera's ``VideoStream``."""
        return f"/world/{self.rig}/{self.camera}/pinhole/video"


@dataclass(frozen=True, slots=True)
class CatalogRigLayout:
    """Every rig camera with a video stream, in rig / camera index order."""

    cameras: tuple[CatalogCamera, ...]
    """Cameras sorted by numeric rig index, then camera index."""

    @property
    def rigs(self) -> tuple[str, ...]:
        """Rig entity ids with at least one video camera, in rig index order."""
        return tuple(dict.fromkeys(camera.rig for camera in self.cameras))


def _camera_sort_key(camera: CatalogCamera) -> tuple[int, int]:
    return int(camera.rig.removeprefix("rig_")), int(camera.camera.removeprefix("cam_"))


def parse_rig_layout(components: Iterable[CatalogComponent], rig_kinds: Mapping[str, RigKind] | None = None) -> CatalogRigLayout:
    """Discover the rig cameras described by a catalog schema.

    Args:
        components: The schema's component columns (see :func:`catalog_components`).
        rig_kinds: Static ``kind`` metadata per rig id, read separately from the data
            (the schema only names the column).

    Returns:
        The rig layout. Only cameras whose video entity carries ``VideoStream:sample``
        are reported; a codec column alone is not a video.
    """
    kinds: Mapping[str, RigKind] = rig_kinds if rig_kinds is not None else {}
    video_cameras: set[tuple[str, str]] = set()
    moving_rigs: set[str] = set()
    static_transform_cameras: set[tuple[str, str]] = set()
    pinhole_cameras: set[tuple[str, str]] = set()
    camera_node_components: dict[tuple[str, str], set[str]] = {}
    for item in components:
        video_match: re.Match[str] | None = _CAMERA_VIDEO_PATH.fullmatch(item.entity_path)
        if video_match is not None and item.archetype == "VideoStream" and item.field == "sample":
            video_cameras.add((video_match.group(1), video_match.group(2)))
            continue
        rig_match: re.Match[str] | None = _RIG_NODE_PATH.fullmatch(item.entity_path)
        if rig_match is not None and item.archetype == "Transform3D" and not item.is_static:
            moving_rigs.add(rig_match.group(1))
            continue
        camera_match: re.Match[str] | None = _CAMERA_NODE_PATH.fullmatch(item.entity_path)
        if camera_match is not None and item.is_static:
            key: tuple[str, str] = (camera_match.group(1), camera_match.group(2))
            camera_node_components.setdefault(key, set()).add(item.component)
            if item.archetype == "Transform3D":
                static_transform_cameras.add(key)
            continue
        pinhole_match: re.Match[str] | None = _CAMERA_PINHOLE_PATH.fullmatch(item.entity_path)
        if pinhole_match is not None and item.archetype == "Pinhole":
            pinhole_cameras.add((pinhole_match.group(1), pinhole_match.group(2)))
    cameras: list[CatalogCamera] = [
        CatalogCamera(
            rig=rig,
            camera=camera,
            rig_kind=kinds.get(rig),
            rig_is_moving=rig in moving_rigs,
            has_calibration=(rig, camera) in static_transform_cameras and (rig, camera) in pinhole_cameras,
            camera_node_components=frozenset(camera_node_components.get((rig, camera), set())),
        )
        for rig, camera in video_cameras
    ]
    return CatalogRigLayout(cameras=tuple(sorted(cameras, key=_camera_sort_key)))


def read_rig_kinds(dataset: DatasetEntry, segment_id: str, rigs: Iterable[str]) -> dict[str, RigKind]:
    """Read the static ``kind`` metadata of the given rig nodes (the schema only names the column).

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        rigs: Rig entity ids such as ``rig_00`` (see :attr:`CatalogRigLayout.rigs`).

    Returns:
        Kind per rig id, for the rigs whose node carries one.

    Raises:
        ValueError: If a rig carries a kind outside :data:`RIG_KINDS`.
    """
    rig_paths: list[str] = sorted(f"/world/{rig}" for rig in rigs)
    if not rig_paths:
        return {}
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents(rig_paths).reader(index=None).to_arrow_table()
    kinds: dict[str, RigKind] = {}
    for rig_path in rig_paths:
        column_name: str = f"{rig_path}:kind"
        if column_name not in table.column_names:
            continue
        column: pa.ChunkedArray = table.column(column_name).drop_null()
        if len(column) == 0:
            continue
        kind_name: str = str(np.asarray(first_valid_value(column, component_name=column_name), dtype=str).reshape(-1)[0])
        kind: RigKind | None = RIG_KIND_BY_NAME.get(kind_name)
        if kind is None:
            raise ValueError(f"{column_name} = {kind_name!r}; expected one of {RIG_KINDS}")
        kinds[rig_path.removeprefix("/world/")] = kind
    return kinds
