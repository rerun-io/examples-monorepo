"""Atomic publication of layer artifacts: tmp file → ``os.replace``.

Half-written files poison batch runs and the catalog server, so a target either
exists complete or not at all ("exists = done"). Never save over an rrd (or an
``.rbl``) that a catalog server has registered — truncation poisons the server;
re-runs write a fresh tmp and atomically replace, and the catalog re-registers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from rerun.recording_stream import RecordingStream

from dataforge import schema
from dataforge.identity import SequenceIdentity

SEGMENT_LINK_COLUMN: str = "recording link"
"""The segment table's generated URI column; the one the table blueprint turns into a preview."""


@dataclass(frozen=True, slots=True)
class TableField:
    """One segment-table column a layout shows by default, under a readable header."""

    column: str
    """Physical column name, e.g. ``property:episode:action``."""
    name: str
    """Header the card or table shows instead of the physical name."""


@dataclass(frozen=True, slots=True)
class TableFields:
    """The columns each segment-table layout shows by default, in display order.

    A layout with no fields keeps the viewer's default (every property column); a layout
    with fields shows exactly those and hides every other non-system column, which the
    viewer's column menu can still bring back.
    """

    cards: tuple[TableField, ...] = ()
    """Card body fields, below the title and the preview."""
    table: tuple[TableField, ...] = ()
    """Table columns after the recording link."""


def blueprint_views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Every view in the blueprint's container tree, in layout order."""
    views: list[rrb.View] = []

    def walk(node: rrb.View | rrb.Container) -> None:
        if isinstance(node, rrb.View):
            views.append(node)
        else:
            for child in node.contents:
                walk(child)

    walk(blueprint.root_container)
    return views


def save_table_blueprint(
    blueprint: rrb.Blueprint, target: Path, *, timeline: str, fields: TableFields, columns: Sequence[str]
) -> None:
    """Write a Rerun 0.38 segment-table blueprint: the views plus the ``/table`` entities.

    0.38 redesigned table blueprints (the 0.37 files are ignored): the card and table layouts
    each name the preview column, and ``TableColumnPreview`` lists the views a preview renders.
    Every view is embedded as in a plain ``.rbl``; the extra entities are logged on the
    blueprint timeline through the low-level archetypes, as the SDK's ``table_blueprints``
    example does until a Python API exists. The recording link column comes first in the
    table layout and is the card's link; the card title is the recording name property.

    Args:
        blueprint: The preview views, laid out as the card shows them.
        target: ``.rbl`` to write; replaced atomically.
        timeline: Timeline the previews play on.
        fields: Columns each layout shows by default; see ``TableFields``.
        columns: Every column of the segment table; a layout with fields hides the undeclared
            ones by name (the viewer shows every property column otherwise). ``rerun_*`` system
            columns keep the viewer default (hidden).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(target) as temp_path, RecordingStream._from_native(
        bindings.new_blueprint(application_id=APPLICATION_ID, make_default=False, make_thread_default=False, default_enabled=True)
    ) as stream:
        stream.save(str(temp_path))
        stream.set_time("blueprint", sequence=0)
        blueprint._log_to_stream(stream)
        link: str = rr.escape_entity_path_part(SEGMENT_LINK_COLUMN)
        view_paths: list[str] = [view.blueprint_path() for view in blueprint_views(blueprint)]
        for prefix, shown in (("/table/layouts/table/columns", fields.table), ("/table/layouts/cards/fields", fields.cards)):
            stream.log(f"{prefix}/{link}", rrb.experimental.TableColumn(cell_kind=rrb.components.TableCellKind.Preview))
            stream.log(f"{prefix}/{link}", rrb.experimental.TableColumnPreview(views=view_paths))
            if not shown:
                continue  # no declared fields: the viewer shows every property column
            for field in shown:
                stream.log(f"{prefix}/{rr.escape_entity_path_part(field.column)}", rrb.experimental.TableColumn(visible=True, name=field.name))
            declared: set[str] = {field.column for field in shown}
            for column in columns:
                if column not in declared and column != SEGMENT_LINK_COLUMN and not column.startswith("rerun_"):
                    stream.log(f"{prefix}/{rr.escape_entity_path_part(column)}", rrb.experimental.TableColumn(visible=False))
        stream.log("/table", rrb.experimental.PreviewsConfig(timeline=timeline))
        stream.log("/table/layouts/table", rrb.experimental.TableLayout(column_order=[SEGMENT_LINK_COLUMN, *(f.column for f in fields.table)]))
        stream.log(
            "/table/layouts/cards",
            rrb.experimental.CardLayout(
                field_order=[SEGMENT_LINK_COLUMN, *(f.column for f in fields.cards)], title="property:RecordingInfo:name", link=SEGMENT_LINK_COLUMN
            ),
        )

APPLICATION_ID: str = "dataforge"
"""Rerun application id of every recording this package writes; one package, one app."""


def should_skip(target: Path, *, force: bool) -> bool:
    """Idempotency check: an existing target is done unless ``--force``."""
    return target.exists() and not force


@contextmanager
def atomic_write(target: Path) -> Iterator[Path]:
    """Yield a temp path beside ``target`` that replaces it only on clean exit.

    Args:
        target: Final location; its parent directory is created if missing.

    Yields:
        A temp path in ``target.parent`` for the caller to write into.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, suffix=f"{target.suffix}.tmp")
    os.close(descriptor)
    temp_path: Path = Path(temp_name)
    try:
        yield temp_path
        temp_path.chmod(0o644)  # mkstemp temps are 0600; NFS mounts and the catalog server need world-readable files
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


@contextmanager
def recording_to(
    path: Path,
    *,
    application_id: str = APPLICATION_ID,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
    send_properties: bool = True,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to exactly ``path``; the caller owns publication.

    The lower half of ``atomic_recording``, for a converter writing **several**
    layers that must be published together: it stages each one under its own
    ``atomic_write`` and writes into the temp paths, which ``atomic_recording``'s
    single-target contract cannot express. The recording is flushed and closed on
    exit, so ``path`` is complete and readable by the time the caller returns —
    which is what lets a derived layer read the staged base rrd it was just
    handed.

    Args:
        path: Exact file to save into; nothing is replaced or cleaned up here.
        application_id: Rerun application id; one per package.
        recording_id: Shared across a sequence's layers — it is what stacks them.
        default_blueprint: Layout embedded in the file, if any.
        send_properties: Whether Rerun adds its own ``RecordingInfo`` (the name
            and the wall-clock start time). See ``atomic_recording``.
    """
    with rr.RecordingStream(application_id=application_id, recording_id=recording_id, send_properties=send_properties) as recording:
        recording.save(path, default_blueprint=default_blueprint)
        yield recording


@contextmanager
def atomic_recording(
    target: Path,
    *,
    application_id: str = APPLICATION_ID,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
    send_properties: bool = True,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to a temp file that replaces ``target`` on success.

    Args:
        target: Final location, replaced atomically on a clean exit.
        application_id: Rerun application id; one per package.
        recording_id: Shared across a sequence's layers — it is what stacks them.
        default_blueprint: Layout embedded in the file, if any.
        send_properties: Whether Rerun adds its own ``RecordingInfo``. A **base**
            layer wants it: its wall-clock ``start_time`` is when the capture was
            converted, and the viewer shows it. A **derived** layer passes
            ``False`` — it is the same recording as the base it stacks onto, so a
            second name and a second start time are duplicate, and its own
            ``start_time`` would be whenever it was last rebuilt. Explicit
            ``send_property``/``send_recording_name`` calls still land either way
            (verified on rerun 0.37.0), so a layer's own property group is
            unaffected.
    """
    # The recording stream is the inner context on purpose: it flushes and closes
    # before atomic_write reaches its os.replace.
    with (
        atomic_write(target) as temp_path,
        recording_to(
            temp_path,
            application_id=application_id,
            recording_id=recording_id,
            default_blueprint=default_blueprint,
            send_properties=send_properties,
        ) as recording,
    ):
        yield recording


def select_segments(entry: DatasetEntry, requested: Sequence[str]) -> list[str]:
    """Registered segment IDs to work on: the requested ones, or every segment when none are named.

    Raises:
        ValueError: If a requested ID is absent from the dataset or is not a plain file stem.
    """
    registered: set[str] = set(entry.segment_ids())
    selected: list[str] = sorted(requested or registered)
    missing: set[str] = set(selected) - registered
    if missing:
        raise ValueError(f"segments absent from {entry.name}: {sorted(missing)}")
    for segment in selected:
        if Path(segment).name != segment:
            raise ValueError(f"invalid segment ID: {segment}")
    return selected


def write_segment_layer(
    entry: DatasetEntry,
    layer: str,
    output_dir: Path,
    segments: Sequence[str],
    log: Callable[[str, rr.RecordingStream], None],
    *,
    application_id: str = "dataforge",
    register: bool = True,
) -> list[str]:
    """Write one ``<segment>.rrd`` per segment under ``output_dir`` and register them as ``layer``.

    Each file is written atomically with the segment's own recording id, so it
    joins that segment; an existing copy of the layer is replaced. A layer never
    restates recording properties: a second ``/__properties`` chunk collides with
    the base segment's when the catalog merges layers. Returns the registered
    file URIs (``register=False`` only prepares the files).
    """
    outputs: list[str] = []
    for segment in segments:
        output: Path = output_dir.resolve() / f"{segment}.rrd"
        with atomic_recording(output, application_id=application_id, recording_id=segment, send_properties=False) as recording:
            log(segment, recording)
        outputs.append(output.as_uri())
    if register and outputs:
        entry.register(outputs, layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    return outputs


def send_capture_properties(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    *,
    num_cameras: int,
    num_frames: int,
    **extra: Any,
) -> None:
    """Name the recording and stamp it with the dataforge capture/convert properties.

    Args:
        recording: Destination recording stream.
        identity: Sequence identity; its ``recording_id`` becomes the recording name.
        num_cameras: Cameras actually logged in this recording.
        num_frames: Longest per-camera video sample count.
        **extra: Dataset-specific capture keys (``start_time_ns``, ``task``, …);
            ``None`` values are dropped rather than written as empty values.
    """
    recording.send_recording_name(identity.recording_id)
    present: dict[str, Any] = {key: value for key, value in extra.items() if value is not None}
    recording.send_property(
        "capture",
        rr.AnyValues(schema=schema.DATAFORGE_SCHEMA_VERSION, num_frames=num_frames, num_cameras=num_cameras, **present),
    )
    recording.send_property("convert", rr.AnyValues(version="1"))
