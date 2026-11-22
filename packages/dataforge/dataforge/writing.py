"""Atomic publication of layer artifacts: tmp file → ``os.replace``.

Half-written files poison batch runs and the catalog server, so a target either
exists complete or not at all ("exists = done"). Never save over an rrd (or an
``.rbl``) that a catalog server has registered — truncation poisons the server;
re-runs write a fresh tmp and atomically replace, and the catalog re-registers.
A writer that must never replace a file publishes with a hard link instead (``atomic_write(exclusive=True)``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings
from rerun.catalog import DatasetEntry
from rerun.recording_stream import RecordingStream
from simplecv import catalog_layers
from simplecv.catalog_layers import atomic_write

from dataforge import schema
from dataforge.identity import SequenceIdentity

CONVERT_SCHEMA_VERSION: str = "2"
"""Schema version shared by recording properties and conversion timing records."""

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
    hidden: tuple[str, ...] = ()
    """Columns a layout with fields hides even when the segment table does not list them at publish time."""
    sort_by: str | None = None
    """Column that orders the rows, ascending; ignored when the table does not have it. None keeps the data order."""


@dataclass(frozen=True, slots=True)
class PublishedColumns:
    """``TableFields`` applied to the columns a segment table has at publish time."""

    table_hidden: tuple[str, ...]
    """Columns the table layout hides."""
    cards_hidden: tuple[str, ...]
    """Columns the card layout hides."""
    sort_by: str | None
    """Column that orders the rows, when the table has it."""


def published_columns(fields: TableFields, columns: Sequence[str]) -> PublishedColumns:
    """Resolve what each segment-table layout hides and sorts by.

    A layout with fields hides every column it does not declare, the system columns and ``fields.hidden`` too,
    except the recording link. A layout with no fields hides nothing.

    Args:
        fields: Columns each layout shows by default.
        columns: Every column of the segment table at publish time.
    """

    def hidden(shown: tuple[TableField, ...]) -> tuple[str, ...]:
        declared: set[str] = {field.column for field in shown}
        return tuple(column for column in dict.fromkeys((*columns, *fields.hidden)) if shown and column not in declared and column != SEGMENT_LINK_COLUMN)

    return PublishedColumns(hidden(fields.table), hidden(fields.cards), fields.sort_by if fields.sort_by in columns else None)


def episode_table_fields(*names: str) -> TableFields:
    """Cards and table both show ``property:episode:<name>`` for each of ``names``, then the frame count."""
    fields: tuple[TableField, ...] = (*(TableField(f"property:episode:{name}", name) for name in names), TableField("property:capture:num_frames", "frames"))
    return TableFields(cards=fields, table=fields)


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
    """Write a Rerun 0.41 segment-table blueprint: the views plus the ``/table`` entities.

    The card and table layouts each name the preview column; ``TableColumnPreview`` marks it as
    a preview column and lists the views a preview renders.
    Every view is embedded as in a plain ``.rbl``; the extra entities are logged on the
    blueprint timeline through the low-level archetypes, as the SDK's ``table_blueprints``
    example does until a Python API exists. The recording link column comes first in the
    table layout and is the card's link; the card title is the recording name property.

    Args:
        blueprint: The preview views, laid out as the card shows them.
        target: ``.rbl`` to write; replaced atomically.
        timeline: Timeline the previews play on.
        fields: Columns each layout shows by default; see ``TableFields``.
        columns: Every column of the segment table at publish time; see ``published_columns``.
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
        published: PublishedColumns = published_columns(fields, columns)
        layouts: tuple[tuple[str, tuple[TableField, ...], tuple[str, ...]], ...] = (
            ("/table/layouts/table/columns", fields.table, published.table_hidden),
            ("/table/layouts/cards/fields", fields.cards, published.cards_hidden),
        )
        for prefix, shown, hidden in layouts:
            stream.log(f"{prefix}/{link}", rrb.experimental.TableColumnPreview(views=view_paths))
            for field in shown:
                stream.log(f"{prefix}/{rr.escape_entity_path_part(field.column)}", rrb.experimental.TableColumn(visible=True, name=field.name))
            for column in hidden:
                stream.log(f"{prefix}/{rr.escape_entity_path_part(column)}", rrb.experimental.TableColumn(visible=False))
        stream.log("/table", rrb.experimental.PreviewsConfig(timeline=timeline))
        if published.sort_by is not None:
            stream.log("/table", rrb.experimental.TableBlueprint(sort_by=rrb.table.SortBy(published.sort_by, direction="Ascending")))
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
        # The footer lets the catalog serve each chunk whole (meshes.log_mesh_batch relies on it).
        recording.save(path, default_blueprint=default_blueprint, write_footer=True)
        yield recording


atomic_recording = partial(catalog_layers.atomic_recording, application_id=APPLICATION_ID)
"""``simplecv.catalog_layers.atomic_recording`` with this package's application id (a caller may still pass another)."""


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

    Use ``dataforge-swap-layer`` for gated replacement on the shared server.

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
    if register:
        catalog_layers.register_layer(entry, [output_dir.resolve() / f"{segment}.rrd" for segment in segments], layer)
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
    recording.send_property("convert", rr.AnyValues(version=CONVERT_SCHEMA_VERSION))
