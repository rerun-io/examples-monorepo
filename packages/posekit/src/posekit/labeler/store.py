"""Reviewed hand labels of one segment: held in memory, written to and read back from the ``hand_boxes`` layer.

The layer is the only place labels persist. ``save_layer`` writes one ``.rrd`` whose recording id is the
segment id, so the catalog stacks it on the segment's other layers; ``CatalogLayers.register`` registers it
with REPLACE; ``load_layer`` rebuilds a store from the registered layer, so a server restart keeps the work.

Layout per label (``video_time``, a duration timeline in ns), at ``paths.label_path(cam, side)``:

- ``Boxes2D`` with the box (stored by Rerun as centers + half sizes), class id 100/101 and a text label;
  ``no_hand``/``rejected`` log an empty ``Boxes2D``, which clears any earlier box at that entity.
- ``AnyValues`` ``status``, ``source``, ``reviewer`` on the same entity and time.

The recording property ``hand_boxes:split`` says whether the segment is train or eval.

The reviewer's action log (``schema.ActionEvent``) sits in the same file:

- ``/labeler/actions``: one ``TextLog`` row per action (``"draw_box cam_02 right by astra"``) on ``video_time`` = the
  action's grid time and on the ``wall_time`` timestamp timeline, with the event's fields as ``AnyValues``
  (``wall_ns``, ``t_ns``, ``t1_ns``, ``cam``, ``side``, ``kind``, ``reviewer``, ``n_labels``); read back on ``wall_time``,
  which is unique per action.
- ``/labeler/interventions/cumulative``: ``Scalars``, the number of interventions at or before each ``video_time``.
- Recording properties ``hand_boxes_effort``: the ``EffortSummary`` fields (undefined ratios left out).

Autosave writes the same layout to a second file (``autosave_path``) that is never registered: rewriting
a file the catalog has registered, in place, could hand the catalog bytes that disagree with its manifest.
``CatalogLayers.open_store`` recovers from the autosave file when it is newer than the committed layer.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetSelection, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader, StoreEntry
from serde import from_dict, to_dict

from posekit.labeler.paths import class_id, label_path
from posekit.labeler.schema import (
    HAND_SIDES,
    LABEL_LAYER,
    PASSIVE_ACTIONS,
    ActionEvent,
    EffortSummary,
    FrameLabels,
    HandSide,
    SegmentSplit,
    ViewLabel,
)

TIMELINE: str = "video_time"
"""Timeline the labels sit on (a duration timeline, ns)."""
LAYER_DIR: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-labeler/layers") / LABEL_LAYER
"""Where committed ``hand_boxes`` layer files go, one ``<segment>.rrd`` each."""
N_CAMS: int = 6
"""Cameras on a RoboCap rig; the layer reader asks for these entity paths."""
WALL_TIMELINE: str = "wall_time"
"""Timestamp timeline of the action log (wall clock of each reviewer action)."""
ACTIONS_ENTITY: str = "/labeler/actions"
"""Entity of the action log."""
CUMULATIVE_ENTITY: str = "/labeler/interventions/cumulative"
"""Entity of the cumulative intervention count."""
EFFORT_PROPERTY: str = "hand_boxes_effort"
"""Recording property group of the effort summary."""


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One stored label with its time, the unit the layer writer and reader exchange."""

    t_ns: int
    """``video_time``, ns."""
    label: ViewLabel
    """The decision."""


@dataclass(frozen=True, slots=True)
class LayerContents:
    """Everything a ``hand_boxes`` layer holds for one segment."""

    rows: list[LabelRow]
    """Labels, sorted by (t_ns, cam, side)."""
    actions: list[ActionEvent]
    """Action log, sorted by wall time."""


@dataclass(frozen=True, slots=True)
class NewWrite:
    """Outcome of ``LabelStore.put_new``."""

    written: int
    """Labels written where there was none."""
    kept: int
    """Views that already had a label, left untouched."""


@dataclass(frozen=True, slots=True)
class RangeWrite:
    """Outcome of ``LabelStore.put_range`` at one time."""

    stored: int
    """Range labels written."""
    open_views: int
    """Views left without any label (no reviewed label, no range decision)."""


class LabelStore:
    """Thread-safe labels of one segment, keyed by (t_ns, cam, side)."""

    def __init__(self, segment_id: str, *, split: SegmentSplit, application_id: str) -> None:
        """Start an empty store.

        Args:
            segment_id: Catalog segment id; also the recording id of the layer file.
            split: train or eval, written as the ``hand_boxes:split`` recording property.
            application_id: Application id of the segment's base layer, reused by the layer file.
        """
        self.segment_id: str = segment_id
        self.split: SegmentSplit = split
        self.application_id: str = application_id
        self._labels: dict[int, dict[tuple[int, HandSide], ViewLabel]] = {}
        """Labels by time, then by (cam, side)."""
        self._actions: list[ActionEvent] = []
        """Action log, in logging order (wall times strictly increasing)."""
        self._unsaved: int = 0
        self._lock = threading.Lock()

    def put(self, frame: FrameLabels) -> int:
        """Store every label of ``frame``, replacing earlier labels of the same (t_ns, cam, side).

        Returns:
            Number of labels stored.
        """
        if frame.segment_id != self.segment_id:
            raise ValueError(f"labels for segment {frame.segment_id!r} posted to store of {self.segment_id!r}")
        with self._lock:
            at_time: dict[tuple[int, HandSide], ViewLabel] = self._labels.setdefault(frame.t_ns, {})
            for label in frame.labels:
                at_time[(label.cam, label.side)] = label
            self._unsaved += len(frame.labels)
        return len(frame.labels)

    def put_new(self, labels: list[LabelRow]) -> NewWrite:
        """Write each label only where its (t_ns, cam, side) has no label yet; one critical section, so a label posted meanwhile is never overwritten."""
        written: int = 0
        kept: int = 0
        with self._lock:
            for row in labels:
                at_time: dict[tuple[int, HandSide], ViewLabel] = self._labels.setdefault(row.t_ns, {})
                key: tuple[int, HandSide] = (row.label.cam, row.label.side)
                if key in at_time:
                    kept += 1
                    continue
                at_time[key] = row.label
                written += 1
            self._unsaved += written
        return NewWrite(written, kept)

    def log_actions(self, events: list[ActionEvent]) -> list[ActionEvent]:
        """Append actions to the log; a wall time not after the previous one is moved 1 ns past it (so wall times are unique).

        Returns:
            The events as logged.
        """
        logged: list[ActionEvent] = []
        with self._lock:
            for event in events:
                last: int = self._actions[-1].wall_ns if self._actions else -1
                stored: ActionEvent = event if event.wall_ns > last else dataclasses.replace(event, wall_ns=last + 1)
                self._actions.append(stored)
                logged.append(stored)
            self._unsaved += len(logged)
        return logged

    def actions(self) -> list[ActionEvent]:
        """Snapshot of the action log, in logging order."""
        with self._lock:
            return list(self._actions)

    def put_range(self, t_ns: int, decisions: dict[tuple[int, HandSide], ViewLabel | None]) -> RangeWrite:
        """Apply a range fill's decisions at one time, atomically with respect to every other write.

        A view whose stored label is not a ``range`` label is left as it is. Otherwise a decision replaces the
        stored label, and an undecided view (``None``) loses its stale ``range`` label.

        Args:
            t_ns: Grid time.
            decisions: Per (cam, side): the ``range`` label to store, or None when the fill cannot decide.
        """
        stored: int = 0
        open_views: int = 0
        with self._lock:
            at_time: dict[tuple[int, HandSide], ViewLabel] = self._labels.setdefault(t_ns, {})
            for key, decided in decisions.items():
                existing: ViewLabel | None = at_time.get(key)
                if existing is not None and existing.source != "range":
                    continue
                if decided is not None:
                    at_time[key] = decided
                    stored += 1
                    self._unsaved += 1
                    continue
                open_views += 1
                if existing is not None:
                    del at_time[key]
                    self._unsaved += 1
            if not at_time:
                del self._labels[t_ns]
        return RangeWrite(stored, open_views)

    def get(self, t_ns: int) -> list[ViewLabel]:
        """Labels at one time, sorted by (cam, side)."""
        with self._lock:
            found: list[ViewLabel] = list(self._labels.get(t_ns, {}).values())
        return sorted(found, key=lambda label: (label.cam, label.side))

    def get_view(self, t_ns: int, cam: int, side: HandSide) -> ViewLabel | None:
        """The label of one (t_ns, cam, side), or None."""
        with self._lock:
            return self._labels.get(t_ns, {}).get((cam, side))

    def rows(self) -> list[LabelRow]:
        """Snapshot of every label, sorted by (t_ns, cam, side)."""
        with self._lock:
            rows: list[LabelRow] = [LabelRow(t_ns, label) for t_ns, at_time in self._labels.items() for label in at_time.values()]
        return sorted(rows, key=lambda row: (row.t_ns, row.label.cam, row.label.side))

    def __len__(self) -> int:
        with self._lock:
            return sum(len(at_time) for at_time in self._labels.values())

    @property
    def unsaved(self) -> int:
        """Labels put or removed since the last ``save_layer``."""
        with self._lock:
            return self._unsaved

    def save_layer(self, path: Path, effort: EffortSummary | None = None) -> int:
        """Write the ``hand_boxes`` layer (labels, action log, effort properties) to ``path`` atomically (temp file + rename, mode 0644).

        Returns:
            Number of labels written.
        """
        with self._lock:
            unsaved_before: int = self._unsaved
        rows: list[LabelRow] = self.rows()
        actions: list[ActionEvent] = self.actions()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".rrd.tmp", dir=path.parent)
        os.close(handle)
        temp_path: Path = Path(temp_name)
        try:
            write_layer_rrd(
                temp_path, rows, segment_id=self.segment_id, split=self.split, application_id=self.application_id, actions=actions, effort=effort
            )
            temp_path.chmod(0o644)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        with self._lock:
            self._unsaved -= unsaved_before
        return len(rows)

    def load_rows(self, rows: list[LabelRow], actions: list[ActionEvent] | None = None) -> None:
        """Replace the store's labels with ``rows`` and its action log with ``actions`` (used when rebuilding from a layer)."""
        with self._lock:
            self._labels = {}
            for row in rows:
                self._labels.setdefault(row.t_ns, {})[(row.label.cam, row.label.side)] = row.label
            self._actions = sorted(actions or [], key=lambda event: event.wall_ns)
            self._unsaved = 0


def action_text(event: ActionEvent) -> str:
    """One line for the ``TextLog`` of an action, e.g. ``draw_box cam_02 right by astra``."""
    if event.kind == "accept_range":
        return f"accept_range to {event.t1_ns} ns: {event.n_labels} labels by {event.reviewer}"
    if event.cam is None or event.side is None:
        return f"{event.kind}: {event.n_labels} labels by {event.reviewer}"
    return f"{event.kind} cam_{event.cam:02d} {event.side} by {event.reviewer}"


def write_actions(recording: rr.RecordingStream, actions: list[ActionEvent]) -> None:
    """Log the action log and the cumulative intervention count into ``recording``; see the module docstring."""
    if not actions:
        return
    ordered: list[ActionEvent] = sorted(actions, key=lambda event: event.wall_ns)
    video_ns: Int64[ndarray, "n"] = np.array([event.t_ns for event in ordered], dtype=np.int64)
    wall_ns: Int64[ndarray, "n"] = np.array([event.wall_ns for event in ordered], dtype=np.int64)
    values: rr.ComponentColumnList = rr.AnyValues.columns(
        wall_ns=wall_ns,
        t_ns=video_ns,
        t1_ns=pa.array([event.t1_ns for event in ordered], type=pa.int64()),
        cam=pa.array([event.cam for event in ordered], type=pa.int64()),
        side=pa.array([event.side for event in ordered], type=pa.string()),
        kind=[event.kind for event in ordered],
        reviewer=[event.reviewer for event in ordered],
        n_labels=np.array([event.n_labels for event in ordered], dtype=np.int64),
    )
    texts: rr.ComponentColumnList = rr.TextLog.columns(text=[action_text(event) for event in ordered], level=["INFO"] * len(ordered))
    rr.send_columns(
        ACTIONS_ENTITY,
        indexes=[
            rr.TimeColumn(TIMELINE, duration=video_ns.astype("timedelta64[ns]")),
            rr.TimeColumn(WALL_TIMELINE, timestamp=wall_ns.astype("datetime64[ns]")),
        ],
        columns=[*texts, *values],
        recording=recording,
    )
    interventions: list[ActionEvent] = sorted(
        (event for event in ordered if event.kind not in PASSIVE_ACTIONS), key=lambda event: (event.t_ns, event.wall_ns)
    )
    # Several interventions at one time: the count after the last of them.
    cumulative: dict[int, int] = {event.t_ns: count for count, event in enumerate(interventions, start=1)}
    if cumulative:
        times_ns: Int64[ndarray, "m"] = np.array(list(cumulative), dtype=np.int64)
        rr.send_columns(
            CUMULATIVE_ENTITY,
            indexes=[rr.TimeColumn(TIMELINE, duration=times_ns.astype("timedelta64[ns]"))],
            columns=rr.Scalars.columns(scalars=np.array(list(cumulative.values()), dtype=np.float64)),
            recording=recording,
        )


def write_layer_rrd(
    path: Path,
    rows: list[LabelRow],
    *,
    segment_id: str,
    split: SegmentSplit,
    application_id: str,
    actions: list[ActionEvent] | None = None,
    effort: EffortSummary | None = None,
) -> None:
    """Write ``rows`` (and the action log and effort summary) as a ``hand_boxes`` layer recording to exactly ``path`` (no atomicity here)."""
    # send_properties=False: a layer must not restate the base layer's RecordingInfo (name, start time).
    with rr.RecordingStream(application_id, recording_id=segment_id, send_properties=False) as recording:
        recording.save(path)
        recording.send_property(LABEL_LAYER, rr.AnyValues(split=split))
        if effort is not None:
            properties: rr.AnyValues = rr.AnyValues()
            for name, value in to_dict(effort).items():
                if value is not None:  # an undefined ratio (nothing covered or measured yet) is left out
                    properties = properties.with_component_from_data(name, value)
            recording.send_property(EFFORT_PROPERTY, properties)
        write_actions(recording, actions or [])
        by_entity: dict[tuple[int, HandSide], list[LabelRow]] = {}
        for row in rows:
            by_entity.setdefault((row.label.cam, row.label.side), []).append(row)
        for (cam, side), entity_rows in sorted(by_entity.items()):
            times_ns: Int64[ndarray, "n"] = np.array([row.t_ns for row in entity_rows], dtype=np.int64)
            boxed: list[LabelRow] = [row for row in entity_rows if row.label.box is not None]
            corners: Float32[ndarray, "m 4"] = np.array(
                [[row.label.box.x0, row.label.box.y0, row.label.box.x1, row.label.box.y1] for row in boxed if row.label.box is not None],
                dtype=np.float32,
            ).reshape(-1, 4)
            # XYXY corners -> the centers/half sizes Boxes2D stores; an empty partition clears the entity's box.
            centers: Float32[ndarray, "m 2"] = (corners[:, :2] + corners[:, 2:]) / 2.0
            half_sizes: Float32[ndarray, "m 2"] = (corners[:, 2:] - corners[:, :2]) / 2.0
            lengths: list[int] = [1 if row.label.box is not None else 0 for row in entity_rows]
            boxes: rr.ComponentColumnList = rr.Boxes2D.columns(
                centers=centers,
                half_sizes=half_sizes,
                class_ids=np.full(len(boxed), class_id(side), dtype=np.uint16),
                labels=[f"{side} {row.label.status}" for row in boxed],
            ).partition(lengths)
            values: rr.ComponentColumnList = rr.AnyValues.columns(
                status=[row.label.status for row in entity_rows],
                source=[row.label.source for row in entity_rows],
                reviewer=[row.label.reviewer for row in entity_rows],
            )
            time_column: rr.TimeColumn = rr.TimeColumn(TIMELINE, duration=times_ns.astype("timedelta64[ns]"))
            rr.send_columns(label_path(cam, side), indexes=[time_column], columns=[*boxes, *values], recording=recording)


def label_entities(n_cams: int = N_CAMS) -> list[str]:
    """Every label entity path of a rig with ``n_cams`` cameras."""
    return [label_path(cam, side) for cam in range(n_cams) for side in HAND_SIDES]


def rows_from_table(table: pa.Table, n_cams: int = N_CAMS) -> list[LabelRow]:
    """Decode a catalog reader table (index ``video_time``) of the label entities into label rows, sorted by (t_ns, cam, side).

    Each reader row carries one entity's event; other entities' columns are null on it.
    """
    rows: list[LabelRow] = []
    if table.num_rows == 0 or TIMELINE not in table.column_names:
        return rows
    times_ns: list[int] = table.column(TIMELINE).cast(pa.int64()).to_pylist()
    for cam in range(n_cams):
        for side in HAND_SIDES:
            entity: str = label_path(cam, side)
            status_name: str = f"{entity}:status"
            if status_name not in table.column_names:
                continue
            statuses: list[list[str] | None] = table.column(status_name).to_pylist()
            sources: list[list[str] | None] = table.column(f"{entity}:source").to_pylist()
            reviewers: list[list[str] | None] = table.column(f"{entity}:reviewer").to_pylist()
            centers_name: str = f"{entity}:Boxes2D:centers"
            has_boxes: bool = centers_name in table.column_names
            centers: list[list[list[float]] | None] = table.column(centers_name).to_pylist() if has_boxes else [None] * table.num_rows
            halves: list[list[list[float]] | None] = (
                table.column(f"{entity}:Boxes2D:half_sizes").to_pylist() if has_boxes else [None] * table.num_rows
            )
            for index, status in enumerate(statuses):
                if not status:
                    continue
                center: list[list[float]] | None = centers[index]
                half: list[list[float]] | None = halves[index]
                box: dict[str, float] | None = None
                if center and half:
                    cx, cy = center[0]
                    hx, hy = half[0]
                    box = {"x0": cx - hx, "y0": cy - hy, "x1": cx + hx, "y1": cy + hy}
                source: list[str] | None = sources[index]
                reviewer: list[str] | None = reviewers[index]
                fields: dict[str, object] = {
                    "cam": cam,
                    "side": side,
                    "status": status[0],
                    "box": box,
                    "source": source[0] if source else "",
                    "reviewer": reviewer[0] if reviewer else "",
                }
                # pyserde checks the Literal fields and ViewLabel's status/box rule on the way in.
                rows.append(LabelRow(int(times_ns[index]), from_dict(ViewLabel, fields)))
    return sorted(rows, key=lambda row: (row.t_ns, row.label.cam, row.label.side))


def actions_from_table(table: pa.Table) -> list[ActionEvent]:
    """Decode a catalog reader table (index ``wall_time``) of the action entity into events, sorted by wall time."""
    events: list[ActionEvent] = []
    kind_name: str = f"{ACTIONS_ENTITY}:kind"
    if table.num_rows == 0 or kind_name not in table.column_names:
        return events
    names: list[str] = ["wall_ns", "t_ns", "t1_ns", "cam", "side", "kind", "reviewer", "n_labels"]
    columns: dict[str, list[list[object] | None]] = {name: table.column(f"{ACTIONS_ENTITY}:{name}").to_pylist() for name in names}
    for index in range(table.num_rows):
        if not columns["kind"][index]:
            continue
        fields: dict[str, object] = {"segment_id": "", **{name: (columns[name][index] or [None])[0] for name in names}}
        events.append(from_dict(ActionEvent, fields))
    return sorted(events, key=lambda event: event.wall_ns)


def load_layer(dataset: DatasetEntry, segment_id: str, *, n_cams: int = N_CAMS) -> list[LabelRow]:
    """Read the registered ``hand_boxes`` labels of one segment from the catalog (one round trip)."""
    entities: list[str] = label_entities(n_cams)
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents(entities).reader(index=TIMELINE).to_arrow_table()
    return rows_from_table(table, n_cams)


def load_actions(dataset: DatasetEntry, segment_id: str) -> list[ActionEvent]:
    """Read the registered action log of one segment (empty when the dataset has no ``wall_time`` timeline)."""
    view: DatasetSelection = dataset.filter_segments(segment_id)
    if WALL_TIMELINE not in [column.name for column in view.schema().index_columns()]:
        return []
    table: pa.Table = view.filter_contents([ACTIONS_ENTITY]).reader(index=WALL_TIMELINE).to_arrow_table()
    return [dataclasses.replace(event, segment_id=segment_id) for event in actions_from_table(table)]


def load_layer_contents(dataset: DatasetEntry, segment_id: str, *, n_cams: int = N_CAMS) -> LayerContents:
    """Labels and action log of one segment from the catalog."""
    return LayerContents(load_layer(dataset, segment_id, n_cams=n_cams), load_actions(dataset, segment_id))


def load_layer_file_contents(path: Path, *, n_cams: int = N_CAMS) -> LayerContents:
    """Read labels and actions from an unregistered layer file through a private in-process catalog server.

    The same decoders as ``load_layer_contents`` run on it, so the file and the catalog cannot disagree on layout.
    """
    store_entry: StoreEntry = RrdReader(path).recordings()[0]
    with rr.server.Server(host="127.0.0.1", port=0) as server:
        client: CatalogClient = server.client()
        dataset: DatasetEntry = client.create_dataset("hand_boxes_file")
        dataset.register([path.resolve().as_uri()], layer_name=LABEL_LAYER).wait()
        return load_layer_contents(dataset, store_entry.recording_id, n_cams=n_cams)


def load_layer_file(path: Path, *, n_cams: int = N_CAMS) -> list[LabelRow]:
    """Labels of an unregistered layer file (see ``load_layer_file_contents``)."""
    return load_layer_file_contents(path, n_cams=n_cams).rows


def base_application_id(dataset: DatasetEntry, segment_id: str) -> str:
    """Application id of the segment's ``base`` layer, read from the rrd header the catalog points at.

    Raises:
        ValueError: If the segment has no file-backed base layer holding exactly one recording.
    """
    table: pa.Table = dataset.segment_table().to_arrow_table()
    for row in table.select(["rerun_segment_id", "rerun_layer_names", "rerun_storage_urls"]).to_pylist():
        if row["rerun_segment_id"] != segment_id:
            continue
        for layer, url in zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True):
            if layer == "base" and url.startswith("file://"):
                recordings: list[StoreEntry] = RrdReader(Path(url.removeprefix("file://"))).recordings()
                if len(recordings) == 1:
                    return recordings[0].application_id
    raise ValueError(f"{segment_id}: no file-backed base layer with one recording on {dataset.name}")


class CatalogLayers:
    """The real layer backend: catalog dataset + layer directory."""

    def __init__(self, catalog_url: str, dataset_name: str, layer_dir: Path = LAYER_DIR) -> None:
        """Connect to the catalog.

        Args:
            catalog_url: For example ``rerun+http://127.0.0.1:9897``.
            dataset_name: For example ``robocap-labeling``.
            layer_dir: Directory of committed layer files.
        """
        self.client: CatalogClient = CatalogClient(catalog_url)
        self.dataset: DatasetEntry = self.client.get_dataset(dataset_name)
        self.layer_dir: Path = layer_dir

    def layer_path(self, segment_id: str) -> Path:
        """Committed layer file of a segment."""
        return self.layer_dir / f"{segment_id}.rrd"

    def autosave_path(self, segment_id: str) -> Path:
        """Unregistered autosave file of a segment."""
        return self.layer_dir / "autosave" / f"{segment_id}.rrd"

    def open_store(self, segment_id: str, split: SegmentSplit) -> LabelStore:
        """A store filled from the registered layer, or from the autosave file when that is newer."""
        store: LabelStore = LabelStore(segment_id, split=split, application_id=base_application_id(self.dataset, segment_id))
        autosave: Path = self.autosave_path(segment_id)
        committed: Path = self.layer_path(segment_id)
        if autosave.exists() and (not committed.exists() or autosave.stat().st_mtime > committed.stat().st_mtime):
            print(f"[labeler] {segment_id}: recovering labels from autosave {autosave}")
            contents: LayerContents = load_layer_file_contents(autosave)
        else:
            contents = load_layer_contents(self.dataset, segment_id)
        store.load_rows(contents.rows, contents.actions)
        return store

    def register(self, path: Path) -> None:
        """Register a committed layer file as ``hand_boxes`` with REPLACE and wait for it."""
        self.dataset.register([path.resolve().as_uri()], layer_name=LABEL_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
