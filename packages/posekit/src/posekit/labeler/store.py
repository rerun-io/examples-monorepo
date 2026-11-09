"""Reviewed hand labels of one segment: held in memory, written to and read back from the ``hand_boxes`` layer.

The layer is the only place labels persist. ``save_layer`` writes one ``.rrd`` whose recording id is the
segment id, so the catalog stacks it on the segment's other layers; ``CatalogLayers.register`` registers it
with REPLACE; ``load_layer`` rebuilds a store from the registered layer, so a server restart keeps the work.

Layout per label (``video_time``, a duration timeline in ns), at ``paths.label_path(cam, side)``:

- ``Boxes2D`` with the box (stored by Rerun as centers + half sizes), class id 100/101 and a text label;
  ``no_hand``/``rejected`` log an empty ``Boxes2D``, which clears any earlier box at that entity.
- ``AnyValues`` ``status``, ``source``, ``reviewer`` on the same entity and time.

The recording property ``hand_boxes:split`` says whether the segment is train or eval.

Autosave writes the same layout to a second file (``autosave_path``) that is never registered: rewriting
a file the catalog has registered, in place, could hand the catalog bytes that disagree with its manifest.
``CatalogLayers.open_store`` recovers from the autosave file when it is newer than the committed layer.
"""

from __future__ import annotations

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
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader, StoreEntry
from serde import from_dict

from posekit.labeler.paths import class_id, label_path
from posekit.labeler.schema import HAND_SIDES, LABEL_LAYER, FrameLabels, HandSide, SegmentSplit, ViewLabel

TIMELINE: str = "video_time"
"""Timeline the labels sit on (a duration timeline, ns)."""
LAYER_DIR: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-labeler/layers") / LABEL_LAYER
"""Where committed ``hand_boxes`` layer files go, one ``<segment>.rrd`` each."""
N_CAMS: int = 6
"""Cameras on a RoboCap rig; the layer reader asks for these entity paths."""


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One stored label with its time, the unit the layer writer and reader exchange."""

    t_ns: int
    """``video_time``, ns."""
    label: ViewLabel
    """The decision."""


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
        """Labels put since the last ``save_layer``."""
        with self._lock:
            return self._unsaved

    def save_layer(self, path: Path) -> int:
        """Write the ``hand_boxes`` layer to ``path`` atomically (temp file + rename, mode 0644).

        Returns:
            Number of labels written.
        """
        with self._lock:
            unsaved_before: int = self._unsaved
        rows: list[LabelRow] = self.rows()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".rrd.tmp", dir=path.parent)
        os.close(handle)
        temp_path: Path = Path(temp_name)
        try:
            write_layer_rrd(temp_path, rows, segment_id=self.segment_id, split=self.split, application_id=self.application_id)
            temp_path.chmod(0o644)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        with self._lock:
            self._unsaved -= unsaved_before
        return len(rows)

    def load_rows(self, rows: list[LabelRow]) -> None:
        """Replace the store's contents with ``rows`` (used when rebuilding from a layer)."""
        with self._lock:
            self._labels = {}
            for row in rows:
                self._labels.setdefault(row.t_ns, {})[(row.label.cam, row.label.side)] = row.label
            self._unsaved = 0


def write_layer_rrd(path: Path, rows: list[LabelRow], *, segment_id: str, split: SegmentSplit, application_id: str) -> None:
    """Write ``rows`` as a ``hand_boxes`` layer recording to exactly ``path`` (no atomicity here)."""
    # send_properties=False: a layer must not restate the base layer's RecordingInfo (name, start time).
    with rr.RecordingStream(application_id, recording_id=segment_id, send_properties=False) as recording:
        recording.save(path)
        recording.send_property(LABEL_LAYER, rr.AnyValues(split=split))
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


def load_layer(dataset: DatasetEntry, segment_id: str, *, n_cams: int = N_CAMS) -> list[LabelRow]:
    """Read the registered ``hand_boxes`` labels of one segment from the catalog (one round trip)."""
    entities: list[str] = label_entities(n_cams)
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents(entities).reader(index=TIMELINE).to_arrow_table()
    return rows_from_table(table, n_cams)


def load_layer_file(path: Path, *, n_cams: int = N_CAMS) -> list[LabelRow]:
    """Read labels from an unregistered layer file through a private in-process catalog server.

    The same decoder as ``load_layer`` runs on it, so the file and the catalog cannot disagree on layout.
    """
    store_entry: StoreEntry = RrdReader(path).recordings()[0]
    with rr.server.Server(host="127.0.0.1", port=0) as server:
        client: CatalogClient = server.client()
        dataset: DatasetEntry = client.create_dataset("hand_boxes_file")
        dataset.register([path.resolve().as_uri()], layer_name=LABEL_LAYER).wait()
        return load_layer(dataset, store_entry.recording_id, n_cams=n_cams)


def base_application_id(dataset: DatasetEntry, segment_id: str) -> str:
    """Application id of the segment's ``base`` layer file, read from its rrd header."""
    table: pa.Table = dataset.segment_table().to_arrow_table()
    for row in table.select(["rerun_segment_id", "rerun_layer_names", "rerun_storage_urls"]).to_pylist():
        if row["rerun_segment_id"] != segment_id:
            continue
        for layer, url in zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True):
            if layer == "base":
                if not url.startswith("file://"):
                    raise ValueError(f"base layer of {segment_id} is not a local file: {url}")
                return RrdReader(Path(url.removeprefix("file://"))).recordings()[0].application_id
    raise ValueError(f"segment {segment_id!r} has no base layer in dataset {dataset.name!r}")


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
            store.load_rows(load_layer_file(autosave))
        else:
            store.load_rows(load_layer(self.dataset, segment_id))
        return store

    def register(self, path: Path) -> None:
        """Register a committed layer file as ``hand_boxes`` with REPLACE and wait for it."""
        self.dataset.register([path.resolve().as_uri()], layer_name=LABEL_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
