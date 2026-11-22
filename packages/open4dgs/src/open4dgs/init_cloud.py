"""The trainer's init contract (``InitCloud``) and its ``.rrd`` record.

An init cloud is the set of 4D points TRiGS starts from (``trigs.init.from_colmap.build_canonical_from_npz``): rows
grouped by keyframe, each with a world position, a velocity in world units per VIDEO frame, a colour and a normalised
time. Body-surface init and the upstream RoMa v2 + MEMFOF init both produce one.

Record layout (one entity, default ``/world/init_cloud``): at each keyframe on ``frame_index`` (the segment's frame
index), ``Points3D`` (positions, uint8 colours for the viewer) plus the ``InitCloud`` archetype with the float columns
that the trainer reads (``colors``, ``velocities``, ``times``, ``frames`` and the optional per-row columns). The
scalars and the provenance (dataset, segment, method) are the recording property ``init_cloud``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Int32
from numpy import ndarray

INIT_CLOUD_ENTITY: str = "/world/init_cloud"
"""Default entity path of the init cloud in a recording."""
FRAME_TIMELINE: str = "frame_index"
"""The segment's frame timeline."""
_ARCHETYPE: str = "InitCloud"
_PROPERTY: str = "init_cloud"
_OPTIONAL_FLAGS: tuple[str, ...] = ("has_velocity", "is_static", "dynamic_protect", "dynamic_allowed", "stage_keep")


@dataclass(frozen=True, slots=True)
class InitCloud:
    """Init cloud for TRiGS, rows grouped by keyframe in ascending ``frames``.

    Frames are LOGICAL trainer frames: 0 is the first frame of the training window, ``frame_end`` its last.
    """

    positions: Float32[ndarray, "n 3"]
    """World positions."""
    velocities: Float32[ndarray, "n 3"]
    """World units per VIDEO frame (``velocity_units='world_per_video_frame'``)."""
    colors: Float32[ndarray, "n 3"]
    """RGB in [0, 1]."""
    times: Float32[ndarray, "n 1"]
    """Row time normalised over the window: ``frames / (frame_end - frame_start)``."""
    frames: Int32[ndarray, " n"]
    """Logical trainer frame (keyframe) of each row."""
    frame_start: int
    """First logical frame of the window (0)."""
    frame_end: int
    """Last logical frame of the window (num_frames - 1)."""
    keyframe_step: int
    """Frames between keyframes."""
    fps: float
    """Video frame rate."""
    source_frame_start: int
    """Segment ``frame_index`` of logical frame 0 (the window start in the segment)."""
    dataset: str
    """Catalog dataset of the source segment."""
    segment_id: str
    """Source segment id."""
    method: str
    """How the cloud was made, e.g. ``body_surface`` or ``roma_v2_memfof``."""
    durations: Float32[ndarray, "n 1"] | None = None
    """Optional temporal extent per row, normalised like ``times``."""
    has_velocity: Bool[ndarray, " n"] | None = None
    is_static: Bool[ndarray, " n"] | None = None
    dynamic_protect: Bool[ndarray, " n"] | None = None
    dynamic_allowed: Bool[ndarray, " n"] | None = None
    stage_keep: Bool[ndarray, " n"] | None = None

    def __post_init__(self) -> None:
        if self.frame_end <= self.frame_start:
            raise ValueError(f"frame_end={self.frame_end} must exceed frame_start={self.frame_start}")
        if self.frames.size and np.any(np.diff(self.frames) < 0):
            raise ValueError("rows must be grouped by keyframe in ascending frame order")

    @property
    def num_rows(self) -> int:
        """Number of rows (points over all keyframes)."""
        return int(self.positions.shape[0])

    @property
    def keyframes(self) -> Int32[ndarray, " k"]:
        """The distinct logical keyframes, ascending."""
        return np.unique(self.frames)

    def to_trigs_mapping(self) -> dict[str, ndarray]:
        """The arrays ``build_canonical_from_npz`` accepts in place of a keyframes NPZ (recipe patch 0002).

        No ``source_frame_start`` key: that would switch the trainer to its macro-NPZ window filter (Mode B).
        """
        mapping: dict[str, ndarray] = {
            "positions": self.positions,
            "velocities": self.velocities,
            "colors": self.colors,
            "times": self.times,
            "frame_start": np.asarray(self.frame_start),
            "frame_end": np.asarray(self.frame_end),
            "keyframe_step": np.asarray(self.keyframe_step),
            "fps": np.asarray(self.fps),
            "velocity_units": np.asarray("world_per_video_frame"),
            "time_units": np.asarray("normalized_0_1_over_source_frame_range"),
        }
        if self.durations is not None:
            mapping["durations"] = self.durations
        for name in _OPTIONAL_FLAGS:
            value: Bool[ndarray, " n"] | None = getattr(self, name)
            if value is not None:
                mapping[name] = value
        return mapping


def _vec3(values: Float32[ndarray, "n 3"]) -> pa.FixedSizeListArray:
    """An (n, 3) float32 array as an Arrow fixed-size list column (one cell per row)."""
    return pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(values).reshape(-1), type=pa.float32()), 3)


def log_init_cloud(cloud: InitCloud, recording: rr.RecordingStream, entity: str = INIT_CLOUD_ENTITY) -> None:
    """Log ``cloud`` as one ``Points3D`` + ``InitCloud`` row per keyframe on ``frame_index``, scalars as properties."""
    bounds: Int32[ndarray, " k1"] = np.searchsorted(cloud.frames, np.append(cloud.keyframes, cloud.frames.max() + 1)).astype(np.int32)
    for keyframe, start, end in zip(cloud.keyframes.tolist(), bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
        rows: slice = slice(start, end)
        components: dict[str, ndarray | pa.Array] = {
            "colors": _vec3(cloud.colors[rows]),
            "velocities": _vec3(cloud.velocities[rows]),
            "times": cloud.times[rows, 0],
            "frames": cloud.frames[rows],
        }
        if cloud.durations is not None:
            components["durations"] = cloud.durations[rows, 0]
        for name in _OPTIONAL_FLAGS:
            value: Bool[ndarray, " n"] | None = getattr(cloud, name)
            if value is not None:
                components[name] = value[rows]
        recording.set_time(FRAME_TIMELINE, sequence=cloud.source_frame_start + keyframe)
        recording.log(
            entity,
            rr.Points3D(cloud.positions[rows], colors=(np.clip(cloud.colors[rows], 0.0, 1.0) * 255.0).round().astype(np.uint8)),
            rr.DynamicArchetype(_ARCHETYPE, components=components),
        )
    recording.send_property(
        _PROPERTY,
        rr.AnyValues(
            entity=entity,
            frame_start=cloud.frame_start,
            frame_end=cloud.frame_end,
            keyframe_step=cloud.keyframe_step,
            fps=cloud.fps,
            source_frame_start=cloud.source_frame_start,
            dataset=cloud.dataset,
            segment_id=cloud.segment_id,
            method=cloud.method,
            num_rows=cloud.num_rows,
        ),
    )


def _concat(table_column: list[list[object] | None], dtype: type, width: int | None) -> ndarray:
    """Concatenate the per-keyframe list cells of one column (null cells are frames without rows)."""
    rows: list[object] = [item for cell in table_column if cell is not None for item in cell]
    array: ndarray = np.asarray(rows, dtype=dtype)
    return array.reshape(-1, width) if width is not None else array.reshape(-1)


def read_init_cloud(rrd_path: Path, entity: str = INIT_CLOUD_ENTITY) -> InitCloud:
    """Read the init cloud that ``log_init_cloud`` wrote to ``rrd_path`` (the arrays come back bit-exact)."""
    with rr.server.Server(datasets={"init": [str(rrd_path)]}) as server:
        dataset = server.client().get_dataset("init")
        segments: Mapping[str, list[object]] = dataset.segment_table().to_arrow_table().to_pydict()
        table = dataset.reader(index=FRAME_TIMELINE).to_arrow_table().sort_by(FRAME_TIMELINE)
    # Property cells come back as one list per segment holding one value: [[value]].
    prop: dict[str, object] = {
        name.removeprefix(f"property:{_PROPERTY}:"): np.asarray(cells[0], dtype=object).reshape(-1)[0]
        for name, cells in segments.items()
        if name.startswith(f"property:{_PROPERTY}:")
    }
    if prop.get("entity") != entity:
        raise ValueError(f"{rrd_path} holds no init cloud at {entity} (property init_cloud: {prop})")

    def column(component: str, dtype: type, width: int | None = None, archetype: str = _ARCHETYPE) -> ndarray | None:
        name: str = f"{entity}:{archetype}:{component}"
        if name not in table.column_names:
            return None
        return _concat(table.column(name).to_pylist(), dtype, width)

    def required(component: str, dtype: type, width: int | None = None, archetype: str = _ARCHETYPE) -> ndarray:
        value: ndarray | None = column(component, dtype, width, archetype)
        if value is None:
            raise ValueError(f"{rrd_path}: init cloud column {component} is missing")
        return value

    durations: ndarray | None = column("durations", np.float32)
    flags: dict[str, ndarray | None] = {name: column(name, np.bool_) for name in _OPTIONAL_FLAGS}
    return InitCloud(
        positions=required("positions", np.float32, 3, "Points3D"),
        velocities=required("velocities", np.float32, 3),
        colors=required("colors", np.float32, 3),
        times=required("times", np.float32).reshape(-1, 1),
        frames=required("frames", np.int32),
        frame_start=int(prop["frame_start"]),  # type: ignore[arg-type]
        frame_end=int(prop["frame_end"]),  # type: ignore[arg-type]
        keyframe_step=int(prop["keyframe_step"]),  # type: ignore[arg-type]
        fps=float(prop["fps"]),  # type: ignore[arg-type]
        source_frame_start=int(prop["source_frame_start"]),  # type: ignore[arg-type]
        dataset=str(prop["dataset"]),
        segment_id=str(prop["segment_id"]),
        method=str(prop["method"]),
        durations=durations.reshape(-1, 1) if durations is not None else None,
        **flags,
    )


def read_layer_init_cloud(catalog_url: str, dataset: str, segment_id: str, layer: str) -> tuple[InitCloud, tuple[str, ...]]:
    """The init cloud of ``segment_id``'s catalog layer ``layer`` and the held-out cameras it was built without.

    The held-out cameras are the layer's ``<layer>:holdout`` property (``open4dgs-init-roma`` writes it).

    Raises:
        ValueError: When the segment has no such layer, the layer is not a local file, or it records no held-out cameras.
    """
    from urllib.parse import unquote, urlparse

    from rerun.catalog import CatalogClient

    client: CatalogClient = CatalogClient(catalog_url)  # keep the client alive while the table is read (DataFusion FFI)
    table = client.get_dataset(dataset).segment_table().to_arrow_table()
    rows: list[dict[str, object]] = [row for row in table.to_pylist() if row["rerun_segment_id"] == segment_id]
    if not rows:
        raise ValueError(f"{dataset}: no segment {segment_id}")
    names: list[str] = list(rows[0]["rerun_layer_names"])  # type: ignore[arg-type]
    if layer not in names:
        raise ValueError(f"{dataset}/{segment_id}: no layer {layer!r} (layers {names})")
    url = urlparse(list(rows[0]["rerun_storage_urls"])[names.index(layer)])  # type: ignore[arg-type]
    if url.scheme != "file":
        raise ValueError(f"{dataset}/{segment_id}: layer {layer!r} is not a local file ({url.geturl()})")
    holdout = rows[0].get(f"property:{layer}:holdout")
    if not holdout:
        raise ValueError(f"{dataset}/{segment_id}: layer {layer!r} records no held-out cameras (property {layer}:holdout)")
    return read_init_cloud(Path(unquote(url.path))), tuple(str(name) for name in holdout)  # type: ignore[union-attr]


def check_init_matches_run(
    init: InitCloud, init_holdout: tuple[str, ...], holdout_names: tuple[str, ...], frame_start: int, num_frames: int
) -> None:
    """Refuse an init built for another split or window.

    Raises:
        ValueError: When the init's held-out cameras differ from the run's (as sets), or its window
            (``source_frame_start``, ``frame_end``) differs from the run's.
    """
    if set(init_holdout) != set(holdout_names):
        raise ValueError(
            f"the init held out {sorted(init_holdout)} but the run holds out {sorted(holdout_names)}: "
            "an init that saw a held-out camera would leak it; pass --holdout-cameras " + " ".join(init_holdout)
        )
    if init.source_frame_start != frame_start or init.frame_end != num_frames - 1:
        raise ValueError(
            f"the init covers frames {init.source_frame_start}..{init.source_frame_start + init.frame_end}, the run "
            f"{frame_start}..{frame_start + num_frames - 1}: use --frame-start {init.source_frame_start} --num-frames {init.frame_end + 1}"
        )
