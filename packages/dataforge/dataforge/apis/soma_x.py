"""Write segments' ``soma_x`` layers from their pose layers: no raw data, the pose layers carry the source parameters.

Run in a ``dataforge-soma-x`` environment (CUDA):

    pixi run -e dataforge-soma-x --frozen dataforge-soma-x --dataset epfl --tree /mnt/nas/datasets/dataforge-epfl-rrd \\
        --output-dir /mnt/nas/ssd-8tb/scratch/dataforge-soma-x/smoke/epfl --segment epfl__test__YH2003__2023_05_17_09_08_58 --max-seconds 30

Writes ``<output-dir>/soma_x/<segment>.rrd`` with the segment's recording id, so it stacks onto the segment's other layers.
Each dataset adapter turns its pose layer into windows of posed source meshes (``SourceWindow``); the shared driver fits
them (``dataforge.soma_x_fit``) and writes them (``dataforge.soma_x``) window by window, so a 30-minute session never
holds more than one window of source vertices.
"""

import fcntl
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
import torch
from dataforge import schema, soma_x
from dataforge.soma_x_fit import BODY_MAX_MEDIAN_ERROR_MM, HAND_MAX_MEDIAN_ERROR_MM, BodyFitter, HandFitter, ModelFit, Side, mano_model
from dataforge.writing import atomic_recording
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.mano.mano_np import MANOLayerNP

SOMA_X_LAYER: str = "soma_x"
"""Layer name of every dataset's SOMA-X fit (decision 14)."""
SIDES: tuple[Side, ...] = ("left", "right")
WINDOW_ROWS: int = 2048
"""Segment rows per window: bounds the source vertices in memory (2048 SMPL rows = 170 MB)."""
Dataset = Literal["hocap", "gsi-six-view", "genrobot", "gen-humanego", "hot3d-aria", "hot3d-quest3"]


@dataclass(frozen=True, slots=True)
class Config:
    dataset: Dataset
    """Source dataset: decides which pose-layer entities hold the source parameters and which model poses them."""
    tree: Path
    """The dataset's dataforge output tree (``<tree>/<layer>/<segment>.rrd``)."""
    output_dir: Path
    """Output tree; the layer goes to ``<output-dir>/soma_x/<segment>.rrd``."""
    segment: list[str] = field(default_factory=list)
    """Segment ids; empty means every segment of the tree's pose layer."""
    max_seconds: float | None = None
    """Keep only the first seconds of each segment (smoke tests); None keeps everything."""
    skip_existing: bool = True
    """Leave segments whose layer already exists (a resumable run); False refits them."""
    hand_bar_mm: float = HAND_MAX_MEDIAN_ERROR_MM
    """A hand row is trusted (gets a mesh) when its median fit error is at most this; the default is decision 11's bar."""
    body_bar_mm: float = BODY_MAX_MEDIAN_ERROR_MM
    """A body row is trusted (gets a mesh) when its median fit error is at most this; the default is decision 11's bar."""
    reverse: bool = False
    """Walk the segments last to first: a second machine then meets the first in the middle (``skip_existing`` skips the done ones)."""
    shard: tuple[int, int] = (0, 1)
    """(index, count): this run takes every count-th segment starting at index, so several runs split a dataset."""
    gpu_lock: Path | None = None
    """Lock file held exclusively while each segment fits (a shared GPU's timed runs slot in between segments); None takes no lock."""


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """Consecutive segment rows and the posed source meshes of their present rows."""

    times_ns: Int64[ndarray, "w"]
    """``video_time`` of each row."""
    frame_indices: Int64[ndarray, "w"]
    """Source frame index of each row."""
    present: Bool[ndarray, "w"]
    """Rows with a usable source fit."""
    vertices: Float32[ndarray, "k s 3"]
    """Posed source vertices of the present rows, world metres."""
    centres: Float32[ndarray, "k 3"]
    """Point each present row is fitted around (the MANO wrist; unused for bodies)."""
    betas: Float32[ndarray, "k c"]
    """Source shape of each present row: the SOMA-X identity."""
    temporal: dict[str, Float32[ndarray, "w c"]]
    """The source parameters of every row, shared vocabulary; NaN rows where the source has none."""


@dataclass(frozen=True, slots=True)
class Stream:
    """One body's or hand's source: where it goes, what it is, and its windows."""

    root: str
    """Model root the ``soma_x`` subtree goes under."""
    fitter: HandFitter | BodyFitter
    """The fitter of its model."""
    source: soma_x.SourceParameters
    """Source model, encodings and static fields (``temporal`` comes per window)."""
    albedo: soma_x.Rgba
    """Mesh colour."""
    windows: Callable[[], Iterator[SourceWindow]]
    """The windows, produced lazily."""


def fixed_rows(table: pa.Table, column: str, width: int) -> Float32[ndarray, "t c"]:
    """One list per row as floats (a fixed-size-list cell ``[[...]]`` or a flat AnyValues array ``[...]``); a missing row or value is NaN."""
    cells: list[list[object] | None] = table.column(column).to_pylist()
    values: list[list[object]] = [[] if not cell else (cell[0] if isinstance(cell[0], list) else cell) for cell in cells]
    return np.array(
        [[np.nan] * width if len(row) != width else [np.nan if value is None else value for value in row] for row in values], dtype=np.float32
    ).reshape(len(cells), width)


def scalar_rows(table: pa.Table, column: str) -> Float32[ndarray, "t"]:
    """One scalar per row (a ``Scalars`` cell is a flat list); a missing row is NaN."""
    cells: list[list[float | None] | None] = table.column(column).to_pylist()
    return np.array([np.nan if not cell or cell[0] is None else cell[0] for cell in cells], dtype=np.float32)


def read_rows(path: Path, contents: str | list[str], max_seconds: float | None) -> tuple[rrc.ChunkStore, pa.Table]:
    """A layer and the ``video_time``-sorted rows of one entity subtree, cut to ``max_seconds``."""
    store: rrc.ChunkStore = rrc.ChunkStore.from_chunks(list(rrc.RrdReader(path).stream()))
    table: pa.Table = store.reader(index=schema.TIMELINE, contents=contents).to_arrow_table().sort_by(schema.TIMELINE)
    if max_seconds is not None and table.num_rows:
        times: Int64[ndarray, "t"] = table.column(schema.TIMELINE).cast(pa.int64()).to_numpy()
        table = table.slice(0, int(np.searchsorted(times, times[0] + int(max_seconds * 1e9), side="right")))
    return store, table


def index_rows(table: pa.Table) -> tuple[Int64[ndarray, "t"], Int64[ndarray, "t"]]:
    """``video_time`` and ``frame_index`` of each row."""
    times_ns: Int64[ndarray, "t"] = table.column(schema.TIMELINE).cast(pa.int64()).to_numpy()
    frames: Int64[ndarray, "t"] = table.column(schema.FRAME_INDEX).to_numpy(zero_copy_only=False).astype(np.int64)
    return times_ns, frames


def fit_window(fitter: HandFitter | BodyFitter, window: SourceWindow) -> soma_x.SomaXFit:
    """Fit the present rows run by run of equal shape; a row is trusted when its fit meets the fitter's error bar."""
    count: int = len(window.times_ns)
    joints: int = len(fitter.topology.parents)
    rotations: Float32[ndarray, "w j 3"] = np.full((count, joints, 3), np.nan, dtype=np.float32)
    translation: Float32[ndarray, "w 3"] = np.full((count, 3), np.nan, dtype=np.float32)
    identity: Float32[ndarray, "w c"] = np.full((count, window.betas.shape[1]), np.nan, dtype=np.float32)
    error_mm: Float32[ndarray, "w"] = np.full(count, np.nan, dtype=np.float32)
    rows: Int64[ndarray, "k"] = np.flatnonzero(window.present)
    vertices: list[Float32[ndarray, "r v 3"]] = []
    joint_rows: list[Float32[ndarray, "r j 3"]] = []
    start: int = 0
    while start < len(rows):
        stop: int = start + 1
        while stop < len(rows) and np.array_equal(window.betas[stop], window.betas[start]):
            stop += 1
        part: slice = slice(start, stop)
        fitted: ModelFit = (
            fitter.fit(window.betas[start], window.vertices[part], window.centres[part])
            if isinstance(fitter, HandFitter)
            else fitter.fit(window.betas[start], window.vertices[part])
        )
        target: Int64[ndarray, "r"] = rows[part]
        rotations[target] = fitted.rotations
        translation[target] = fitted.root_translation
        identity[target] = fitted.identity_coeffs
        error_mm[target] = fitted.error_mm
        vertices.append(fitted.vertices)
        joint_rows.append(fitted.joints)
        start = stop
    trusted: Bool[ndarray, "w"] = np.isfinite(error_mm) & (error_mm <= fitter.max_median_error_mm)
    keep: Bool[ndarray, "k"] = trusted[rows]
    empty_vertices: Float32[ndarray, "0 v 3"] = np.zeros((0, int(fitter.topology.faces.max()) + 1, 3), dtype=np.float32)
    return soma_x.SomaXFit(
        times_ns=window.times_ns,
        frame_indices=window.frame_indices,
        trusted=trusted,
        vertices=(np.concatenate(vertices) if vertices else empty_vertices)[keep],
        joints=(np.concatenate(joint_rows) if joint_rows else np.zeros((0, joints, 3), dtype=np.float32))[keep],
        rotations=rotations,
        root_translation=translation,
        identity_coeffs=identity,
        fit_error_mm=error_mm,
    )


def empty_window(
    times_ns: Int64[ndarray, "w"],
    frame_indices: Int64[ndarray, "w"],
    present: Bool[ndarray, "w"],
    vertex_count: int,
    temporal: dict[str, Float32[ndarray, "t c"]],
    rows: slice,
) -> SourceWindow:
    """A window whose rows all lack a source fit: no mesh to pose, every row written untrusted."""
    return SourceWindow(
        times_ns,
        frame_indices,
        present,
        np.zeros((0, vertex_count, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        temporal["betas"][rows][present],
        {name: values[rows] for name, values in temporal.items()},
    )


def write_stream(recording: rr.RecordingStream, stream: Stream) -> Float32[ndarray, "n"]:
    """Fit and write one body or hand, window by window; return the fit errors of its fitted rows."""
    soma_x.log_soma_x_static(
        recording, stream.root, topology=stream.fitter.topology, model=stream.fitter.model(), source=stream.source, albedo=stream.albedo
    )
    errors: list[Float32[ndarray, "k"]] = []
    for window in stream.windows():
        fit: soma_x.SomaXFit = fit_window(stream.fitter, window)
        temporal: dict[str, Float32[ndarray, "w c"]] = window.temporal
        soma_x.log_soma_x_rows(
            recording, stream.root, fit=fit, topology=stream.fitter.topology, source_temporal=temporal, mesh_rows=soma_x.MESH_ROWS[stream.fitter.kind]
        )
        errors.append(fit.fit_error_mm[np.isfinite(fit.fit_error_mm)])
    return np.concatenate(errors) if errors else np.zeros(0, dtype=np.float32)


# ── HO-Cap ─────────────────────────────────────────────────────────────────


def hocap_streams(config: Config, segment: str, fitters: "Fitters") -> Iterator[Stream]:
    """HO-Cap: MANO PCA-45 with hands_mean, one shape per subject (``hocap_layers.write_hands``)."""
    store, table = read_rows(config.tree / "hand_pose" / f"{segment}.rrd", "/world/gt/hands/**", config.max_seconds)
    times_ns, frames = index_rows(table)
    for side in SIDES:
        path: str = schema.hand_mano_path(side)
        statics: dict[str, list[object]] = store.reader(index=None, contents=path).to_arrow_table().to_pylist()[0]
        if not statics.get(f"{path}:present", [False])[0]:
            continue
        betas: Float32[ndarray, "c"] = np.asarray(statics[f"{path}:betas"], dtype=np.float32)
        global_orient: Float32[ndarray, "t 3"] = fixed_rows(table, f"{path}:global_orient", 3)
        hand_pose: Float32[ndarray, "t 45"] = fixed_rows(table, f"{path}:pca_coefficients", 45)
        transl: Float32[ndarray, "t 3"] = fixed_rows(table, f"{path}:translation", 3)
        mano: MANOLayerNP = MANOLayerNP(side=side, betas=betas, use_pca=True)

        def windows(
            mano: MANOLayerNP = mano,
            betas: Float32[ndarray, "c"] = betas,
            go: Float32[ndarray, "t 3"] = global_orient,
            hp: Float32[ndarray, "t 45"] = hand_pose,
            tr: Float32[ndarray, "t 3"] = transl,
        ) -> Iterator[SourceWindow]:
            for start in range(0, len(times_ns), WINDOW_ROWS):
                rows: slice = slice(start, start + WINDOW_ROWS)
                present: Bool[ndarray, "w"] = np.isfinite(np.concatenate([go[rows], hp[rows], tr[rows]], axis=1)).all(axis=1)
                posed: tuple[Float32[ndarray, "k 778 3"], Float32[ndarray, "k 21 3"]] = mano(
                    np.concatenate([go[rows], hp[rows]], axis=1)[present], tr[rows][present]
                )
                yield SourceWindow(
                    times_ns[rows],
                    frames[rows],
                    present,
                    posed[0],
                    posed[1][:, 0],
                    np.repeat(betas[None], int(present.sum()), axis=0),
                    {"global_orient": go[rows], "hand_pose": hp[rows], "transl": tr[rows]},
                )

        yield Stream(
            schema.hands_path(side),
            fitters.hand(side),
            soma_x.SourceParameters(
                model="MANO",
                encoding={
                    "global_orient": "axis-angle, radians",
                    "hand_pose": "45 PCA coefficients of MANO's hands_components, hands_mean added",
                    "transl": "metres, added to the posed mesh",
                },
                static={"betas": betas},
                temporal={},
            ),
            soma_x.HAND_ALBEDO[side],
            windows,
        )


# ── MANO hands from the logged hand_mesh layer (GSI, DAS-Ego) ──────────────

IDENTITY_PER_SEGMENT: frozenset[str] = frozenset({"gsi-six-view"})
"""Families whose betas change on every frame (GSI: per-frame shape estimates, up to ~0.5 apart): one SOMA-X identity per hand
and segment, the median of its rows' betas, instead of one identity preparation per frame. The fit still targets each
frame's own source mesh. Measured on a 1,291-frame GSI segment: fit error 1.85 / 1.50 mm median (left / right) per-frame vs
1.86 / 1.52 median, worst frame 2.77 / 2.08 vs 2.81 / 2.23 mm, and 15 → 5,400 fps. The per-frame betas stay in ``source``."""

MESH_SOURCES: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {
    "gsi-six-view": (
        "MANO (vertices skinned by GSI; params in GSI's rig frame)",
        {
            "betas": ("betas", "10 shape coefficients, per row as shipped"),
            "global_orient": ("global_orient", "axis-angle, GSI rig frame"),
            "hand_pose": ("hand_pose", "as shipped by GSI"),
            "transl": ("transl", "metres, GSI rig frame"),
        },
    ),
    "das-ego": (
        "MANO full pose as rotation matrices (left shape fix, left fingers mirrored; das_ego.pose)",
        {
            "betas": ("betas", "10 shape coefficients, per row as shipped"),
            "global_orient": ("global_orient_rotmat", "3x3 rotation matrix, row-major"),
            "hand_pose": ("hand_pose_rotmat", "15 x 3x3 rotation matrices, row-major"),
            "transl": ("global_transl", "metres"),
        },
    ),
}
"""Datasets whose SOMA-X fit inverts the hand mesh they already log: the exact mesh shown today, with no re-posing convention to repeat."""


def width(table: pa.Table, column: str) -> int:
    """Length of a fixed-size list column's first non-empty cell."""
    for cell in table.column(column).to_pylist():
        if cell:
            return len(cell[0]) if isinstance(cell[0], list) else len(cell)
    return 0


def mesh_layer_streams(config: Config, segment: str, fitters: "Fitters") -> Iterator[Stream]:
    """MANO hands whose posed vertices the dataset's ``hand_mesh`` layer holds, betas from its pose layer."""
    family: str = "gsi-six-view" if config.dataset == "gsi-six-view" else "das-ego"
    model, fields = MESH_SOURCES[family]
    _, pose_table = read_rows(config.tree / POSE_LAYER[config.dataset] / f"{segment}.rrd", "/world/gt/hands/**", config.max_seconds)
    times_ns, frames = index_rows(pose_table)
    mesh_store, _ = read_rows(config.tree / "hand_mesh" / f"{segment}.rrd", "/world/gt/hands/**", None)
    for side in SIDES:
        path: str = schema.hand_mano_path(side)
        if f"{path}:betas" not in pose_table.schema.names:
            continue
        temporal: dict[str, Float32[ndarray, "t c"]] = {
            name: fixed_rows(pose_table, f"{path}:{column}", width(pose_table, f"{path}:{column}")) for name, (column, _) in fields.items()
        }
        mesh_table: pa.Table = (
            mesh_store.reader(index=schema.TIMELINE, contents=schema.hand_mesh_path(side)).to_arrow_table().sort_by(schema.TIMELINE)
        )
        column: str = f"{schema.hand_mesh_path(side)}:Mesh3D:vertex_positions"
        if column not in mesh_table.schema.names:
            continue
        mesh_times: Int64[ndarray, "m"] = mesh_table.column(schema.TIMELINE).cast(pa.int64()).to_numpy()
        cells: list[list[list[float]] | None] = mesh_table.column(column).to_pylist()
        at: Int64[ndarray, "t"] = np.clip(np.searchsorted(mesh_times, times_ns), 0, max(len(mesh_times) - 1, 0))
        same: Bool[ndarray, "t"] = (mesh_times[at] == times_ns) if len(mesh_times) else np.zeros(len(times_ns), dtype=bool)
        has_mesh: Bool[ndarray, "t"] = same & np.array([bool(cells[i]) and len(cells[i]) == 778 for i in at], dtype=bool)
        present: Bool[ndarray, "t"] = has_mesh & np.isfinite(temporal["betas"]).all(axis=1)
        identity: Float32[ndarray, "t c"] = temporal["betas"]
        if family in IDENTITY_PER_SEGMENT and present.any():
            identity = np.broadcast_to(np.median(temporal["betas"][present], axis=0), temporal["betas"].shape).astype(np.float32)

        def windows(
            present: Bool[ndarray, "t"] = present,
            at: Int64[ndarray, "t"] = at,
            cells: list[list[list[float]] | None] = cells,
            temporal: dict[str, Float32[ndarray, "t c"]] = temporal,
            identity: Float32[ndarray, "t c"] = identity,
        ) -> Iterator[SourceWindow]:
            for start in range(0, len(times_ns), WINDOW_ROWS):
                rows: slice = slice(start, start + WINDOW_ROWS)
                keep: Bool[ndarray, "w"] = present[rows]
                vertices: Float32[ndarray, "k 778 3"] = np.array([cells[i] for i in at[rows][keep]], dtype=np.float32).reshape(-1, 778, 3)
                yield SourceWindow(
                    times_ns[rows],
                    frames[rows],
                    keep,
                    vertices,
                    vertices.mean(axis=1),
                    identity[rows][keep],
                    {name: values[rows] for name, values in temporal.items()},
                )

        yield Stream(
            schema.hands_path(side),
            fitters.hand(side, fix_left_shapedirs=family == "das-ego"),
            soma_x.SourceParameters(model=model, encoding={name: text for name, (_, text) in fields.items()}, static={}, temporal={}),
            soma_x.HAND_ALBEDO[side],
            windows,
        )


# ── HOT3D (shipped MANO, 15 PCA) ───────────────────────────────────────────


def hot3d_streams(config: Config, segment: str, fitters: "Fitters") -> Iterator[Stream]:
    """HOT3D: the shipped MANO (15 PCA coefficients, mean hand added, world wrist pose as global orient + transl, left shape fix).

    dataforge never drew this MANO; the convention was measured against the UmeTrack mesh HOT3D's hand_mesh layer logs
    (Quest3 P0002: median surface distance 3.8 mm left, 4.0 mm right; the other conventions were 15-90 mm off).
    """
    import smplx
    from dataforge.soma_x_fit import mano_model

    # Only the MANO entities: the hands subtree also holds pose tracks pyarrow cannot sort.
    _, table = read_rows(config.tree / "hand_pose" / f"{segment}.rrd", [schema.hand_mano_path(side) for side in SIDES], config.max_seconds)
    times_ns, frames = index_rows(table)
    for side in SIDES:
        path: str = schema.hand_mano_path(side)
        if f"{path}:pca_coefficients" not in table.schema.names:
            continue
        temporal: dict[str, Float32[ndarray, "t c"]] = {
            "betas": fixed_rows(table, f"{path}:betas", 10),
            "global_orient": fixed_rows(table, f"{path}:wrist_quaternion_wxyz", 4),
            "hand_pose": fixed_rows(table, f"{path}:pca_coefficients", 15),
            "transl": fixed_rows(table, f"{path}:wrist_translation", 3),
        }
        present: Bool[ndarray, "t"] = np.isfinite(np.concatenate(list(temporal.values()), axis=1)).all(axis=1)

        def windows(
            side: Side = side, temporal: dict[str, Float32[ndarray, "t c"]] = temporal, present: Bool[ndarray, "t"] = present
        ) -> Iterator[SourceWindow]:
            layer = smplx.MANO(str(mano_model(side)), is_rhand=side == "right", use_pca=True, num_pca_comps=15, flat_hand_mean=False).to("cuda")
            if side == "left":
                cast(torch.Tensor, layer.shapedirs)[:, 0, :] *= -1.0
            for start in range(0, len(times_ns), WINDOW_ROWS):
                rows: slice = slice(start, start + WINDOW_ROWS)
                keep: Bool[ndarray, "w"] = present[rows]
                if not keep.any():  # smplx cannot pose an empty batch
                    yield empty_window(times_ns[rows], frames[rows], keep, int(cast(torch.Tensor, layer.v_template).shape[0]), temporal, rows)
                    continue
                quaternions: Float32[ndarray, "k 4"] = temporal["global_orient"][rows][keep]
                orient: Float32[ndarray, "k 3"] = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]].astype(np.float64)).as_rotvec().astype(np.float32)

                def cuda(values: Float32[ndarray, "k c"]) -> torch.Tensor:
                    return torch.from_numpy(np.ascontiguousarray(values)).cuda()

                with torch.no_grad():
                    output = layer(
                        betas=cuda(temporal["betas"][rows][keep]),
                        global_orient=cuda(orient),
                        hand_pose=cuda(temporal["hand_pose"][rows][keep]),
                        transl=cuda(temporal["transl"][rows][keep]),
                    )
                yield SourceWindow(
                    times_ns[rows],
                    frames[rows],
                    keep,
                    output.vertices.float().cpu().numpy(),
                    output.joints[:, 0].float().cpu().numpy(),
                    temporal["betas"][rows][keep],
                    {name: values[rows] for name, values in temporal.items()},
                )

        yield Stream(
            schema.hands_path(side),
            fitters.hand(side, fix_left_shapedirs=True),
            soma_x.SourceParameters(
                model="MANO, HOT3D convention (15 PCA, mean hand added, left shape fix)",
                encoding={
                    "betas": "10 shape coefficients, per row as shipped",
                    "global_orient": "unit quaternion wxyz of the world wrist pose (MANO global orient)",
                    "hand_pose": "15 PCA coefficients of MANO's hands_components, hands_mean added",
                    "transl": "metres, world wrist translation (MANO transl)",
                },
                static={},
                temporal={},
            ),
            soma_x.HAND_ALBEDO[side],
            windows,
        )


# ── driver ─────────────────────────────────────────────────────────────────


class Fitters:
    """One fitter per model, built on first use and kept for every segment."""

    def __init__(self, *, hand_bar_mm: float, body_bar_mm: float) -> None:
        self.hand_bar_mm: float = hand_bar_mm
        self.body_bar_mm: float = body_bar_mm
        self.hands: dict[tuple[Side, bool], HandFitter] = {}
        self.bodies: dict[tuple[str, Path], BodyFitter] = {}

    def hand(self, side: Side, *, fix_left_shapedirs: bool = False) -> HandFitter:
        key: tuple[Side, bool] = (side, fix_left_shapedirs)
        if key not in self.hands:
            mano_model(side)
            self.hands[key] = HandFitter(side, fix_left_shapedirs=fix_left_shapedirs)
            self.hands[key].max_median_error_mm = self.hand_bar_mm
        return self.hands[key]

    def body(self, backend: Literal["smpl", "smplh", "smplx"], model_path: Path, *, gender: str = "neutral", num_betas: int = 10) -> BodyFitter:
        if (backend, model_path) not in self.bodies:
            self.bodies[backend, model_path] = BodyFitter(backend, model_path=model_path, gender=gender, num_betas=num_betas)
            self.bodies[backend, model_path].max_median_error_mm = self.body_bar_mm
        return self.bodies[backend, model_path]


STREAMS: dict[Dataset, Callable[[Config, str, Fitters], Iterator[Stream]]] = {
    "hocap": hocap_streams,
    "gsi-six-view": mesh_layer_streams,
    "genrobot": mesh_layer_streams,
    "gen-humanego": mesh_layer_streams,
    "hot3d-aria": hot3d_streams,
    "hot3d-quest3": hot3d_streams,
}
POSE_LAYER: dict[Dataset, str] = {
    "hocap": "hand_pose",
    "gsi-six-view": "pose",
    "genrobot": "pose",
    "gen-humanego": "pose",
    "hot3d-aria": "hand_pose",
    "hot3d-quest3": "hand_pose",
}
"""Layer whose segment files list the dataset's segments."""


def write_segment(config: Config, fitters: Fitters, segment: str) -> dict[str, Float32[ndarray, "n"]]:
    """Fit and write one segment's ``soma_x`` layer; return each stream's fit errors by model root."""
    target: Path = config.output_dir / SOMA_X_LAYER / f"{segment}.rrd"
    target.parent.mkdir(parents=True, exist_ok=True)
    errors: dict[str, Float32[ndarray, "n"]] = {}
    with atomic_recording(target, recording_id=segment, send_properties=False) as recording:
        for stream in STREAMS[config.dataset](config, segment, fitters):
            errors[stream.root] = write_stream(recording, stream)
    target.chmod(0o644)
    return errors


class _FileLock(AbstractContextManager[None]):
    """An exclusive ``flock`` on a file for the length of a ``with`` block."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def __enter__(self) -> None:
        self.handle = self.path.open("a")
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def gpu_lock(path: Path | None) -> AbstractContextManager[None]:
    """Hold ``path`` exclusively, or nothing."""
    return nullcontext() if path is None else _FileLock(path)


def main(config: Config) -> None:
    """Write every requested segment and print each stream's error percentiles."""
    segments: list[str] = (config.segment or sorted(path.stem for path in (config.tree / POSE_LAYER[config.dataset]).glob("*.rrd")))[
        config.shard[0] :: config.shard[1]
    ]
    if config.reverse:
        segments = segments[::-1]
    fitters: Fitters = Fitters(hand_bar_mm=config.hand_bar_mm, body_bar_mm=config.body_bar_mm)
    for number, segment in enumerate(segments, start=1):
        if config.skip_existing and (config.output_dir / SOMA_X_LAYER / f"{segment}.rrd").is_file():
            continue
        with gpu_lock(config.gpu_lock):
            errors: dict[str, Float32[ndarray, "n"]] = write_segment(config, fitters, segment)
            torch.cuda.empty_cache()  # hand the cached blocks back to the GPU's other users between segments
        summary: str = "; ".join(
            f"{root}: {len(values)} rows, median {np.median(values):.2f} mm, p95 {np.percentile(values, 95):.2f}, max {values.max():.2f}"
            for root, values in errors.items()
            if len(values)
        )
        print(f"[{number}/{len(segments)}] {segment}: {summary or 'no source rows to fit (layer written, every row untrusted)'}", flush=True)
