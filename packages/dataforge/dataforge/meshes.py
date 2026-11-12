"""Model-independent temporal mesh output; skinning stays in source adapters.

Two layouts. The SOMA-X layers (``log_mesh_topology`` + ``log_mesh_frames``, the EgoHand layout) log the triangle
list and albedo once, statically, and per row either the vertices or a non-recursive ``Clear``. On Rerun 0.42 the
viewer did not warn on that layout in 48 catalog loads of EPFL YH2037, while the per-chunk layout below warned
("num_positions % 3", vertices without their triangle list) in 2 of 15 loads of the same segment (2026-11-12).

Per-chunk convention (``log_mesh_batch``, the older layers): the triangle list rides on the vertex chunks and is never
logged statically. A viewer that streams chunks from a catalog can receive a vertex chunk before any
other; without indices in that chunk it reads the vertices as a triangle soup (778 MANO vertices are not a
multiple of 3) and warns. A static triangle list does not help: static data wins every latest-at query,
so while the static chunk is known but not yet loaded it hides the per-chunk copy. Log only fields that
never change per frame (albedo, ``log_mesh_static``) statically.

Each ``log_mesh_batch`` call is one chunk in the rrd, and the catalog serves it whole as long as two
conditions hold (verified against Rerun 0.38.1):

- the rrd has a footer (``writing.recording_to`` writes one), so the server reads chunks lazily by byte span
  instead of re-inserting them into an eager store that splits large chunks into row pieces;
- no pass rewrites or rechunks the files between conversion and registration.

The viewer itself splits a large received chunk into row pieces, but it inserts all pieces at once, so the
later pieces still find the triangle list of the first by latest-at. Rerun loading chunks together with
their topology would make this convention unnecessary.
"""

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Int64, UInt32
from numpy import ndarray
from rerun.datatypes import Rgba32Like

from dataforge.logging_toolkit import frame_index_column, time_column

BODY_MESH_STRIDE: int = 3
"""body_mesh keeps every third 30 Hz frame (10 Hz): a display layer, by decision (2026-09-25).

Full-rate SMPL vertices cost 82 KB per frame (4.3 GB for a 29-min EPFL session, 4x its videos), and
Rerun 0.38 has no mesh skinning to pose one logged mesh from joint transforms. The body parameters
and keypoints stay at full rate."""
BODY_MESH_BATCH_ROWS: int = 32
"""Body mesh rows per chunk (``log_mesh_rows``): an SMPL row is 82 KB, an SMPL-X row 126 KB, so ~4 MB plus the triangle list."""


class SparseComponentColumn(rr.ComponentColumn):
    """One component batch on one row of a column, null on every other row.

    ``rr.ComponentColumn`` builds its list array without a validity mask, so a zero-length row is an empty
    list, which latest-at returns as a value. A null row is skipped instead.
    """

    def __init__(self, column: rr.ComponentColumn, *, row: int, num_rows: int) -> None:
        count: int = len(column.component_batch.as_arrow_array())
        super().__init__(column.descriptor, column.component_batch, lengths=[count if index == row else 0 for index in range(num_rows)])
        self.row: int = row
        self.num_rows: int = num_rows

    def as_arrow_array(self) -> pa.Array:
        """The laid-out column with every row but ``row`` null."""
        dense: pa.ListArray = super().as_arrow_array()
        return pa.ListArray.from_arrays(dense.offsets, dense.values, mask=pa.array([index != self.row for index in range(self.num_rows)]))

    def partition(self, lengths: object) -> rr.ComponentColumn:
        """Refuse: the inherited version rebuilds a dense column and turns the null rows into empty lists."""
        raise TypeError("SparseComponentColumn is laid out once; repartitioning would drop its null rows")


class MaskedComponentColumn(rr.ComponentColumn):
    """A component column with a null on every row that ``present`` marks absent, so latest-at skips that row."""

    def __init__(self, column: rr.ComponentColumn, *, lengths: list[int], present: list[bool]) -> None:
        super().__init__(column.descriptor, column.component_batch, lengths=lengths)
        self.present: list[bool] = present

    def as_arrow_array(self) -> pa.Array:
        """The laid-out column with every absent row null."""
        dense: pa.ListArray = super().as_arrow_array()
        return pa.ListArray.from_arrays(dense.offsets, dense.values, mask=pa.array([not ok for ok in self.present]))

    def partition(self, lengths: object) -> rr.ComponentColumn:
        """Refuse: the inherited version rebuilds a dense column and turns the null rows into empty lists."""
        raise TypeError("MaskedComponentColumn is laid out once; repartitioning would drop its null rows")


def log_mesh_topology(recording: rr.RecordingStream, path: str, *, topology: UInt32[ndarray, "f 3"], albedo_factor: Rgba32Like) -> None:
    """Log a per-frame mesh's triangle list and albedo once, statically (EgoHand's SOMA layers); ``log_mesh_frames`` adds the vertices."""
    rr.log(path, rr.Mesh3D.from_fields(triangle_indices=topology, albedo_factor=albedo_factor), static=True, recording=recording)


def log_mesh_frames(
    recording: rr.RecordingStream,
    path: str,
    *,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    vertices: Float32[ndarray, "k v 3"],
    trusted: list[bool],
) -> None:
    """Write one chunk of a static-topology mesh: vertices on trusted rows, a non-recursive ``Clear`` on the others.

    The Clear hides the mesh the way posekit's EgoHand hides a lost hand: latest-at finds no vertices, so the viewer
    draws nothing and does not warn (an empty vertex list against the static triangle list would). The other column
    is null on each row, so every timeline row stays in the dataframe.

    Args:
        recording: Destination stream; the caller logs ``log_mesh_topology`` once.
        path: Mesh entity.
        times_ns: Int64[ndarray, "t"] times, including untrusted rows.
        frame_indices: Int64[ndarray, "t"] source indices.
        vertices: Float32[ndarray, "k v 3"] for trusted rows only, in metres.
        trusted: One flag per timeline row; sum must equal k.
    """
    if sum(trusted) != len(vertices):
        raise ValueError("vertices must contain exactly the trusted rows")
    untrusted: list[bool] = [not ok for ok in trusted]
    positions: rr.ComponentColumn = next(iter(rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3))))
    clears: rr.ComponentColumn = next(iter(rr.Clear.columns(is_recursive=[False] * sum(untrusted))))
    columns: list[rr.ComponentColumn] = [
        MaskedComponentColumn(positions, lengths=[vertices.shape[1] if ok else 0 for ok in trusted], present=trusted),
        MaskedComponentColumn(clears, lengths=[int(gone) for gone in untrusted], present=untrusted),
    ]
    rr.send_columns(path, indexes=[time_column(times_ns), frame_index_column(frame_indices)], columns=columns, recording=recording)


def log_mesh_static(recording: rr.RecordingStream, path: str, *, albedo_factor: Rgba32Like) -> None:
    """Log a per-frame mesh's only static field; its triangle list rides on the vertex chunks (module docstring)."""
    rr.log(path, rr.Mesh3D.from_fields(albedo_factor=albedo_factor), static=True, recording=recording)


def log_mesh_batch(
    recording: rr.RecordingStream,
    path: str,
    *,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    vertices: Float32[ndarray, "k v 3"],
    trusted: list[bool],
    topology: UInt32[ndarray, "f 3"],
) -> None:
    """Write one chunk: empty vertex rows where tracking is lost, the triangle list on the first trusted row.

    Rows other than the first trusted one hold a null triangle list, which latest-at skips; a batch without a
    trusted row writes no triangle list at all.

    Args:
        recording: Destination stream; the caller logs ``log_mesh_static`` once, never the triangle list.
        path: Mesh entity.
        times_ns: Int64[ndarray, "t"] times, including missing rows.
        frame_indices: Int64[ndarray, "t"] source indices.
        vertices: Float32[ndarray, "k v 3"] for trusted rows only, in metres.
        trusted: One flag per timeline row; sum must equal k.
        topology: UInt32[ndarray, "f 3"] triangle list of the mesh.
    """
    if sum(trusted) != len(vertices):
        raise ValueError("vertices must contain exactly the trusted rows")
    lengths: list[int] = [vertices.shape[1] if ok else 0 for ok in trusted]
    columns: list[rr.ComponentColumn] = list(rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3)).partition(lengths))
    if any(trusted):
        triangles: rr.ComponentColumn = rr.ComponentColumn(rr.Mesh3D.descriptor_triangle_indices(), rr.components.TriangleIndicesBatch(topology))
        columns.append(SparseComponentColumn(triangles, row=trusted.index(True), num_rows=len(trusted)))
    rr.send_columns(path, indexes=[time_column(times_ns), frame_index_column(frame_indices)], columns=columns, recording=recording)


def log_mesh_rows(
    recording: rr.RecordingStream,
    path: str,
    *,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    vertices: Float32[ndarray, "k v 3"],
    present: Bool[ndarray, "t"],
    topology: UInt32[ndarray, "f 3"],
    albedo_factor: Rgba32Like,
) -> None:
    """A body mesh over its rows: the static albedo, then one ``log_mesh_batch`` chunk per ``BODY_MESH_BATCH_ROWS`` rows.

    ``vertices`` holds the present rows only, in row order; an absent row is an empty mesh.
    """
    log_mesh_static(recording, path, albedo_factor=albedo_factor)
    before: Int64[ndarray, "t"] = np.cumsum(present) - present  # present rows ahead of each row
    for start in range(0, len(present), BODY_MESH_BATCH_ROWS):
        rows: slice = slice(start, start + BODY_MESH_BATCH_ROWS)
        trusted: list[bool] = present[rows].tolist()
        first: int = int(before[start])
        log_mesh_batch(
            recording,
            path,
            times_ns=times_ns[rows],
            frame_indices=frame_indices[rows],
            vertices=vertices[first : first + sum(trusted)],
            trusted=trusted,
            topology=topology,
        )
