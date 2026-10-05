"""Model-independent temporal mesh output; skinning stays in source adapters.

Topology convention for every per-frame mesh: the triangle list rides on the vertex chunks and is never
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

import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Int64, UInt32
from numpy import ndarray
from rerun.datatypes import Rgba32Like

from dataforge.logging_toolkit import frame_index_column, time_column


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
