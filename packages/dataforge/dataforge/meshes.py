"""Model-independent temporal mesh output; skinning stays in source adapters."""

import rerun as rr
from jaxtyping import Float32, Int64
from numpy import ndarray

from dataforge.logging_toolkit import frame_index_column, time_column


def log_mesh_batch(
    recording: rr.RecordingStream,
    path: str,
    *,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    vertices: Float32[ndarray, "k v 3"],
    trusted: list[bool],
) -> None:
    """Write one batch with empty rows where tracking is lost.

    Args:
        recording: Destination stream; caller logs static topology and albedo once.
        path: Mesh entity.
        times_ns: Int64[ndarray, "t"] times, including missing rows.
        frame_indices: Int64[ndarray, "t"] source indices.
        vertices: Float32[ndarray, "k v 3"] for trusted rows only, in metres.
        trusted: One flag per timeline row; sum must equal k.
    """
    if sum(trusted) != len(vertices):
        raise ValueError("vertices must contain exactly the trusted rows")
    lengths: list[int] = [vertices.shape[1] if ok else 0 for ok in trusted]
    rr.send_columns(
        path, indexes=[time_column(times_ns), frame_index_column(frame_indices)], columns=rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3)).partition(lengths), recording=recording
    )
