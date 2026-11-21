"""The frozen reference of the dynamic gate: per-frame SMPL-X vertices of running_jumping, in one ``.rrd``.

Layout (the stream logger's mesh layout): ``/world/meshes/person_0`` holds ``Mesh3D:vertex_positions`` per frame on the
``frame`` sequence timeline and the shared ``time`` timeline (``frame / fps``); the recording property
``/__properties/reference`` names the source. Values are float32 and read back bit-exactly.

The original-DAG output (``ma_3d_dynamic``) was lost in 2026-11; the reference since is the dataset's own MAMMA
prediction (``pred/params_00.npz``) forwarded through SMPL-X v1.1 neutral, frozen by ``tools/freeze_dynamic_reference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

ENTITY: str = "/world/meshes/person_0"
"""Entity of the reference mesh (``WORLD_TAG/meshes/person_<id>`` as the stream logger writes it)."""
FRAME_TIMELINE: str = "frame"
"""Sequence timeline holding each row's video frame index."""


def write_reference(path: Path, frames: Int64[ndarray, "f"], vertices: Float32[ndarray, "f v 3"], fps: float, source: str) -> None:
    """Write ``vertices`` (one row per frame in ``frames``) to ``path``."""
    import rerun as rr

    if len(frames) != len(vertices):
        raise ValueError(f"{len(frames)} frames for {len(vertices)} vertex rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream("mamma-dynamic-reference")
    recording.save(path)
    recording.send_columns(
        ENTITY,
        indexes=[rr.TimeColumn(FRAME_TIMELINE, sequence=frames), rr.TimeColumn("time", duration=frames / fps)],
        columns=rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3)).partition([vertices.shape[1]] * len(frames)),
    )
    recording.send_property("reference", rr.AnyValues(source=source, fps=fps))
    recording.flush()


def load_reference(path: Path) -> tuple[Int64[ndarray, "f"], Float32[ndarray, "f v 3"], str]:
    """Frames, vertices and source of the reference at ``path``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file holds no reference mesh or no source property.
    """
    from rerun.chunk import RrdReader

    if not path.is_file():
        raise FileNotFoundError(
            f"dynamic reference {path} is missing (hf download pablovela5620/mamma-streaming-data --repo-type dataset --include 'golden/**')"
        )
    frames: list[Int64[ndarray, "n"]] = []
    vertices: list[Float32[ndarray, "n v 3"]] = []
    source: str | None = None
    for chunk in RrdReader(path).stream():
        batch = chunk.to_record_batch()
        if chunk.entity_path == ENTITY:
            column = batch.column("Mesh3D:vertex_positions")
            vertices.append(np.asarray(column.flatten().flatten().to_numpy(), dtype=np.float32).reshape(len(column), -1, 3))
            frames.append(np.asarray(batch.column(FRAME_TIMELINE).to_numpy(), dtype=np.int64))
        elif chunk.entity_path == "/__properties/reference":
            source = str(batch.column("source")[0].as_py()[0])
    if not vertices or source is None:
        raise ValueError(f"{path}: no {ENTITY} mesh rows or no reference source property")
    all_frames: Int64[ndarray, "f"] = np.concatenate(frames)
    order: Int64[ndarray, "f"] = np.argsort(all_frames, kind="stable")
    return all_frames[order], np.concatenate(vertices)[order], source
