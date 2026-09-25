"""Shared UmeTrack handedness and batched mesh writer for raw UmeTrack and SHOW3D."""

from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from simplecv.umetrack_temp.generic_hand_model_numpy import LEFT_HAND_INDEX, RIGHT_HAND_INDEX, HandModelNumpy, skin_mesh, wrist_for_hand

from dataforge import hands, meshes, schema

HAND_SIDES: tuple[Literal["left", "right"], ...] = ("left", "right")
assert (LEFT_HAND_INDEX, RIGHT_HAND_INDEX) == (0, 1)


_SKINNING_BATCH_SIZE: int = 256
"""Bound skinning workspace to less than 20 MB."""


def log_hand_meshes(
    recording: rr.RecordingStream,
    side: Literal["left", "right"],
    hand_index: int,
    model: HandModelNumpy,
    angles: Float32[ndarray, "n 22"],
    wrists: Float32[ndarray, "n 4 4"],
    trusted: Bool[ndarray, "n"],
    *,
    times_ns: Int64[ndarray, "n"],
    frame_indices: Int64[ndarray, "n"],
) -> None:
    """Log the static template, then skin trusted rows in bounded batches.

    Wrists are world-from-wrist in millimetres; untrusted rows are ignored and written
    as empty meshes so the viewer's latest-at never holds a stale hand.
    """
    path: str = schema.hand_mesh_path(side)
    rr.log(path, rr.Mesh3D.from_fields(triangle_indices=model.mesh_triangles, albedo_factor=hands.HAND_ALBEDO[side]), static=True, recording=recording)
    for start in range(0, len(trusted), _SKINNING_BATCH_SIZE):
        batch: slice = slice(start, start + _SKINNING_BATCH_SIZE)
        keep: Bool[ndarray, "b"] = trusted[batch]
        vertices: Float32[ndarray, "k v 3"] = np.zeros((0, len(model.mesh_vertices), 3), dtype=np.float32)
        if keep.any():
            vertices = skin_mesh(model, angles[batch][keep], wrist_for_hand(wrists[batch][keep], hand_index)) * np.float32(0.001)
        meshes.log_mesh_batch(recording, path, times_ns=times_ns[batch], frame_indices=frame_indices[batch], vertices=vertices, trusted=keep.tolist())
