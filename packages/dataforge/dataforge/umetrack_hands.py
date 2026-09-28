"""Shared UmeTrack subject hand-profile reader and batched mesh writer (raw UmeTrack, SHOW3D, HOT3D)."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Int64, UInt32
from numpy import ndarray
from serde import serde
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    NUM_JOINTS_PER_HAND,
    NUM_LANDMARKS_PER_HAND,
    HandModelNumpy,
    skin_mesh,
    wrist_for_hand,
)

from dataforge import hands, meshes, schema
from dataforge.records import read_json

SKINNING_BATCH_SIZE: int = 256
"""Bound skinning workspace to less than 20 MB."""


@serde
@dataclass(frozen=True, slots=True)
class HandProfile:
    """Full typed subject model; preserve its source text separately."""

    hand_model: HandModelNumpy
    """Float32 UmeTrack rest geometry and int64 topology."""

    def __post_init__(self) -> None:
        if len(self.hand_model.landmark_rest_positions) != NUM_LANDMARKS_PER_HAND:
            raise ValueError("hand profile requires 21 rest landmarks")
        if len(self.hand_model.joint_rotation_axes) != NUM_JOINTS_PER_HAND:
            raise ValueError("hand profile requires 22 joint rotation axes")


class HandProfileDoc(NamedTuple):
    """Verbatim profile text and its validated model."""

    text: str
    model: HandModelNumpy


def read_hand_profile(path: Path) -> HandProfileDoc:
    """Read and validate a profile once, retaining its verbatim document."""
    text: str = path.read_text()
    profile: HandProfile = read_json(path, HandProfile, text=text)
    return HandProfileDoc(text, profile.hand_model)


def log_hand_meshes(
    recording: rr.RecordingStream,
    side: hands.HandSide,
    model: HandModelNumpy,
    angles: Float32[ndarray, "n 22"],
    wrists: Float32[ndarray, "n 4 4"],
    trusted: Bool[ndarray, "n"],
    *,
    times_ns: Int64[ndarray, "n"],
    frame_indices: Int64[ndarray, "n"],
    path: str | None = None,
) -> None:
    """Log the static albedo, then skin trusted rows in bounded batches (topology rides on each batch).

    Wrists are world-from-wrist in millimetres; untrusted rows are ignored and written
    as empty meshes so the viewer's latest-at never holds a stale hand. ``path`` defaults
    to the ground-truth mesh entity; a predicted hand passes its own.
    """
    entity: str = schema.hand_mesh_path(side.name) if path is None else path
    faces: UInt32[ndarray, "f 3"] = np.asarray(model.mesh_triangles, dtype=np.uint32)
    meshes.log_mesh_static(recording, entity, albedo_factor=hands.HAND_ALBEDO[side.name])
    for start in range(0, len(trusted), SKINNING_BATCH_SIZE):
        batch: slice = slice(start, start + SKINNING_BATCH_SIZE)
        keep: Bool[ndarray, "b"] = trusted[batch]
        vertices: Float32[ndarray, "k v 3"] = np.zeros((0, len(model.mesh_vertices), 3), dtype=np.float32)
        if keep.any():
            vertices = skin_mesh(model, angles[batch][keep], wrist_for_hand(wrists[batch][keep], side.model_index)) * np.float32(0.001)
        meshes.log_mesh_batch(
            recording, entity, times_ns=times_ns[batch], frame_indices=frame_indices[batch], vertices=vertices, trusted=keep.tolist(),
            topology=faces,
        )
