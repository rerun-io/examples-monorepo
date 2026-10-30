"""Stable skinning-based display crop; lower-body parameters may remain in the fit."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64, Int32, Int64
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class MeshSelection:
    vertices: Int64[ndarray, "v"]
    """Original vertex indices retained by the cropped faces."""
    faces: Int32[ndarray, "f 3"]
    """Retained triangles remapped into the selected vertex array."""


def lower_body_mask(names: tuple[str, ...], parents: Int64[ndarray, "j"]) -> Bool[ndarray, "j"]:
    """Mark the hip-to-knee bones and all descendants by ancestry, not index ranges."""
    lower = np.array([name in ("LeftLeg", "RightLeg") for name in names])
    for _ in range(len(names)):
        inherited = np.array([parent >= 0 and parent != child and lower[parent] for child, parent in enumerate(parents)])
        updated = lower | inherited
        if np.array_equal(updated, lower):
            break
        lower = updated
    return lower


def visible_joint_mask(names: tuple[str, ...], parents: Int64[ndarray, "j"]) -> Bool[ndarray, "j"]:
    """Keep the hip joint centres while hiding the knees and their descendants."""
    return ~lower_body_mask(names, parents) | np.array([name in ("LeftLeg", "RightLeg") for name in names])


def mesh_selection(
    weights: Float64[ndarray, "v j"], names: tuple[str, ...], parents: Int64[ndarray, "j"], faces: Int32[ndarray, "f 3"]
) -> MeshSelection:
    """Drop lower-body-dominated vertices and every triangle touching one of them."""
    lower = lower_body_mask(names, parents)
    keep = weights[:, lower].sum(axis=1) < 0.5
    triangles = faces[keep[faces].all(axis=1)]
    vertices = np.unique(triangles).astype(np.int64)
    remap = np.full(len(weights), -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices), dtype=np.int64)
    return MeshSelection(vertices, remap[triangles].astype(np.int32))
