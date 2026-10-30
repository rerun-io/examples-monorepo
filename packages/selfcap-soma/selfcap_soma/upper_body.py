"""Fixed rest-pose waist crop; lower-body parameters may remain in the fit."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float32, Int32, Int64
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class MeshSelection:
    vertices: Int64[ndarray, "v"]
    """Original vertex indices retained by the cropped faces."""
    faces: Int32[ndarray, "f 3"]
    """Retained triangles remapped into the selected vertex array."""
    rest_offsets: Float32[ndarray, "v 3"]
    """Display-only displacement for retained vertices touching removed faces."""


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


def mesh_selection(rest: Float32[ndarray, "v 3"], faces: Int32[ndarray, "f 3"], waist_height_m: float) -> MeshSelection:
    """Keep rest-pose vertices above a horizontal waist plane and drop crossing faces.

    Selection is made once from rest coordinates, never from posed geometry.
    """
    keep: Bool[ndarray, "v"] = rest[:, 1] >= waist_height_m
    triangles: Int32[ndarray, "f 3"] = faces[keep[faces].all(axis=1)]
    vertices: Int64[ndarray, "v"] = np.unique(triangles).astype(np.int64)
    remap: Int64[ndarray, "v"] = np.full(len(rest), -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices), dtype=np.int64)
    crossing: Int32[ndarray, "f 3"] = faces[~keep[faces].all(axis=1)]
    rim: Int64[ndarray, "v"] = np.intersect1d(vertices, crossing.ravel())
    offsets: Float32[ndarray, "v 3"] = np.zeros_like(rest)
    offsets[rim, 1] = waist_height_m - rest[rim, 1]
    return MeshSelection(vertices, remap[triangles].astype(np.int32), offsets)
