"""Camera rig container and camera-frame geometry shared by every stage."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64
from numpy import ndarray


@dataclass(slots=True)
class RigCameras:
    """A camera rig read back from a catalog layer (GT base layer or an exocalib layer)."""

    names: tuple[str, ...]
    """Camera entity names, e.g. ``rig_00/cam_00``."""
    intrinsics: Float64[ndarray, "v 3 3"]
    """Pinhole intrinsics at the native video resolution."""
    cam_T_world: Float64[ndarray, "v 4 4"]
    """World-to-camera transforms (RDF, metric, +Z-up world)."""


def camera_centers(cam_T_world: Float64[ndarray, "v 4 4"]) -> Float64[ndarray, "v 3"]:
    """World-frame camera centres ``-Rᵀt`` of world-to-camera transforms."""
    return -np.einsum("vji,vj->vi", cam_T_world[:, :3, :3], cam_T_world[:, :3, 3])
