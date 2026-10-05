"""Pinch as mesh contact: the minimum distance from the thumb's distal pad to the index finger's last two phalanges on the skinned mesh.

The fingertip landmarks sit 22-25 mm past the DIP joint, so a tip-to-tip rule misses pad-to-pad pinches; contact on the mesh does not.
Contact under ``CONTACT_CLOSED_MM`` is a pinch, over ``CONTACT_OPEN_MM`` open, in between ambiguous (not a label).
"""

import numpy as np
import torch
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

THUMB_DISTAL_BONE: int = 4
"""Skinning frame of the thumb's distal phalanx (the thumb-tip landmark's bone)."""
INDEX_CONTACT_BONES: tuple[int, int] = (6, 7)
"""The index finger's middle and distal phalanges (the index-tip landmark's bone is 7)."""
CONTACT_CLOSED_MM: float = 5.0
CONTACT_OPEN_MM: float = 15.0


def contact_vertices(model: HandModelTorch) -> tuple[Int64[ndarray, "a"], Int64[ndarray, "b"]]:
    """The thumb-pad and index vertex indices (each vertex belongs to its dominant skinning frame)."""
    bones: Int64[ndarray, "v"] = model.dense_bone_weights.argmax(dim=1).numpy()
    return np.flatnonzero(bones == THUMB_DISTAL_BONE), np.flatnonzero(np.isin(bones, INDEX_CONTACT_BONES))


def contact_mm(mesh: Float32[Tensor, "k v 3"], model: HandModelTorch) -> Float32[Tensor, "k"]:
    """Per skinned mesh (metres): the minimum thumb-pad to index vertex distance in mm (NaN meshes give NaN)."""
    thumb, index = contact_vertices(model)
    distances: Float32[Tensor, "k a b"] = torch.cdist(mesh[:, torch.from_numpy(thumb).to(mesh.device)], mesh[:, torch.from_numpy(index).to(mesh.device)])
    return distances.amin(dim=(-1, -2)) * 1000.0
