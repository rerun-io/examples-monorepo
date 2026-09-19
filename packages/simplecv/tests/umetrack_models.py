"""Shared synthetic UmeTrack hand models for backend tests."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from jaxtyping import Float32, Int64
from numpy import ndarray

from simplecv.umetrack_temp import generic_hand_model_numpy as numpy_model

if TYPE_CHECKING:
    from simplecv.umetrack_temp import generic_hand_model_torch as torch_model


def synthetic_hand_model() -> numpy_model.HandModelNumpy:
    """A synthetic left hand with nonzero joint pivots and blended bone weights."""
    rng: np.random.Generator = np.random.default_rng(42)
    axes: Float32[ndarray, "22 3"] = rng.normal(size=(22, 3)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    indices: Int64[ndarray, "21 3"] = np.stack([np.arange(21) % 17, (np.arange(21) + 5) % 17, (np.arange(21) + 9) % 17], axis=-1)
    weights: Float32[ndarray, "21 3"] = np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (21, 1))
    dense: Float32[ndarray, "31 17"] = rng.uniform(size=(31, 17)).astype(np.float32)
    dense /= dense.sum(axis=-1, keepdims=True)
    return numpy_model.HandModelNumpy(
        joint_rotation_axes=axes,
        joint_rest_positions=rng.uniform(-0.1, 0.1, (22, 3)).astype(np.float32),
        joint_frame_index=np.arange(22, dtype=np.int64) % 17,
        joint_parent=np.full(22, -1, dtype=np.int64),
        joint_first_child=np.full(22, -1, dtype=np.int64),
        joint_next_sibling=np.full(22, -1, dtype=np.int64),
        landmark_rest_positions=rng.uniform(-0.1, 0.1, (21, 3)).astype(np.float32),
        landmark_rest_bone_weights=weights,
        landmark_rest_bone_indices=indices,
        hand_scale=np.array(1.0, dtype=np.float32),
        mesh_vertices=rng.uniform(-0.1, 0.1, (31, 3)).astype(np.float32),
        mesh_triangles=np.array([[0, 1, 2]], dtype=np.int64),
        dense_bone_weights=dense,
        joint_limits=np.tile(np.array([-1.0, 1.0], dtype=np.float32), (22, 1)),
    )


def to_torch(model: numpy_model.HandModelNumpy) -> torch_model.HandModelTorch:
    """Convert a copy so tensor mutations cannot change the NumPy model."""
    pytest.importorskip("torch", reason="UmeTrack parity checks require torch")
    from simplecv.umetrack_temp import generic_hand_model_torch as torch_model

    return torch_model.hand_model_numpy_to_tensor(deepcopy(model))
