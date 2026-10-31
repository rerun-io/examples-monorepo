"""Frozen train/test splits and body-frame landmark calibration."""

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray


def heldout_blocks(times: Float64[ndarray, "n"]) -> Bool[ndarray, "n"]:
    """Hold out [0,2), [4,6), ... including the occlusion example."""
    return (np.floor(times / 2.0).astype(np.int64) % 2) == 0


def learn_body_offsets(
    predicted: Float64[ndarray, "n j 3"],
    target: Float64[ndarray, "n j 3"],
    rotations: Float64[ndarray, "n 3 3"],
    train: Bool[ndarray, "n"],
) -> Float64[ndarray, "j 3"]:
    """Learn median local offsets only on train rows; cap each at 10 cm."""
    local: Float64[ndarray, "n j 3"] = np.einsum("nji,nik->njk", target - predicted, rotations)
    offsets: Float64[ndarray, "j 3"] = np.zeros((predicted.shape[1], 3))
    for joint in range(predicted.shape[1]):
        valid: Bool[ndarray, "n"] = train & np.isfinite(local[:, joint]).all(-1)
        if valid.sum() >= 2:
            offsets[joint] = np.median(local[valid, joint], axis=0)
    lengths: Float64[ndarray, "j"] = np.linalg.norm(offsets, axis=-1)
    return offsets * np.minimum(1.0, 0.1 / np.maximum(lengths, 1e-12))[:, None]


def apply_body_offsets(
    predicted: Float64[ndarray, "n j 3"], rotations: Float64[ndarray, "n 3 3"], offsets: Float64[ndarray, "j 3"]
) -> Float64[ndarray, "n j 3"]:
    """Map frozen body-frame offsets back to world without target data."""
    return predicted + np.einsum("nik,jk->nji", rotations, offsets)
