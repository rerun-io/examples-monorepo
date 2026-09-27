import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133, assembly21_to_coco133_batch


@pytest.mark.parametrize("dimensions", [2, 3])
def test_batch_matches_single_frame_map(dimensions: int) -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    joints: Float32[ndarray, "n 2 21 d"] = rng.normal(size=(64, 2, 21, dimensions)).astype(np.float32)
    joints[rng.random(joints.shape) < 0.05] = np.nan  # scattered missing coordinates
    joints[3, 0] = np.nan  # a missing left hand
    joints[4, 1] = np.nan  # a missing right hand
    joints[5, :, 5] = np.nan  # missing wrists: both thumb bases undefined
    batch: Float32[ndarray, "n 133 d"] = assembly21_to_coco133_batch(joints)
    assert batch.dtype == np.float32
    for frame, frame_joints in enumerate(joints):
        padded: Float32[ndarray, "2 21 3"] = np.zeros((2, 21, 3), dtype=np.float32)
        padded[..., :dimensions] = frame_joints
        expected: Float32[ndarray, "133 d"] = assembly21_to_coco133(padded)[:, :dimensions]
        np.testing.assert_array_equal(batch[frame].view(np.uint32), expected.view(np.uint32))
