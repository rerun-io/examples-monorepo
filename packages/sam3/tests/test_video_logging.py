"""The GPU-built instance id map equals the former CPU construction."""

import numpy as np
import pytest
import torch
from jaxtyping import Float32
from numpy import ndarray

from sam3.api.video_logging import segmentation_id_map


def cpu_id_map(masks: Float32[ndarray, "n h w"]) -> ndarray:
    """The demos' previous per-instance np.where loop (last instance wins)."""
    seg_map: ndarray = np.zeros(masks.shape[1:], dtype=np.uint8 if masks.shape[0] <= 255 else np.uint16)
    for idx in range(masks.shape[0]):
        seg_map = np.where(masks[idx] >= 0.5, seg_map.dtype.type(idx + 1), seg_map)
    return seg_map


@pytest.mark.parametrize("num_instances", [1, 4, 300])
def test_id_map_matches_cpu_loop(num_instances: int) -> None:
    rng: np.random.Generator = np.random.default_rng(num_instances)
    masks: Float32[ndarray, "n 24 32"] = rng.uniform(0.0, 1.0, size=(num_instances, 24, 32)).astype(np.float32)
    ours: ndarray = segmentation_id_map(torch.from_numpy(masks))
    reference: ndarray = cpu_id_map(masks)
    assert ours.dtype == reference.dtype
    np.testing.assert_array_equal(ours, reference)


def test_id_map_accepts_boolean_masks() -> None:
    masks: torch.Tensor = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, :2] = True
    masks[1, 1:3] = True
    np.testing.assert_array_equal(segmentation_id_map(masks)[:, 0], [1, 2, 2, 0])
