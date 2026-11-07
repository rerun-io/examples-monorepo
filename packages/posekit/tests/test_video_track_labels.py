"""The video-track segmentation label map is built on the device and matches the host loop it replaced (CPU-only)."""

import numpy as np
import pytest
import torch
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.apis.video_track import track_label_map


def host_label_map(masks: Bool[ndarray, "n h w"], track_ids: Int64[ndarray, "n"], frame_hw: tuple[int, int]) -> UInt8[ndarray, "h w"]:
    """The previous per-row host loop: later rows overwrite earlier ones where masks overlap."""
    label_map: UInt8[ndarray, "h w"] = np.zeros(frame_hw, dtype=np.uint8)
    for row in range(len(track_ids)):
        label_map[masks[row]] = int(track_ids[row]) + 1
    return label_map


@pytest.mark.parametrize("rows", [0, 1, 3, 6])
def test_label_map_matches_the_host_loop_with_overlaps(rows: int) -> None:
    generator: torch.Generator = torch.Generator().manual_seed(rows)
    frame_hw: tuple[int, int] = (37, 53)
    masks: Bool[Tensor, "n h w"] = torch.rand((rows, *frame_hw), generator=generator) > 0.5
    track_ids: Int64[Tensor, "n"] = torch.randperm(8, generator=generator)[:rows]
    expected: UInt8[ndarray, "h w"] = host_label_map(masks.numpy(), track_ids.numpy(), frame_hw)
    labels: UInt8[ndarray, "h w"] = track_label_map(masks, track_ids, frame_hw)
    assert labels.dtype == np.uint8 and labels.shape == frame_hw
    np.testing.assert_array_equal(labels, expected)
    if rows > 1:
        assert np.any(masks.numpy().sum(axis=0) > 1), "the case must contain overlaps for last-writer order to matter"
