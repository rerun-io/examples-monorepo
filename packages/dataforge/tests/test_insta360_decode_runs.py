"""The cube and panorama lens decoder's plan (simplecv's ``decode_runs`` capped at ``RUN_GOPS``) and its rebatching."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Int64
from numpy import ndarray


def x5_gops(frames: int = 2498, gop: int = 60) -> Int64[ndarray, " g"]:
    return np.arange(0, frames, gop, dtype=np.int64)


def test_every_frame_of_the_x5_clip_is_decoded_once_in_runs_of_eight_gops() -> None:
    """Stride 1 on the X5 (GOP 60): 2,498 samples decoded, where a decoder per batch of 4 decoded 19,898."""
    from simplecv.catalog_video_decode import decode_runs

    from dataforge.datasets.insta360 import RUN_GOPS

    indices = np.arange(2498, dtype=np.int64)
    runs = decode_runs(indices, x5_gops(), max_gops=RUN_GOPS)
    assert sum(run.stop - run.start for run in runs) == 2498
    assert [run.start for run in runs] == [0, 480, 960, 1440, 1920, 2400]
    assert np.array_equal(np.concatenate([run.wanted for run in runs]), indices)


@given(sizes=st.lists(st.integers(1, 7), min_size=0, max_size=12), batch=st.integers(1, 9))
def test_rebatching_keeps_every_picture_in_order_in_full_batches(sizes: list[int], batch: int) -> None:
    import torch

    from dataforge.datasets.insta360 import rebatch

    chunks = []
    start = 0
    for size in sizes:
        chunks.append(torch.arange(start, start + size, dtype=torch.uint8).reshape(size, 1, 1, 1).expand(size, 3, 1, 1).contiguous())
        start += size
    batches = list(rebatch(iter(chunks), batch))
    assert [b.shape[0] for b in batches[:-1]] == [batch] * (len(batches) - 1) and (not batches or 0 < batches[-1].shape[0] <= batch)
    assert torch.cat(batches)[:, 0, 0, 0].tolist() == list(range(start)) if batches else start == 0
