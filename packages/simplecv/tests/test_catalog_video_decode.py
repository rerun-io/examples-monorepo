"""`simplecv.catalog_video_decode.PrefetchedChunks`: what each stream's decoder is asked to mux."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from simplecv import catalog_video_decode
from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_decode import PrefetchedChunks


def test_the_decoders_mux_no_further_than_the_last_wanted_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller passes the end of the clip while its chunks stop early (a sample pack of a long clip): the decoder muxes
    `[keyframe, end)` into memory, so `end` is the last wanted frame, not the clip's."""
    spans: list[tuple[int, int]] = []

    class FakeDecoder:
        def __init__(self, video: CatalogVideo, start: int, end: int, device: str, fps: int = 30) -> None:
            spans.append((start, end))

        def frames(self, indices: list[int]) -> torch.Tensor:
            return torch.zeros((len(indices), 1, 1, 3), dtype=torch.uint8)

    monkeypatch.setattr(catalog_video_decode, "StreamDecoder", FakeDecoder)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: types.SimpleNamespace(synchronize=lambda: None))
    video = CatalogVideo(times=np.arange(1000, dtype=np.int64), samples=[], keyframes=[], codec="av1")

    batches = list(PrefetchedChunks((video, video), [[10, 20], [30]], 10, 1000, "cpu"))
    assert spans == [(10, 31), (10, 31)]
    assert [[len(frames) for frames in per_stream] for per_stream in batches] == [[2, 2], [1, 1]]

    spans.clear()
    assert list(PrefetchedChunks((video,), [[10, 20]], 10, 15, "cpu")) and spans == [(10, 15)], "a caller's earlier end still bounds it"
