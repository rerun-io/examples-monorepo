from pathlib import Path

import numpy as np
import pytest
import torch
from test_stream import _fake_segment, _GrayDecoder

from handtrack.data.cache import DetNetCache, read_manifest, write_cache
from handtrack.data.catalog import UMETRACK
from handtrack.data.stream import CatalogStream, StreamConfig


def _build(directory: Path) -> None:
    segments = {str(i): _fake_segment(str(i), i) for i in range(5)}
    with CatalogStream(StreamConfig(datasets=(UMETRACK,), device='cpu', nets='detnet', producers=1, fetchers=1, detnet_buffer=4, min_fill=0.0),
                       segments=tuple(info for info, _ in segments.values()), read_segment=lambda info: segments[info.segment_id][1],
                       open_decoder=lambda *_args: _GrayDecoder()) as stream:
        write_cache(stream, directory, 'training', draw=3)


def test_cache_holds_one_epoch_and_every_epoch_visits_each_sample_once(tmp_path: Path) -> None:
    _build(tmp_path)
    manifest = read_manifest(tmp_path)
    assert (manifest.samples, manifest.segments, manifest.datasets) == (10, 5, (UMETRACK,))
    assert np.load(tmp_path / 'pooled.npy').dtype == np.uint8
    # Tag each row so the draws can be traced back to it.
    np.save(tmp_path / 'circle.npy', np.arange(10 * 6, dtype=np.float32).reshape(10, 2, 3))
    cache = DetNetCache(tmp_path, batch_size=4, device='cpu', seed=3)
    orders = []
    for epoch in range(2):
        cache.start_epoch(epoch)
        seen = []
        while (batch := cache.next_detnet_batch()) is not None:
            assert batch.pooled.shape[1:] == (1, 120, 160) and batch.pooled.dtype == torch.float32
            seen += (batch.circle[:, 0, 0] / 6).long().tolist()
        assert len(seen) == 8 and len(set(seen)) == 8 and set(seen) <= set(range(10))  # two full batches; the remainder is dropped
        orders.append(seen)
    assert orders[0] != orders[1]
    cache.close()


def test_cancel_ends_a_blocked_draw_and_incomplete_caches_are_refused(tmp_path: Path) -> None:
    _build(tmp_path)
    cache = DetNetCache(tmp_path, batch_size=2, device='cpu', prefetch=1)
    cache.start_epoch(0)
    cache.cancel()
    assert cache.next_detnet_batch() is None
    cache.close()
    (tmp_path / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='manifest'):
        DetNetCache(tmp_path, batch_size=2, device='cpu')


def test_oversized_circles_lose_their_target_but_keep_presence_and_other_splits_are_refused(tmp_path: Path) -> None:
    _build(tmp_path)
    circle = np.load(tmp_path / 'circle.npy')
    circle[:] = (0.5, 0.5, 0.1)
    circle[1, 0] = (0.5, 0.5, 32.9)   # a hand at the lens: radius of 33 image widths
    circle[2, 1] = (-27.5, 0.5, 0.1)  # a centre far left of the image
    np.save(tmp_path / 'circle.npy', circle)
    np.save(tmp_path / 'circle_mask.npy', np.ones((10, 2), dtype=bool))
    np.save(tmp_path / 'presence.npy', np.ones((10, 2), dtype=np.float32))
    cache = DetNetCache(tmp_path, batch_size=10, device='cpu')
    assert cache.dropped_circles == 2
    cache.start_epoch(0)
    batch = cache.next_detnet_batch()
    assert batch is not None and int(batch.circle_mask.sum()) == 18 and float(batch.presence.sum()) == 20.0
    cache.close()
    manifest = (tmp_path / 'manifest.json').read_text().replace('"split":"training"', '"split":"validation"')
    (tmp_path / 'manifest.json').write_text(manifest)
    with pytest.raises(ValueError, match='training cache'):
        DetNetCache(tmp_path, batch_size=2, device='cpu')
