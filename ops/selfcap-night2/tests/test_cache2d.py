import numpy as np
import pyarrow.parquet as pq
import pytest

import cache2d
from selfcap_curate.keypoints import Keypoints2D, ViewDetections


def detections():
    view = ViewDetections(np.zeros((2, 1, 4), np.float32), np.ones((2, 1), np.float32),
                          np.ones((2, 1, 133, 2), np.float32), np.ones((2, 1, 133), np.float32))
    return Keypoints2D(np.array([0.0, 0.1]), {'one': view, 'two': view}, 10.0)


def test_complete_cache_roundtrip_and_missing_phone_rejected(tmp_path):
    cache2d.save(tmp_path / 'cache', detections())
    restored = cache2d.load(tmp_path / 'cache')
    np.testing.assert_array_equal(restored.views['one'].xy, detections().views['one'].xy)
    (tmp_path / 'cache/two.parquet').unlink()
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / 'cache')


def test_failed_save_does_not_publish_partial_cache(tmp_path, monkeypatch):
    original = pq.write_table
    def fail_second(table, path):
        if path.name == 'two.parquet':
            raise OSError('interrupted')
        original(table, path)
    monkeypatch.setattr(pq, 'write_table', fail_second)
    with pytest.raises(OSError):
        cache2d.save(tmp_path / 'cache', detections())
    assert not (tmp_path / 'cache').exists()


def test_legacy_cache_requires_session_and_matching_grids(tmp_path):
    cache2d.save(tmp_path / 'cache', detections())
    (tmp_path / 'cache/complete.json').unlink()
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / 'cache')
    assert set(cache2d.load(tmp_path / 'cache', expected_phones=('one', 'two'), expected_rate=10.0).views) == {'one', 'two'}
    path = tmp_path / 'cache/two.parquet'
    table = pq.read_table(path)
    table = table.set_column(0, 'time_s', [__import__('pyarrow').array([0.0, 0.2])])
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / 'cache', expected_phones=('one', 'two'), expected_rate=10.0)
