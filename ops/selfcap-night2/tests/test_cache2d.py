import cache2d
import numpy as np
import pyarrow.parquet as pq
import pytest
from selfcap_curate.keypoints import Keypoints2D, ViewDetections


def detections():
    view = ViewDetections(
        np.arange(8, dtype=np.float64).reshape(2, 1, 4),
        np.ones((2, 1), np.float64),
        np.arange(532, dtype=np.float64).reshape(2, 1, 133, 2),
        np.ones((2, 1, 133), np.float64),
    )
    return Keypoints2D(np.array([0.0, 0.1]), {"one": view, "two": view}, 10.0)


def test_complete_cache_roundtrip_and_missing_phone_rejected(tmp_path):
    cache2d.save(tmp_path / "cache", detections())
    restored = cache2d.load(tmp_path / "cache")
    expected = detections()
    np.testing.assert_array_equal(restored.times, expected.times, strict=True)
    assert restored.rate == expected.rate
    for phone in expected.views:
        for field in ("boxes", "box_scores", "xy", "scores"):
            np.testing.assert_array_equal(getattr(restored.views[phone], field), getattr(expected.views[phone], field), strict=True)
    (tmp_path / "cache/two.parquet").unlink()
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / "cache")


@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_does_not_publish_partial_cache(tmp_path, monkeypatch, existing):
    if existing:
        cache2d.save(tmp_path / "cache", detections())
    original = pq.write_table

    def fail_second(table, path):
        if path.name == "two.parquet":
            raise OSError("interrupted")
        original(table, path)

    monkeypatch.setattr(pq, "write_table", fail_second)
    with pytest.raises(OSError):
        cache2d.save(tmp_path / "cache", detections())
    if existing:
        assert set(cache2d.load(tmp_path / "cache").views) == {"one", "two"}
    else:
        assert not (tmp_path / "cache").exists()


def test_legacy_cache_requires_session_and_matching_grids(tmp_path):
    cache2d.save(tmp_path / "cache", detections())
    (tmp_path / "cache/complete.json").unlink()
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / "cache")
    assert set(cache2d.load(tmp_path / "cache", expected_phones=("one", "two"), expected_rate=10.0).views) == {"one", "two"}
    path = tmp_path / "cache/two.parquet"
    table = pq.read_table(path)
    table = table.set_column(0, "time_s", [__import__("pyarrow").array([0.0, 0.2])])
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        cache2d.load(tmp_path / "cache", expected_phones=("one", "two"), expected_rate=10.0)
