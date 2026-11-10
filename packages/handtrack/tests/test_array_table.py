"""Named arrays that share their first axis go to one Parquet table and come back with the same dtype and shape."""
from pathlib import Path

import numpy as np
import pytest

from handtrack.data.array_table import read_arrays, write_arrays


def test_arrays_round_trip_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arrays = {"pooled_u8": rng.integers(0, 256, (5, 120, 160), dtype=np.uint8), "center": rng.random((5, 2, 2), dtype=np.float32),
              "dataset": np.arange(5, dtype=np.int64)}
    write_arrays(tmp_path / "set.parquet", arrays)
    back = read_arrays(tmp_path / "set.parquet")
    assert list(back) == list(arrays)
    for name, value in arrays.items():
        assert back[name].dtype == value.dtype and back[name].shape == value.shape
        np.testing.assert_array_equal(back[name], value)


def test_arrays_must_share_their_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rows"):
        write_arrays(tmp_path / "bad.parquet", {"a": np.zeros((3, 2)), "b": np.zeros((4,))})
