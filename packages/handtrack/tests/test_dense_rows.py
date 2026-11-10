"""Fixed-length list rows of an Arrow column become one dense array without a Python loop over frames."""
import numpy as np
import pyarrow as pa

from handtrack.data.rerun_source import dense_rows

POINT = pa.list_(pa.list_(pa.float32(), 2))


def test_rows_of_the_expected_length_fill_the_array_and_others_stay_nan() -> None:
    column = pa.chunked_array([pa.array([[[1, 2], [3, 4]], None], type=POINT), pa.array([[], [[5, 6]], [[7, 8], [9, 10]]], type=POINT)])
    values, ok = dense_rows(column, length=2, inner=2)
    assert ok.tolist() == [True, False, False, False, True]
    assert values.dtype == np.float32 and values.shape == (5, 2, 2)
    assert values[0].tolist() == [[1, 2], [3, 4]] and values[4].tolist() == [[7, 8], [9, 10]]
    assert np.isnan(values[1:4]).all()


def test_scalar_lists_come_back_without_an_inner_axis() -> None:
    column = pa.chunked_array([pa.array([[0.5, 0.25, 1.0], [0.1]], type=pa.list_(pa.float32()))])
    values, ok = dense_rows(column, length=3)
    assert ok.tolist() == [True, False] and values.shape == (2, 3)
    assert values[0].tolist() == [0.5, 0.25, 1.0]


def test_a_sliced_chunk_keeps_its_own_rows() -> None:
    whole = pa.array([[[0, 0]], [[1, 1]], [[2, 2]]], type=POINT)
    values, ok = dense_rows(pa.chunked_array([whole.slice(1)]), length=1, inner=2)
    assert ok.tolist() == [True, True] and values[:, 0, 0].tolist() == [1, 2]
