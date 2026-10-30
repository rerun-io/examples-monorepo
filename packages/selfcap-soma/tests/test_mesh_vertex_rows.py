"""Mesh decoding respects Arrow slices without expanding points into Python."""

import numpy as np
import pyarrow as pa
import pytest

from selfcap_soma.overlay import mesh_vertex_rows


def test_mesh_vertex_rows_preserves_slice_offsets_dtype_and_nan_without_copy():
    values = np.arange(36, dtype=np.float32)
    values[19] = np.nan
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(values), 3)
    rows = pa.ListArray.from_arrays(pa.array([0, 3, 6, 9, 12]), vectors)
    decoded = mesh_vertex_rows(rows.slice(1, 2))
    np.testing.assert_array_equal(decoded, values[9:27].reshape(2, 3, 3))
    assert decoded.dtype == np.float32
    assert np.shares_memory(decoded, values)


def test_mesh_vertex_rows_handles_chunked_rows_and_rejects_ragged_meshes():
    kind = pa.list_(pa.list_(pa.float32(), 3))
    rows = pa.array([[[1., 2., 3.]], [[4., 5., 6.]]], type=kind)
    np.testing.assert_array_equal(mesh_vertex_rows(pa.chunked_array([rows[:1], rows[1:]])), [[[1., 2., 3.]], [[4., 5., 6.]]])
    with pytest.raises(ValueError, match="uniform"):
        mesh_vertex_rows(pa.array([[[1., 2., 3.]], []], type=kind))
    with pytest.raises(ValueError, match="null"):
        mesh_vertex_rows(pa.array([None, [[1., 2., 3.]]], type=kind))
