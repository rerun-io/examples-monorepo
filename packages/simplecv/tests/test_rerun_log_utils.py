"""Tests for shared Rerun logging geometry helpers."""

import numpy as np

from simplecv.rerun_log_utils import compute_vertex_normals


def test_compute_vertex_normals_on_regular_tetrahedron() -> None:
    """A centered regular tetrahedron has radial unit vertex normals."""
    vertices = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        dtype=np.float32,
    )
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)

    normals = compute_vertex_normals(vertices, faces)

    assert normals.dtype == np.float32
    assert np.allclose(normals, vertices / np.sqrt(np.float32(3.0)), atol=1e-6)
