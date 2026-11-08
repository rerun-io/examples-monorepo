"""fuse_keyframes: keyframes at their Sim3 poses become one front-faced TSDF mesh."""

import lietorch_mojo as lietorch
import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

pytest.importorskip("tsdf_rs", reason="the map surface needs the tsdf-rs extension from the mast3r-slam envs")

from mast3r_slam.map_surface import KeyframeDepth, SurfaceSettings, fuse_keyframes  # noqa: E402

H: int = 48
W: int = 64


def _plane_keyframe(depth: float) -> KeyframeDepth:
    """A fronto-parallel wall at ``depth`` (before scale) filling a pinhole with f=W."""
    k_33: Float32[ndarray, "3 3"] = np.array([[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    return KeyframeDepth(
        depth_hw=np.full((H, W), depth, dtype=np.float32),
        k_33=k_33,
        rgb_hw3=np.full((H, W, 3), 200, dtype=np.uint8),
    )


def _sim3_data(scale: float) -> Float32[ndarray, "8"]:
    data: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32).copy()
    data[7] = scale
    return data


def test_mesh_lands_at_the_scaled_depth_and_shows_front_faces_only() -> None:
    """A Sim3 scale of 2 puts a wall seen at Z=1 in its camera frame at Z=2 in the world."""
    mesh: rr.Mesh3D | None = fuse_keyframes(
        [(_plane_keyframe(1.0), _sim3_data(2.0)), (_plane_keyframe(1.0), _sim3_data(2.0))],
        SurfaceSettings(voxel_size=0.02),
    )
    assert mesh is not None
    assert mesh.vertex_positions is not None
    vertices_z: Float32[ndarray, "v"] = np.asarray(mesh.vertex_positions.pa_array.to_pylist(), dtype=np.float32)[:, 2]
    np.testing.assert_allclose(np.median(vertices_z), 2.0, atol=0.03)
    assert mesh.face_rendering is not None
    assert mesh.face_rendering.pa_array.to_pylist() == [rr.components.MeshFaceRendering.Front.value]


def test_no_depth_gives_no_mesh() -> None:
    """Keyframes whose depth is all filtered out produce no surface."""
    assert fuse_keyframes([(_plane_keyframe(0.0), _sim3_data(1.0))]) is None
