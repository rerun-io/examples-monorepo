"""Unit tests for the cube-face join/crop geometry on a synthetic two-lens rig and a scene at infinity."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation

from monopriors.cube_faces import (
    CubeFace,
    FullFaceWindow,
    depth_png_mm,
    full_face_window,
    join_half_faces,
    normal_png_rgb,
    opposite_half_face,
)

N: int = 100
"""Full face side in the synthetic rig."""

# Rows = face axes (x right, y down, z forward) in lens coordinates (RDF).
FACE_R_LENS: dict[str, Float64[ndarray, "3 3"]] = {
    "front": np.eye(3),
    "left": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "right": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "down": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


def _face(lens: int, direction: str, lens_R_rig: Float64[ndarray, "3 3"], pixel_centre_px: float) -> CubeFace:
    """A face as the layer describes it: the 0.6N part towards the lens's front, off-centre principal point."""
    face_R_lens: Float64[ndarray, "3 3"] = FACE_R_LENS[direction]
    width: int = N
    height: int = N
    x0: int = 0
    y0: int = 0
    if direction != "front":
        # The lens's forward axis (lens z) in face coordinates decides which side of the full face it sees.
        forward_face: Float64[ndarray, "3"] = face_R_lens[:, 2]
        if abs(forward_face[0]) > 0.5:
            width = int(0.6 * N)
            x0 = N - width if forward_face[0] > 0 else 0
        else:
            height = int(0.6 * N)
            y0 = N - height if forward_face[1] > 0 else 0
    centre_px: float = (N - 1) / 2 + pixel_centre_px
    k_matrix: Float64[ndarray, "3 3"] = np.array([[N / 2, 0.0, centre_px - x0], [0.0, N / 2, centre_px - y0], [0.0, 0.0, 1.0]])
    return CubeFace(
        entity=f"/world/rig_00/cam_{lens}{direction}",
        name=f"cube_{lens}_{direction}",
        k_matrix=k_matrix,
        width=width,
        height=height,
        face_R_rig=face_R_lens @ lens_R_rig,
    )


def _rig(lens1_error_deg: float = 1.0, pixel_centre_px: float = 0.0) -> tuple[CubeFace, ...]:
    """Lens 0 = rig axes; lens 1 looks the other way (180 deg about y) with a small calibration error."""
    lens1_R_rig: Float64[ndarray, "3 3"] = Rotation.from_euler("xyz", [lens1_error_deg, -lens1_error_deg, 0.5], degrees=True).as_matrix() @ np.diag(
        [-1.0, 1.0, -1.0]
    )
    return tuple(
        _face(lens, direction, rotation, pixel_centre_px) for lens, rotation in ((0, np.eye(3)), (1, lens1_R_rig)) for direction in FACE_R_LENS
    )


def _render(
    k_matrix: Float64[ndarray, "3 3"], face_R_rig: Float64[ndarray, "3 3"], width: int, height: int, pixel_centre_px: float
) -> UInt8[ndarray, "h w 3"]:
    """A smooth scene at infinity: colour is a function of the rig-frame ray direction."""
    u: Float64[ndarray, "h w"]
    v: Float64[ndarray, "h w"]
    u, v = np.meshgrid(np.arange(width) + pixel_centre_px, np.arange(height) + pixel_centre_px)
    rays_cam: Float64[ndarray, "h w 3"] = np.stack(
        [(u - k_matrix[0, 2]) / k_matrix[0, 0], (v - k_matrix[1, 2]) / k_matrix[1, 1], np.ones_like(u)], -1
    )
    rays_rig: Float64[ndarray, "h w 3"] = rays_cam @ face_R_rig  # row vectors: R^T applied
    rays_rig /= np.linalg.norm(rays_rig, axis=-1, keepdims=True)
    rgb: Float64[ndarray, "h w 3"] = 127.5 + 100.0 * np.stack(
        [np.sin(3.0 * rays_rig[..., 0]), np.cos(2.0 * rays_rig[..., 1] + 1.0), np.sin(2.5 * rays_rig[..., 2] + rays_rig[..., 0])], -1
    )
    return np.round(rgb).astype(np.uint8)


def _image(face: CubeFace, pixel_centre_px: float) -> UInt8[ndarray, "h w 3"]:
    return _render(face.k_matrix, face.face_R_rig, face.width, face.height, pixel_centre_px)


def test_full_face_window_places_half_faces() -> None:
    faces: dict[str, CubeFace] = {face.name: face for face in _rig()}
    assert full_face_window(faces["cube_0_front"]) == FullFaceWindow(side_px=N, x0_px=0, y0_px=0, width=N, height=N)
    assert full_face_window(faces["cube_0_left"]) == FullFaceWindow(side_px=N, x0_px=40, y0_px=0, width=60, height=N)
    assert full_face_window(faces["cube_0_right"]) == FullFaceWindow(side_px=N, x0_px=0, y0_px=0, width=60, height=N)
    assert full_face_window(faces["cube_0_up"]) == FullFaceWindow(side_px=N, x0_px=0, y0_px=40, width=N, height=60)


def test_full_face_window_rejects_bad_intrinsics() -> None:
    front: CubeFace = _rig()[0]
    with pytest.raises(ValueError, match="not an integer"):
        full_face_window(front, pixel_centre_px=0.5)  # the wrong pixel-centre convention
    face: CubeFace = _rig()[1]
    with pytest.raises(ValueError, match="not an integer"):
        full_face_window(
            CubeFace(
                face.entity,
                face.name,
                face.k_matrix + np.diag([0.0, 0.0, 0.0]) + np.array([[0, 0, 0.3], [0, 0, 0], [0, 0, 0]]),
                face.width,
                face.height,
                face.face_R_rig,
            )
        )
    with pytest.raises(ValueError, match="does not fit"):
        full_face_window(CubeFace(face.entity, face.name, face.k_matrix, face.width + 50, face.height, face.face_R_rig))


def test_opposite_half_face_pairs_by_view_direction() -> None:
    faces: tuple[CubeFace, ...] = _rig()
    by_name: dict[str, CubeFace] = {face.name: face for face in faces}
    expected: dict[str, str] = {"left": "right", "right": "left", "up": "up", "down": "down"}
    for direction, partner in expected.items():
        assert opposite_half_face(by_name[f"cube_0_{direction}"], faces).name == f"cube_1_{partner}"
        assert opposite_half_face(by_name[f"cube_1_{partner}"], faces).name == f"cube_0_{direction}"
    with pytest.raises(ValueError, match="front"):
        opposite_half_face(by_name["cube_0_front"], faces)


@pytest.mark.parametrize("pixel_centre_px", [0.0, 0.5])
@pytest.mark.parametrize("name", ["cube_0_left", "cube_0_right", "cube_0_up", "cube_0_down", "cube_1_left", "cube_1_down"])
def test_join_matches_the_full_face_and_crops_back_exactly(name: str, pixel_centre_px: float) -> None:
    faces: tuple[CubeFace, ...] = _rig(pixel_centre_px=pixel_centre_px)
    face: CubeFace = next(f for f in faces if f.name == name)
    other: CubeFace = opposite_half_face(face, faces)
    own_rgb: UInt8[ndarray, "h w 3"] = _image(face, pixel_centre_px)
    joined: UInt8[ndarray, "n n 3"] = join_half_faces(own_rgb, _image(other, pixel_centre_px), face, other, pixel_centre_px)
    window: FullFaceWindow = full_face_window(face, pixel_centre_px)
    assert joined.shape == (N, N, 3)
    # Own pixels are exact, so a prior cropped back from the full face lines up with the training image.
    np.testing.assert_array_equal(window.crop(joined), own_rgb)
    # The context part agrees with a direct render of the full face (bilinear warp error only), and has no hole.
    full_k: Float64[ndarray, "3 3"] = face.k_matrix.copy()
    full_k[:2, 2] = (N - 1) / 2 + pixel_centre_px
    direct: UInt8[ndarray, "n n 3"] = _render(full_k, face.face_R_rig, N, N, pixel_centre_px)
    error: Float32[ndarray, "n n"] = np.abs(joined.astype(np.float32) - direct.astype(np.float32)).max(-1)
    assert np.percentile(error, 99) <= 3.0
    assert not np.any(np.all(joined == 0, axis=-1))


def test_depth_png_mm_marks_invalid_as_zero() -> None:
    depth_m: Float32[ndarray, "2 3"] = np.array([[1.0, 2.5, -1.0], [70.0, np.nan, 0.0004]], dtype=np.float32)
    valid: np.ndarray = np.array([[True, False, True], [True, True, True]])
    np.testing.assert_array_equal(depth_png_mm(depth_m, valid), np.array([[1000, 0, 0], [0, 0, 0]], dtype=np.uint16))


def test_normal_png_rgb_encoding() -> None:
    normals: Float32[ndarray, "1 2 3"] = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    valid: np.ndarray = np.array([[True, False]])
    np.testing.assert_array_equal(normal_png_rgb(normals, valid), np.array([[[128, 128, 0], [0, 0, 0]]], dtype=np.uint8))
