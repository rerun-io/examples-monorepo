"""Cube faces of an Insta360 lens: face geometry, the remap tables through the Rust Mei model, and the cross-lens seam."""

from __future__ import annotations

import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, UInt8
from numpy import ndarray

LENS: int = 384
"""Synthetic lens image size: a fifth of the X4 Air's 1920."""
FACE: int = 160
"""Face size of the synthetic tests (side faces 96 = 0.6 N)."""
XI: float = 2.315
"""Clip B's xi."""
FOCAL: float = 1835.37 * LENS / 1920
"""Clip B's focal at the synthetic lens size."""
HALF_FIELD: float = np.deg2rad(100.0)
"""The X4 Air's half field."""


def camera() -> insta360_io.MeiCamera:
    """A distortion-free Mei lens shaped like clip B's, so its inverse is closed form."""
    centre: float = (LENS - 1) / 2.0
    return insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [XI, HALF_FIELD] + [0.0] * 13)


def unproject(u: Float64[ndarray, "h w"], v: Float64[ndarray, "h w"]) -> Float64[ndarray, "h w 3"]:
    """Unit rays of lens pixels for ``camera()`` (the Mei inverse without distortion)."""
    centre: float = (LENS - 1) / 2.0
    mx, my = (u - centre) / FOCAL, (v - centre) / FOCAL
    r2: Float64[ndarray, "h w"] = mx * mx + my * my
    factor: Float64[ndarray, "h w"] = (XI + np.sqrt(np.maximum(1.0 + (1.0 - XI * XI) * r2, 0.0))) / (1.0 + r2)
    return np.stack([factor * mx, factor * my, factor - XI], axis=-1)


def colour(rays: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3"]:
    """A smooth colour per direction: ``127.5 + 120 d`` per channel of the unit direction."""
    unit: Float64[ndarray, "... 3"] = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return 127.5 + 120.0 * unit


def test_the_default_face_size_is_800_at_1920_and_every_side_is_a_multiple_of_32() -> None:
    from dataforge.datasets.insta360_cube import default_face_size, lens_faces, side_size

    assert default_face_size(1920) == 800
    assert side_size(800) == 480
    assert default_face_size(3840) == 1600
    for n in (160, 800, 1216, 1600):
        for face in lens_faces(0, n):
            assert face.width % 32 == 0 and face.height % 32 == 0
    assert {(f.direction, f.width, f.height) for f in lens_faces(1, 800)} == {
        ("front", 800, 800),
        ("left", 480, 800),
        ("right", 480, 800),
        ("up", 800, 480),
        ("down", 800, 480),
    }
    with pytest.raises(ValueError, match="multiple of 32"):
        lens_faces(0, 810)


def test_face_rotations_are_proper_and_each_face_looks_along_its_direction() -> None:
    from dataforge.datasets.insta360_cube import lens_faces

    looks: dict[str, tuple[float, float, float]] = {"front": (0, 0, 1), "right": (1, 0, 0), "left": (-1, 0, 0), "down": (0, 1, 0), "up": (0, -1, 0)}
    for face in lens_faces(0, FACE):
        rotation: Float64[ndarray, "3 3"] = face.lens_from_face
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(rotation[:, 2], looks[face.direction])
        assert face.intrinsics[0, 0] == face.intrinsics[1, 1] == FACE / 2
        assert face.name == f"cube_0_{face.direction}"


def test_only_the_middle_of_a_side_face_s_far_edge_lies_past_the_100_deg_field() -> None:
    """Tangent 0.2 past the seam is 11.3 deg on the face's centre line, beyond the 100 deg half field there only."""
    from dataforge.datasets.insta360_cube import lens_faces, remap_table

    for face in lens_faces(0, FACE):
        rays: Float64[ndarray, "h w 3"] = face.rays() @ face.lens_from_face.T
        off_axis_deg: Float64[ndarray, "h w"] = np.degrees(np.arccos(rays[..., 2] / np.linalg.norm(rays, axis=-1)))
        valid = remap_table(face, camera()).valid
        np.testing.assert_array_equal(valid, off_axis_deg <= 100.0)
        if face.direction == "front":
            assert off_axis_deg.max() < 55.0
        else:
            assert off_axis_deg.max() == pytest.approx(101.1, abs=0.3)
            assert 0.98 < valid.mean() < 1.0
            # The far edge: the corners stay inside the field (98 deg), the middle does not.
            far = valid[:, -1] if face.direction == "right" else valid[:, 0] if face.direction == "left" else valid[-1] if face.direction == "down" else valid[0]
            assert far[0] and far[-1] and not far[len(far) // 2]


def test_a_face_remapped_from_a_painted_lens_shows_the_colour_of_its_own_rays() -> None:
    from dataforge.datasets.insta360_cube import lens_faces, remap, remap_table

    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_image: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(unproject(u, v))), 0, 255).astype(np.uint8)
    for face in lens_faces(0, FACE):
        table = remap_table(face, camera())
        expected: Float64[ndarray, "h w 3"] = colour(face.rays() @ face.lens_from_face.T)
        error: Float64[ndarray, "n 3"] = np.abs(remap(lens_image, table).astype(np.float64) - expected)[table.valid]
        # Bilinear interpolation of a smooth field plus the 0.5 rounding of the painted lens.
        assert error.max() < 2.0, f"{face.name}: {error.max():.2f}"


def test_a_ray_outside_the_half_field_maps_to_black() -> None:
    from dataforge.datasets.insta360_cube import CubeFace, remap, remap_table

    centre: float = (FACE - 1) / 2
    backward: CubeFace = CubeFace(0, "back", np.diag([-1.0, 1.0, -1.0]), np.array([[FACE / 2, 0, centre], [0, FACE / 2, centre], [0, 0, 1.0]]), FACE, FACE)
    table = remap_table(backward, camera())
    assert not table.valid.any()
    assert (table.map_x == -1).all()
    assert not remap(np.full((LENS, LENS, 3), 200, np.uint8), table).any()


def test_the_seam_strips_of_the_two_lenses_see_the_same_directions() -> None:
    """Lens 1 looks the other way (half a turn about y): each lens-0 side face has a lens-1 twin over the same strip."""
    from dataforge.datasets.insta360_cube import lens_faces

    lens0_from_lens1: Float64[ndarray, "3 3"] = np.diag([-1.0, 1.0, -1.0])
    twins: dict[str, str] = {"right": "left", "left": "right", "up": "up", "down": "down"}
    faces1 = {face.direction: face for face in lens_faces(1, FACE)}
    for face0 in lens_faces(0, FACE)[1:]:
        face1 = faces1[twins[face0.direction]]
        np.testing.assert_allclose(lens0_from_lens1 @ face1.lens_from_face[:, 2], face0.lens_from_face[:, 2], atol=1e-12)
        rays0: Float64[ndarray, "n 3"] = face0.rays().reshape(-1, 3)
        across: int = 0 if face0.direction in ("left", "right") else 1
        strip: Float64[ndarray, "m 3"] = rays0[np.abs(rays0[:, across]) <= 0.19]
        assert len(strip) > 0
        in_face1: Float64[ndarray, "m 3"] = (face1.lens_from_face.T @ lens0_from_lens1.T @ face0.lens_from_face @ strip.T).T
        pixels: Float64[ndarray, "m 3"] = (face1.intrinsics @ (in_face1 / in_face1[:, 2:]).T).T
        assert (pixels[:, 0] > -0.5).all() and (pixels[:, 0] < face1.width - 0.5).all()
        assert (pixels[:, 1] > -0.5).all() and (pixels[:, 1] < face1.height - 0.5).all()


def test_a_face_pose_in_the_rig_keeps_the_lens_centre() -> None:
    from dataforge.datasets.insta360_cube import face_from_rig, lens_faces

    lens_from_rig: Float64[ndarray, "4 4"] = np.eye(4)
    lens_from_rig[:3, :3] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    lens_from_rig[:3, 3] = [0.01, -0.02, 0.0166]
    centre_in_rig: Float64[ndarray, "3"] = -lens_from_rig[:3, :3].T @ lens_from_rig[:3, 3]
    for face in lens_faces(0, FACE):
        pose: Float64[ndarray, "4 4"] = face_from_rig(face, lens_from_rig)
        np.testing.assert_allclose(-pose[:3, :3].T @ pose[:3, 3], centre_in_rig, atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3].T[:, 2], lens_from_rig[:3, :3].T @ face.lens_from_face[:, 2], atol=1e-12)
