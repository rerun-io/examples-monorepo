"""The six-face cubemap of an Insta360 frameset: face geometry, the seam, the joined tables through the Rust Mei model."""

from __future__ import annotations

import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, UInt8
from numpy import ndarray

LENS: int = 384
"""Synthetic lens image size: a fifth of the X4 Air's 1920."""
FACE: int = 160
"""Face size of the synthetic tests."""
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


LENS1_FROM_LENS0: Float64[ndarray, "3 3"] = np.diag([-1.0, 1.0, -1.0]) @ np.array(
    [[1.0, 0.0, 0.0], [0.0, np.cos(0.01), -np.sin(0.01)], [0.0, np.sin(0.01), np.cos(0.01)]]
)
"""Lens 1 looks the other way (half a turn about y), tilted 0.6 deg so the rotation path is exercised."""


def test_the_default_face_size_is_800_at_1920_and_a_face_size_must_be_a_multiple_of_32() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, default_face_size

    assert default_face_size(1920) == 800
    assert default_face_size(3840) == 1600
    assert [face.size for face in cube_faces(800)] == [800] * 6
    with pytest.raises(ValueError, match="multiple of 32"):
        cube_faces(810)


def test_the_six_faces_are_proper_rotations_that_look_along_each_cube_axis_once() -> None:
    from dataforge.datasets.insta360_cube import cube_faces

    faces = cube_faces(FACE)
    assert [face.name for face in faces] == ["cube6_front0", "cube6_front1", "cube6_left", "cube6_right", "cube6_up", "cube6_down"]
    looks: list[tuple[float, ...]] = []
    for face in faces:
        rotation: Float64[ndarray, "3 3"] = face.cube_from_face
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        looks.append(tuple(rotation[:, 2]))
        np.testing.assert_allclose(face.intrinsics, [[FACE / 2, 0, (FACE - 1) / 2], [0, FACE / 2, (FACE - 1) / 2], [0, 0, 1]])
    assert looks == [(0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    # Every direction is seen by exactly one face: the largest |coordinate| of a ray picks its face.
    rng = np.random.default_rng(0)
    for ray in rng.normal(size=(200, 3)):
        seen: list[bool] = []
        for face in faces:
            local: Float64[ndarray, "3"] = face.cube_from_face.T @ ray
            seen.append(bool(local[2] > 0 and abs(local[0]) < local[2] and abs(local[1]) < local[2]))
        assert sum(seen) == 1


def test_the_seam_is_the_centre_line_of_the_four_side_faces() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_table

    for face in cube_faces(FACE):
        use0 = face_table(face, [camera(), camera()], LENS1_FROM_LENS0).use_lens0
        half: int = FACE // 2
        expected: dict[str, bool] = {"cube6_front0": True, "cube6_front1": False}
        if face.name in expected:
            assert (use0 == expected[face.name]).all()
            continue
        # Lens 0 holds the half toward +Z0: image right on left, image left on right, image bottom on up, top on down.
        sides = {
            "cube6_left": (slice(None), slice(half, None)),
            "cube6_right": (slice(None), slice(None, half)),
            "cube6_up": (slice(half, None), slice(None)),
            "cube6_down": (slice(None, half), slice(None)),
        }
        lens0 = np.zeros_like(use0)
        lens0[sides[face.name]] = True
        np.testing.assert_array_equal(use0, lens0)


def test_a_face_pixel_outside_the_lens_it_is_read_from_is_refused() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_table

    centre: float = (LENS - 1) / 2.0
    narrow = insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [XI, np.deg2rad(80.0)] + [0.0] * 13)
    with pytest.raises(ValueError, match="outside the lens"):
        face_table(cube_faces(FACE)[2], [narrow, narrow], LENS1_FROM_LENS0)


def test_a_scene_at_infinity_joins_into_the_colour_of_each_face_s_own_rays() -> None:
    """Paint each lens from one colour-of-direction function (cube frame); every joined face shows it on its rays."""
    from dataforge.datasets.insta360_cube import cube_faces, face_table, join

    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_rays: Float64[ndarray, "h w 3"] = unproject(u, v)
    lens0: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays)), 0, 255).astype(np.uint8)
    lens1: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays @ LENS1_FROM_LENS0)), 0, 255).astype(np.uint8)
    for face in cube_faces(FACE):
        table = face_table(face, [camera(), camera()], LENS1_FROM_LENS0)
        expected: Float64[ndarray, "h w 3"] = colour(face.rays() @ face.cube_from_face.T)
        error: Float64[ndarray, "h w 3"] = np.abs(join((lens0, lens1), table).astype(np.float64) - expected)
        # Bilinear interpolation of a smooth field plus the 0.5 rounding of the painted lenses; no seam step.
        assert error.max() < 2.0, f"{face.name}: {error.max():.2f}"


def test_a_lens_mask_joins_nearest_onto_the_face_that_sees_it() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_table, join

    centre: float = (LENS - 1) / 2.0
    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    disc: UInt8[ndarray, "h w"] = (np.hypot(u - centre, v - centre) <= 12.0).astype(np.uint8)
    empty: UInt8[ndarray, "h w"] = np.zeros_like(disc)
    for face in cube_faces(FACE):
        table = face_table(face, [camera(), camera()], LENS1_FROM_LENS0)
        mask: UInt8[ndarray, "n n"] = join((empty, disc), table, nearest=True)
        assert set(np.unique(mask)) <= {0, 1}
        if face.name != "cube6_front1":
            assert not mask.any(), face.name
            continue
        distance: Float64[ndarray, "n n"] = np.hypot(table.map_x[1] - centre, table.map_y[1] - centre)
        assert (distance[mask == 1] <= 12.0 + 0.75).all()
        assert (mask[distance <= 12.0 - 0.75] == 1).all()


def test_every_face_sits_at_the_midpoint_of_the_two_lens_centres() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_from_rig

    lens_from_rig: list[Float64[ndarray, "4 4"]] = [np.eye(4), np.eye(4)]
    lens_from_rig[0][:3, :3] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    lens_from_rig[1][:3, :3] = LENS1_FROM_LENS0 @ lens_from_rig[0][:3, :3]
    centres: list[Float64[ndarray, "3"]] = [np.array([0.01, -0.0166, 0.002]), np.array([0.01, 0.0166, 0.0])]
    for pose, centre in zip(lens_from_rig, centres, strict=True):
        pose[:3, 3] = -pose[:3, :3] @ centre
    for face in cube_faces(FACE):
        pose = face_from_rig(face, lens_from_rig)
        np.testing.assert_allclose(-pose[:3, :3].T @ pose[:3, 3], (centres[0] + centres[1]) / 2, atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3].T, lens_from_rig[0][:3, :3].T @ face.cube_from_face, atol=1e-12)
