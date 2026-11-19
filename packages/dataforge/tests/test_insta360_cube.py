"""The six-face cubemap of an Insta360 frameset: face geometry, the seam, the joined tables through the Rust Mei model."""

from __future__ import annotations

import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

LENS: int = 384
"""Synthetic lens image size: a fifth of the X4 Air's 1920."""
FACE: int = 160
"""Face size of the synthetic tests."""
XI: float = 2.315
"""An X4 Air lens's xi."""
FOCAL: float = 1835.37 * LENS / 1920
"""An X4 Air lens's focal at the synthetic lens size."""
HALF_FIELD: float = np.deg2rad(100.0)
"""The X4 Air's half field."""


def camera() -> insta360_io.MeiCamera:
    """A distortion-free Mei lens shaped like an X4 Air lens, so its inverse is closed form."""
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


def lens_pair() -> tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]]:
    """Each lens painted from one colour-of-direction function (cube frame)."""
    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_rays: Float64[ndarray, "h w 3"] = unproject(u, v)
    lens0: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays)), 0, 255).astype(np.uint8)
    lens1: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays @ LENS1_FROM_LENS0)), 0, 255).astype(np.uint8)
    return lens0, lens1


def join_faces(device: str, lens0: UInt8[ndarray, "h w 3"], lens1: UInt8[ndarray, "h w 3"]) -> UInt8[ndarray, "f n n 3"]:
    """``CubeJoin.faces`` of one lens pair on ``device``, back as ``f n n 3``."""
    import torch

    from dataforge.datasets.insta360_cube import CubeJoin, cube_faces, face_table

    tables = [face_table(face, [camera(), camera()], LENS1_FROM_LENS0) for face in cube_faces(FACE)]
    join = CubeJoin(tables, (LENS, LENS), torch.device(device))
    pair = [torch.from_numpy(image).permute(2, 0, 1)[None].to(device) for image in (lens0, lens1)]
    return join.faces(pair[0], pair[1])[:, 0].permute(0, 2, 3, 1).cpu().numpy()


def test_a_scene_at_infinity_joins_into_the_colour_of_each_face_s_own_rays() -> None:
    """Every joined face shows the colour of its own rays."""
    from dataforge.datasets.insta360_cube import cube_faces

    for face, joined in zip(cube_faces(FACE), join_faces("cpu", *lens_pair()), strict=True):
        expected: Float64[ndarray, "h w 3"] = colour(face.rays() @ face.cube_from_face.T)
        error: Float64[ndarray, "h w 3"] = np.abs(joined.astype(np.float64) - expected)
        # Bilinear interpolation of a smooth field plus the 0.5 rounding of the painted lenses; no seam step.
        assert error.max() < 2.0, f"{face.name}: {error.max():.2f}"


def test_one_remap_of_the_stacked_pair_is_cv2_s_remap_of_the_lens_each_pixel_reads() -> None:
    """Noise lenses, so every interpolation weight shows: within 1 of ``cv2.remap`` (fixed-point weights) everywhere."""
    import cv2

    from dataforge.datasets.insta360_cube import cube_faces, face_table

    rng = np.random.default_rng(0)
    lenses = rng.integers(0, 256, size=(2, LENS, LENS, 3), dtype=np.uint8)
    for face, joined in zip(cube_faces(FACE), join_faces("cpu", lenses[0], lenses[1]), strict=True):
        table = face_table(face, [camera(), camera()], LENS1_FROM_LENS0)
        remapped = [cv2.remap(lens, table.map_x, table.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for lens in lenses]
        expected: UInt8[ndarray, "n n 3"] = np.where(table.use_lens0[..., None], remapped[0], remapped[1])
        assert np.abs(joined.astype(np.int16) - expected).max() <= 1, face.name


@pytest.mark.parametrize("mask_size", [LENS, LENS // 4])
def test_a_mask_face_is_cv2_s_nearest_remap_of_the_lens_mask_upsampled_by_contract_v2(mask_size: int) -> None:
    """Random labels at lens size and at a quarter of it: identical to ``cv2.INTER_NEAREST`` of the contract-v2 upsample."""
    import cv2

    from dataforge.datasets.insta360_cube import MaskJoin, cube_faces, face_table

    rng = np.random.default_rng(1)
    masks = rng.integers(0, 4, size=(2, mask_size, mask_size), dtype=np.uint8)
    pick: Int64[ndarray, " h"] = (2 * np.arange(LENS) + 1) * mask_size // (2 * LENS)
    tables = [face_table(face, [camera(), camera()], LENS1_FROM_LENS0) for face in cube_faces(FACE)]
    joined: UInt8[ndarray, "f n n"] = MaskJoin(tables, (LENS, LENS)).faces(masks[0], masks[1])
    for table, face_mask in zip(tables, joined, strict=True):
        remapped = [cv2.remap(mask[np.ix_(pick, pick)], table.map_x, table.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for mask in masks]
        np.testing.assert_array_equal(face_mask, np.where(table.use_lens0, remapped[0], remapped[1]))


@pytest.mark.integration
def test_the_cuda_join_is_the_cpu_join_within_one_level() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    lenses = lens_pair()
    assert np.abs(join_faces("cuda", *lenses).astype(np.int16) - join_faces("cpu", *lenses)).max() <= 1


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
