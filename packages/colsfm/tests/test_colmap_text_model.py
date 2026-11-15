# SPDX-License-Identifier: Apache-2.0
"""Behaviour of `colsfm.colmap_text_model`, the pycolmap-free COLMAP text reader.

The defect this pins: `images.txt` writes two lines per image, and an image with
no observations still gets its (empty) second line. A reader that strips empty
lines before taking every second one pairs every later image with an observation
line and drops it. Against the pre-fix parser the two-image fixture below returns
one image; it must return two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from colsfm.colmap_text_model import (
    ColmapImage,
    ColmapModel,
    parse_colmap_images_text,
    parse_colmap_points_text,
    read_colmap_model,
)

IMAGES_TXT_HEADER: str = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    "# Number of images: 2, mean observations per image: 0.5\n"
)
"""The exact comment block COLMAP writes above the image records."""

FIRST_IMAGE_UNOBSERVED: str = (
    f"{IMAGES_TXT_HEADER}"
    "1 1 0 0 0 0 0 0 1 cam_00/000000.jpg\n"
    "\n"  # image 1 observed nothing: COLMAP still writes its (empty) 2D line
    "2 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 cam_01/000000.jpg\n"
    "100.5 200.5 7\n"
)
"""Two-image model whose first image has an empty observation line."""

POINTS_TXT: str = (
    "# 3D point list with one line of data per point:\n"
    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
    "1 0.5 1.5 2.5 10 20 30 0.1 2 0\n"
)
"""One point with a track, as COLMAP writes it."""


def test_empty_observation_line_does_not_shift_pairing() -> None:
    """An image with zero observations must not drop every later image."""
    images: list[ColmapImage] = parse_colmap_images_text(FIRST_IMAGE_UNOBSERVED)

    assert [image.image_id for image in images] == [1, 2]
    assert [image.name for image in images] == ["cam_00/000000.jpg", "cam_01/000000.jpg"]


def test_pose_is_inverted_into_world_from_camera() -> None:
    """``world_T_cam`` is the inverse of COLMAP's stored ``cam_T_world``."""
    images: list[ColmapImage] = parse_colmap_images_text(FIRST_IMAGE_UNOBSERVED)

    np.testing.assert_allclose(images[0].world_T_cam, np.eye(4), atol=1e-12)

    # 90 degrees about +Z, then a translation of (1, 2, 3), inverted.
    cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_T_world[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(images[1].world_T_cam, np.linalg.inv(cam_T_world), atol=1e-9)


def test_an_image_name_may_contain_spaces() -> None:
    """``NAME`` is the last field and is not split, so a spaced path survives."""
    text: str = f"{IMAGES_TXT_HEADER}1 1 0 0 0 0 0 0 1 cam 00/frame 000000.jpg\n\n"

    assert parse_colmap_images_text(text)[0].name == "cam 00/frame 000000.jpg"


def test_trailing_blank_lines_are_tolerated() -> None:
    """Newlines at the end of the file yield no phantom image."""
    images: list[ColmapImage] = parse_colmap_images_text(f"{FIRST_IMAGE_UNOBSERVED}\n\n\n")

    assert [image.image_id for image in images] == [1, 2]


def test_blank_lines_between_the_header_and_the_first_record_are_skipped() -> None:
    """Padding before the records would otherwise offset every pair."""
    images: list[ColmapImage] = parse_colmap_images_text(IMAGES_TXT_HEADER + "\n\n" + FIRST_IMAGE_UNOBSERVED[len(IMAGES_TXT_HEADER) :])

    assert [image.image_id for image in images] == [1, 2]


def test_header_only_model_parses_to_nothing() -> None:
    """A comment-only ``images.txt`` is a valid, empty model."""
    assert parse_colmap_images_text(IMAGES_TXT_HEADER) == []


def test_points_carry_their_positions_and_colours() -> None:
    """Fields 1..3 are the position and 4..6 the colour; the track is ignored."""
    positions, colours = parse_colmap_points_text(POINTS_TXT)

    np.testing.assert_allclose(positions, [[0.5, 1.5, 2.5]])
    np.testing.assert_array_equal(colours, [[10, 20, 30]])
    assert colours.dtype == np.uint8


def test_a_pointless_model_yields_empty_arrays_of_the_right_shape() -> None:
    """Downstream code indexes columns, so an empty cloud is still `[0, 3]`."""
    positions, colours = parse_colmap_points_text("")

    assert positions.shape == (0, 3)
    assert colours.shape == (0, 3)


def test_read_colmap_model_keys_every_registered_image(tmp_path: Path) -> None:
    """The on-disk reader keeps both images, keyed by name, alongside the points."""
    (tmp_path / "images.txt").write_text(FIRST_IMAGE_UNOBSERVED)
    (tmp_path / "points3D.txt").write_text(POINTS_TXT)

    model: ColmapModel = read_colmap_model(tmp_path)

    assert sorted(model.world_T_cam) == ["cam_00/000000.jpg", "cam_01/000000.jpg"]
    np.testing.assert_allclose(model.points_xyz, [[0.5, 1.5, 2.5]])
    np.testing.assert_array_equal(model.points_rgb, [[10, 20, 30]])


def test_a_model_without_points_still_reads(tmp_path: Path) -> None:
    """cuSFM writes `images.txt` first; a missing cloud is empty, not an error."""
    (tmp_path / "images.txt").write_text(FIRST_IMAGE_UNOBSERVED)

    model: ColmapModel = read_colmap_model(tmp_path)

    assert len(model.world_T_cam) == 2
    assert model.points_xyz.shape == (0, 3)


def test_a_binary_model_is_named_rather_than_misread(tmp_path: Path) -> None:
    """Reading `images.bin` needs pycolmap, which is the whole point of not having it."""
    (tmp_path / "images.bin").write_bytes(b"\x00")

    with pytest.raises(NotImplementedError, match="binary COLMAP model"):
        read_colmap_model(tmp_path)


def test_an_empty_directory_is_a_missing_model(tmp_path: Path) -> None:
    """No model at all is a different failure from a model this reader cannot read."""
    with pytest.raises(FileNotFoundError, match="no COLMAP model"):
        read_colmap_model(tmp_path)
