"""Unit tests for the colsfm -> spirula COLMAP text rewrite."""

import numpy as np
import pytest

from monopriors.apis.spirula_dataset import ColmapImageKey, parse_image_key, read_colmap_cameras, rewrite_image_names

IMAGES_TXT: str = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3
0 1 0 0 0 0.1 0.2 0.3 2 cube_0_front/2262305829
10.5 20.25 7 30 40 -1
5 0.7071 0 0.7071 0 1 2 3 3 cube_0_left/2262305829

7 1 0 0 0 0 0 0 2 cube_0_front/2295639162
1 2 -1
"""


def test_rewrite_points_names_at_jpegs_and_keeps_records() -> None:
    text, keys, camera_ids = rewrite_image_names(IMAGES_TXT)
    lines: list[str] = [line for line in text.splitlines() if not line.startswith("#")]
    assert lines == [
        "0 1 0 0 0 0.1 0.2 0.3 2 cube_0_front/2262305829.jpg",
        "10.5 20.25 7 30 40 -1",
        "5 0.7071 0 0.7071 0 1 2 3 3 cube_0_left/2262305829.jpg",
        "",
        "7 1 0 0 0 0 0 0 2 cube_0_front/2295639162.jpg",
        "1 2 -1",
    ]
    assert keys == {0: ColmapImageKey("cube_0_front", 2262305829), 5: ColmapImageKey("cube_0_left", 2262305829), 7: ColmapImageKey("cube_0_front", 2295639162)}
    assert camera_ids == {0: 2, 5: 3, 7: 2}


def test_rewrite_handles_empty_points_line_at_end() -> None:
    _, keys, _ = rewrite_image_names("1 1 0 0 0 0 0 0 1 cube_1_up/5\n\n")
    assert keys == {1: ColmapImageKey("cube_1_up", 5)}


@pytest.mark.parametrize("name", ["cube_0_front", "cube_0_front/12.jpg", "/12", "cube_0_front/abc"])
def test_parse_image_key_rejects_non_keys(name: str) -> None:
    with pytest.raises(ValueError, match="not <face>/<t_ns>"):
        parse_image_key(name)


def test_read_colmap_cameras() -> None:
    cameras = read_colmap_cameras("# c\n2 PINHOLE 480 800 400 400 79.5 399.5\n")
    assert cameras[2].model == "PINHOLE" and (cameras[2].width, cameras[2].height) == (480, 800)
    np.testing.assert_array_equal(cameras[2].params, [400.0, 400.0, 79.5, 399.5])
