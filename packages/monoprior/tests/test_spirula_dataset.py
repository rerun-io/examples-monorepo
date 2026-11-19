"""Unit tests for the colsfm -> spirula COLMAP text rewrite."""

import numpy as np
import pytest

from monopriors.apis.spirula_dataset import (
    COLMAP_PIXEL_CENTRE_SHIFT_PX,
    ColmapImageKey,
    colmap_camera_line,
    keep_mask,
    parse_image_key,
    read_colmap_cameras,
    rewrite_image_names,
)

IMAGES_TXT: str = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3
0 1 0 0 0 0.1 0.2 0.3 2 cube6_front0/2262305829
10.5 20.25 7 30 40 -1
5 0.7071 0 0.7071 0 1 2 3 3 cube6_left/2262305829

7 1 0 0 0 0 0 0 2 cube6_front0/2295639162
1 2 -1
"""


def test_rewrite_points_names_at_jpegs_and_keeps_records() -> None:
    text, keys, camera_ids = rewrite_image_names(IMAGES_TXT, points_shift_px=0.0)
    lines: list[str] = [line for line in text.splitlines() if not line.startswith("#")]
    assert lines == [
        "0 1 0 0 0 0.1 0.2 0.3 2 cube6_front0/2262305829.jpg",
        "10.5 20.25 7 30 40 -1",
        "5 0.7071 0 0.7071 0 1 2 3 3 cube6_left/2262305829.jpg",
        "",
        "7 1 0 0 0 0 0 0 2 cube6_front0/2295639162.jpg",
        "1 2 -1",
    ]
    assert keys == {
        0: ColmapImageKey("cube6_front0", 2262305829),
        5: ColmapImageKey("cube6_left", 2262305829),
        7: ColmapImageKey("cube6_front0", 2295639162),
    }
    assert camera_ids == {0: 2, 5: 3, 7: 2}


def test_rewrite_handles_empty_points_line_at_end() -> None:
    _, keys, _ = rewrite_image_names("1 1 0 0 0 0 0 0 1 cube6_up/5\n\n", points_shift_px=0.0)
    assert keys == {1: ColmapImageKey("cube6_up", 5)}


@pytest.mark.parametrize("name", ["cube6_front0", "cube6_front0/12.jpg", "/12", "cube6_front0/abc"])
def test_parse_image_key_rejects_non_keys(name: str) -> None:
    with pytest.raises(ValueError, match="not <face>/<t_ns>"):
        parse_image_key(name)


def test_read_colmap_cameras() -> None:
    cameras = read_colmap_cameras("# c\n2 PINHOLE 480 800 400 400 79.5 399.5\n")
    assert cameras[2].model == "PINHOLE" and (cameras[2].width, cameras[2].height) == (480, 800)
    np.testing.assert_array_equal(cameras[2].params, [400.0, 400.0, 79.5, 399.5])


def test_camera_line_moves_the_principal_point_to_colmaps_pixel_centres() -> None:
    camera = read_colmap_cameras("3 PINHOLE 480 800 400 400 79.5 399.5\n")[3]
    assert COLMAP_PIXEL_CENTRE_SHIFT_PX == 0.5
    assert colmap_camera_line(camera, COLMAP_PIXEL_CENTRE_SHIFT_PX) == "3 PINHOLE 480 800 400.0 400.0 80.0 400.0"
    assert colmap_camera_line(camera, 0.0) == "3 PINHOLE 480 800 400.0 400.0 79.5 399.5"
    with pytest.raises(ValueError, match="PINHOLE only"):
        colmap_camera_line(read_colmap_cameras("1 SIMPLE_RADIAL 10 10 5 5 5 0.1\n")[1], 0.5)


def test_keep_mask_drops_people() -> None:
    np.testing.assert_array_equal(keep_mask(np.array([[0, 1, 0]], dtype=np.uint8)), [[255, 0, 255]])


def test_exported_observations_stay_on_the_exported_projection() -> None:
    camera = read_colmap_cameras("2 PINHOLE 800 800 400 400 399.5 399.5\n")[2]
    point_cam: list[float] = [0.3, -0.2, 2.0]  # identity pose: world = camera frame
    u: float = 400 * 0.3 / 2.0 + 399.5
    v: float = 400 * -0.2 / 2.0 + 399.5
    images_txt: str = f"0 1 0 0 0 0 0 0 2 cube6_front0/5\n{u!r} {v!r} 11\n"

    text, _, _ = rewrite_image_names(images_txt, points_shift_px=COLMAP_PIXEL_CENTRE_SHIFT_PX)
    fx, fy, cx, cy = (float(value) for value in colmap_camera_line(camera, COLMAP_PIXEL_CENTRE_SHIFT_PX).split()[4:])
    x, y, point_id = text.splitlines()[-1].split()

    assert point_id == "11"
    assert float(x) == pytest.approx(fx * point_cam[0] / point_cam[2] + cx, abs=1e-9)
    assert float(y) == pytest.approx(fy * point_cam[1] / point_cam[2] + cy, abs=1e-9)
