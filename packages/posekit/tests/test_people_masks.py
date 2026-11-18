"""Unit tests for the people-mask post-processing (synthetic arrays; no GPU, no weights)."""

import numpy as np
import pytest
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray


def test_union_instances_ors_per_frame_and_drops_small_instances() -> None:
    from posekit.ops.people_masks import union_instances

    masks: Bool[ndarray, "n h w"] = np.zeros((3, 8, 8), dtype=bool)
    masks[0, 0:4, 0:4] = True  # frame 0, 16 px
    masks[1, 6:8, 6:7] = True  # frame 0, 2 px: below min area
    masks[2, 2:6, 2:6] = True  # frame 2, 16 px
    frame_indices: Int64[ndarray, "n"] = np.array([0, 0, 2], dtype=np.int64)

    union: Bool[ndarray, "f h w"] = union_instances(masks, frame_indices, n_frames=3, min_area_px=4)

    assert union.shape == (3, 8, 8)
    assert union[0].sum() == 16 and not union[0, 6:8, 6:7].any()
    assert not union[1].any()
    assert union[2].sum() == 16


def test_union_instances_without_detections_is_empty() -> None:
    from posekit.ops.people_masks import union_instances

    union: Bool[ndarray, "f h w"] = union_instances(np.zeros((0, 5, 7), dtype=bool), np.zeros((0,), dtype=np.int64), n_frames=2, min_area_px=0)
    assert union.shape == (2, 5, 7) and not union.any()


@pytest.mark.parametrize(("height", "width"), [(1920, 1920), (800, 480), (480, 800), (7, 5)])
def test_tile_windows_cover_the_image_with_overlap(height: int, width: int) -> None:
    from posekit.ops.people_masks import tile_windows

    windows = tile_windows(height, width, grid=2, overlap=0.2)
    assert len(windows) == 4
    covered: Int64[ndarray, "h w"] = np.zeros((height, width), dtype=np.int64)
    for window in windows:
        assert 0 <= window.y0 < window.y1 <= height and 0 <= window.x0 < window.x1 <= width
        covered[window.y0 : window.y1, window.x0 : window.x1] += 1
    assert covered.min() >= 1
    assert covered.max() == 4  # the centre lies in all four tiles
    # One tile size per image, so tiles batch together.
    assert len({(w.y1 - w.y0, w.x1 - w.x0) for w in windows}) == 1


def test_tile_windows_grid_one_is_empty() -> None:
    from posekit.ops.people_masks import tile_windows

    assert tile_windows(100, 100, grid=1, overlap=0.2) == []


def test_paste_tiles_ors_tile_masks_into_place() -> None:
    from posekit.ops.people_masks import paste_tiles, tile_windows

    full: Bool[ndarray, "h w"] = np.zeros((10, 10), dtype=bool)
    full[0, 0] = True
    windows = tile_windows(10, 10, grid=2, overlap=0.2)
    tile_h: int = windows[0].y1 - windows[0].y0
    tile_w: int = windows[0].x1 - windows[0].x0
    tile_masks: Bool[ndarray, "t th tw"] = np.zeros((4, tile_h, tile_w), dtype=bool)
    tile_masks[3, -1, -1] = True  # bottom-right pixel of the bottom-right tile

    merged: Bool[ndarray, "h w"] = paste_tiles(full, tile_masks, windows)

    assert merged[0, 0] and merged[9, 9]
    assert merged.sum() == 2
    assert full.sum() == 1  # the input is not modified


def test_dilate_mask_grows_by_radius_and_zero_is_identity() -> None:
    from posekit.ops.people_masks import dilate_mask

    mask: Bool[ndarray, "h w"] = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    assert np.array_equal(dilate_mask(mask, 0), mask)
    grown: Bool[ndarray, "h w"] = dilate_mask(mask, 3)
    assert grown[10, 13] and grown[7, 10] and not grown[10, 14]
    assert not grown[7, 7]  # disc, not square


def test_selection_windows_group_frames_that_share_a_decode_run() -> None:
    from posekit.ops.people_masks import selection_windows

    keyframes: list[bool] = [i % 4 == 0 for i in range(16)]
    windows = selection_windows(keyframes, [1, 2, 9, 15])
    assert [(w.first, w.indices) for w in windows] == [(0, (1, 2)), (8, (9,)), (12, (15,))]


def test_selection_windows_rejects_out_of_range_and_missing_keyframe() -> None:
    from posekit.ops.people_masks import selection_windows

    with pytest.raises(ValueError, match="out of range"):
        selection_windows([True, False], [2])
    with pytest.raises(ValueError, match="keyframe"):
        selection_windows([False, True], [0])


def test_overlay_tints_masked_pixels_red_only() -> None:
    from posekit.ops.people_masks import overlay_mask

    rgb: UInt8[ndarray, "h w 3"] = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask: Bool[ndarray, "h w"] = np.array([[True, False], [False, False]])
    tinted: UInt8[ndarray, "h w 3"] = overlay_mask(rgb, mask, alpha=0.5)
    assert tinted[0, 0, 0] > 100 and tinted[0, 0, 1] < 100
    assert np.array_equal(tinted[1, 1], rgb[1, 1])
