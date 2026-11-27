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


def test_tile_to_image_places_masks_and_shifts_boxes() -> None:
    import torch

    from posekit.ops.people_masks import tile_to_image, tile_windows

    windows = tile_windows(10, 10, grid=2, overlap=0.2)
    window = windows[3]
    tile_h: int = window.y1 - window.y0
    tile_w: int = window.x1 - window.x0
    masks = torch.zeros((1, tile_h, tile_w), dtype=torch.bool)
    masks[0, -1, -1] = True  # bottom-right pixel of the bottom-right tile
    xyxy = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    placed = tile_to_image(masks, xyxy, window, height=10, width=10)

    assert placed[0].shape == (1, 10, 10)
    assert placed[0][0, 9, 9] and int(placed[0].sum()) == 1
    assert placed[1].tolist() == [[1.0 + window.x0, 2.0 + window.y0, 3.0 + window.x0, 4.0 + window.y0]]


def test_dilate_mask_grows_by_radius_and_zero_is_identity() -> None:
    from posekit.ops.people_masks import dilate_mask

    mask: Bool[ndarray, "h w"] = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    assert np.array_equal(dilate_mask(mask, 0), mask)
    grown: Bool[ndarray, "h w"] = dilate_mask(mask, 3)
    assert grown[10, 13] and grown[7, 10] and not grown[10, 14]
    assert not grown[7, 7]  # disc, not square


def test_dilate_masks_matches_opencv_dilation() -> None:
    import torch

    from posekit.ops.people_masks import dilate_mask, dilate_masks

    masks: Bool[ndarray, "n h w"] = np.random.default_rng(0).random((3, 40, 56)) > 0.995
    masks[0, 0, 0] = masks[1, -1, -1] = True  # borders grow inward only
    for radius in (0, 1, 8):
        grown: Bool[ndarray, "n h w"] = dilate_masks(torch.from_numpy(masks), radius).numpy()
        assert np.array_equal(grown, np.stack([dilate_mask(m, radius) for m in masks]))



@pytest.mark.integration
def test_dilate_masks_on_cuda_matches_opencv_dilation() -> None:
    import torch

    from posekit.ops.people_masks import dilate_mask, dilate_masks

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (the float16 conv path)")
    masks: Bool[ndarray, "n h w"] = np.random.default_rng(1).random((2, 480, 512)) > 0.98
    for radius in (1, 8, 30):
        grown: Bool[ndarray, "n h w"] = dilate_masks(torch.from_numpy(masks).cuda(), radius).cpu().numpy()
        assert np.array_equal(grown, np.stack([dilate_mask(m, radius) for m in masks]))


def test_overlay_tints_masked_pixels_red_only() -> None:
    from posekit.ops.people_masks import overlay_mask

    rgb: UInt8[ndarray, "h w 3"] = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask: Bool[ndarray, "h w"] = np.array([[True, False], [False, False]])
    tinted: UInt8[ndarray, "h w 3"] = overlay_mask(rgb, mask, alpha=0.5)
    assert tinted[0, 0, 0] > 100 and tinted[0, 0, 1] < 100
    assert np.array_equal(tinted[1, 1], rgb[1, 1])


def test_select_entities_globs_excludes_and_rejects_unmatched_patterns() -> None:
    from posekit.apis.people_masks import select_entities

    available: list[str] = [f"/world/rig_00/cam_{i:02d}/pinhole/video" for i in range(12)]
    selected: list[str] = select_entities(available, ("/world/rig_00/cam_*/pinhole/video",), ("/world/rig_00/cam_00/*", "/world/rig_00/cam_01/*"))
    assert selected == available[2:]
    assert select_entities(available, ("/world/rig_00/cam_05/pinhole/video",), ()) == [available[5]]
    with pytest.raises(ValueError, match="matches no VideoStream"):
        select_entities(available, ("/world/rig_01/*",), ())


def test_mask_entity_is_a_sibling_of_the_video() -> None:
    from posekit.apis.people_masks import mask_entity

    assert mask_entity("/world/rig_00/cam_03/pinhole/video") == "/world/rig_00/cam_03/pinhole/people_mask"
