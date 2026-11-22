"""Pure logic of the foreground masks: entity paths, person crop boxes, gap filling, and shared resampling."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_mask_entity_sits_under_the_pinhole() -> None:
    from open4dgs.masks import mask_entity

    assert mask_entity("rig_03/cam_00") == "/world/rig_03/cam_00/pinhole/masks/foreground"


def _camera() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.eye(4, dtype=np.float32), K


def test_person_box_pads_and_clips() -> None:
    from open4dgs.masks import person_boxes

    world_T_cam, K = _camera()
    # Two points at z=2: x = +-0.2 -> u = 50 +- 10; y = 0 / 0.4 -> v = 40 / 60.
    vertices = np.array([[[-0.2, 0.0, 2.0], [0.2, 0.4, 2.0]]], dtype=np.float32)
    boxes = person_boxes(vertices, world_T_cam, K, (100, 80), pad_fraction=0.5, pad_px=2.0)
    np.testing.assert_allclose(boxes[0], [40 - 12, 40 - 12, 60 + 12, 60 + 12], atol=1e-4)
    clipped = person_boxes(vertices, world_T_cam, K, (100, 80), pad_fraction=5.0, pad_px=0.0)
    np.testing.assert_allclose(clipped[0], [0, 0, 100, 80], atol=1e-4)


def test_person_box_is_nan_without_points_in_front() -> None:
    from open4dgs.masks import person_boxes

    world_T_cam, K = _camera()
    vertices = np.array([[[0.0, 0.0, -1.0]], [[np.nan, np.nan, np.nan]]], dtype=np.float32)
    assert np.isnan(person_boxes(vertices, world_T_cam, K, (100, 80), 0.1, 1.0)).all()


def test_fill_missing_boxes_takes_the_union_of_neighbours() -> None:
    from open4dgs.masks import fill_missing_boxes

    nan = [np.nan] * 4
    boxes = np.array([[0, 0, 10, 10], nan, nan, [5, 2, 20, 8], nan], dtype=np.float32)
    filled = fill_missing_boxes(boxes)
    np.testing.assert_array_equal(filled[1], [0, 0, 20, 10])
    np.testing.assert_array_equal(filled[2], [0, 0, 20, 10])
    np.testing.assert_array_equal(filled[4], [5, 2, 20, 8])
    np.testing.assert_array_equal(filled[0], boxes[0])


def test_masks_and_frames_share_one_resampling() -> None:
    """A mask resampled with resample_images lands on the same pixels as the frame resampled with resample_frames."""
    from open4dgs.catalog_clip import ViconRadialLens, resample_frames, resample_images, undistortion_grid

    lens = ViconRadialLens(centre=np.array([31.5, 23.5]), radial=np.array([2e-5, 0.0, 0.0]), native_width=64, native_height=48)
    grid = undistortion_grid(lens, (64, 48), torch.device("cpu"))
    square = torch.zeros((1, 1, 48, 64))
    square[..., 10:30, 20:44] = 1.0
    frame = (square.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
    mask = resample_images(square, grid, (32, 24))
    rgb = resample_frames(frame, grid, (32, 24))
    np.testing.assert_allclose(mask[0, 0].numpy() * 255, rgb[0, 0].float().numpy(), atol=0.5)
