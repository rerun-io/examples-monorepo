"""Pure logic of the foreground masks: entity paths, person crop boxes, gap filling, and shared resampling."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_mask_entity_sits_under_the_pinhole() -> None:
    from open4dgs.masks import mask_entity

    assert mask_entity("rig_03/cam_00") == "/world/rig_03/cam_00/pinhole/people_matte"


def test_matte_leaf_is_posekits() -> None:
    """open4dgs keeps a copy of posekit's leaf name (posekit is not in the open4dgs envs); where posekit imports, they match."""
    from open4dgs.masks import MATTE_NAME

    assert MATTE_NAME == "people_matte"
    mask_layer = pytest.importorskip("posekit.mask_layer", reason="posekit is not in this env")
    assert mask_layer.MATTE_NAME == MATTE_NAME


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


def _lens() -> object:
    from open4dgs.catalog_clip import ViconRadialLens

    return ViconRadialLens(centre=np.array([40.3, 29.1]), radial=np.array([3e-5, -2e-10, 1e-15]), native_width=80, native_height=60)


def _soft_mattes(frames: int, height: int, width: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    blobs = np.zeros((frames, height, width), dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    for t in range(frames):
        cx, cy = rng.uniform(0.3, 0.7) * width, rng.uniform(0.3, 0.7) * height
        blobs[t] = np.clip(1.5 - np.hypot(xs - cx, ys - cy) / (0.2 * width), 0.0, 1.0)
    return torch.from_numpy((blobs * 255).round().astype(np.uint8))


def test_grid_cache_returns_the_same_tensor() -> None:
    from open4dgs.catalog_clip import ViconRadialLens, clear_grid_cache, undistortion_grid

    clear_grid_cache()
    lens = _lens()
    first = undistortion_grid(lens, (80, 60), torch.device("cpu"))
    equal_lens = ViconRadialLens(centre=lens.centre.copy(), radial=lens.radial.copy(), native_width=80, native_height=60)
    assert undistortion_grid(equal_lens, (80, 60), torch.device("cpu")) is first
    other = ViconRadialLens(centre=lens.centre, radial=lens.radial * 2.0, native_width=80, native_height=60)
    assert undistortion_grid(other, (80, 60), torch.device("cpu")) is not first
    assert undistortion_grid(lens, (40, 30), torch.device("cpu")) is not first
    clear_grid_cache()
    rebuilt = undistortion_grid(lens, (80, 60), torch.device("cpu"))
    assert rebuilt is not first and torch.equal(rebuilt, first)


def test_grid_cache_evicts_past_its_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import open4dgs.catalog_clip as catalog_clip

    catalog_clip.clear_grid_cache()
    monkeypatch.setattr(catalog_clip, "GRID_CACHE_BYTES", 80 * 60 * 2 * 4 + 1)  # one 80x60 grid
    first = catalog_clip.undistortion_grid(_lens(), (80, 60), torch.device("cpu"))
    catalog_clip.undistortion_grid(_lens(), (40, 30), torch.device("cpu"))
    assert len(catalog_clip._GRID_CACHE) == 1
    assert catalog_clip.undistortion_grid(_lens(), (80, 60), torch.device("cpu")) is not first
    catalog_clip.clear_grid_cache()


def test_roi_grid_is_cached_per_virtual_camera() -> None:
    from open4dgs.catalog_clip import clear_grid_cache, roi_grid
    from open4dgs.roi_cameras import fit_virtual_camera

    clear_grid_cache()
    K = np.array([[60.0, 0.0, 39.5], [0.0, 60.0, 29.5], [0.0, 0.0, 1.0]])
    points = np.array([[0.1, 0.0, 2.0], [0.3, 0.2, 2.1]], dtype=np.float32)
    camera = fit_virtual_camera(points, K, np.eye(4), (80, 60), 0.15, None)
    grid = roi_grid(_lens(), camera, torch.device("cpu"))
    assert roi_grid(_lens(), camera, torch.device("cpu")) is grid
    assert roi_grid(None, camera, torch.device("cpu")) is not grid
    clear_grid_cache()


def test_chunked_mask_resample_equals_one_shot() -> None:
    """resample_masks (chunked, any output dtype) equals resample_images over the whole stack."""
    from open4dgs.catalog_clip import RESAMPLE_CHUNK, clear_grid_cache, resample_images, undistortion_grid
    from open4dgs.masks import resample_masks

    clear_grid_cache()
    stored = _soft_mattes(RESAMPLE_CHUNK + 5, 60, 80)
    grid = undistortion_grid(_lens(), (80, 60), torch.device("cpu"))
    cpu = torch.device("cpu")
    expected = resample_images(stored[:, None].float() / 255.0, grid, (40, 30))[:, 0].clamp(0.0, 1.0)
    got = resample_masks(stored, grid, (40, 30), cpu, cpu, torch.float32)
    assert got.dtype == torch.float32 and got.shape == (RESAMPLE_CHUNK + 5, 30, 40)
    assert torch.equal(got, expected)
    assert torch.equal(resample_masks(stored, grid, (40, 30), cpu, cpu, torch.float16), expected.half())
    clear_grid_cache()


@pytest.mark.integration
def test_gpu_masks_match_cpu_masks_within_one_level() -> None:
    """Grid built and masks resampled on the GPU agree with the CPU path to 1/255 (needs a GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from open4dgs.catalog_clip import clear_grid_cache, undistortion_grid
    from open4dgs.masks import resample_masks

    clear_grid_cache()
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    stored = _soft_mattes(20, 60, 80)
    cpu_grid = undistortion_grid(_lens(), (80, 60), cpu)
    gpu_grid = undistortion_grid(_lens(), (80, 60), cuda)
    assert gpu_grid.is_cuda
    torch.testing.assert_close(gpu_grid.cpu(), cpu_grid, atol=1e-6, rtol=0.0)
    on_cpu = resample_masks(stored, cpu_grid, (40, 30), cpu, cpu, torch.float32)
    on_gpu = resample_masks(stored, gpu_grid, (40, 30), cuda, cpu, torch.float32)
    assert not on_gpu.is_cuda
    assert float((on_gpu - on_cpu).abs().max()) <= 1.0 / 255.0
    resident = resample_masks(stored, gpu_grid, (40, 30), cuda, cuda, torch.float16)
    assert resident.is_cuda and resident.dtype == torch.float16
    clear_grid_cache()


def test_threaded_png_decode_equals_serial() -> None:
    import cv2

    from open4dgs.masks import decode_mask_pngs

    stored = _soft_mattes(24, 60, 80).numpy()
    blobs = [cv2.imencode(".png", image)[1].tobytes() for image in stored] + [b"not a png"]
    serial = decode_mask_pngs(blobs, threads=1)
    threaded = decode_mask_pngs(blobs, threads=8)
    assert serial[-1] is None and threaded[-1] is None
    for a, b, original in zip(serial[:-1], threaded[:-1], stored, strict=True):
        assert a is not None and b is not None
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, original)
