"""Unit tests for the pure parts of the catalog clip reader: frame selection, K scaling, the Vicon lens and its remap."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from open4dgs.catalog_clip import (
    ViconRadialLens,
    fit_max_side,
    resample_frames,
    scale_intrinsics,
    select_frame_indices,
    undistortion_grid,
    valid_mask,
)

# A lens of MAMMA's magnitude (IOI_01 of the breakdance segment, native 2056x1504).
LENS: ViconRadialLens = ViconRadialLens(
    centre=np.array([1033.76, 719.77]), radial=np.array([1.98e-8, -1.2e-14, -3.0e-22]), native_width=2056, native_height=1504
)


def test_select_frame_indices_steps_from_start() -> None:
    available = np.arange(100, dtype=np.int64)
    assert select_frame_indices(10, 4, 3, available).tolist() == [10, 13, 16, 19]


def test_select_frame_indices_rejects_missing_frames() -> None:
    with pytest.raises(ValueError, match="not in the segment"):
        select_frame_indices(95, 4, 2, np.arange(100, dtype=np.int64))
    with pytest.raises(ValueError, match="frame_step"):
        select_frame_indices(0, 4, 0, np.arange(100, dtype=np.int64))


def test_fit_max_side_keeps_aspect() -> None:
    assert fit_max_side(2056, 1504, 1028) == (1028, 752)
    assert fit_max_side(704, 1280, 640) == (352, 640)
    assert fit_max_side(800, 600, None) == (800, 600)
    assert fit_max_side(800, 600, 2000) == (800, 600)


def test_scale_intrinsics_moves_projections_with_the_resize() -> None:
    """A world point's projection under the scaled K is the resized image's coordinate of the same point."""
    K = np.array([[1788.7, 0.0, 1033.76], [0.0, 1790.1, 719.77], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(K, (2056, 1504), (1028, 752))
    point = np.array([0.3, -0.2, 4.0])
    uv = (K @ point)[:2] / point[2]
    uv_scaled = (scaled @ point)[:2] / point[2]
    np.testing.assert_allclose(uv_scaled, (uv + 0.5) * 0.5 - 0.5, atol=1e-9)
    # The image's outer corner maps to the outer corner.
    np.testing.assert_allclose(scaled[0, 2], (1033.76 + 0.5) / 2 - 0.5)


def test_lens_round_trip_on_a_grid() -> None:
    ys, xs = torch.meshgrid(torch.linspace(0, 1503, 40, dtype=torch.float64), torch.linspace(0, 2055, 50, dtype=torch.float64), indexing="ij")
    distorted = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    undistorted = LENS.undistort(distorted)
    assert (undistorted - distorted).abs().max() > 5.0  # the lens is not the identity at the border
    np.testing.assert_allclose(LENS.distort(undistorted).numpy(), distorted.numpy(), atol=1e-6)


def test_lens_matches_dataforge_vicon_lens() -> None:
    """The torch port gives dataforge's ``ViconLens`` answers (both directions)."""
    mamma_source = pytest.importorskip("dataforge.datasets.mamma_source", reason="dataforge is not in this env")
    reference = mamma_source.ViconLens(centre=LENS.centre, radial=LENS.radial)
    points = np.random.default_rng(0).uniform([0.0, 0.0], [2056.0, 1504.0], size=(500, 2))
    np.testing.assert_allclose(LENS.undistort(torch.from_numpy(points)).numpy(), reference.undistort(points), atol=1e-9)
    np.testing.assert_allclose(LENS.distort(torch.from_numpy(points)).numpy(), reference.distort(points), atol=1e-9)


def test_undistortion_grid_resamples_a_synthetic_distorted_image() -> None:
    """Render dots at known undistorted pixels into a distorted image; the remap puts them back where K says."""
    width, height = 2056, 1504
    lens = ViconRadialLens(centre=LENS.centre, radial=LENS.radial * 3.0, native_width=width, native_height=height)
    targets = torch.tensor([[150.0, 120.0], [1900.0, 1400.0], [1030.0, 720.0], [120.0, 1380.0]], dtype=torch.float64)
    sources = lens.distort(targets)
    image = torch.zeros((1, 3, height, width), dtype=torch.uint8)
    for x, y in sources.round().long().tolist():
        image[0, :, y - 3 : y + 4, x - 3 : x + 4] = 255
    grid = undistortion_grid(lens, (width, height), torch.device("cpu"))
    out = resample_frames(image, grid, (width, height))[0, 0].float()
    for x, y in targets.tolist():
        patch = out[int(y) - 12 : int(y) + 13, int(x) - 12 : int(x) + 13]
        ys, xs = torch.nonzero(patch > 64, as_tuple=True)
        assert len(xs) > 0
        assert abs(xs.float().mean().item() - 12) < 1.5 and abs(ys.float().mean().item() - 12) < 1.5


def test_undistortion_grid_with_a_half_size_store() -> None:
    """A video stored at half the lens grid: the stored-grid remap equals the native remap downsampled."""
    lens = ViconRadialLens(centre=np.array([100.0, 80.0]), radial=np.array([2e-6, 0.0, 0.0]), native_width=200, native_height=160)
    native = undistortion_grid(lens, (200, 160), torch.device("cpu"))
    half = undistortion_grid(lens, (100, 80), torch.device("cpu"))
    pooled = F.avg_pool2d(native.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    torch.testing.assert_close(half, pooled, atol=2e-3, rtol=0.0)


def test_pinhole_camera_keeps_every_pixel() -> None:
    frames = torch.randint(0, 256, (2, 3, 40, 60), dtype=torch.uint8)
    torch.testing.assert_close(resample_frames(frames, None, (60, 40)), frames)
    assert resample_frames(frames, None, (30, 20)).shape == (2, 3, 20, 30)
    assert bool(valid_mask(None, (60, 40), (30, 20)).all())


def test_valid_mask_marks_the_lens_border() -> None:
    lens = ViconRadialLens(centre=np.array([100.0, 80.0]), radial=np.array([-8e-6, 0.0, 0.0]), native_width=200, native_height=160)
    mask = valid_mask(undistortion_grid(lens, (200, 160), torch.device("cpu")), (200, 160), (100, 80))
    assert mask.shape == (80, 100)
    assert bool(mask[40, 50]) and not bool(mask.all())


def test_mesh_vertices_equal_the_python_list_read() -> None:
    """The buffer-sliced vertex read equals the old to_pylist read: chunked column, null row, short row, missing frame."""
    import pyarrow as pa

    from open4dgs.catalog_clip import mesh_vertices

    rng = np.random.default_rng(1)
    vec3 = pa.list_(pa.field("item", pa.float32(), nullable=False), 3)
    meshes = [rng.normal(size=(5, 3)).astype(np.float32) for _ in range(4)]
    rows_a = [meshes[0].tolist(), None]
    rows_b = [meshes[1][:3].tolist(), meshes[2].tolist(), meshes[3].tolist()]
    column = pa.chunked_array([pa.array(rows_a, type=pa.list_(vec3)), pa.array(rows_b, type=pa.list_(vec3)).slice(0, 3)])
    rows = [3, None, 0, 1, 2, 4]
    vertices, present = mesh_vertices(column, rows, 5)
    cells = column.to_pylist()
    for slot, row in enumerate(rows):
        cell = None if row is None else cells[row]
        if isinstance(cell, list) and len(cell) == 5:
            assert present[slot]
            np.testing.assert_array_equal(vertices[slot], np.asarray(cell, dtype=np.float32))
        else:
            assert not present[slot] and np.isnan(vertices[slot]).all()
    assert present.tolist() == [True, False, True, False, False, True]
