import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from monopriors.third_party import utils3d


def _manual_torch_sliding_window(
    x: torch.Tensor,
    window_size: tuple[int, int],
    stride: tuple[int, int],
) -> torch.Tensor:
    """Build spatial windows from Tensor.unfold semantics, width first."""
    windows_width_last: torch.Tensor = x.unfold(-1, window_size[1], stride[1])
    windows_reversed: torch.Tensor = windows_width_last.unfold(-3, window_size[0], stride[0])
    return torch.movedim(windows_reversed, (-2, -1), (-1, -2))


@settings(max_examples=25, deadline=None)
@given(
    data=st.data(),
    channels=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
)
def test_numpy_sliding_window_matches_numpy_builtin(
    data: st.DataObject,
    channels: int | None,
    height: int,
    width: int,
) -> None:
    window_height: int = data.draw(st.integers(min_value=1, max_value=height), label="window_height")
    window_width: int = data.draw(st.integers(min_value=1, max_value=width), label="window_width")
    stride_height: int = data.draw(st.integers(min_value=1, max_value=3), label="stride_height")
    stride_width: int = data.draw(st.integers(min_value=1, max_value=3), label="stride_width")
    shape: tuple[int, ...] = (height, width) if channels is None else (channels, height, width)
    element_count: int = int(np.prod(shape))
    values: list[float] = data.draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
            min_size=element_count,
            max_size=element_count,
        ),
        label="values",
    )
    x: np.ndarray = np.asarray(values, dtype=np.float32).reshape(shape)
    window_size: tuple[int, int] = (window_height, window_width)
    stride: tuple[int, int] = (stride_height, stride_width)

    actual: np.ndarray = utils3d.np.sliding_window(x, window_size, stride, axis=(-2, -1))
    builtin_windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=(-2, -1))
    stride_slices: tuple[slice, ...] = (slice(None),) * (x.ndim - 2) + (
        slice(None, None, stride_height),
        slice(None, None, stride_width),
    )
    expected: np.ndarray = builtin_windows[stride_slices]

    np.testing.assert_array_equal(actual, expected)


@settings(max_examples=25, deadline=None)
@given(
    data=st.data(),
    channels=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
)
def test_torch_sliding_window_matches_unfold_and_numpy(
    data: st.DataObject,
    channels: int | None,
    height: int,
    width: int,
) -> None:
    window_height: int = data.draw(st.integers(min_value=1, max_value=height), label="window_height")
    window_width: int = data.draw(st.integers(min_value=1, max_value=width), label="window_width")
    stride_height: int = data.draw(st.integers(min_value=1, max_value=3), label="stride_height")
    stride_width: int = data.draw(st.integers(min_value=1, max_value=3), label="stride_width")
    shape: tuple[int, ...] = (height, width) if channels is None else (channels, height, width)
    element_count: int = int(np.prod(shape))
    values: list[float] = data.draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
            min_size=element_count,
            max_size=element_count,
        ),
        label="values",
    )
    x_numpy: np.ndarray = np.asarray(values, dtype=np.float32).reshape(shape)
    x_torch: torch.Tensor = torch.from_numpy(x_numpy)
    window_size: tuple[int, int] = (window_height, window_width)
    stride: tuple[int, int] = (stride_height, stride_width)

    actual: torch.Tensor = utils3d.pt.sliding_window(x_torch, window_size, stride, dim=(-2, -1))
    expected_unfold: torch.Tensor = _manual_torch_sliding_window(x_torch, window_size, stride)
    expected_numpy: np.ndarray = utils3d.np.sliding_window(x_numpy, window_size, stride, axis=(-2, -1))

    torch.testing.assert_close(actual, expected_unfold, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual.numpy(), expected_numpy)


@settings(max_examples=25, deadline=None)
@given(
    height=st.integers(min_value=1, max_value=32),
    width=st.integers(min_value=1, max_value=32),
)
def test_torch_uv_map_uses_normalized_pixel_centers(height: int, width: int) -> None:
    expected_u: torch.Tensor = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=torch.float32)
    expected_v: torch.Tensor = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=torch.float32)
    expected_u_grid: torch.Tensor
    expected_v_grid: torch.Tensor
    expected_u_grid, expected_v_grid = torch.meshgrid(expected_u, expected_v, indexing="xy")
    expected: torch.Tensor = torch.stack((expected_u_grid, expected_v_grid), dim=-1)

    actual: torch.Tensor = utils3d.pt.uv_map(height, width)
    actual_numpy: np.ndarray = utils3d.np.uv_map(height, width)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_numpy, actual.numpy(), rtol=0.0, atol=np.finfo(np.float32).eps)


@settings(max_examples=25, deadline=None)
@given(
    fx=st.floats(min_value=0.125, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
    fy=st.floats(min_value=0.125, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
    cx=st.floats(min_value=0.125, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
    cy=st.floats(min_value=0.125, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
)
def test_intrinsics_from_scalar_tensors_matches_explicit_matrix(fx: float, fy: float, cx: float, cy: float) -> None:
    fx_tensor: torch.Tensor = torch.tensor(fx, dtype=torch.float32)
    fy_tensor: torch.Tensor = torch.tensor(fy, dtype=torch.float32)
    cx_tensor: torch.Tensor = torch.tensor(cx, dtype=torch.float32)
    cy_tensor: torch.Tensor = torch.tensor(cy, dtype=torch.float32)
    expected: torch.Tensor = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    actual: torch.Tensor = utils3d.pt.intrinsics_from_focal_center(fx_tensor, fy_tensor, cx_tensor, cy_tensor)

    assert actual.shape == (3, 3)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@settings(max_examples=25, deadline=None)
@given(
    values=st.lists(
        st.tuples(
            st.floats(min_value=0.125, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
            st.floats(min_value=0.125, max_value=100.0, allow_nan=False, allow_infinity=False, width=32),
            st.floats(min_value=0.125, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
            st.floats(min_value=0.125, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_intrinsics_from_batched_tensors_preserves_batch(values: list[tuple[float, float, float, float]]) -> None:
    parameters: torch.Tensor = torch.tensor(values, dtype=torch.float32)
    batch_size: int = len(values)
    expected: torch.Tensor = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
    expected[:, 0, 0] = parameters[:, 0]
    expected[:, 1, 1] = parameters[:, 1]
    expected[:, 0, 2] = parameters[:, 2]
    expected[:, 1, 2] = parameters[:, 3]
    expected[:, 2, 2] = 1.0

    actual: torch.Tensor = utils3d.pt.intrinsics_from_focal_center(
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
    )

    assert actual.shape == (batch_size, 3, 3)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@settings(max_examples=25, deadline=None)
@given(
    data=st.data(),
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
    focal_x=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False, width=32),
    focal_y=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False, width=32),
    center_x=st.floats(min_value=0.125, max_value=0.875, allow_nan=False, allow_infinity=False, width=32),
    center_y=st.floats(min_value=0.125, max_value=0.875, allow_nan=False, allow_infinity=False, width=32),
)
def test_depth_map_to_point_map_round_trips_normalized_pixel_centers(
    data: st.DataObject,
    height: int,
    width: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
) -> None:
    pixel_count: int = height * width
    depth_values: list[float] = data.draw(
        st.lists(
            st.floats(min_value=0.125, max_value=10.0, allow_nan=False, allow_infinity=False, width=32),
            min_size=pixel_count,
            max_size=pixel_count,
        ),
        label="depth_values",
    )
    depth: torch.Tensor = torch.tensor(depth_values, dtype=torch.float32).reshape(height, width)
    intrinsics: torch.Tensor = torch.tensor(
        [[focal_x, 0.0, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    points: torch.Tensor = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

    torch.testing.assert_close(points[..., 2], depth, rtol=0.0, atol=1.0e-6)
    normalized_points: torch.Tensor = points / points[..., 2, None]
    projected: torch.Tensor = torch.matmul(intrinsics, normalized_points.unsqueeze(-1)).squeeze(-1)
    expected_u: torch.Tensor = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=torch.float32)
    expected_v: torch.Tensor = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=torch.float32)
    expected_u_grid: torch.Tensor
    expected_v_grid: torch.Tensor
    expected_u_grid, expected_v_grid = torch.meshgrid(expected_u, expected_v, indexing="xy")
    expected_uv: torch.Tensor = torch.stack((expected_u_grid, expected_v_grid), dim=-1)

    # The shim uses normalized intrinsics: K projects to pixel-center UV values in [0, 1], not pixel units.
    torch.testing.assert_close(projected[..., :2], expected_uv, rtol=0.0, atol=1.0e-5)


@settings(max_examples=25, deadline=None)
@given(
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
    target_height=st.integers(min_value=1, max_value=12),
    target_width=st.integers(min_value=1, max_value=12),
)
def test_masked_nearest_resize_full_mask_selects_input_values(
    height: int,
    width: int,
    target_height: int,
    target_width: int,
) -> None:
    image_a: np.ndarray = np.arange(height * width, dtype=np.float32).reshape(height, width)
    image_b: np.ndarray = np.stack((image_a + 1000.0, image_a + 2000.0), axis=-1)
    mask: np.ndarray = np.ones((height, width), dtype=bool)

    result: tuple[np.ndarray, np.ndarray, np.ndarray] = utils3d.np.masked_nearest_resize(
        image_a,
        image_b,
        mask=mask,
        size=(target_height, target_width),
    )
    output_a: np.ndarray = result[0]
    output_b: np.ndarray = result[1]
    output_mask: np.ndarray = result[2]

    assert np.all(output_mask)
    assert np.all(np.isin(output_a, image_a))
    assert np.all(np.isin(output_b[..., 0], image_b[..., 0]))
    assert np.all(np.isin(output_b[..., 1], image_b[..., 1]))


@settings(max_examples=25, deadline=None)
@given(
    data=st.data(),
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=2, max_value=8),
    target_height=st.integers(min_value=1, max_value=12),
    target_width=st.integers(min_value=1, max_value=12),
)
def test_masked_nearest_resize_never_selects_masked_nan_values(
    data: st.DataObject,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
) -> None:
    pixel_count: int = height * width
    valid_ids: set[int] = data.draw(
        st.sets(st.integers(min_value=0, max_value=pixel_count - 1), min_size=1, max_size=pixel_count - 1),
        label="valid_ids",
    )
    mask: np.ndarray = np.zeros(pixel_count, dtype=bool)
    mask[list(valid_ids)] = True
    mask = mask.reshape(height, width)
    image_a: np.ndarray = np.arange(pixel_count, dtype=np.float32).reshape(height, width)
    image_b: np.ndarray = np.stack((image_a + 1000.0, image_a + 2000.0), axis=-1)
    image_a[~mask] = np.nan
    image_b[~mask] = np.nan

    result: tuple[np.ndarray, np.ndarray, np.ndarray] = utils3d.np.masked_nearest_resize(
        image_a,
        image_b,
        mask=mask,
        size=(target_height, target_width),
    )
    output_a: np.ndarray = result[0]
    output_b: np.ndarray = result[1]
    output_mask: np.ndarray = result[2]

    assert not np.isnan(output_a[output_mask]).any()
    assert not np.isnan(output_b[output_mask]).any()


@settings(max_examples=25, deadline=None)
@given(
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
    target_height=st.integers(min_value=1, max_value=12),
    target_width=st.integers(min_value=1, max_value=12),
)
def test_masked_nearest_resize_empty_mask_marks_every_output_invalid(
    height: int,
    width: int,
    target_height: int,
    target_width: int,
) -> None:
    image_a: np.ndarray = np.arange(height * width, dtype=np.float32).reshape(height, width)
    image_b: np.ndarray = np.stack((image_a + 1000.0, image_a + 2000.0), axis=-1)
    mask: np.ndarray = np.zeros((height, width), dtype=bool)

    result: tuple[np.ndarray, np.ndarray, np.ndarray] = utils3d.np.masked_nearest_resize(
        image_a,
        image_b,
        mask=mask,
        size=(target_height, target_width),
    )
    output_mask: np.ndarray = result[2]

    # With no valid source, sampled values are unspecified; the mask marks every output invalid.
    assert not np.any(output_mask)
