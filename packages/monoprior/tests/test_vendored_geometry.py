import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from monopriors.third_party.moge.utils.geometry_torch import depth_map_to_point_map, intrinsics_from_focal_center


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

    actual: torch.Tensor = intrinsics_from_focal_center(fx_tensor, fy_tensor, cx_tensor, cy_tensor)

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

    actual: torch.Tensor = intrinsics_from_focal_center(
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
    depth_hw: torch.Tensor = torch.tensor(depth_values, dtype=torch.float32).reshape(height, width)
    intrinsics_33: torch.Tensor = torch.tensor(
        [[focal_x, 0.0, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    points_hw3: torch.Tensor = depth_map_to_point_map(depth_hw, intrinsics=intrinsics_33)

    torch.testing.assert_close(points_hw3[..., 2], depth_hw, rtol=0.0, atol=1.0e-6)
    normalized_points_hw3: torch.Tensor = points_hw3 / points_hw3[..., 2, None]
    projected_hw3: torch.Tensor = torch.matmul(intrinsics_33, normalized_points_hw3.unsqueeze(-1)).squeeze(-1)
    expected_u: torch.Tensor = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=torch.float32)
    expected_v: torch.Tensor = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=torch.float32)
    expected_u_grid: torch.Tensor
    expected_v_grid: torch.Tensor
    expected_u_grid, expected_v_grid = torch.meshgrid(expected_u, expected_v, indexing="xy")
    expected_uv_hw2: torch.Tensor = torch.stack((expected_u_grid, expected_v_grid), dim=-1)

    torch.testing.assert_close(projected_hw3[..., :2], expected_uv_hw2, rtol=0.0, atol=1.0e-5)
