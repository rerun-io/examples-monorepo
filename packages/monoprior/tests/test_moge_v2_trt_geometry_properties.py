"""Property tests for the pure pieces of batched MoGe v2 geometry."""

import math

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jaxtyping import Bool, Float32
from torch import Tensor

from monopriors.models.moge_v2_trt_shared import ASPECT_BUCKETS, aspect_buckets, select_aspect_bucket, token_grid_hw
from monopriors.models.monoprior.moge_v2_trt import (
    MoGeV2GeometryGraphOutput,
    MoGeV2GeometryOutput,
    postprocess_moge_v2_geometry,
)
from monopriors.third_party.moge.utils.geometry_torch import normalized_view_plane_uv


@settings(max_examples=200, deadline=None)
@given(
    aspect_ratio=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
    num_tokens=st.integers(min_value=100, max_value=5000),
)
def test_token_grid_is_patch_aligned_and_near_budget(aspect_ratio: float, num_tokens: int) -> None:
    """The grid is 14-pixel aligned, near the token budget, and near the requested aspect."""
    network_hw: tuple[int, int] = token_grid_hw(aspect_ratio, num_tokens)
    token_rows: int = network_hw[0] // 14
    token_cols: int = network_hw[1] // 14

    assert network_hw[0] % 14 == 0 and network_hw[1] % 14 == 0
    assert token_rows >= 1 and token_cols >= 1
    # Rounding each side by at most 0.5 token bounds the product error.
    assert abs(token_rows * token_cols - num_tokens) <= 0.5 * (token_rows + token_cols) + 0.25
    assert abs(math.log(token_cols / token_rows) - math.log(aspect_ratio)) <= math.log(1.0 + 0.5 / min(token_rows, token_cols)) * 2.0


@settings(max_examples=100, deadline=None)
@given(num_tokens=st.integers(min_value=1, max_value=5000))
def test_aspect_buckets_match_each_pinned_aspect_grid(num_tokens: int) -> None:
    """Every generated bucket is the token grid for its declared aspect."""
    expected_buckets: tuple[tuple[int, int], ...] = tuple(
        token_grid_hw(aspect_ratio, num_tokens) for aspect_ratio in (4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0, 1.0)
    )

    assert aspect_buckets(num_tokens) == expected_buckets
    assert len(aspect_buckets(num_tokens)) == 5


@settings(max_examples=200, deadline=None)
@given(
    height=st.integers(min_value=1, max_value=8192),
    width=st.integers(min_value=1, max_value=8192),
    scale=st.integers(min_value=1, max_value=16),
)
def test_select_aspect_bucket_is_scale_invariant_and_nearest(height: int, width: int, scale: int) -> None:
    """Selection returns a bucket, ignores resolution, and minimizes the log-aspect distance."""
    selected_hw: tuple[int, int] = select_aspect_bucket((height, width), ASPECT_BUCKETS)

    assert selected_hw in ASPECT_BUCKETS
    assert select_aspect_bucket((height * scale, width * scale), ASPECT_BUCKETS) == selected_hw
    input_log_aspect: float = math.log(width / height)
    selected_distance: float = abs(math.log(selected_hw[1] / selected_hw[0]) - input_log_aspect)
    for bucket_hw in ASPECT_BUCKETS:
        assert selected_distance <= abs(math.log(bucket_hw[1] / bucket_hw[0]) - input_log_aspect) + 1e-12


def test_every_bucket_selects_itself() -> None:
    """Feeding a bucket's own size back in returns that bucket."""
    for bucket_hw in ASPECT_BUCKETS:
        assert select_aspect_bucket(bucket_hw, ASPECT_BUCKETS) == bucket_hw


@settings(max_examples=10, deadline=None)
@given(
    focal=st.floats(min_value=0.6, max_value=3.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    metric_scale=st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False),
    base_depth=st.floats(min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False),
    slope_u=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    slope_v=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    batch_size=st.integers(min_value=1, max_value=3),
)
def test_postprocess_recovers_synthetic_pinhole_camera(
    focal: float,
    shift: float,
    metric_scale: float,
    base_depth: float,
    slope_u: float,
    slope_v: float,
    batch_size: int,
) -> None:
    """A noiseless tilted-plane scene yields the true focal, shift, depth, and metric scale.

    The scene is built with MoGe's own conventions: view-plane ``uv`` is
    normalized by the image diagonal and ``focal`` is relative to that
    diagonal, so ``points = depth * [u / focal, v / focal, 1]``. The graph
    output is that map divided by ``metric_scale`` with ``shift`` removed
    from Z, exactly what the network is trained to emit.
    """
    assume(abs(slope_u) + abs(slope_v) > 0.2)  # a fronto-parallel plane has a focal/shift ambiguity
    network_hw: tuple[int, int] = (48, 64)
    aspect_ratio: float = network_hw[1] / network_hw[0]
    uv_hw2: Float32[Tensor, "48 64 2"] = normalized_view_plane_uv(network_hw[1], network_hw[0], dtype=torch.float32)
    true_depth_hw: Float32[Tensor, "48 64"] = base_depth + slope_u * uv_hw2[..., 0] + slope_v * uv_hw2[..., 1]
    assume(true_depth_hw.min().item() > 0.3)
    rays_hw3: Float32[Tensor, "48 64 3"] = torch.cat((uv_hw2 / focal, torch.ones_like(uv_hw2[..., :1])), dim=-1)
    true_points_hw3: Float32[Tensor, "48 64 3"] = true_depth_hw[..., None] * rays_hw3
    affine_points_hw3: Float32[Tensor, "48 64 3"] = true_points_hw3 / metric_scale
    affine_points_hw3[..., 2] -= shift
    assume(affine_points_hw3[..., 2].min().item() > 0.1)  # the network emits positive affine Z; the solver assumes it
    normals_hw3: Float32[Tensor, "48 64 3"] = torch.nn.functional.normalize(torch.randn(*network_hw, 3), dim=-1)

    graph_output: MoGeV2GeometryGraphOutput = MoGeV2GeometryGraphOutput(
        points_bhw3=affine_points_hw3[None].expand(batch_size, -1, -1, -1).contiguous(),
        normals_bhw3=normals_hw3[None].expand(batch_size, -1, -1, -1).contiguous(),
        mask_bhw=torch.full((batch_size, *network_hw), 0.9),
        metric_scale_b=torch.full((batch_size,), metric_scale),
    )
    prediction: MoGeV2GeometryOutput = postprocess_moge_v2_geometry(graph_output, network_hw)

    expected_fx: float = focal / 2.0 * math.sqrt(1.0 + aspect_ratio**2) / aspect_ratio
    expected_fy: float = focal / 2.0 * math.sqrt(1.0 + aspect_ratio**2)
    torch.testing.assert_close(prediction.shift_b, torch.full((batch_size,), shift), rtol=0.0, atol=1e-2)
    torch.testing.assert_close(prediction.intrinsics_b33[:, 0, 0], torch.full((batch_size,), expected_fx), rtol=1e-2, atol=0.0)
    torch.testing.assert_close(prediction.intrinsics_b33[:, 1, 1], torch.full((batch_size,), expected_fy), rtol=1e-2, atol=0.0)
    torch.testing.assert_close(prediction.intrinsics_b33[:, :2, 2], torch.full((batch_size, 2), 0.5), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(prediction.depth_bhw, true_depth_hw[None].expand(batch_size, -1, -1), rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(prediction.points_bhw3, true_points_hw3[None].expand(batch_size, -1, -1, -1), rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(prediction.metric_scale_b, torch.full((batch_size,), metric_scale), rtol=0.0, atol=0.0)
    torch.testing.assert_close(prediction.points_bhw3[..., 2], prediction.depth_bhw, rtol=0.0, atol=0.0)
    torch.testing.assert_close(prediction.normals_bhw3, normals_hw3[None].expand(batch_size, -1, -1, -1), rtol=0.0, atol=1e-5)


@settings(max_examples=30, deadline=None)
@given(
    shift=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    output_hw=st.tuples(st.integers(min_value=8, max_value=96), st.integers(min_value=8, max_value=96)),
)
def test_postprocess_invariants_hold_for_arbitrary_graph_output(shift: float, output_hw: tuple[int, int]) -> None:
    """Shapes, unit normals, centered intrinsics, and the depth/mask coupling hold for any input.

    Focal positivity is deliberately not asserted here: the vendored SciPy solve
    can return a negative focal for non-geometric (random) point maps.
    """
    network_hw: tuple[int, int] = (32, 40)
    generator: torch.Generator = torch.Generator().manual_seed(int(abs(shift) * 1000) + output_hw[0] * 100 + output_hw[1])
    points_bhw3: Float32[Tensor, "2 32 40 3"] = torch.randn(2, *network_hw, 3, generator=generator)
    points_bhw3[..., 2] += shift
    graph_output: MoGeV2GeometryGraphOutput = MoGeV2GeometryGraphOutput(
        points_bhw3=points_bhw3,
        normals_bhw3=torch.randn(2, *network_hw, 3, generator=generator),
        mask_bhw=torch.rand(2, *network_hw, generator=generator),
        metric_scale_b=torch.rand(2, generator=generator) + 0.1,
    )
    prediction: MoGeV2GeometryOutput = postprocess_moge_v2_geometry(graph_output, output_hw)

    assert prediction.depth_bhw.shape == (2, *output_hw)
    assert prediction.points_bhw3.shape == (2, *output_hw, 3)
    assert prediction.normals_bhw3.shape == (2, *output_hw, 3)
    assert prediction.mask_bhw.shape == (2, *output_hw)
    for output_tensor in prediction:
        assert torch.isfinite(output_tensor).all()
    torch.testing.assert_close(torch.linalg.vector_norm(prediction.normals_bhw3, dim=-1), torch.ones(2, *output_hw), rtol=0.0, atol=1e-5)
    torch.testing.assert_close(prediction.intrinsics_b33[:, :2, 2], torch.full((2, 2), 0.5), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(prediction.points_bhw3[..., 2], prediction.depth_bhw, rtol=0.0, atol=0.0)
    non_positive_bhw: Bool[Tensor, "2 h w"] = prediction.depth_bhw <= 0.0
    assert (prediction.mask_bhw[non_positive_bhw] == 0.0).all()
    assert (prediction.mask_bhw >= 0.0).all() and (prediction.mask_bhw <= 1.0).all()
