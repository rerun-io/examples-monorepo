"""Analytic contracts for GausSurf-compatible Gaussian surface rendering."""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from gauss_surf.gaussurf_geometry import (
    gaussian_plane_features,
    plane_depth_from_rendered_features,
    smallest_axis_world_normals,
)

PLANE_DENOMINATOR_CUTOFF: float = 1e-4
"""Denominators below this explicit cutoff are treated as degenerate."""


def _rotation_xyz(rx: float, ry: float, rz: float) -> torch.Tensor:
    """Return a float64 XYZ Euler rotation on CPU."""
    cx: float = math.cos(rx)
    sx: float = math.sin(rx)
    cy: float = math.cos(ry)
    sy: float = math.sin(ry)
    cz: float = math.cos(rz)
    sz: float = math.sin(rz)
    rotation_x_33: torch.Tensor = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=torch.float64,
        device="cpu",
    )
    rotation_y_33: torch.Tensor = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=torch.float64,
        device="cpu",
    )
    rotation_z_33: torch.Tensor = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
        device="cpu",
    )
    return rotation_z_33 @ rotation_y_33 @ rotation_x_33


def _quaternion_align_z(normal_3: torch.Tensor) -> torch.Tensor:
    """Return a WXYZ quaternion that maps the positive Z axis to ``normal_3``."""
    z_axis_3: torch.Tensor = torch.tensor(
        [0.0, 0.0, 1.0], dtype=torch.float64, device="cpu"
    )
    dot: float = float(torch.dot(z_axis_3, normal_3).clamp(-1.0, 1.0).item())
    if dot < -1.0 + 1e-12:
        return torch.tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=torch.float64, device="cpu"
        )
    cross_3: torch.Tensor = torch.linalg.cross(z_axis_3, normal_3)
    quaternion_wxyz_4: torch.Tensor = torch.cat(
        (
            torch.tensor([1.0 + dot], dtype=torch.float64, device="cpu"),
            cross_3,
        )
    )
    return quaternion_wxyz_4 / torch.linalg.vector_norm(quaternion_wxyz_4)


@settings(max_examples=64, deadline=None)
@given(
    rx=st.floats(-math.pi, math.pi, allow_nan=False, allow_infinity=False),
    ry=st.floats(-math.pi, math.pi, allow_nan=False, allow_infinity=False),
    rz=st.floats(-math.pi, math.pi, allow_nan=False, allow_infinity=False),
    tx=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    ty=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    tz=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    ray_x=st.floats(-1.5, 1.5, allow_nan=False, allow_infinity=False),
    ray_y=st.floats(-1.5, 1.5, allow_nan=False, allow_infinity=False),
    denominator=st.one_of(
        st.sampled_from((-2e-4, -1.1e-4, -1.01e-4, 1.01e-4, 1.1e-4, 2e-4)),
        st.floats(-0.95, -2e-4, allow_nan=False, allow_infinity=False),
        st.floats(2e-4, 0.95, allow_nan=False, allow_infinity=False),
    ),
    exact_depth=st.floats(0.25, 100.0, allow_nan=False, allow_infinity=False),
)
def test_plane_depth_matches_exact_world_ray_plane_intersection(
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
    ray_x: float,
    ray_y: float,
    denominator: float,
    exact_depth: float,
) -> None:
    """Plane depth equals the analytic world-space intersection, including near-grazing rays.

    The generated denominator is always greater than ``1e-4`` in magnitude,
    which is the renderer's actual default cutoff. This exercises stable
    near-grazing intersections without silently relaxing the degenerate-plane
    threshold. The ``2e-7`` absolute oracle tolerance covers cancellation when
    the random camera translation is removed in world-space float64 arithmetic.
    """
    rotation_world_from_camera_33: torch.Tensor = _rotation_xyz(rx, ry, rz)
    camera_origin_world_3: torch.Tensor = torch.tensor(
        [tx, ty, tz], dtype=torch.float64, device="cpu"
    )
    ray_camera_3: torch.Tensor = torch.tensor(
        [ray_x, ray_y, 1.0], dtype=torch.float64, device="cpu"
    )
    ray_length: torch.Tensor = torch.linalg.vector_norm(ray_camera_3)
    ray_unit_3: torch.Tensor = ray_camera_3 / ray_length
    perpendicular_3: torch.Tensor = torch.tensor(
        [1.0, 0.0, -ray_x], dtype=torch.float64, device="cpu"
    )
    perpendicular_3 = perpendicular_3 / torch.linalg.vector_norm(perpendicular_3)
    parallel_coefficient: torch.Tensor = torch.tensor(
        denominator, dtype=torch.float64, device="cpu"
    ) / ray_length
    normal_camera_3: torch.Tensor = (
        parallel_coefficient * ray_unit_3
        + math.sqrt(1.0 - parallel_coefficient.item() ** 2) * perpendicular_3
    )
    gaussian_camera_3: torch.Tensor = ray_camera_3 * exact_depth
    normal_world_3: torch.Tensor = rotation_world_from_camera_33 @ normal_camera_3
    tangent_camera_3: torch.Tensor = torch.linalg.cross(
        normal_camera_3, ray_camera_3
    )
    tangent_camera_3 = tangent_camera_3 / torch.linalg.vector_norm(
        tangent_camera_3
    )
    bitangent_camera_3: torch.Tensor = torch.linalg.cross(
        normal_camera_3, tangent_camera_3
    )
    gaussian_means_camera_n3: torch.Tensor = torch.stack(
        (
            gaussian_camera_3,
            gaussian_camera_3 + 0.25 * tangent_camera_3,
            gaussian_camera_3 - 0.4 * tangent_camera_3 + 0.1 * bitangent_camera_3,
        )
    )
    gaussian_means_world_n3: torch.Tensor = (
        gaussian_means_camera_n3 @ rotation_world_from_camera_33.T
        + camera_origin_world_3
    )
    plane_offset_world: torch.Tensor = normal_world_3 @ gaussian_means_world_n3[0]
    ray_world_3: torch.Tensor = rotation_world_from_camera_33 @ ray_camera_3
    oracle_depth: torch.Tensor = (
        plane_offset_world - normal_world_3 @ camera_origin_world_3
    ) / (normal_world_3 @ ray_world_3)

    world_to_camera_44: torch.Tensor = torch.eye(
        4, dtype=torch.float64, device="cpu"
    )
    world_to_camera_44[:3, :3] = rotation_world_from_camera_33.T
    world_to_camera_44[:3, 3] = (
        -rotation_world_from_camera_33.T @ camera_origin_world_3
    )
    log_scales_n3: torch.Tensor = torch.tensor(
        [[0.0, 0.0, -2.0]], dtype=torch.float64, device="cpu"
    ).repeat(3, 1)
    quaternion_wxyz_4: torch.Tensor = _quaternion_align_z(normal_world_3)
    quaternions_wxyz_n4: torch.Tensor = quaternion_wxyz_4.repeat(3, 1)
    gaussian_features_n4: torch.Tensor = gaussian_plane_features(
        gaussian_means_world_n3,
        log_scales_n3,
        quaternions_wxyz_n4,
        world_to_camera_44,
    )
    gaussian_weights_n1: torch.Tensor = torch.tensor(
        [[0.2], [0.3], [0.5]], dtype=torch.float64, device="cpu"
    )
    features_hw4: torch.Tensor = (
        gaussian_features_n4 * gaussian_weights_n1
    ).sum(dim=0).reshape(1, 1, 4)
    alpha_hw1: torch.Tensor = torch.ones(
        (1, 1, 1), dtype=torch.float64, device="cpu"
    )
    intrinsics_33: torch.Tensor = torch.tensor(
        [[1.0, 0.0, 0.5 - ray_x], [0.0, 1.0, 0.5 - ray_y], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
        device="cpu",
    )

    result = plane_depth_from_rendered_features(
        features_hw4,
        alpha_hw1,
        intrinsics_33,
        min_denominator=PLANE_DENOMINATOR_CUTOFF,
        min_depth=0.0,
        max_depth=1e6,
    )

    assert result.valid_hw1.item()
    torch.testing.assert_close(
        gaussian_means_world_n3 @ normal_world_3,
        plane_offset_world.expand(3),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.depth_hw1[0, 0, 0], oracle_depth, rtol=2e-9, atol=2e-7
    )


def test_plane_depth_uses_the_documented_degenerate_cutoff() -> None:
    """A denominator below the exact cutoff is invalid; a larger grazing ray is valid."""
    intrinsics_33: torch.Tensor = torch.eye(
        3, dtype=torch.float64, device="cpu"
    )
    alpha_hw1: torch.Tensor = torch.ones(
        (1, 2, 1), dtype=torch.float64, device="cpu"
    )
    below_cutoff: float = PLANE_DENOMINATOR_CUTOFF / 2.0
    above_cutoff: float = PLANE_DENOMINATOR_CUTOFF * 2.0
    features_hw4: torch.Tensor = torch.tensor(
        [
            [
                [1.0, 0.0, -0.5 + below_cutoff, 2.0 * below_cutoff],
                [1.0, 0.0, -1.5 + above_cutoff, 2.0 * above_cutoff],
            ]
        ],
        dtype=torch.float64,
        device="cpu",
    )

    result = plane_depth_from_rendered_features(
        features_hw4,
        alpha_hw1,
        intrinsics_33,
        min_denominator=PLANE_DENOMINATOR_CUTOFF,
        min_depth=0.0,
    )

    assert not result.valid_hw1[0, 0, 0].item()
    assert result.valid_hw1[0, 1, 0].item()
    torch.testing.assert_close(
        result.depth_hw1[0, 1, 0],
        torch.tensor(2.0, dtype=torch.float64, device="cpu"),
    )


def test_smallest_axis_normal_uses_rotation_column() -> None:
    log_scales = torch.tensor([[0.0, -3.0, -1.0]])
    identity_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    normal = smallest_axis_world_normals(log_scales, identity_wxyz)

    torch.testing.assert_close(normal, torch.tensor([[0.0, 1.0, 0.0]]))


def test_gaussian_plane_features_face_along_camera_ray() -> None:
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    log_scales = torch.tensor([[1.0, 1.0, -2.0], [1.0, 1.0, -2.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    world_to_camera = torch.eye(4)

    features = gaussian_plane_features(means, log_scales, quats, world_to_camera)

    torch.testing.assert_close(features[:, :3], torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1))
    torch.testing.assert_close(features[:, 3], torch.tensor([2.0, 3.0]))


def test_plane_intersection_depth_matches_frontal_and_slanted_planes() -> None:
    intrinsics = torch.tensor(
        [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]
    )
    alpha = torch.ones((3, 3, 1))

    frontal_features = torch.zeros((3, 3, 4))
    frontal_features[..., 2] = 1.0
    frontal_features[..., 3] = 2.0
    frontal = plane_depth_from_rendered_features(frontal_features, alpha, intrinsics)
    torch.testing.assert_close(frontal.depth_hw1, torch.full((3, 3, 1), 2.0))
    assert frontal.valid_hw1.all()

    root_two = 2.0**0.5
    slanted_features = torch.zeros((3, 3, 4))
    slanted_features[..., 0] = 1.0 / root_two
    slanted_features[..., 2] = 1.0 / root_two
    slanted_features[..., 3] = 2.0 / root_two
    slanted = plane_depth_from_rendered_features(slanted_features, alpha, intrinsics)
    # The right-most center-row ray is [0.5, 0, 1], so z = 2 / 1.5.
    torch.testing.assert_close(slanted.depth_hw1[1, 2, 0], torch.tensor(4.0 / 3.0))


def test_plane_depth_masks_low_alpha_grazing_and_negative_intersections() -> None:
    intrinsics = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
    )
    alpha = torch.tensor([[[1.0], [0.001], [1.0]]])
    features = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, -2.0],
        ]]
    )

    result = plane_depth_from_rendered_features(
        features,
        alpha,
        intrinsics,
        min_alpha=0.01,
    )

    assert not result.valid_hw1.any()
    assert torch.isfinite(result.depth_hw1).all()
    torch.testing.assert_close(result.depth_hw1, torch.zeros_like(result.depth_hw1))


def test_plane_depth_has_finite_gradients() -> None:
    intrinsics = torch.eye(3)
    alpha = torch.ones((1, 1, 1))
    features = torch.tensor([[[0.0, 0.0, 1.0, 2.0]]], requires_grad=True)

    result = plane_depth_from_rendered_features(features, alpha, intrinsics)
    result.depth_hw1.sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
