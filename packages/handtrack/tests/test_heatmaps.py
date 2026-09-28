import torch
from jaxtyping import Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
from handtrack.labels.heatmaps import (
    DISTANCE_RANGE_MM,
    crop_to_heatmap,
    decode_distance,
    decode_heatmaps,
    heatmap_to_crop,
    render_distance,
    render_heatmaps,
)


def measure_generic_distance_extent(n: int = 4096) -> float:
    """Seed 234; uniform joint-limit poses, camera directions and 0.25..1 m distances."""
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(234)
    angles: Float32[Tensor, 'b 22'] = model.joint_limits[:, 0] + torch.rand(n, 22, generator=generator) * (model.joint_limits[:, 1] - model.joint_limits[:, 0])
    direction: Float32[Tensor, 'b 3'] = torch.randn(n, 3, generator=generator)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    translation: Float32[Tensor, 'b 3'] = direction * (0.25 + 0.75 * torch.rand(n, 1, generator=generator))
    pose: HandPose = HandPose(torch.eye(3).expand(n, -1, -1), translation, angles)
    distances: Float32[Tensor, 'b 21'] = torch.linalg.vector_norm(landmarks(model, pose, Side.LEFT), dim=-1) * 1000.0
    return float((distances - distances.mean(dim=-1, keepdim=True)).abs().max())


def test_distance_range_has_margin_for_generic_geometry() -> None:
    maximum: float = measure_generic_distance_extent()
    print(f'Generic-hand sampled maximum |d_rel|: {maximum:.6f} mm')
    assert 50.0 < maximum < DISTANCE_RANGE_MM / 1.1


def test_pixel_centres_and_heatmap_round_trip_including_edges() -> None:
    corners: Float32[Tensor, '2 2'] = torch.tensor([[-0.5, -0.5], [95.5, 95.5]])
    torch.testing.assert_close(crop_to_heatmap(corners), torch.tensor([[-0.5, -0.5], [17.5, 17.5]]))
    points: Float32[Tensor, '32 21 2'] = torch.rand(32, 21, 2, generator=torch.Generator().manual_seed(19)) * 96
    points[0, :4] = torch.tensor([[0.0, 0.0], [95.99, 95.99], [0.0, 95.99], [95.99, 0.0]])
    torch.testing.assert_close(heatmap_to_crop(crop_to_heatmap(points)), points, atol=1e-5, rtol=1e-5)
    decoded: tuple[Float32[Tensor, '32 21 2'], Float32[Tensor, '32 21']] = decode_heatmaps(render_heatmaps(points))
    error: float = float((decoded[0] - points).abs().max())
    print(f'Heatmap max crop-coordinate error: {error:.8f} px')
    assert error < 0.5
    assert (decoded[1] > 0).all() and (decoded[1] <= 1).all()


def test_gaussian_peak_tail_and_empty_heatmaps() -> None:
    points: Float32[Tensor, '1 3 2'] = heatmap_to_crop(torch.tensor([[[8.0, 9.0], [-1.0, 5.0], [-1000.0, -1000.0]]]))
    maps: Float32[Tensor, '1 3 18 18'] = render_heatmaps(points)
    assert maps[0, 0, 9, 8] == 1
    torch.testing.assert_close(maps[0, 0, 9, 9], torch.tensor(0.60653066))
    assert 0 < maps[0, 1].max() < 1
    assert maps[0, 2].max() == 0
    assert torch.isfinite(decode_heatmaps(maps)[0]).all()


def test_distance_round_trip_and_clamped_endpoints() -> None:
    values: Float32[Tensor, '30 21'] = (torch.rand(30, 21, generator=torch.Generator().manual_seed(20)) * 2 - 1) * DISTANCE_RANGE_MM
    error: float = float((decode_distance(render_distance(values)) - values).abs().max())
    print(f'Distance max error: {error:.8f} mm')
    assert error < 0.1 * 2 * DISTANCE_RANGE_MM / 17
    torch.testing.assert_close(decode_distance(render_distance(torch.tensor([[-1000.0, 1000.0]]))), torch.tensor([[-DISTANCE_RANGE_MM, DISTANCE_RANGE_MM]]))
    assert torch.isfinite(decode_distance(torch.zeros(1, 1, 18))).all()
