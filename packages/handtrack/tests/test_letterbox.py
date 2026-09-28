from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float32, UInt8
from torch import Tensor

from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH, Letterbox, letterbox_for


@pytest.mark.parametrize(('width', 'height', 'expected'), [(636, 480, [[2.0, 0.0], [102.0, 200.0]]), (640, 480, [[0.0, 0.0], [100.0, 200.0]]), (1024, 1280, [[619.265625, -0.265625], [525.515625, 46.609375]])])
def test_coordinate_maps_and_inverse(width: int, height: int, expected: list[list[float]]) -> None:
    box: Letterbox = letterbox_for(width, height)
    points: Float32[Tensor, '2 2'] = torch.tensor([[0.0, 0.0], [100.0, 200.0]])
    torch.testing.assert_close(box.to_net(points), torch.tensor(expected))
    torch.testing.assert_close(box.from_net(box.to_net(points)), points)


@pytest.mark.parametrize(('width', 'height'), [(636, 480), (640, 480), (1024, 1280)])
def test_image_and_point_map_agree(width: int, height: int) -> None:
    box: Letterbox = letterbox_for(width, height)
    image: UInt8[Tensor, '2 1 h w'] = torch.zeros((2, 1, height, width), dtype=torch.uint8)
    image[..., 200, 100] = 255
    result: UInt8[Tensor, '2 1 480 640'] = box.apply(image)
    assert result.shape == (2, 1, NET_HEIGHT, NET_WIDTH)
    index: int = int(result[0, 0].flatten().argmax())
    peak: Float32[Tensor, '2'] = torch.tensor([index % NET_WIDTH, index // NET_WIDTH], dtype=torch.float32)
    assert (peak - box.to_net(torch.tensor([100.0, 200.0]))).abs().max() <= 1.0
    assert result.max() > 0
    assert box.apply(image[0, 0]).shape == (480, 640)


def test_unknown_resolution_is_rejected() -> None:
    with pytest.raises(ValueError):
        letterbox_for(800, 600)


@pytest.mark.parametrize('axis', [0, 1])
def test_show3d_linear_ramp_matches_inverse_point_map(axis: int) -> None:
    box: Letterbox = letterbox_for(1024, 1280)
    length: int = box.source_width if axis == 0 else box.source_height
    ramp: Float32[Tensor, 'n'] = torch.linspace(0.0, 255.0, length)
    image: UInt8[Tensor, '1280 1024'] = (ramp[None, :].expand(1280, -1) if axis == 0 else ramp[:, None].expand(-1, 1024)).round().to(torch.uint8)
    pixels: Float32[Tensor, 'n 2'] = torch.stack(torch.meshgrid(torch.arange(30, 610, dtype=torch.float32), torch.arange(10, 470, dtype=torch.float32), indexing='xy'), dim=-1).reshape(-1, 2)
    expected: Float32[Tensor, 'n'] = box.from_net(pixels)[:, axis] * (255.0 / (length - 1))
    with patch('handtrack.geometry.letterbox.F.interpolate', wraps=F.interpolate) as resize:
        output: UInt8[Tensor, '480 640'] = box.apply(image)
    continuous: Float32[Tensor, '1280 1024'] = ramp[None, :].expand(1280, -1) if axis == 0 else ramp[:, None].expand(-1, 1024)
    sampled: Float32[Tensor, '1 1 480 600'] = F.interpolate(torch.rot90(continuous, -1, (-2, -1))[None, None], **resize.call_args.kwargs)
    assert (sampled[0, 0, pixels[:, 1].long(), pixels[:, 0].long() - 20] - expected).abs().max() < 0.05
    # Account explicitly for source and output uint8 rounding (0.5 each).
    assert (output[pixels[:, 1].long(), pixels[:, 0].long()].float() - expected).abs().max() < 1.05


@pytest.mark.parametrize(('width', 'height', 'scale', 'pad'), [(640, 480, 0.5, 0.0), (636, 480, 1.0, 2.5), (641, 480, 1.0, 0.0), (640, 480, float('nan'), 0.0), (640, 480, 1.0, -1.0)])
def test_letterbox_rejects_inconsistent_frame(width: int, height: int, scale: float, pad: float) -> None:
    with pytest.raises(ValueError):
        Letterbox(width, height, False, scale, pad)


@pytest.mark.parametrize(('width', 'height'), [(636, 480), (640, 480), (1024, 1280)])
def test_point_maps_match_reference_scalar_padding(width: int, height: int) -> None:
    box: Letterbox = letterbox_for(width, height)
    points: Float32[Tensor, '3 21 2'] = torch.rand(3, 21, 2, generator=torch.Generator().manual_seed(61)) * 1000.0
    rotated: Float32[Tensor, '3 21 2'] = torch.stack((height - 1 - points[..., 1], points[..., 0]), dim=-1) if box.quarter_turn_cw else points
    expected: Float32[Tensor, '3 21 2'] = (rotated + 0.5) * box.scale - 0.5 + torch.tensor([box.pad_x, 0.0])
    torch.testing.assert_close(box.to_net(points), expected, atol=1e-6, rtol=0.0)
    inverse: Float32[Tensor, '3 21 2'] = (points - torch.tensor([box.pad_x, 0.0]) + 0.5) / box.scale - 0.5
    expected = torch.stack((inverse[..., 1], height - 1 - inverse[..., 0]), dim=-1) if box.quarter_turn_cw else inverse
    torch.testing.assert_close(box.from_net(points), expected, atol=1e-6, rtol=0.0)
