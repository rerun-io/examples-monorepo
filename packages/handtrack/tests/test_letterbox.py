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


# Worked by hand: net = (uv + 0.5) * scale - 0.5 + pad.
# GSI 1920x1080: scale 1/3 -> 640x360, bars of 60 px above and below (the RoboCap live geometry).
# Gen-HumanEgo 1600x1300: scale 0.36 -> 576x468, pads 32 (x) and 6 (y).
@pytest.mark.parametrize(('width', 'height', 'expected'), [
    (1920, 1080, [[-1.0 / 3.0, 60.0 - 1.0 / 3.0], [1919.5 / 3.0 - 0.5, 1079.5 / 3.0 - 0.5 + 60.0]]),
    (1600, 1300, [[0.18 - 0.5 + 32.0, 0.18 - 0.5 + 6.0], [1599.5 * 0.36 - 0.5 + 32.0, 1299.5 * 0.36 - 0.5 + 6.0]]),
])
def test_wide_and_tall_frames_letterbox_with_bars(width: int, height: int, expected: list[list[float]]) -> None:
    box: Letterbox = letterbox_for(width, height)
    corners: Float32[Tensor, '2 2'] = torch.tensor([[0.0, 0.0], [width - 1.0, height - 1.0]])
    torch.testing.assert_close(box.to_net(corners), torch.tensor(expected))
    torch.testing.assert_close(box.from_net(box.to_net(corners)), corners)


@pytest.mark.parametrize(('width', 'height', 'inside'), [(1920, 1080, (slice(60, 420), slice(0, 640))), (1600, 1300, (slice(6, 474), slice(32, 608)))])
def test_bars_are_black_and_the_image_fills_the_rest(width: int, height: int, inside: tuple[slice, slice]) -> None:
    image: UInt8[Tensor, 'h w'] = torch.full((height, width), 200, dtype=torch.uint8)
    result: UInt8[Tensor, '480 640'] = letterbox_for(width, height).apply(image)
    mask: Tensor = torch.zeros((NET_HEIGHT, NET_WIDTH), dtype=torch.bool)
    mask[inside] = True
    assert bool((result[mask] == 200).all())
    assert bool((result[~mask] == 0).all())


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


@pytest.mark.parametrize(('width', 'height', 'scale', 'pad', 'pad_y'), [
    (640, 480, 1.5, 0.0, 0.0), (636, 480, 1.0, 2.5, 0.0), (641, 480, 1.0, 0.0, 0.0), (640, 480, float('nan'), 0.0, 0.0),
    (640, 480, 1.0, -1.0, 0.0), (1920, 1080, 1.0 / 3.0, 0.0, 60.5), (1920, 1080, 1.0 / 3.0, 0.0, 121.0), (1600, 1300, 0.369, 32.0, 6.0),
])
def test_letterbox_rejects_inconsistent_frame(width: int, height: int, scale: float, pad: float, pad_y: float) -> None:
    with pytest.raises(ValueError):
        Letterbox(width, height, False, scale, pad, pad_y)


@pytest.mark.parametrize(('width', 'height'), [(636, 480), (640, 480), (1024, 1280)])
def test_point_maps_match_reference_scalar_padding(width: int, height: int) -> None:
    box: Letterbox = letterbox_for(width, height)
    points: Float32[Tensor, '3 21 2'] = torch.rand(3, 21, 2, generator=torch.Generator().manual_seed(61)) * 1000.0
    rotated: Float32[Tensor, '3 21 2'] = torch.stack((height - 1 - points[..., 1], points[..., 0]), dim=-1) if box.quarter_turns == 1 else points
    expected: Float32[Tensor, '3 21 2'] = (rotated + 0.5) * box.scale - 0.5 + torch.tensor([box.pad_x, 0.0])
    torch.testing.assert_close(box.to_net(points), expected, atol=1e-6, rtol=0.0)
    inverse: Float32[Tensor, '3 21 2'] = (points - torch.tensor([box.pad_x, 0.0]) + 0.5) / box.scale - 0.5
    expected = torch.stack((inverse[..., 1], height - 1 - inverse[..., 0]), dim=-1) if box.quarter_turns == 1 else inverse
    torch.testing.assert_close(box.from_net(points), expected, atol=1e-6, rtol=0.0)


# Upright turns, worked by hand for a 640x480 source and the source pixel (100, 200). One clockwise turn maps (u, v) of a W x H image
# to (H - 1 - v, u) of the H x W image. Odd turns fit 480x640 into the net frame at scale 0.75 (360x480) with 140 px bars left/right.
@pytest.mark.parametrize(('turns', 'expected'), [
    (0, [100.0, 200.0]),
    (1, [(279 + 0.5) * 0.75 - 0.5 + 140, (100 + 0.5) * 0.75 - 0.5]),
    (2, [539.0, 279.0]),
    (3, [(200 + 0.5) * 0.75 - 0.5 + 140, (539 + 0.5) * 0.75 - 0.5]),
])
def test_quarter_turns_move_a_known_point(turns: int, expected: list[float]) -> None:
    box: Letterbox = letterbox_for(640, 480, quarter_turns=turns)
    point: Float32[Tensor, '2'] = torch.tensor([100.0, 200.0])
    torch.testing.assert_close(box.to_net(point), torch.tensor(expected))
    torch.testing.assert_close(box.from_net(box.to_net(point)), point)


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
@pytest.mark.parametrize(('width', 'height'), [(636, 480), (640, 480), (1024, 1280), (1920, 1080), (1600, 1300)])
def test_every_turn_moves_pixels_and_points_together(width: int, height: int, turns: int) -> None:
    box: Letterbox = letterbox_for(width, height, quarter_turns=turns)
    image: UInt8[Tensor, 'h w'] = torch.zeros((height, width), dtype=torch.uint8)
    image[200:204, 100:104] = 255  # a 4 px block survives the downscale
    result: UInt8[Tensor, '480 640'] = box.apply(image)
    index: int = int(result.flatten().argmax())
    peak: Float32[Tensor, '2'] = torch.tensor([index % NET_WIDTH, index // NET_WIDTH], dtype=torch.float32)
    assert (peak - box.to_net(torch.tensor([101.5, 201.5]))).abs().max() <= 1.5


def test_an_upright_portrait_quest_frame_keeps_its_orientation_with_side_bars() -> None:
    # 1024x1280 at 0.375 -> 384x480, bars of 128 px left and right.
    box: Letterbox = letterbox_for(1024, 1280, quarter_turns=0)
    torch.testing.assert_close(box.to_net(torch.tensor([[0.0, 0.0], [1023.0, 1279.0]])),
                               torch.tensor([[0.5 * 0.375 - 0.5 + 128, 0.5 * 0.375 - 0.5], [1023.5 * 0.375 - 0.5 + 128, 1279.5 * 0.375 - 0.5]]))


def test_without_turns_the_legacy_map_turns_quest_frames_once() -> None:
    """The convention repro-1 and the deployed DetNet-F were trained with."""
    assert letterbox_for(1024, 1280) == letterbox_for(1024, 1280, quarter_turns=1)
    assert letterbox_for(640, 480) == letterbox_for(640, 480, quarter_turns=0)


def test_a_hot3d_quest_frame_fits_like_a_half_size_show3d_frame() -> None:
    # HOT3D stores Quest 3 frames at 512x640: upright 0.75 -> 384x480 with 128 px side bars; turned 0.9375 -> 600x480, 20 px bars.
    upright: Letterbox = letterbox_for(512, 640, quarter_turns=0)
    torch.testing.assert_close(upright.to_net(torch.tensor([[0.0, 0.0], [511.0, 639.0]])),
                               torch.tensor([[0.5 * 0.75 - 0.5 + 128, 0.5 * 0.75 - 0.5], [511.5 * 0.75 - 0.5 + 128, 639.5 * 0.75 - 0.5]]))
    assert (letterbox_for(512, 640, quarter_turns=1).scale, letterbox_for(512, 640, quarter_turns=1).pad_x) == (0.9375, 20.0)
    assert letterbox_for(512, 640) == letterbox_for(512, 640, quarter_turns=1)
