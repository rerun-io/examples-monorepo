import pytest
import torch
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
