"""Half correlation rounds the product and sum separately, including subnormals."""

import pytest
from jaxtyping import Float16, Float32, Int64

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="build the Mojo kernels first")
from torch import Tensor  # noqa: E402

from dpvo.mojo_backend import corr  # noqa: E402
from dpvo.mojo_backend._launch import launch  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
@pytest.mark.parametrize("channels", [2, 128])
def test_half_correlation_keeps_separate_rounding(device: str, channels: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    generator: torch.Generator = torch.Generator().manual_seed(42)
    patches: Float16[Tensor, "1 16 channels 1 1"] = torch.randn(1, 16, channels, 1, 1, generator=generator).half()
    features: Float16[Tensor, "1 16 channels 2 2"] = torch.randn(1, 16, channels, 2, 2, generator=generator).half()
    # Separate rounding gives zero; an FMA keeps the small product remainder.
    patches[0, 0, :, 0, 0] = 0.0
    patches[0, 0, :2, 0, 0] = torch.tensor([-1.0, 1.0009765625], dtype=torch.float16)
    features[0, 0, 0] = 1.0
    features[0, 0, 1] = 0.99951171875
    # Products straddle the half subnormal and normal ranges, with both signs.
    patches[:, 1:8] *= 2.0**-12
    features[:, 1:8] *= 2.0**-4
    expected: Float16[Tensor, "16"] = torch.zeros(16, dtype=torch.float16)
    for channel in range(channels):
        product: Float16[Tensor, "16"] = patches[0, :, channel, 0, 0] * features[0, :, channel, 0, 0]
        expected = expected + product
    assert expected[0] == 0.0
    assert bool((expected[1:8].abs() < torch.finfo(torch.float16).tiny).any())
    coordinates: Float32[Tensor, "1 16 2 1 1"] = torch.zeros(1, 16, 2, 1, 1, device=device)
    indices: Int64[Tensor, "16"] = torch.arange(16, device=device)
    (actual,) = corr.forward(patches.to(device), features.to(device), coordinates, indices, indices, radius=0)
    torch.testing.assert_close(actual.flatten().cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.integration
@pytest.mark.parametrize("radius", [0, 1, 3])
@pytest.mark.parametrize("width", [3, 4])
def test_half_correlation_matches_cpu_at_image_borders(radius: int, width: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    generator: torch.Generator = torch.Generator().manual_seed(19)
    patches: Float16[Tensor, "2 5 7 2 3"] = torch.randn(2, 5, 7, 2, 3, generator=generator).half()
    features: Float16[Tensor, "2 3 7 3 width"] = torch.randn(2, 3, 7, 3, width, generator=generator).half()
    coordinates: Float32[Tensor, "2 7 2 2 3"] = torch.zeros(2, 7, 2, 2, 3)
    coordinates[:, :, 0] = torch.tensor([-2.2, -0.2, 0.0, 1.7, 2.0, 3.4, 5.0])[None, :, None, None]
    coordinates[:, :, 1] = torch.tensor([1.0, 0.1, -1.0, 1.3, 2.0, 3.1, 1.0])[None, :, None, None]
    source: Int64[Tensor, "7"] = torch.arange(7) % 5
    target: Int64[Tensor, "7"] = torch.arange(7) % 3
    size: int = 2 * radius + 2
    expected: Float16[Tensor, "2 7 d d 2 3"] = torch.empty(2, 7, size, size, 2, 3, dtype=torch.float16)
    launch("corr", (True, radius), (patches, features, coordinates, source, target, expected))
    actual: Float16[Tensor, "2 7 d d 2 3"] = torch.empty_like(expected, device="cuda")
    launch("corr", (True, radius), (patches.cuda(), features.cuda(), coordinates.cuda(), source.cuda(), target.cuda(), actual))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)
