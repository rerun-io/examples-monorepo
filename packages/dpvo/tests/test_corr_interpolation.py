"""CUDA correlation keeps the CPU reference's interpolation and output axis order."""

import pytest
from jaxtyping import Float16, Float32, Int64

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="build the Mojo kernels first")
from torch import Tensor  # noqa: E402

from dpvo.mojo_backend import corr  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("radius", [0, 1, 3])
def test_cuda_correlation_keeps_fractional_coordinates_and_axis_order(dtype: torch.dtype, radius: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    generator: torch.Generator = torch.Generator().manual_seed(61)
    patches: Float16[Tensor, "2 3 7 3 3"] | Float32[Tensor, "2 3 7 3 3"] = torch.randn(2, 3, 7, 3, 3, generator=generator).to(dtype)
    features: Float16[Tensor, "2 4 7 5 4"] | Float32[Tensor, "2 4 7 5 4"] = torch.randn(2, 4, 7, 5, 4, generator=generator).to(dtype)
    coordinates: Float32[Tensor, "2 5 2 3 3"] = torch.rand(2, 5, 2, 3, 3, generator=generator) * 7.0 - 2.0
    source: Int64[Tensor, "5"] = torch.arange(5) % 3
    target: Int64[Tensor, "5"] = torch.arange(5) % 4
    for offset in (0.0, 0.3125):
        (expected,) = corr.forward(patches, features, coordinates + offset, source, target, radius)
        (actual,) = corr.forward(patches.cuda(), features.cuda(), (coordinates + offset).cuda(), source.cuda(), target.cuda(), radius)
        tolerance: float = 0.0 if dtype == torch.float16 else 1e-5
        torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    (empty,) = corr.forward(patches.cuda(), features.cuda(), coordinates[:, :0].cuda(), source[:0].cuda(), target[:0].cuda(), radius)
    assert empty.shape == (2, 0, 2 * radius + 1, 2 * radius + 1, 3, 3)
