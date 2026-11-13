"""Fused inference motion agrees with the differentiable projective reference."""

import pytest
from jaxtyping import Float32, Int64

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")

from lietorch_mojo import SE3  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo import projective_ops  # noqa: E402
from dpvo.mojo_backend.projective import flow_magnitude  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_fused_motion_keeps_per_camera_intrinsics_and_near_plane_clamping(device: str, beta: float) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    torch.manual_seed(23)
    poses: SE3 = SE3.exp(torch.randn(1, 4, 6, device=device) * 0.1)
    poses.data[0, 3, 2] = -2.0  # Include patches behind the target camera.
    poses.data[0, 1, 3:] *= 1.02  # SE3 operations normalize stored quaternion drift.
    patches: Float32[Tensor, "1 8 3 3 3"] = torch.rand(1, 8, 3, 3, 3, device=device)
    patches[:, :, :2] *= 100.0
    patches[:, :, 2] += 0.5
    intrinsics: Float32[Tensor, "1 4 4"] = torch.tensor(
        [[[80.0, 90.0, 50.0, 40.0], [82.0, 89.0, 48.0, 42.0], [75.0, 85.0, 45.0, 35.0], [81.0, 91.0, 49.0, 41.0]]], device=device
    )
    source: Int64[Tensor, "6"] = torch.tensor([0, 1, 2, 0, 3, 1], device=device)
    target: Int64[Tensor, "6"] = torch.tensor([1, 0, 3, 0, 2, 3], device=device)
    patch: Int64[Tensor, "6"] = torch.tensor([0, 2, 4, 1, 6, 3], device=device)
    expected, _ = projective_ops.flow_mag(poses, patches, intrinsics, source, target, patch, beta=beta)
    actual: Float32[Tensor, "1 6 3 3"] = flow_magnitude(poses.data, patches, intrinsics, source, target, patch, beta=beta)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
    empty: Float32[Tensor, "1 0 3 3"] = flow_magnitude(poses.data, patches, intrinsics, source[:0], target[:0], patch[:0], beta=beta)
    assert empty.shape == (1, 0, 3, 3)
