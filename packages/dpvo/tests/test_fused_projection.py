"""Inference projection preserves camera calibration, quaternion handling, and the near plane."""

import pytest
from jaxtyping import Float32, Int64

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")
from lietorch_mojo import SE3  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo import projective_ops  # noqa: E402
from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402
from dpvo.mojo_backend.projective import project_points  # noqa: E402
from dpvo.net import VONet  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
def test_fused_projection_keeps_per_camera_intrinsics_and_near_plane(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    torch.manual_seed(23)
    poses: SE3 = SE3.exp(torch.randn(1, 4, 6, device=device) * 0.1)
    poses.data[0, 3, 2] = -2.0
    poses.data[0, 1, 3:] *= 1.02
    patches: Float32[Tensor, "1 8 3 3 3"] = torch.rand(1, 8, 3, 3, 3, device=device)
    patches[:, :, :2] *= 100.0
    patches[:, :, 2] += 0.5
    intrinsics: Float32[Tensor, "1 4 4"] = torch.tensor(
        [[[80.0, 90.0, 50.0, 40.0], [82.0, 89.0, 48.0, 42.0], [75.0, 85.0, 45.0, 35.0], [81.0, 91.0, 49.0, 41.0]]], device=device
    )
    source: Int64[Tensor, "6"] = torch.tensor([0, 1, 2, 0, 3, 1], device=device)
    target: Int64[Tensor, "6"] = torch.tensor([1, 0, 3, 0, 2, 3], device=device)
    patch: Int64[Tensor, "6"] = torch.tensor([0, 2, 4, 1, 6, 3], device=device)
    expected: Float32[Tensor, "1 6 3 3 2"] = projective_ops.transform(poses, patches, intrinsics, source, target, patch)
    actual: Float32[Tensor, "1 6 3 3 2"] = project_points(poses.data, patches, intrinsics, source, target, patch)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
    empty: Float32[Tensor, "1 0 3 3 2"] = project_points(poses.data, patches, intrinsics, source[:0], target[:0], patch[:0])
    assert empty.shape == (1, 0, 3, 3, 2)


def test_tracker_projection_keeps_autograd_when_requested() -> None:
    tracker: DPVO = DPVO(DPVOConfig(buffer_size=8, patches_per_frame=2), VONet(), ht=32, wd=32, device=torch.device("cpu"))
    tracker.pg.intrinsics_[:] = torch.tensor([10.0, 10.0, 4.0, 4.0])
    tracker.pg.patches_[:, :, :2] = 2.0
    tracker.pg.patches_[:, :, 2] = 1.0
    tracker.pg.poses_.requires_grad_()
    source: Int64[Tensor, "1"] = torch.tensor([0])
    target: Int64[Tensor, "1"] = torch.tensor([1])
    with torch.enable_grad():
        projected: Float32[Tensor, "1 1 2 3 3"] = tracker.reproject(indicies=(source, target, source))
        projected.sum().backward()
    assert tracker.pg.poses_.grad is not None
    assert bool(torch.isfinite(tracker.pg.poses_.grad).all())
    assert float(tracker.pg.poses_.grad.abs().sum()) > 0.0
