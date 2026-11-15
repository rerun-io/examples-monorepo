"""`lietorch_mojo.pose_retr.pose_retr`, MASt3R-SLAM's Sim3 retraction: poses[k] = Exp(dx[k - num_fix]) * poses[k] for k >= num_fix."""

import pytest
import torch
from conftest import DEVICES, DTYPES, require_mojo_extension
from torch import Tensor

require_mojo_extension()

from lietorch_mojo import Sim3  # noqa: E402
from lietorch_mojo.pose_retr import pose_retr  # noqa: E402


def _poses(n: int, device: str, dtype: torch.dtype, seed: int = 0) -> Tensor:
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    return Sim3.exp(torch.randn(n, 7, generator=gen, dtype=torch.float64)).data.to(device=device, dtype=dtype).contiguous()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("n", "num_fix"), [(1, 0), (10, 1), (10, 3), (5000, 2)])
def test_equals_sim3_retr_on_free_poses(n: int, num_fix: int, dtype: torch.dtype, device: str) -> None:
    poses: Tensor = _poses(n, device, dtype)
    gen: torch.Generator = torch.Generator().manual_seed(1)
    dx: Tensor = (0.3 * torch.randn(n - num_fix, 7, generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
    expected: Tensor = Sim3(poses[num_fix:].contiguous()).retr(dx).data
    out: Tensor = poses.clone()
    pose_retr(out, dx, num_fix)
    assert torch.equal(out[:num_fix], poses[:num_fix])
    # retr stores Exp(dx) and re-normalises its quaternion on load; the fused kernel does not, so allow a few ulp.
    err: Tensor = (out[num_fix:] - expected).abs().amax(-1) / expected.abs().amax(-1).clamp_min(1.0)
    assert err.max().item() <= 8 * torch.finfo(dtype).eps


@pytest.mark.parametrize("device", DEVICES)
def test_all_fixed_is_a_no_op(device: str) -> None:
    poses: Tensor = _poses(4, device, torch.float32)
    out: Tensor = poses.clone()
    pose_retr(out, torch.empty(0, 7, device=device), 4)
    assert torch.equal(out, poses)


def test_rejects_bad_inputs() -> None:
    poses: Tensor = _poses(4, "cpu", torch.float32)
    with pytest.raises(ValueError, match="dx must have shape"):
        pose_retr(poses, torch.zeros(4, 7), 1)
    with pytest.raises(ValueError, match="poses must have shape"):
        pose_retr(poses[:, :7], torch.zeros(3, 7), 1)
    with pytest.raises(TypeError, match="dtype"):
        pose_retr(poses, torch.zeros(3, 7, dtype=torch.float64), 1)
    with pytest.raises(ValueError, match="contiguous"):
        pose_retr(torch.zeros(8, 4).t(), torch.zeros(3, 7), 1)
    with pytest.raises(ValueError, match="num_fix"):
        pose_retr(poses, torch.zeros(5, 7), -1)
