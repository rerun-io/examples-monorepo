from __future__ import annotations

import pytest
import torch
from torch import Tensor

cuda_ba = pytest.importorskip("dpvo._cuda_ba", reason="DPVO CUDA fastba extension is not built")
mojo_ba = pytest.importorskip("dpvo_fastba_mojo.backend", reason="dpvo_fastba_mojo is not importable")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _assert_close(name: str, got: Tensor, expected: Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    diff = (got - expected).abs()
    assert torch.allclose(got, expected, atol=atol, rtol=rtol), (
        f"{name} mismatch: max_abs={float(diff.max()):.6g}, mean_abs={float(diff.mean()):.6g}, "
        f"shape={tuple(got.shape)}"
    )


def _inputs(seed: int = 120) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    poses = torch.zeros(1, 6, 7, device="cuda", dtype=torch.float32)
    poses[..., 6] = 1.0
    poses[..., :3] = torch.randn(1, 6, 3, device="cuda", generator=gen) * 0.01

    patch_xy = torch.tensor(
        [
            [[316.0, 320.0, 324.0], [316.0, 320.0, 324.0], [316.0, 320.0, 324.0]],
            [[236.0, 236.0, 236.0], [240.0, 240.0, 240.0], [244.0, 244.0, 244.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    patches = torch.empty(1, 32, 3, 3, 3, device="cuda", dtype=torch.float32)
    patches[:, :, :2] = patch_xy
    patches[:, :, 2] = 1.0 + torch.rand(1, 32, 3, 3, device="cuda", generator=gen) * 0.05

    intrinsics = torch.tensor([[[320.0, 320.0, 320.0, 240.0]]], device="cuda", dtype=torch.float32)
    edges = 64
    ii = torch.randint(0, 6, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    jj = torch.randint(0, 6, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    same = ii == jj
    jj[same] = (jj[same] + 1) % 6
    kk = torch.randint(0, 32, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    return poses, patches, intrinsics, ii, jj, kk


def test_neighbors_matches_cuda() -> None:
    kk = torch.tensor([0, 0, 1, 0, 1, 2, 2], device="cuda", dtype=torch.int64)
    jj = torch.tensor([3, 1, 0, 2, 2, 5, 4], device="cuda", dtype=torch.int64)
    expected = cuda_ba.neighbors(kk, jj)
    got = mojo_ba.neighbors(kk, jj)
    assert len(got) == len(expected) == 2
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_reproject_matches_cuda() -> None:
    poses, patches, intrinsics, ii, jj, kk = _inputs()
    expected = cuda_ba.reproject(poses, patches, intrinsics, ii, jj, kk)
    torch.cuda.synchronize()
    got = mojo_ba.reproject(poses, patches, intrinsics, ii, jj, kk)
    torch.cuda.synchronize()
    _assert_close("reproject", got, expected, atol=1e-5, rtol=1e-5)


def test_ba_dense_matches_cuda() -> None:
    poses, patches, intrinsics, ii, jj, kk = _inputs(seed=121)
    target = cuda_ba.reproject(poses, patches, intrinsics, ii, jj, kk)[:, :, :, 1, 1].contiguous()
    target = target + torch.randn_like(target) * 0.01
    weight = torch.ones_like(target)
    lmbda = torch.as_tensor([1e-4], device="cuda")

    expected_poses = poses.clone()
    expected_patches = patches.clone()
    got_poses = poses.clone()
    got_patches = patches.clone()
    cuda_ba.forward(expected_poses.data, expected_patches, intrinsics, target, weight, lmbda, ii, jj, kk, -1, 1, 6, 1, False)
    mojo_ba.forward(got_poses.data, got_patches, intrinsics, target, weight, lmbda, ii, jj, kk, -1, 1, 6, 1, False)
    torch.cuda.synchronize()
    _assert_close("ba_dense_poses", got_poses, expected_poses, atol=2e-4, rtol=2e-4)
    _assert_close("ba_dense_patches", got_patches, expected_patches, atol=2e-4, rtol=2e-4)


def test_solve_system_requires_explicit_cuda() -> None:
    gen = torch.Generator(device="cuda").manual_seed(122)
    edges = 3
    j_i = torch.randn(edges, 7, 7, device="cuda", generator=gen) * 0.01
    j_j = torch.randn(edges, 7, 7, device="cuda", generator=gen) * 0.01
    ii = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)
    jj = torch.tensor([1, 2, 2], device="cuda", dtype=torch.int64)
    res = torch.randn(edges, 7, device="cuda", generator=gen) * 0.01
    with pytest.raises(NotImplementedError, match="DPVO_FASTBA_BACKEND=cuda"):
        mojo_ba.solve_system(j_i, j_j, ii, jj, res, 1e-3, 1e-3, -1)


def test_ba_eff_impl_true_requires_explicit_cuda() -> None:
    poses, patches, intrinsics, ii, jj, kk = _inputs(seed=123)
    kk = kk % 24
    patches = patches[:, :24].contiguous()
    target = cuda_ba.reproject(poses, patches, intrinsics, ii, jj, kk)[:, :, :, 1, 1].contiguous()
    weight = torch.ones_like(target)
    lmbda = torch.as_tensor([1e-4], device="cuda")

    got_poses = poses.clone()
    got_patches = patches.clone()
    with pytest.raises(NotImplementedError, match="DPVO_FASTBA_BACKEND=cuda"):
        mojo_ba.forward(got_poses.data, got_patches, intrinsics, target, weight, lmbda, ii, jj, kk, 8, 1, 6, 1, True)
