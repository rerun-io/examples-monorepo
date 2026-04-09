from __future__ import annotations

import lietorch
import pytest
import torch

cuda_backends = pytest.importorskip("mast3r_slam._backends", reason="CUDA backends not built")
mojo_backends = pytest.importorskip("mast3r_slam_mojo_backends", reason="Mojo backends not built")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = torch.device("cuda")
POSE_ATOL = 1e-4
POSE_RTOL = 1e-4


def _sync() -> None:
    torch.cuda.synchronize()


def _make_representative_rays_inputs() -> tuple[torch.Tensor | float | int, ...]:
    """Build one synthetic GN rays fixture shaped like the real base workload."""
    torch.manual_seed(1)
    n_poses = 10
    h = 512
    w = 512
    hw = h * w

    twc = lietorch.Sim3.exp(0.02 * torch.randn(n_poses, 7, device=DEVICE)).data.contiguous().float()
    xs = torch.randn(n_poses, hw, 3, device=DEVICE, dtype=torch.float32)
    xs[..., 2].abs_().add_(1.0)
    cs = (0.5 + torch.rand(n_poses, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()

    undirected: list[tuple[int, int]] = [(i, i + 1) for i in range(n_poses - 1)]
    cursor = 0
    loop_span = 2
    while len(undirected) < 16:
        i = cursor % max(n_poses - loop_span, 1)
        j = i + loop_span
        edge = (i, j)
        if j < n_poses and edge not in undirected:
            undirected.append(edge)
        else:
            loop_span = 3 if loop_span == 2 else 2
            cursor += 1
            continue
        cursor += 1

    ii_fwd = torch.tensor([edge[0] for edge in undirected], device=DEVICE, dtype=torch.long)
    jj_fwd = torch.tensor([edge[1] for edge in undirected], device=DEVICE, dtype=torch.long)
    ii = torch.cat([ii_fwd, jj_fwd], dim=0).contiguous()
    jj = torch.cat([jj_fwd, ii_fwd], dim=0).contiguous()
    n_edges = int(ii.shape[0])

    idx = torch.arange(hw, device=DEVICE, dtype=torch.long).unsqueeze(0).repeat(n_edges, 1).contiguous()
    valid = (torch.rand(n_edges, hw, 1, device=DEVICE) < 0.14).contiguous()
    q = (1.55 + 0.35 * torch.rand(n_edges, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()

    return (
        twc,
        xs,
        cs,
        ii,
        jj,
        idx,
        valid,
        q,
        0.5,
        0.25,
        0.1,
        0.1,
        5,
        1e-6,
    )


def test_gauss_newton_rays_matches_cuda_on_representative_base_shape() -> None:
    args = _make_representative_rays_inputs()

    twc_cuda = args[0].clone()
    dx_cuda = cuda_backends.gauss_newton_rays(twc_cuda, *args[1:])[0]
    _sync()

    twc_mojo = args[0].clone()
    dx_mojo = mojo_backends.gauss_newton_rays_impl_idiomatic((twc_mojo, *args[1:]))[0]
    _sync()

    assert torch.allclose(dx_mojo, dx_cuda, atol=POSE_ATOL, rtol=POSE_RTOL)
    assert torch.allclose(twc_mojo, twc_cuda, atol=POSE_ATOL, rtol=POSE_RTOL)
