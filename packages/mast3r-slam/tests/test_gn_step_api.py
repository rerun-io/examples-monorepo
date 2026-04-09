from __future__ import annotations

import pytest
import torch

from mast3r_slam import _backends
from mast3r_slam import gn_backends


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _make_fixture() -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    Twc = torch.randn(4, 8, device=device, generator=gen, dtype=torch.float32).contiguous()
    Twc[:, 6] = 1.0
    Twc[:, 7] = 1.0
    Xs = torch.randn(4, 64, 3, device=device, generator=gen, dtype=torch.float32).contiguous()
    Cs = (torch.rand(4, 64, 1, device=device, generator=gen, dtype=torch.float32) + 0.5).contiguous()
    ii = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    idx_ii2jj = torch.randint(0, 64, (3, 64), device=device, generator=gen, dtype=torch.long).contiguous()
    valid_match = (torch.rand(3, 64, 1, device=device, generator=gen) > 0.2).contiguous()
    Q = (torch.rand(3, 64, 1, device=device, generator=gen, dtype=torch.float32) * 2.0 + 1.0).contiguous()
    K = torch.tensor([[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    return Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, K


def test_rays_step_shapes() -> None:
    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, _K = _make_fixture()
    Hs, gs = _backends.gauss_newton_rays_step(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5)
    assert Hs.shape == (4, ii.numel(), 7, 7)
    assert gs.shape == (2, ii.numel(), 7)


def test_selected_rays_step_matches_cuda() -> None:
    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, _K = _make_fixture()
    hs_cuda, gs_cuda = _backends.gauss_newton_rays_step(
        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5
    )
    hs_sel, gs_sel = gn_backends.gauss_newton_rays_step(
        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5
    )
    torch.testing.assert_close(hs_sel, hs_cuda, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gs_sel, gs_cuda, atol=1e-3, rtol=1e-4)


def test_calib_step_shapes() -> None:
    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, K = _make_fixture()
    Hs, gs = _backends.gauss_newton_calib_step(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q, 64, 64, -10, 1e-6, 1.0, 10.0, 0.0, 1.5)
    assert Hs.shape == (4, ii.numel(), 7, 7)
    assert gs.shape == (2, ii.numel(), 7)


def test_points_step_shapes() -> None:
    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, _K = _make_fixture()
    Hs, gs = _backends.gauss_newton_points_step(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.05, 0.0, 1.5)
    assert Hs.shape == (4, ii.numel(), 7, 7)
    assert gs.shape == (2, ii.numel(), 7)


def test_rays_public_matches_cuda_shape() -> None:
    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, _K = _make_fixture()
    dx_cuda = _backends.gauss_newton_rays(
        Twc.clone(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5, 3, 1e-8
    )[0]
    dx_selected = gn_backends.gauss_newton_rays(
        Twc.clone(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5, 3, 1e-8
    )[0]
    assert dx_selected.shape == dx_cuda.shape


def test_pose_retr_updates_tensor_in_place() -> None:
    if not hasattr(gn_backends, "pose_retr"):
        pytest.skip("pose_retr not available")

    Twc, *_rest = _make_fixture()
    dx = torch.zeros(3, 7, device=Twc.device, dtype=Twc.dtype)
    dx[0, 0] = 0.1
    before = Twc.clone()
    gn_backends.pose_retr(Twc, dx, 1)
    assert not torch.allclose(Twc[1:], before[1:])
