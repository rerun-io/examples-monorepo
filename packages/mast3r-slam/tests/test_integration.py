"""Integration lane: the Gauss-Newton Op's Mojo kernels against independent float64 references (GPU; no CUDA extension
needed) and the Rerun round trips of the edge Types through an in-process server (a catalog). Parity of the whole loop
with the CUDA extension is in test_mojo_vs_cuda.py."""

import lietorch
import numpy as np
import pytest
import torch
from dagkit.testing import rerun_roundtrip
from torch import Tensor

from mast3r_slam.ops.optim import gauss_newton_rays
from mast3r_slam.types import KeyframeCloud, LocalOptConfig, MapUpdate, TrackedFrame

pytestmark = pytest.mark.integration

# (keyframes, points, valid fraction): 4 partial blocks per edge, 1 block per edge (sparse), 8 blocks (the clamp).
GN_GRAPHS = [(12, 50176, 0.3), (60, 4096, 0.05), (3, 140000, 0.3)]
RayGraph = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]  # poses, xs, cs, ii, jj, idx_ii2jj, valid, q


def _mojo():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return pytest.importorskip("mast3r_slam_mojo_backends", reason="Mojo kernels: pixi task _mast3r-slam-build-mojo-kernels")


def _ray_graph(n_kf: int, hw: int, valid_frac: float) -> RayGraph:
    """Chain + skip-one two-way edges, random correspondences (idx_ii2jj), production-like confidences."""
    torch.manual_seed(n_kf)
    device = "cuda:0"
    poses = lietorch.Sim3.exp(0.05 * torch.randn(n_kf, 7, device=device)).data.contiguous().float()
    xs = torch.randn(n_kf, hw, 3, device=device)
    xs[..., 2].abs_().add_(1.0)
    cs = (0.5 + torch.rand(n_kf, hw, 1, device=device)).contiguous()
    forward = [(i, i + 1) for i in range(n_kf - 1)] + [(i, i + 2) for i in range(n_kf - 2)]
    ii_f, jj_f = (torch.tensor([e[k] for e in forward], device=device) for k in (0, 1))
    ii, jj = torch.cat([ii_f, jj_f]).contiguous(), torch.cat([jj_f, ii_f]).contiguous()
    idx = torch.randint(0, hw, (len(ii), hw), device=device)
    valid = (torch.rand(len(ii), hw, 1, device=device) < valid_frac).contiguous()
    q = (1.55 + 0.35 * torch.rand(len(ii), hw, 1, device=device)).contiguous()
    return poses, xs, cs, ii, jj, idx, valid, q


def _ray_step_reference(graph: RayGraph, cfg: LocalOptConfig, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """float64 J^T W J / J^T W r of the ray + distance cost per edge: residuals through lietorch, Jacobians by central
    differences of left perturbations Exp(xi) * T of both world poses, Huber-weighted like the kernel."""
    poses, xs, cs, ii, jj, idx, valid, q = graph
    world = lietorch.Sim3(poses.double())
    steps = torch.cat([torch.zeros(1, 14), eps * torch.eye(14), -eps * torch.eye(14)]).to(poses.device, torch.float64)
    sigma = torch.tensor([cfg.sigma_ray] * 3 + [cfg.sigma_dist], device=poses.device, dtype=torch.float64)
    hs, gs = [], []
    for e, (i, j) in enumerate(zip(ii.tolist(), jj.tolist(), strict=True)):
        matched = valid[e, :, 0]
        src = torch.where(matched, idx[e], 0)  # the kernel reads point 0 for unmatched points (weight 0)
        p_i, p_j = xs[i, src].double(), xs[j].double()
        rel = lietorch.Sim3.exp(steps[:, :7]).mul(world[i : i + 1]).inv().mul(lietorch.Sim3.exp(steps[:, 7:]).mul(world[j : j + 1]))
        x = rel[:, None].act(p_j[None])
        assert isinstance(x, Tensor)
        norm_x, norm_i = x.norm(dim=-1, keepdim=True), p_i.norm(dim=-1, keepdim=True)
        residuals = torch.cat([x / norm_x - p_i / norm_i, norm_x - norm_i], -1)  # [1 + 28 steps, points, 4]
        r = residuals[0]
        jac = ((residuals[1:15] - residuals[15:]) / (2 * eps)).permute(1, 2, 0)  # [points, 4, 14]
        ok = matched & (q[e, :, 0] > cfg.Q_conf) & (cs[i, src, 0] > cfg.C_conf) & (cs[j, :, 0] > cfg.C_conf)
        sqrt_w = torch.where(ok, q[e, :, 0].double().sqrt(), 0.0)[:, None] / sigma
        scaled = (sqrt_w * r).abs()
        w = torch.where(scaled < 1.345, 1.0, 1.345 / scaled) * sqrt_w**2
        h = torch.einsum("nak,na,nal->kl", jac, w, jac)
        g = torch.einsum("nak,na,na->k", jac, w, r)
        hs.append(torch.stack([h[:7, :7], h[:7, 7:], h[7:, :7], h[7:, 7:]]))
        gs.append(torch.stack([g[:7], g[7:]]))
    return torch.stack(hs, 1), torch.stack(gs, 1)


@pytest.mark.parametrize(("n_kf", "hw", "valid_frac"), GN_GRAPHS)
def test_gauss_newton_rays_step_matches_a_float64_reference(n_kf: int, hw: int, valid_frac: float) -> None:
    mojo = _mojo()
    graph, cfg = _ray_graph(n_kf, hw, valid_frac), LocalOptConfig()
    hs, gs = mojo.gauss_newton_rays_step((*graph, cfg.sigma_ray, cfg.sigma_dist, cfg.C_conf, cfg.Q_conf))
    for got, want in zip((hs, gs), _ray_step_reference(graph, cfg), strict=True):
        scale = want.abs().amax(dim=tuple(range(2, want.dim())), keepdim=True)  # per (block, edge): float32 sums
        assert torch.all((got.double() - want).abs() <= 1e-4 * scale)


def test_pose_retr_matches_lietorch_and_keeps_the_pinned_pose() -> None:
    mojo = _mojo()
    poses = _ray_graph(12, 16, 0.3)[0]
    dx = 0.05 * torch.randn(11, 7, device=poses.device)
    out = poses.clone()
    mojo.pose_retr(out, dx, 1)
    expected = lietorch.Sim3.exp(dx.double()).mul(lietorch.Sim3(poses[1:].double())).data
    assert torch.equal(out[0], poses[0])
    torch.testing.assert_close(out[1:].double(), expected, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize(("n_kf", "hw", "valid_frac"), GN_GRAPHS[:2])
def test_gauss_newton_rays_moves_only_the_free_poses(n_kf: int, hw: int, valid_frac: float) -> None:
    _mojo()
    poses, xs, cs, ii, jj, idx, valid, q = _ray_graph(n_kf, hw, valid_frac)
    ported = poses.clone()
    gauss_newton_rays(ported, xs, cs, ii, jj, idx, valid, q, LocalOptConfig())
    assert torch.equal(ported[0], poses[0]) and torch.isfinite(ported).all()
    assert (ported[1:] != poses[1:]).any(dim=1).all()  # every free pose moved


def test_tracked_frame_rerun_round_trip() -> None:
    value = TrackedFrame(42, np.array([0.5, -1.0, 2.0, 0.0, 0.6, 0.0, 0.8, 0.25], dtype=np.float32), world=1)
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/tracked/cam_00", "video_time", 1_500_000_000)
    assert t_ns == 1_500_000_000 and back.frame_index == 42 and back.world == 1
    np.testing.assert_array_equal(back.world_sim3_cam, value.world_sim3_cam)


@pytest.mark.parametrize("n_edges", [0, 3])
def test_map_update_rerun_round_trip(n_edges: int) -> None:
    poses = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32), (3, 1))
    poses[:, 0] = [0.0, 1.0, 2.0]
    value = MapUpdate(poses, np.array([0, 1, 0][:n_edges], dtype=np.int64), np.array([1, 2, 2][:n_edges], dtype=np.int64), world=2)
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/map", "video_time", 7)
    assert t_ns == 7 and back.world == 2
    np.testing.assert_array_equal(back.world_sim3_cam, poses)
    np.testing.assert_array_equal(np.stack([back.ii, back.jj]), np.stack([value.ii, value.jj]))


def test_keyframe_cloud_rerun_round_trip() -> None:
    value = KeyframeCloud(12, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32), np.array([[255, 0, 10], [1, 2, 3]], dtype=np.uint8))
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/map/keyframes", "video_time", 9)
    assert t_ns == 9 and back.index == 12
    np.testing.assert_array_equal(back.points, value.points)
    np.testing.assert_array_equal(back.colors, value.colors)
