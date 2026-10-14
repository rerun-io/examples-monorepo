"""Integration lane: the Gauss-Newton Op against the Mojo loop it replaces (GPU, bit-identical) and the Rerun round trips
of the edge Types through an in-process server (a catalog)."""

import lietorch
import numpy as np
import pytest
import torch
from dagkit.types import rerun_roundtrip

from mast3r_slam.ops.optim import gauss_newton_rays
from mast3r_slam.types import KeyframeCloud, LocalOptConfig, MapUpdate, TrackedFrame

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(("n_kf", "hw", "valid_frac"), [(12, 50176, 0.3), (60, 4096, 0.05)])
def test_gauss_newton_rays_equals_the_mojo_loop(n_kf: int, hw: int, valid_frac: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    mojo = pytest.importorskip("mast3r_slam_mojo_backends", reason="Mojo kernels: pixi task _mast3r-slam-build-mojo-kernels")
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
    cfg = LocalOptConfig()
    reference, ported = poses.clone(), poses.clone()
    mojo.gauss_newton_rays((reference, xs, cs, ii, jj, idx, valid, q, cfg.sigma_ray, cfg.sigma_dist, cfg.C_conf, cfg.Q_conf, cfg.max_iters, cfg.delta_norm))
    gauss_newton_rays(ported, xs, cs, ii, jj, idx, valid, q, cfg)
    assert not torch.equal(ported, poses)  # the solve moved the free poses
    assert torch.equal(ported[0], poses[0]) and torch.equal(ported, reference)


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
