"""Synthetic-rig tests for the confidence-weighted keypoint bundle adjustment.

A known 4-camera rig observes a moving point cloud (a stand-in for the dense
MammaNet body landmarks across frames). We perturb the non-anchor cameras
(rotation, translation, focal) and check that BA, driven by the perfect 2D
observations, pulls them back toward ground truth and collapses the
reprojection error.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float32

from mamma.calibration.keypoint_bundle_adjust import (
    BundleAdjustConfig,
    bundle_adjust_cameras,
    project_points,
    so3_exp,
)
from mamma.fitting.triangulation import triangulate_points


def _rig(n_cams: int = 4, radius: float = 3.0, focal: float = 900.0) -> tuple[
    Float32[torch.Tensor, "c 3 3"], Float32[torch.Tensor, "c 4 4"]
]:
    """A ring of ``n_cams`` cameras on the z=1.2m circle looking at the origin."""
    w: int = 1280
    h: int = 720
    k: Float32[torch.Tensor, "3 3"] = torch.tensor(
        [[focal, 0.0, w / 2], [0.0, focal, h / 2], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    k_per: Float32[torch.Tensor, "c 3 3"] = k.unsqueeze(0).repeat(n_cams, 1, 1)
    w2c: list[Float32[torch.Tensor, "4 4"]] = []
    up: Float32[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 1.0])
    for i in range(n_cams):
        ang: float = 2.0 * math.pi * i / n_cams
        cam_pos: Float32[torch.Tensor, "3"] = torch.tensor([radius * math.cos(ang), radius * math.sin(ang), 1.2])
        fwd: Float32[torch.Tensor, "3"] = -cam_pos / torch.linalg.norm(cam_pos)
        right: Float32[torch.Tensor, "3"] = torch.linalg.cross(fwd, up)
        right = right / torch.linalg.norm(right)
        down: Float32[torch.Tensor, "3"] = torch.linalg.cross(fwd, right)
        rot_cam_world: Float32[torch.Tensor, "3 3"] = torch.stack([right, down, fwd], dim=0)
        t: Float32[torch.Tensor, "3"] = -rot_cam_world @ cam_pos
        m: Float32[torch.Tensor, "4 4"] = torch.eye(4)
        m[:3, :3] = rot_cam_world
        m[:3, 3] = t
        w2c.append(m)
    return k_per, torch.stack(w2c, dim=0)


def _project_all(
    pts: Float32[torch.Tensor, "s 3"],
    k_per: Float32[torch.Tensor, "c 3 3"],
    w2c: Float32[torch.Tensor, "c 4 4"],
) -> tuple[Float32[torch.Tensor, "c s 3"], Float32[torch.Tensor, "c s"]]:
    """Project to ``[x, y, logvar=const]`` per cam plus all-visible confidence."""
    n_cams: int = k_per.shape[0]
    kps: list[Float32[torch.Tensor, "s 3"]] = []
    for c in range(n_cams):
        uv, _ = project_points(pts, w2c[c, :3, :3], w2c[c, :3, 3], k_per[c], None)
        logvar: Float32[torch.Tensor, "s 1"] = torch.full((uv.shape[0], 1), -6.0)  # sigma ~ small
        kps.append(torch.cat([uv, logvar], dim=-1))
    vis: Float32[torch.Tensor, "c s"] = torch.ones(n_cams, pts.shape[0])
    return torch.stack(kps, dim=0), vis


def _rotation_angle_deg(r_a: Float32[torch.Tensor, "3 3"], r_b: Float32[torch.Tensor, "3 3"]) -> float:
    rel: Float32[torch.Tensor, "3 3"] = r_a @ r_b.T
    cos: float = float(((torch.trace(rel) - 1.0) / 2.0).clamp(-1.0, 1.0).item())
    return math.degrees(math.acos(cos))


def test_so3_exp_basics() -> None:
    """exp(0) == I and exp about +z by 90 deg rotates x->y."""
    eye: Float32[torch.Tensor, "1 3 3"] = so3_exp(torch.zeros(1, 3))
    assert torch.allclose(eye[0], torch.eye(3), atol=1e-6)
    rot: Float32[torch.Tensor, "3 3"] = so3_exp(torch.tensor([[0.0, 0.0, math.pi / 2]]))[0]
    got: Float32[torch.Tensor, "3"] = rot @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(got, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_bundle_adjust_recovers_perturbed_extrinsics() -> None:
    """Perturb cams 1..3 rotation + translation (focal fixed at GT); BA recovers them.

    Focal is held fixed here so the test targets the *unambiguous* extrinsic
    recovery — with focal free there is a focal<->translation<->depth gauge
    family that admits a small residual translation at identical reprojection
    (that is precisely what the focal anchor guards against in production).
    """
    torch.manual_seed(0)
    k_gt, w2c_gt = _rig()
    n_cams: int = k_gt.shape[0]
    # A blob of points filling the working volume (≈ a person moving over frames).
    pts: Float32[torch.Tensor, "s 3"] = torch.randn(400, 3) * torch.tensor([0.4, 0.4, 0.8]) + torch.tensor([0.0, 0.0, 1.0])
    kps2d, vis = _project_all(pts, k_gt, w2c_gt)

    # Perturbed init: camera 0 stays GT (the gauge); 1..3 get a few-degree /
    # few-cm kick.
    w2c_init: Float32[torch.Tensor, "c 4 4"] = w2c_gt.clone()
    delta_rot: Float32[torch.Tensor, "c 3"] = torch.zeros(n_cams, 3)
    delta_rot[1:] = torch.tensor([[0.05, -0.04, 0.03], [-0.03, 0.05, -0.02], [0.04, 0.02, 0.05]])
    w2c_init[:, :3, :3] = so3_exp(delta_rot) @ w2c_gt[:, :3, :3]
    w2c_init[1:, :3, 3] += torch.tensor([[0.08, -0.05, 0.04], [-0.06, 0.05, -0.03], [0.05, 0.04, 0.06]])

    pts3d_init, _ = triangulate_points(
        [kps2d[c] for c in range(n_cams)], [k_gt[c] for c in range(n_cams)],
        [w2c_init[c] for c in range(n_cams)], [vis[c] for c in range(n_cams)], vis_thresh=0.0,
    )

    # Low anchor weight: this synthetic init is deliberately bad, so let the
    # (perfect) reprojection term dominate and pull the cameras to GT.
    cfg: BundleAdjustConfig = BundleAdjustConfig(
        optimize_focal=False, anchor_rot_weight=0.0, anchor_trans_weight=0.0, n_iters=60, huber_delta=10.0
    )
    res = bundle_adjust_cameras(k_gt, w2c_init, kps2d, vis, pts3d_init, cfg)

    assert res.rmse_px_after < res.rmse_px_before
    assert res.rmse_px_after < 0.5, f"reprojection RMSE should collapse, got {res.rmse_px_after:.3f}px"

    # Per-camera errors shrink vs the perturbed init.
    for c in range(1, n_cams):
        ang_init: float = _rotation_angle_deg(w2c_init[c, :3, :3], w2c_gt[c, :3, :3])
        ang_after: float = _rotation_angle_deg(res.world_to_cam_per_cam[c, :3, :3], w2c_gt[c, :3, :3])
        trans_init: float = float(torch.linalg.norm(w2c_init[c, :3, 3] - w2c_gt[c, :3, 3]).item())
        trans_after: float = float(torch.linalg.norm(res.world_to_cam_per_cam[c, :3, 3] - w2c_gt[c, :3, 3]).item())
        assert ang_after < ang_init and ang_after < 0.2, f"cam {c} rotation off: {ang_init:.2f}->{ang_after:.2f}deg"
        # ~16mm floor with free 3D points + this baseline; >6x better than the
        # ~100mm init, and the production anchor + hundreds of motion frames
        # tighten it further.
        assert trans_after < trans_init and trans_after < 0.025, f"cam {c} translation off: {trans_after * 1000:.1f}mm"


def test_bundle_adjust_refines_focal() -> None:
    """A wrong global focal (all cams) is pulled back toward GT by the keypoints."""
    torch.manual_seed(2)
    k_gt, w2c_gt = _rig(focal=900.0)
    n_cams: int = k_gt.shape[0]
    pts: Float32[torch.Tensor, "s 3"] = torch.randn(500, 3) * torch.tensor([0.5, 0.5, 0.9]) + torch.tensor([0.0, 0.0, 1.0])
    kps2d, vis = _project_all(pts, k_gt, w2c_gt)

    k_init: Float32[torch.Tensor, "c 3 3"] = k_gt.clone()
    k_init[:, 0, 0] *= 0.90  # 10% focal error on every camera
    k_init[:, 1, 1] *= 0.90
    pts3d_init, _ = triangulate_points(
        [kps2d[c] for c in range(n_cams)], [k_init[c] for c in range(n_cams)],
        [w2c_gt[c] for c in range(n_cams)], [vis[c] for c in range(n_cams)], vis_thresh=0.0,
    )
    cfg: BundleAdjustConfig = BundleAdjustConfig(
        optimize_focal=True, anchor_focal_weight=0.0, anchor_rot_weight=0.0, anchor_trans_weight=0.0, n_iters=60, huber_delta=10.0
    )
    res = bundle_adjust_cameras(k_init, w2c_gt, kps2d, vis, pts3d_init, cfg)
    assert res.rmse_px_after < res.rmse_px_before
    f_err_init: float = abs(float(k_init[1, 0, 0]) - 900.0) / 900.0
    f_err_after: float = abs(float(res.k_per_cam[1, 0, 0]) - 900.0) / 900.0
    assert f_err_after < f_err_init, f"focal not improved: {f_err_init:.3f}->{f_err_after:.3f}"
    assert f_err_after < 0.05, f"focal off by {f_err_after * 100:.1f}%"


def test_anchor_keeps_cameras_near_init() -> None:
    """A large anchor weight should freeze the cameras near their init."""
    torch.manual_seed(1)
    k_gt, w2c_gt = _rig()
    n_cams: int = k_gt.shape[0]
    pts: Float32[torch.Tensor, "s 3"] = torch.randn(200, 3) * 0.5 + torch.tensor([0.0, 0.0, 1.0])
    kps2d, vis = _project_all(pts, k_gt, w2c_gt)

    # Init away from GT; huge anchor should prevent BA from moving toward GT.
    w2c_init: Float32[torch.Tensor, "c 4 4"] = w2c_gt.clone()
    delta_rot: Float32[torch.Tensor, "c 3"] = torch.zeros(n_cams, 3)
    delta_rot[1:] = 0.05
    w2c_init[:, :3, :3] = so3_exp(delta_rot) @ w2c_gt[:, :3, :3]
    pts3d_init, _ = triangulate_points(
        [kps2d[c] for c in range(n_cams)], [k_gt[c] for c in range(n_cams)], [w2c_init[c] for c in range(n_cams)],
        [vis[c] for c in range(n_cams)], vis_thresh=0.0,
    )
    cfg: BundleAdjustConfig = BundleAdjustConfig(
        anchor_rot_weight=1e6, anchor_trans_weight=1e6, anchor_focal_weight=1e6, optimize_focal=False, n_iters=30
    )
    res = bundle_adjust_cameras(k_gt, w2c_init, kps2d, vis, pts3d_init, cfg)
    for c in range(1, n_cams):
        moved: float = _rotation_angle_deg(res.world_to_cam_per_cam[c, :3, :3], w2c_init[c, :3, :3])
        assert moved < 0.5, f"cam {c} drifted {moved:.2f}deg despite huge anchor"
