"""Warm-started sliding-window SMPL-X fitter — the causal replacement for the
original whole-sequence 4-stage LBFGS (``optimization/utils/optimization.py``).

Per the realtime handoff: maintain a W-frame window of per-frame pose/trans
(betas shared), bootstrap with a longer solve when the window first fills,
then a few warm-started Adam iterations per tick, emitting the newest frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray

from mamma.fitting.losses import anchor3d_loss, reprojection_loss, shape_prior_loss, temporal_smoothness_loss
from mamma.fitting.smplx_wrapper import NUM_BETAS, build_smplx_neutral, smplx_forward_per_parts


def closest_point_rays(origins: Float32[torch.Tensor, "c 3"], directions: Float32[torch.Tensor, "c 3"]) -> Float32[torch.Tensor, "3"]:
    """Least-squares closest point to a bundle of rays (original helper)."""
    eye: Float32[torch.Tensor, "3 3"] = torch.eye(3, device=origins.device)
    a: Float32[torch.Tensor, "3 3"] = torch.zeros((3, 3), device=origins.device)
    b: Float32[torch.Tensor, "3"] = torch.zeros(3, device=origins.device)
    for o, d in zip(origins, directions, strict=True):
        d = d / torch.norm(d)
        a += eye - torch.outer(d, d)
        b += (eye - torch.outer(d, d)) @ o
    return torch.linalg.solve(a, b)


@dataclass(slots=True)
class FitterConfig:
    """Sliding-window fitter settings."""

    smplx_model_dir: Path = Path("data/body_models")
    """Body-model root containing ``smplx/SMPLX_NEUTRAL.npz`` (smplx.create layout)."""
    downsampled_verts_pkl: Path = Path("data/body_models/downsampled_verts/verts_512.pkl")
    """[512, 10475] vertex subsampling matrix mapping SMPL-X verts to landmarks."""
    window_size: int = 12
    """Frames kept in the optimization window."""
    bootstrap_iters: int = 300
    """Adam iterations for the first-window solve (betas optimized here)."""
    tick_iters: int = 24
    """Adam iterations per steady-state tick (betas frozen). Tuned on
    crossing_arms vs golden: 24 @ lr 0.02 -> 19.8mm MPJPE (16 -> 24.2, 8 -> 36.6)."""
    learning_rate: float = 0.02
    """Adam learning rate (0.03 oscillates, 0.01 under-converges at this budget)."""
    weight_reproj: float = 20.0
    """Reprojection loss weight (golden ``first_run``)."""
    weight_anchor3d: float = 100.0
    """Triangulated-anchor MSE weight (golden ``l2_loss_3d_points``)."""
    weight_shape_prior: float = 1.0
    """Betas shrinkage weight."""
    weight_temp_trans: float = 10.0
    """Window translation smoothness weight."""
    weight_temp_pose: float = 1.0
    """Window pose smoothness weight."""
    device: str = "cuda"
    """Compute device."""


@dataclass(slots=True)
class FitResult:
    """One emitted (newest-in-window) SMPL-X frame."""

    frame_idx: int
    """Source frame index."""
    vertices: Float32[ndarray, "v 3"]
    """Posed SMPL-X vertices in world meters (10475)."""
    joints: Float32[ndarray, "j 3"]
    """SMPL-X joints in world meters (127 with appendage regressors)."""
    pose: Float32[ndarray, "165"]
    """Full axis-angle pose vector (global+body+jaw+eyes+hands)."""
    betas: Float32[ndarray, "nb"]
    """Shared shape coefficients."""
    trans: Float32[ndarray, "3"]
    """Root translation in world meters."""


@dataclass(slots=True)
class _WindowFrame:
    """Internal per-frame observation buffer."""

    frame_idx: int
    pts2d_per_cam: list[Float32[torch.Tensor, "n 3"]]
    vis_per_cam: list[Float32[torch.Tensor, "n"]]
    anchors3d: Float32[torch.Tensor, "n 3"]
    anchors_valid: Float32[torch.Tensor, "n"]
    global_orient: Float32[torch.Tensor, "3"] = field(repr=False)
    body_pose: Float32[torch.Tensor, "63"] = field(repr=False)
    trans: Float32[torch.Tensor, "3"] = field(repr=False)


def _initial_body_pose(device: str) -> Float32[torch.Tensor, "63"]:
    """Bent-elbow initialization from the original ``init_learnable_params``."""
    body: Float32[torch.Tensor, "63"] = torch.zeros(63, device=device)
    body[-3 + 17 * 3 + 2] = np.pi / 4  # right shoulder z
    body[-3 + 16 * 3 + 2] = -np.pi / 4  # left shoulder z
    body[-3 + 17 * 3 + 1] = np.pi / 7  # right shoulder y
    body[-3 + 16 * 3 + 1] = -np.pi / 7  # left shoulder y
    body[-3 + 19 * 3 + 1] = np.pi / 1.7  # right elbow
    body[-3 + 18 * 3 + 1] = -np.pi / 1.7  # left elbow
    return body


class SlidingWindowFitter:
    """Causal SMPL-X fitter for one person."""

    def __init__(
        self,
        k_per_cam: list[Float32[torch.Tensor, "3 3"]],
        world_to_cam_per_cam: list[Float32[torch.Tensor, "4 4"]],
        config: FitterConfig,
    ) -> None:
        import pickle

        self.config: FitterConfig = config
        self.device: str = config.device
        self.k_per_cam = [k.to(config.device) for k in k_per_cam]
        self.world_to_cam_per_cam = [e.to(config.device) for e in world_to_cam_per_cam]
        self.model = build_smplx_neutral(config.smplx_model_dir, device=config.device)
        with open(config.downsampled_verts_pkl, "rb") as f:
            mat: Float32[torch.Tensor, "n v"] = pickle.load(f)
        self.verts_to_landmarks: Float32[torch.Tensor, "n v"] = mat.float().to(config.device)

        self.betas: Float32[torch.Tensor, "1 nb"] = torch.zeros(1, NUM_BETAS, device=config.device, requires_grad=True)
        self.hand_pose_left: Float32[torch.Tensor, "45"] = torch.zeros(45, device=config.device)
        self.hand_pose_right: Float32[torch.Tensor, "45"] = torch.zeros(45, device=config.device)
        self.jaw_pose: Float32[torch.Tensor, "3"] = torch.zeros(3, device=config.device)
        self._window: list[_WindowFrame] = []
        self._bootstrapped: bool = False

    def _init_trans(self) -> Float32[torch.Tensor, "3"]:
        """Initialize root translation at the camera rays' closest point."""
        exts: Float32[torch.Tensor, "c 4 4"] = torch.stack(self.world_to_cam_per_cam, dim=0)
        ray_dir: Float32[torch.Tensor, "c 3"] = exts[:, 2, :3] / torch.linalg.norm(exts[:, 2, :3], dim=-1, keepdim=True)
        ray_origin: Float32[torch.Tensor, "c 3"] = -torch.einsum("bkj,bk->bj", exts[:, :3, :3], exts[:, :3, 3])
        return closest_point_rays(ray_origin, ray_dir)

    def push(
        self,
        frame_idx: int,
        pts2d_per_cam: list[Float32[torch.Tensor, "n 3"]],
        vis_per_cam: list[Float32[torch.Tensor, "n"]],
        anchors3d: Float32[torch.Tensor, "n 3"],
        anchors_valid: Bool[torch.Tensor, "n"],
    ) -> FitResult | None:
        """Add one tick's observations; returns the newest frame's fit (or None
        before the first window fills)."""
        anchors_valid_f: Float32[torch.Tensor, "n"] = anchors_valid.float().detach()
        anchors3d_d: Float32[torch.Tensor, "n 3"] = anchors3d.detach()
        if self._window:
            prev = self._window[-1]
            global_orient: Float32[torch.Tensor, "3"] = prev.global_orient.detach().clone()
            body_pose: Float32[torch.Tensor, "63"] = prev.body_pose.detach().clone()
            trans: Float32[torch.Tensor, "3"] = prev.trans.detach().clone()
        else:
            global_orient = torch.zeros(3, device=self.device)
            body_pose = _initial_body_pose(self.device)
            trans = self._init_trans()
            # Anchor-centroid warm start: when enough triangulated points
            # exist, snap the first translation guess to their centroid.
            if anchors_valid_f.sum() > 50:
                trans = anchors3d_d[anchors_valid_f > 0].mean(dim=0)
        frame = _WindowFrame(
            frame_idx=frame_idx,
            pts2d_per_cam=[p.detach() for p in pts2d_per_cam],
            vis_per_cam=[v.detach() for v in vis_per_cam],
            anchors3d=anchors3d_d,
            anchors_valid=anchors_valid_f,
            global_orient=global_orient,
            body_pose=body_pose,
            trans=trans,
        )
        self._window.append(frame)
        if len(self._window) > self.config.window_size:
            self._window.pop(0)

        if not self._bootstrapped:
            if len(self._window) < self.config.window_size:
                return None
            self._optimize(self.config.bootstrap_iters, optimize_betas=True)
            self._bootstrapped = True
        else:
            self._optimize(self.config.tick_iters, optimize_betas=False)
        return self._emit()

    def _optimize(self, num_iters: int, optimize_betas: bool) -> None:
        t: int = len(self._window)
        global_orient: Float32[torch.Tensor, "t 3"] = torch.stack([f.global_orient for f in self._window]).detach().requires_grad_()
        body_pose: Float32[torch.Tensor, "t 63"] = torch.stack([f.body_pose for f in self._window]).detach().requires_grad_()
        trans: Float32[torch.Tensor, "t 3"] = torch.stack([f.trans for f in self._window]).detach().requires_grad_()
        params: list[torch.Tensor] = [global_orient, body_pose, trans]
        if optimize_betas:
            params.append(self.betas)

        pts2d: list[Float32[torch.Tensor, "t n 3"]] = [
            torch.stack([f.pts2d_per_cam[c] for f in self._window]) for c in range(len(self.k_per_cam))
        ]
        vis: list[Float32[torch.Tensor, "t n"]] = [
            torch.stack([f.vis_per_cam[c] for f in self._window]) for c in range(len(self.k_per_cam))
        ]
        anchors: Float32[torch.Tensor, "t n 3"] = torch.stack([f.anchors3d for f in self._window])
        anchors_valid: Float32[torch.Tensor, "t n"] = torch.stack([f.anchors_valid for f in self._window])

        hands_l: Float32[torch.Tensor, "t 45"] = self.hand_pose_left.expand(t, -1)
        hands_r: Float32[torch.Tensor, "t 45"] = self.hand_pose_right.expand(t, -1)
        jaw: Float32[torch.Tensor, "t 3"] = self.jaw_pose.expand(t, -1)

        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        cfg = self.config
        for _ in range(num_iters):
            optimizer.zero_grad()
            out = smplx_forward_per_parts(self.model, global_orient, body_pose, hands_l, hands_r, jaw, self.betas, trans)
            sampled: Float32[torch.Tensor, "t n 3"] = torch.einsum("ij,bjk->bik", self.verts_to_landmarks, out.vertices)
            loss = (
                reprojection_loss(pts2d, sampled, self.k_per_cam, self.world_to_cam_per_cam, vis, weight=cfg.weight_reproj)
                + anchor3d_loss(sampled, anchors, anchors_valid, weight=cfg.weight_anchor3d)
                + shape_prior_loss(self.betas, weight=cfg.weight_shape_prior)
                + temporal_smoothness_loss(trans, body_pose, cfg.weight_temp_trans, cfg.weight_temp_pose)
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, f in enumerate(self._window):
                f.global_orient = global_orient[i].detach()
                f.body_pose = body_pose[i].detach()
                f.trans = trans[i].detach()

    def _emit(self) -> FitResult:
        f = self._window[-1]
        with torch.no_grad():
            out = smplx_forward_per_parts(
                self.model,
                f.global_orient.unsqueeze(0),
                f.body_pose.unsqueeze(0),
                self.hand_pose_left.unsqueeze(0),
                self.hand_pose_right.unsqueeze(0),
                self.jaw_pose.unsqueeze(0),
                self.betas,
                f.trans.unsqueeze(0),
            )
        pose_full: Float32[ndarray, "165"] = (
            torch.cat([f.global_orient, f.body_pose, self.jaw_pose, torch.zeros(6, device=self.device), self.hand_pose_left, self.hand_pose_right])
            .cpu()
            .numpy()
        )
        return FitResult(
            frame_idx=f.frame_idx,
            vertices=out.vertices[0].cpu().numpy(),
            joints=out.joints[0].cpu().numpy(),
            pose=pose_full,
            betas=self.betas[0].detach().cpu().numpy(),
            trans=f.trans.cpu().numpy(),
        )
