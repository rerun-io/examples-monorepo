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

from mamma.fitting.losses import (
    acceleration_loss,
    anchor3d_loss,
    floor_contact_loss,
    floor_penetration_loss,
    reprojection_loss,
    shape_prior_loss,
    temporal_smoothness_loss,
)
from mamma.fitting.sampled_smplx import SampledSmplx
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
    window_size: int = 8
    """Frames kept in the optimization window."""
    bootstrap_iters: int = 300
    """Adam iterations for the first-window solve (betas optimized here)."""
    tick_iters: int = 48
    """Adam iterations per optimize tick (betas frozen; graph replays are
    ~1 ms/iter so 64 costs ~60 ms on the overlapped fit worker). Shipped
    defaults pass BOTH golden gates: crossing_arms MPJPE/PVE <= 30 mm and
    running_jumping per-frame PVE p95/max <= 30 mm vs the original DAG."""
    learning_rate: float = 0.02
    """Adam learning rate (0.03 oscillates, 0.01 under-converges at this budget)."""
    fit_stride: int = 1
    """Optimize every Nth tick; skipped ticks emit the warm-started forward of
    the previous solution (observations still enter the window)."""
    anchor_warm_start: bool = True
    """Initialize each new frame's root translation by shifting the previous
    solution by the triangulated-anchor centroid motion (pose stays copied).
    Cuts the systematic lag on fast motion without overshooting reversals;
    no effect on quasi-static content."""
    emit_lag: int = 4
    """Emit window frame [-1-emit_lag] instead of the newest (fixed-lag
    smoothing). Each frame is then refined by ``emit_lag`` future ticks before
    leaving the fitter — output latency emit_lag/fps (4 -> 133 ms at 30 fps).
    The bend-window frames where an offline whole-sequence fit most out-informs
    a zero-lag causal one are exactly the ones this recovers."""
    emit_stride: int = 2
    """Emit (full-mesh smplx forward + readout) every Nth tick. Emission is a
    pure read-out — fit quality is unchanged. CAUTION: on skipped ticks push()
    returns the PREVIOUS FitResult, so ``FitResult.frame_idx`` lags the tick —
    consumers indexing results by frame_idx (e.g. a ResultCollector) must set
    1, as validate_golden does."""
    lbfgs_polish: bool = True
    """Eager LBFGS refinement (lr 0.3, strong-wolfe — the original's actual
    optimizer; its YAML 'Adam' field is dead config) on HARD ticks only:
    when the valid-anchor count drops below 75% of its rolling median
    (self-occlusion windows). Curvature + line search close the last few mm
    where fixed-step Adam stalls; firing on ~5% of ticks keeps it off the
    realtime budget."""
    use_cuda_graph: bool = True
    """Run both the bootstrap and the steady-state optimize through captured
    CUDA graphs (steady ticks 92 -> 15.3 ms per 16-iter call; bootstrap
    ~1.8 s -> ~0.4 s). False = eager Adam everywhere (debugging)."""
    weight_reproj: float = 120.0
    """Reprojection loss weight. The original's refinement stages are
    reprojection-dominant (120-300) and drop the anchor term after stage 1;
    matching that balance closed the last ~14 mm on dynamic content."""
    reproj_delta_px: float = 2.7
    """Geman-McClure saturation scale in engine pixels. The GM delta is
    resolution-relative: the original's 8 px applied to 4K-space landmarks;
    at a 1280-wide engine the equivalent robustness is 8 * 1280/3840 = 2.7
    (8 px at 720p let outliers pull 3x harder)."""
    vis_clip_value: float = 0.2
    """Floor on per-landmark visibility weight in the reprojection loss. The
    original's refinement stages use 0.2; the first port hardcoded first_run's
    0.8, which kept self-occluded (low-visibility) landmarks at 80% influence
    — the bend-over failure window traced back to exactly that."""
    weight_anchor3d: float = 20.0
    """Triangulated-anchor weight (saturating MSE). Anchors initialize and
    stabilize depth; the original uses them only in its first stage, so they
    stay subordinate to reprojection here."""
    weight_shape_prior: float = 1.0
    """Betas shrinkage weight."""
    weight_temp_trans: float = 10.0
    """Window translation smoothness weight."""
    weight_temp_pose: float = 2.0
    """Window pose smoothness weight."""
    weight_accel: float = 0.0
    """Positional-acceleration penalty on the sampled 3D points (original
    ``acceleration_loss``, which ships commented-out there too). Off by
    default: at dt=1/30 even small weights crush real jump accelerations
    (measured +29mm mean PVE at 0.002). Kept as an experiment lever."""
    weight_floor_contact: float = 30.0
    """Soft pull-to-ground on landmarks MammaNet predicts in floor contact
    (fused across cameras). The original exports these predictions per stage;
    without the pull our fit floats 50-65 mm above ground during the
    hands-near-ground bend while the original stays planted. Jump frames
    predict no contact, so airborne motion is unaffected."""
    weight_floor: float = 1000.0
    """One-sided floor-penetration penalty (verts below world z=0). Jump
    landings measured 60-70 mm underground without it; airborne frames are
    unaffected (one-sided)."""
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
    floor_contact: Float32[torch.Tensor, "n"]
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
        self.sampled_model: SampledSmplx = SampledSmplx(self.model, self.verts_to_landmarks)
        self._window: list[_WindowFrame] = []
        self._bootstrapped: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        self._last_emit: FitResult | None = None
        self._valid_counts: list[float] = []

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
        floor_contact: Float32[torch.Tensor, "n"] | None = None,
        optimize: bool = True,
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
            if self.config.anchor_warm_start:
                # Measurement-driven warm start: shift the root by the motion of
                # the triangulated landmark centroid since the previous frame.
                # A static warm start makes Adam spend its budget catching up on
                # fast motion (PVE correlates 0.53 with speed); constant-velocity
                # extrapolation overshoots at direction reversals — the anchors
                # track real motion either way.
                both: Float32[torch.Tensor, "n"] = anchors_valid_f * prev.anchors_valid
                if both.sum() > 50:
                    mask = both > 0
                    # Component-wise median of per-anchor motion (robust to a
                    # subset of mis-triangulated anchors), clamped to a sane
                    # per-frame root speed so one glitchy tick can't teleport
                    # the init (~0.12 m/frame = 3.6 m/s; jumps measure ~0.06).
                    per_anchor: Float32[torch.Tensor, "m 3"] = anchors3d_d[mask] - prev.anchors3d[mask]
                    delta: Float32[torch.Tensor, "3"] = per_anchor.median(dim=0).values
                    speed: Float32[torch.Tensor, ""] = torch.linalg.norm(delta)
                    delta = torch.where(speed > 0.12, delta * (0.12 / speed.clamp(min=1e-6)), delta)
                    trans = (prev.trans + delta).detach()
        else:
            global_orient = torch.zeros(3, device=self.device)
            body_pose = _initial_body_pose(self.device)
            trans = self._init_trans()
            # Anchor-centroid warm start: when enough triangulated points
            # exist, snap the first translation guess to their centroid.
            if anchors_valid_f.sum() > 50:
                trans = anchors3d_d[anchors_valid_f > 0].mean(dim=0)
        fc: Float32[torch.Tensor, "n"] = (
            floor_contact.detach() if floor_contact is not None else torch.zeros_like(anchors_valid_f)
        )
        frame = _WindowFrame(
            frame_idx=frame_idx,
            pts2d_per_cam=[p.detach() for p in pts2d_per_cam],
            vis_per_cam=[v.detach() for v in vis_per_cam],
            anchors3d=anchors3d_d,
            anchors_valid=anchors_valid_f,
            floor_contact=fc,
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
            if self.config.use_cuda_graph:
                self._bootstrap_graphed()
            else:
                self._optimize(self.config.bootstrap_iters, optimize_betas=True)
            self._bootstrapped = True
        elif optimize:
            if self.config.use_cuda_graph:
                self._optimize_graphed()
            else:
                self._optimize(self.config.tick_iters, optimize_betas=False)
            if self.config.lbfgs_polish:
                count: float = float(anchors_valid_f.sum().item())
                self._valid_counts.append(count)
                if len(self._valid_counts) > 60:
                    self._valid_counts.pop(0)
                median: float = float(np.median(self._valid_counts))
                if len(self._valid_counts) >= 20 and count < 0.75 * median:
                    self._lbfgs_polish()
        if self.config.emit_stride > 1 and frame_idx % self.config.emit_stride != 0 and self._last_emit is not None:
            return self._last_emit
        emit_index: int = -1 - min(self.config.emit_lag, len(self._window) - 1)
        self._last_emit = self._emit(emit_index)
        return self._last_emit

    # ── CUDA-graph steady state ──────────────────────────────────────────────

    def _capture_graph(self, optimize_betas: bool) -> tuple[torch.cuda.CUDAGraph, dict]:
        """Capture ONE Adam iteration over static window buffers as a CUDA graph.

        Runs 3 warmup iterations + the capture iteration (4 of the caller's
        budget), so callers replay ``budget - 4`` times. With ``optimize_betas``
        the betas join the capturable optimizer and the shape prior joins the
        loss (bootstrap); otherwise betas are frozen (steady state).
        """
        cfg = self.config
        w: int = cfg.window_size
        n: int = self.verts_to_landmarks.shape[0]
        c: int = len(self.k_per_cam)
        dev: str = self.device
        static: dict = {
            "go": torch.zeros(w, 3, device=dev, requires_grad=True),
            "bp": torch.zeros(w, 63, device=dev, requires_grad=True),
            "tr": torch.zeros(w, 3, device=dev, requires_grad=True),
            "pts2d": [torch.zeros(w, n, 3, device=dev) for _ in range(c)],
            "vis": [torch.zeros(w, n, device=dev) for _ in range(c)],
            "anchors": torch.zeros(w, n, 3, device=dev),
            "avalid": torch.zeros(w, n, device=dev),
            "fc": torch.zeros(w, n, device=dev),
            "hl": self.hand_pose_left.expand(w, -1).contiguous(),
            "hr": self.hand_pose_right.expand(w, -1).contiguous(),
            "jaw": self.jaw_pose.expand(w, -1).contiguous(),
        }
        params: list[torch.Tensor] = [static["go"], static["bp"], static["tr"]]
        if optimize_betas:
            params.append(self.betas)
        betas: Float32[torch.Tensor, "1 nb"] = self.betas if optimize_betas else self.betas.detach()
        optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, capturable=True)

        def one_iter() -> None:
            optimizer.zero_grad(set_to_none=False)
            sampled = self.sampled_model.forward(
                static["go"], static["bp"], static["hl"], static["hr"], static["jaw"], betas, static["tr"]
            )
            loss = reprojection_loss(
                static["pts2d"], sampled, self.k_per_cam, self.world_to_cam_per_cam, static["vis"], vis_clip_value=cfg.vis_clip_value, weight=cfg.weight_reproj, delta_px=cfg.reproj_delta_px
            ) + anchor3d_loss(sampled, static["anchors"], static["avalid"], weight=cfg.weight_anchor3d)
            if optimize_betas:
                loss = loss + shape_prior_loss(self.betas, weight=cfg.weight_shape_prior)
            loss = loss + temporal_smoothness_loss(static["tr"], static["bp"], cfg.weight_temp_trans, cfg.weight_temp_pose)
            loss = loss + floor_penetration_loss(sampled, weight=cfg.weight_floor)
            loss = loss + floor_contact_loss(sampled, static["fc"], weight=cfg.weight_floor_contact)
            loss = loss + acceleration_loss(sampled, weight=cfg.weight_accel)
            loss.backward()
            optimizer.step()

        self._copy_window_to_static(static)
        # Side-stream warmup per the torch CUDA-graph recipe, then capture ONE
        # iteration; callers replay it (capturing a whole multi-iteration loop
        # in one graph corrupted state on first replay).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                one_iter()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # thread_local capture: the engine pipelines fit on a worker thread,
        # so other threads legitimately launch kernels during capture.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            one_iter()
        torch.cuda.synchronize()
        return graph, static

    def _bootstrap_graphed(self) -> None:
        """Run the full bootstrap budget through a captured graph (with betas).

        The eager bootstrap costs ~6 ms/iter x 300 — ~1.8 s inside the realtime
        window. Capturing one (fwd+bwd+Adam-with-betas) iteration and replaying
        it keeps the full iteration budget at ~1 ms/iter, so quality is not
        traded for the gate. The graph is single-use; the steady-state graph
        (betas frozen) builds lazily on the first optimize tick.
        """
        graph, static = self._capture_graph(optimize_betas=True)
        for _ in range(max(0, self.config.bootstrap_iters - 4)):
            graph.replay()
        torch.cuda.synchronize()
        self._copy_static_to_window(static)

    def _build_graph(self) -> None:
        """Capture the steady-state (betas-frozen) iteration and keep it for replay."""
        graph, static = self._capture_graph(optimize_betas=False)
        for _ in range(self.config.tick_iters - 4):  # finish the build tick's budget
            graph.replay()
        torch.cuda.synchronize()
        self._graph = graph
        self._static = static

    def _lbfgs_polish(self) -> None:
        """One strong-wolfe LBFGS outer step over the window (hard ticks only)."""
        cfg = self.config
        t: int = len(self._window)
        go: Float32[torch.Tensor, "t 3"] = torch.stack([f.global_orient for f in self._window]).detach().requires_grad_()
        bp: Float32[torch.Tensor, "t 63"] = torch.stack([f.body_pose for f in self._window]).detach().requires_grad_()
        tr: Float32[torch.Tensor, "t 3"] = torch.stack([f.trans for f in self._window]).detach().requires_grad_()
        pts2d = [torch.stack([f.pts2d_per_cam[c] for f in self._window]) for c in range(len(self.k_per_cam))]
        vis = [torch.stack([f.vis_per_cam[c] for f in self._window]) for c in range(len(self.k_per_cam))]
        anchors: Float32[torch.Tensor, "t n 3"] = torch.stack([f.anchors3d for f in self._window])
        avalid: Float32[torch.Tensor, "t n"] = torch.stack([f.anchors_valid for f in self._window])
        fc: Float32[torch.Tensor, "t n"] = torch.stack([f.floor_contact for f in self._window])
        hl: Float32[torch.Tensor, "t 45"] = self.hand_pose_left.expand(t, -1)
        hr: Float32[torch.Tensor, "t 45"] = self.hand_pose_right.expand(t, -1)
        jaw: Float32[torch.Tensor, "t 3"] = self.jaw_pose.expand(t, -1)
        betas_frozen: Float32[torch.Tensor, "1 nb"] = self.betas.detach()
        optimizer = torch.optim.LBFGS([go, bp, tr], lr=0.3, max_iter=10, history_size=10, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            sampled = self.sampled_model.forward(go, bp, hl, hr, jaw, betas_frozen, tr)
            loss = (
                reprojection_loss(
                    pts2d, sampled, self.k_per_cam, self.world_to_cam_per_cam, vis, vis_clip_value=cfg.vis_clip_value, weight=cfg.weight_reproj, delta_px=cfg.reproj_delta_px
                )
                + anchor3d_loss(sampled, anchors, avalid, weight=cfg.weight_anchor3d)
                + temporal_smoothness_loss(tr, bp, cfg.weight_temp_trans, cfg.weight_temp_pose)
                + floor_penetration_loss(sampled, weight=cfg.weight_floor)
                + floor_contact_loss(sampled, fc, weight=cfg.weight_floor_contact)
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            for i, f in enumerate(self._window):
                f.global_orient = go[i].detach().clone()
                f.body_pose = bp[i].detach().clone()
                f.trans = tr[i].detach().clone()

    def _copy_window_to_static(self, static: dict) -> None:
        with torch.no_grad():
            static["go"].copy_(torch.stack([f.global_orient for f in self._window]))
            static["bp"].copy_(torch.stack([f.body_pose for f in self._window]))
            static["tr"].copy_(torch.stack([f.trans for f in self._window]))
            for cam in range(len(self.k_per_cam)):
                static["pts2d"][cam].copy_(torch.stack([f.pts2d_per_cam[cam] for f in self._window]))
                static["vis"][cam].copy_(torch.stack([f.vis_per_cam[cam] for f in self._window]))
            static["anchors"].copy_(torch.stack([f.anchors3d for f in self._window]))
            static["avalid"].copy_(torch.stack([f.anchors_valid for f in self._window]))
            static["fc"].copy_(torch.stack([f.floor_contact for f in self._window]))

    def _copy_static_to_window(self, static: dict) -> None:
        with torch.no_grad():
            for i, f in enumerate(self._window):
                f.global_orient = static["go"][i].detach().clone()
                f.body_pose = static["bp"][i].detach().clone()
                f.trans = static["tr"][i].detach().clone()

    def _optimize_graphed(self) -> None:
        if self._graph is None:
            self._build_graph()
        else:
            self._copy_window_to_static(self._static)
            for _ in range(self.config.tick_iters):
                self._graph.replay()
        self._copy_static_to_window(self._static)

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
        fc: Float32[torch.Tensor, "t n"] = torch.stack([f.floor_contact for f in self._window])

        hands_l: Float32[torch.Tensor, "t 45"] = self.hand_pose_left.expand(t, -1)
        hands_r: Float32[torch.Tensor, "t 45"] = self.hand_pose_right.expand(t, -1)
        jaw: Float32[torch.Tensor, "t 3"] = self.jaw_pose.expand(t, -1)

        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        cfg = self.config
        for _ in range(num_iters):
            optimizer.zero_grad()
            sampled: Float32[torch.Tensor, "t n 3"] = self.sampled_model.forward(
                global_orient, body_pose, hands_l, hands_r, jaw, self.betas, trans
            )
            loss = (
                reprojection_loss(pts2d, sampled, self.k_per_cam, self.world_to_cam_per_cam, vis, vis_clip_value=cfg.vis_clip_value, weight=cfg.weight_reproj, delta_px=cfg.reproj_delta_px)
                + anchor3d_loss(sampled, anchors, anchors_valid, weight=cfg.weight_anchor3d)
                + shape_prior_loss(self.betas, weight=cfg.weight_shape_prior)
                + temporal_smoothness_loss(trans, body_pose, cfg.weight_temp_trans, cfg.weight_temp_pose)
                + floor_penetration_loss(sampled, weight=cfg.weight_floor)
                + floor_contact_loss(sampled, fc, weight=cfg.weight_floor_contact)
                + acceleration_loss(sampled, weight=cfg.weight_accel)
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, f in enumerate(self._window):
                f.global_orient = global_orient[i].detach()
                f.body_pose = body_pose[i].detach()
                f.trans = trans[i].detach()

    def drain(self) -> list[FitResult]:
        """Emit the frames still inside the lag at end of stream (newest last)."""
        if self.config.emit_lag <= 0 or not self._bootstrapped:
            return []
        lag: int = min(self.config.emit_lag, len(self._window) - 1)
        return [self._emit(i) for i in range(-lag, 0)]

    def _emit(self, index: int = -1) -> FitResult:
        f = self._window[index]
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
