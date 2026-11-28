"""Open4DGS's TRiGS trainer: the train-step speed logic, on the seams of recipes/open4dgs-trigs-src.

The recipe keeps only fixes and hooks (patch 0004: host-sync fixes inside ``train_step`` + the ``_const_tensor`` seam;
0005: ``_render`` split into ``_render_inputs`` / ``_rasterize_inputs`` + the ``_camera_group`` seam; 0006: a camera list
in ``_rasterize_inputs``). The logic lives here, one section per lever:

1. Host syncs: ``HostSizedCameraConfig`` (Python-int sizes, no ``.item()`` per camera) and ``Open4dgsTrainer._const_tensor``
   (the time and background scalars built once per value instead of a blocking host-to-device copy per call).
2. Shared-time deform: when every camera of a step renders at one time (no learned time offsets, no rolling shutter, one
   capture time per frame), ``SharedTimeGroup`` runs the active-row query, the row gather and the deform once; each camera
   rasterizes detached leaves of those inputs and ``finish`` carries the summed leaf gradients through deform and gather.
3. Batched cameras: with ``batch_cameras``, the group renders its cameras of one image size in ONE rasterization call,
   sums the camera losses for ONE backward, and adds the relocation's image-space cue per batch (``accumulate_2d`` sums
   over the camera axis). Each camera's losses and the camera-loss hook (the MoGe-2 prior) read its slice of the output.

Renders equal the per-camera path bit for bit; parameter gradients differ only in float summation order.

TRiGS's official sampler draws the step's cameras WITH replacement (``random.choices``): 8 cameras of 6 views are ~4.6
distinct ones. A drawn twice camera renders the same image at the same time, so the group rasterizes each distinct camera
once and the trainer computes its losses (photometric and the camera-loss hook) once per step (``_step_memo``); the loop
still adds them once per draw, so the loss, every gradient and the relocation cue are those of the duplicate renders.

Three levers from PennyGS (arXiv 2611.27525) and HiGS (arXiv 2611.31918), off in the constructor (``TrainConfig`` turns
them on):

4. Strided pixel sampling (``pixel_stride``, PennyGS 3.4): each shared-time step renders one pixel of every 2x2 block, the
   same offset for every camera, cycling through the four offsets so every pixel is supervised once every four steps. The
   step's cameras get intrinsics that put the half-size image's pixel centres on those full-size pixel centres and gsplat's
   2D dilation scaled by 1/4, so the render equals the full-size render at those pixels; the frames and masks the trainer
   reads are sliced the same way (``strided_frames``), and so are the MoGe-2 prior's maps. Recipe patch 0008 passes the
   scaled dilation to gsplat.
5. Bfloat16 Adam moments (``adam_moments``, HiGS 4.2/4.3): ``open4dgs.adam_bf16.AdamBf16`` for every TRiGS Adam.
6. Fused loss (``fused_loss``, PennyGS 3.5's single pass): the masked-composite photometric loss (composite, clamp, L1,
   alpha outside the mask, alpha-to-mask BCE) as one ``torch.compile``d function, so its forward and backward are a few
   fused kernels instead of ~50 elementwise ones per camera; SSIM stays the fused Mojo kernel. Same maths as
   ``TrigsTrainer._photometric_losses`` for the configuration open4dgs trains with.

Two exact fixes found by profiling ride along: the batched cameras' render is unbound once (a per-camera slice
backpropagates into a zero tensor of the whole batch) and the row gather is ``index_select`` (``_slice_rowwise_params``).
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import torch
from jaxtyping import Float32, Float64, Int64
from torch import Tensor, nn
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig, TrigsTrainingResult
from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

from open4dgs.fused_loss import masked_composite_terms, masked_composite_terms_batch
from open4dgs.strided import GSPLAT_EPS2D, STRIDE, STRIDE_OFFSETS, StrideState, strided_intrinsics, strided_pixels

RENDER_INPUT_KEYS: tuple[str, ...] = ("means", "quats", "scales", "opacities", "colors")
"""Per-row rasterizer inputs of ``_render_inputs`` that the cameras of a shared-time step receive as detached leaves."""

CONST_CACHE_LIMIT: int = 4096
"""Entries of the constant-tensor cache before it is cleared (times are per frame: a clip has a few hundred)."""

# ---------------------------------------------------------------------------------------------------------------------
# Lever 1: host syncs
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class HostSizedCameraConfig(CameraConfig):
    """``CameraConfig`` whose ``width_for`` / ``height_for`` return Python ints read once, not ``.item()`` per call
    (a host sync per camera when the sizes are CUDA tensors)."""

    _sizes: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sizes = list(zip(self.widths.tolist(), self.heights.tolist(), strict=True))

    def width_for(self, cam_idx: int) -> int:
        return self._sizes[cam_idx][0]

    def height_for(self, cam_idx: int) -> int:
        return self._sizes[cam_idx][1]


# ---------------------------------------------------------------------------------------------------------------------
# Levers 2 and 3: shared-time deform, batched cameras
# ---------------------------------------------------------------------------------------------------------------------

RenderOut = tuple[Tensor, Tensor, Tensor, dict]
"""What ``TrigsTrainer._render`` returns: colors, alphas, gamma over all rows, info."""


class SizeBatch(NamedTuple):
    """One batched rasterize call of a shared-time step, its outputs unbound per camera (and as rendered, ``batched``)."""

    colors: tuple[Tensor, ...]
    alphas: tuple[Tensor, ...]
    depths: tuple[Tensor, ...] | None
    gamma: Tensor
    info: dict
    cams: list[int]
    batched: tuple[Tensor, Tensor, Tensor | None]
    """(colors, alphas, depth) of all ``cams``: ``(b, h, w, 3)``, ``(b, h, w, 1)``, ``(b, h, w, 1)`` or None."""


@dataclass(frozen=True, slots=True)
class CameraLosses:
    """What the train loop's camera loop leaves behind, from ``SharedTimeGroup.camera_losses`` (recipe patch 0010's seam):
    the step's per-camera loss sums (float64, as the loop's), the summed per-row gamma, the active rows and the
    first camera's log values."""

    rgb_sum: Float64[Tensor, ""]
    alpha_outside_sum: Float64[Tensor, ""]
    alpha_mask_sum: Float64[Tensor, ""]
    gamma_sum: Float32[Tensor, "n 1"]
    active_idx: Int64[Tensor, " a"]
    cam_idx_log: int
    render_log: Float32[Tensor, "h w 3"]
    gt_rgb: Float32[Tensor, "h w 3"] | None
    """The first camera's target in [0, 1]; None unless the loop logs images (``verbose`` / ``tb_stats_every``)."""
    info: dict


class SharedTimeGroup:
    """The cameras of one train step that render at one time: shared inputs once, per-camera or per-size batched rasterize.

    The trainer's camera loop calls ``render(group_i)`` for each camera, ``add_loss(loss)`` with that camera's loss / n,
    and ``finish(...)`` after the loop (recipe patch 0005's seam).
    """

    def __init__(
        self,
        trainer: "Open4dgsTrainer",
        params: nn.ParameterDict,
        cameras: CameraConfig,
        cam_group: Sequence[int],
        t: float,
        sh_degree: int,
        config: TrigsTrainingConfig,
        step: int,
    ) -> None:
        config = trainer.step_config(config, step)
        self._trainer, self._params, self._config = trainer, params, config
        self._cam_group = list(cam_group)
        self._batch_cameras: bool = trainer.batch_cameras
        self._outputs: dict = trainer._render_inputs(params, cameras, self._cam_group[0], t, sh_degree, config, step=step)
        self._leaves: dict = dict(self._outputs)
        for key in RENDER_INPUT_KEYS:
            self._leaves[key] = self._outputs[key].detach().requires_grad_(self._outputs[key].requires_grad)
        self._cameras: CameraConfig = cameras
        if trainer.pixel_stride > 1:  # lever 4: this step's cameras, frames and masks at one strided offset
            offset: tuple[int, int] = STRIDE_OFFSETS[step % len(STRIDE_OFFSETS)]
            trainer.stride_state.offset = offset
            self._cameras = trainer.strided_cameras(cameras, offset)
            self._leaves["raster_kwargs"] = {"eps2d": GSPLAT_EPS2D / STRIDE**2}
        self._last_info: dict | None = None
        self._loss: Tensor | None = None
        self._batches: list[SizeBatch] = []
        self._batch_of: dict[int, tuple[int, int]] = {}
        self._draws: list[tuple[int, ...]] = []
        """Per size batch: how many times the step drew each of its (distinct) cameras."""
        if self._batch_cameras:  # per camera, each draw backpropagates at once: a shared loss would run backward twice
            trainer._step_memo = {}
            by_size: dict[tuple[int, int], list[int]] = {}
            for cam in dict.fromkeys(self._cam_group):  # distinct cameras, in draw order
                by_size.setdefault((self._cameras.width_for(cam), self._cameras.height_for(cam)), []).append(cam)
            for cams in by_size.values():
                for j, cam in enumerate(cams):
                    for group_i, drawn in enumerate(self._cam_group):
                        if drawn == cam:
                            self._batch_of[group_i] = (len(self._batches), j)
                colors, alphas, gamma, info = trainer._rasterize_inputs(params, self._leaves, self._cameras, cams, config)
                # One unbind per output, not a slice per camera: a slice's backward fills a zero tensor of the WHOLE batch
                # and the cameras' gradients are then summed; unbind's backward stacks the cameras' gradients once.
                depth: Tensor | None = info.get("render_depth")
                depths = depth.unbind(0) if depth is not None else None
                self._batches.append(SizeBatch(colors.unbind(0), alphas.unbind(0), depths, gamma, info, cams, (colors, alphas, depth)))
                self._draws.append(tuple(self._cam_group.count(cam) for cam in cams))

    def render(self, group_i: int) -> RenderOut:
        """Camera ``cam_group[group_i]``'s render: its slice of the size batch, or its own rasterize call."""
        if not self._batch_cameras:
            out = self._trainer._rasterize_inputs(self._params, self._leaves, self._cameras, self._cam_group[group_i], self._config)
            out[3]["stride_offset"] = self._trainer.stride_state.offset
            self._last_info = out[3]
            return out
        b, j = self._batch_of[group_i]
        batch: SizeBatch = self._batches[b]
        cam_info: dict = {**batch.info, "stride_offset": self._trainer.stride_state.offset}
        if batch.depths is not None:
            cam_info["render_depth"] = batch.depths[j][None]
        return batch.colors[j][None], batch.alphas[j][None], batch.gamma, cam_info

    def camera_losses(
        self,
        gt_images: Mapping[int, Sequence[Tensor]],
        gt_alphas: Mapping[int, Sequence[Tensor]] | None,
        frame_idx: int,
        config: TrigsTrainingConfig,
        step: int,
    ) -> CameraLosses | None:
        """Every camera's loss of the step in one batched pass (recipe patch 0010's seam), or None for the per-camera loop.

        Batched when the group is one size batch, the fused loss is on and its configuration holds (as in
        ``Open4dgsTrainer._photometric_losses``), and the camera-loss hook (the MoGe-2 prior) has a ``batch`` entry. The
        maths is the per-camera loop's: one fused composite-loss call and one SSIM call for the step's distinct cameras, the
        prior's terms per camera in one pass, each camera's loss weighted by its draws / n (the loop adds a camera drawn k
        times k times) and summed for the one deferred backward (``finish``).
        """
        trainer = self._trainer
        compiled = trainer._compiled_batch_terms
        hook = trainer._camera_loss_hook  # the hook itself (the property wraps it for the loop's per-draw memo)
        if not (
            self._batch_cameras
            and len(self._batches) == 1
            and compiled is not None
            and gt_alphas is not None
            and config.masked_rgb_loss
            and config.masked_rgb_composite
            and not config.masked_rgb_random_bg
            and config.clamp_render_for_loss
            and "cam_gain" not in self._params
            and (hook is None or hasattr(hook, "batch"))
        ):
            return None
        batch: SizeBatch = self._batches[0]
        cams: list[int] = batch.cams
        draws: tuple[int, ...] = self._draws[0]
        colors, alphas, depth = batch.batched
        gt: Tensor = torch.stack([gt_images[frame_idx][c] for c in cams]).to(trainer.device)
        mask: Tensor = torch.stack([gt_alphas[frame_idx][c] for c in cams]).to(trainer.device).float()
        if gt.dtype != torch.uint8 or mask.shape != alphas.shape[:3] or gt.shape != colors.shape:
            return None
        hook_loss: Tensor | None = None
        if hook is not None and depth is None and (trainer.depth_at_step is None or trainer.depth_at_step(step)):
            return None  # the hook needs a depth this config did not render: the per-camera loop
        if hook is not None and depth is not None:  # (no depth on an RGB-only step: the hook adds nothing)
            hook_loss = hook.batch(
                step=step, frame_idx=frame_idx, cams=cams, depth=depth[..., 0], alphas=alphas[..., 0], offset=trainer.stride_state.offset
            )
            if hook_loss is None:
                return None
        bg: Tensor = trainer._const_tensor(tuple(config.masked_rgb_bg_color), colors.device, colors.dtype)
        l1, pred_l, gt_l, alpha_out, alpha_mask = compiled(colors, alphas, gt, mask, bg, float(config.alpha_loss_eps))
        from hexels_mojo.fused_ssim import C1, C2, FusedSSIMMap

        ssim: Tensor = FusedSSIMMap.apply(C1, C2, pred_l, gt_l, "valid", True).mean(dim=(1, 2, 3))  # pyrefly: ignore[missing-attribute]
        rgb: Tensor = l1 * (1.0 - config.ssim_lambda) + (1.0 - ssim) * config.ssim_lambda
        cam_loss: Tensor = config.lambda_render * rgb + config.lambda_alpha_outside_mask * alpha_out + config.lambda_alpha_mask * alpha_mask
        if hook_loss is not None:
            cam_loss = cam_loss + hook_loss
        n_group: int = len(self._cam_group)
        # Per-camera weights draws / n as a cached device constant (a fresh host tensor would be a blocking copy per step).
        weight: Tensor = trainer._const_tensor(tuple(k / n_group for k in draws), colors.device, colors.dtype)
        self._loss = (cam_loss * weight).sum()
        counts: Tensor = trainer._const_tensor(tuple(float(k) for k in draws), colors.device, torch.float64)
        log_images: bool = bool(config.verbose) or config.tb_stats_every > 0
        info: dict = batch.info
        return CameraLosses(
            rgb_sum=(rgb.detach().double() * counts).sum(),
            alpha_outside_sum=(alpha_out.detach().double() * counts).sum(),
            alpha_mask_sum=(alpha_mask.detach().double() * counts).sum(),
            gamma_sum=batch.gamma * float(n_group),
            active_idx=info["active_idx"],
            cam_idx_log=cams[0],
            render_log=colors[0].detach(),
            gt_rgb=gt[0].float().div(255.0) if log_images else None,
            info={"n_active": info["n_active"], "n_total": info["n_total"]},
        )

    def add_loss(self, loss: Tensor, relocation: MotionGuidedRelocationStrategy, reloc_state: dict, n_group: int) -> None:
        """Per camera: backward now (into the leaves) and the relocation cue; batched: keep the sum for one backward."""
        if self._batch_cameras:
            self._loss = loss if self._loss is None else self._loss + loss
            return
        if loss.requires_grad:
            loss.backward()
        relocation.accumulate_2d(reloc_state, self._last_info, batch_size=n_group)

    def finish(self, relocation: MotionGuidedRelocationStrategy, reloc_state: dict, n_group: int) -> None:
        """The deferred camera backward and per-batch relocation cues, then the leaves' gradients through deform + gather."""
        if self._loss is not None and self._loss.requires_grad:
            self._loss.backward()
        for batch, draws in zip(self._batches, self._draws, strict=True):
            relocation.accumulate_2d(reloc_state, batch.info, batch_size=n_group)
            if max(draws) > 1:
                add_duplicate_visibility(reloc_state, batch.info, draws)
        outs = [self._outputs[k] for k in RENDER_INPUT_KEYS if self._leaves[k].grad is not None and self._outputs[k].requires_grad]
        grads = [self._leaves[k].grad for k in RENDER_INPUT_KEYS if self._leaves[k].grad is not None and self._outputs[k].requires_grad]
        if outs:
            torch.autograd.backward(outs, grads)
        self._batches, self._draws, self._loss, self._outputs, self._leaves = [], [], None, {}, {}
        self._trainer.stride_state.offset = None
        self._trainer._step_memo = None


def add_duplicate_visibility(reloc_state: dict, info: dict, draws: tuple[int, ...]) -> None:
    """The relocation's visibility counts of the cameras drawn more than once (``MotionGuidedRelocationStrategy.accumulate_2d``
    counts a row once per rendered camera in which it projects; a camera drawn k times rendered k times before the
    dedupe). Its gradient cue needs no correction: k draws give one render k times the upstream gradient, and the norm of
    that is the sum of the k norms."""
    visible_accum: Tensor | None = reloc_state.get("visible_accum")
    radii: Tensor | None = info.get("radii")
    if visible_accum is None or radii is None:
        return
    vis: Tensor = radii.detach() > 0
    if vis.shape[-1] == 2:
        vis = vis.all(dim=-1)
    # Python-int weights: a host tensor of the draw counts would be a blocking copy (a host sync) every step.
    extra: Tensor | None = None
    for j, k in enumerate(draws):
        if k > 1:
            term: Tensor = (k - 1) * vis[j].to(visible_accum.dtype)
            extra = term if extra is None else extra + term
    if extra is None:
        return
    active_idx: Tensor | None = info.get("active_idx")
    if extra.shape == visible_accum.shape:
        visible_accum += extra
    elif active_idx is not None and int(active_idx.numel()) == int(extra.numel()):
        visible_accum.index_add_(0, active_idx.detach().long().reshape(-1), extra)


# ---------------------------------------------------------------------------------------------------------------------
# Lever 4 (PennyGS 3.4): strided pixel sampling
# ---------------------------------------------------------------------------------------------------------------------


def strided_camera_config(cameras: CameraConfig, offset: tuple[int, int]) -> CameraConfig:
    """``cameras`` rendering only full-size pixels ``(offset[0] + STRIDE * i, offset[1] + STRIDE * j)``.

    gsplat samples pixel q at q + 0.5: the strided pixel q is full-size pixel p = STRIDE * q + o, at p + 0.5, so
    ``f' = f / STRIDE`` and ``c' = (c - o - 0.5) / STRIDE + 0.5``. Sizes are the number of full-size pixels sampled.
    Rasterize with ``eps2d = GSPLAT_EPS2D / STRIDE**2`` to keep the full-size dilation.
    """
    oy, ox = offset
    Ks: Tensor = strided_intrinsics(cameras.Ks, offset)
    n: int = int(Ks.shape[0])
    widths: list[int] = [len(range(ox, cameras.width_for(c), STRIDE)) for c in range(n)]
    heights: list[int] = [len(range(oy, cameras.height_for(c), STRIDE)) for c in range(n)]
    return HostSizedCameraConfig.from_matrices(viewmats=cameras.viewmats, Ks=Ks, widths=widths, heights=heights)


class _StridedRow(list[Tensor]):
    """One frame's per-camera images or masks; indexing slices to the active stride offset when one is set."""

    def __init__(self, items: Sequence[Tensor], state: StrideState) -> None:
        super().__init__(items)
        self._state = state

    def __getitem__(self, cam: int) -> Tensor:  # pyrefly: ignore[bad-override]  # the trainer indexes one camera
        item: Tensor = super().__getitem__(cam)
        return item if self._state.offset is None else strided_pixels(item, self._state.offset)


def strided_frames(frames: Mapping[int, Sequence[Tensor]], state: StrideState) -> dict[int, list[Tensor]]:
    """``frames[f][c]`` as the trainer indexes it, sliced to ``state.offset`` while a strided step runs."""
    return {f: _StridedRow(row, state) for f, row in frames.items()}


class Open4dgsTrainer(TrigsTrainer):
    """``TrigsTrainer`` with the speed levers on the recipe's seams: 1-3 always on, 4-6 behind their constructor flags."""

    def __init__(
        self,
        *args: object,
        batch_cameras: bool = True,
        pixel_stride: Literal[1, 2] = 1,
        adam_moments: Literal["fp32", "bf16"] = "fp32",
        fused_loss: bool = False,
        compile_deform: bool = False,
        **kwargs: object,
    ) -> None:
        self._camera_loss_hook: Any = None
        super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        if pixel_stride not in (1, STRIDE):
            raise ValueError(f"pixel_stride must be 1 or {STRIDE}, got {pixel_stride}")
        self.batch_cameras: bool = batch_cameras
        self.pixel_stride: Literal[1, 2] = pixel_stride
        self.adam_moments: Literal["fp32", "bf16"] = adam_moments
        # Lever 6, None when off. dynamic=True: one compile for every image size (ROI cameras each have their own size;
        # per-size compiles hit dynamo's recompile limit and fall back to eager).
        self._compiled_terms = torch.compile(masked_composite_terms, dynamic=True) if fused_loss else None
        # The batched camera loss (``SharedTimeGroup.camera_losses``): its own graph; dynamic=None makes the camera count
        # (the step's distinct cameras, 1-8) a dynamic dimension at its first change instead of one graph per count.
        self._compiled_batch_terms = torch.compile(masked_composite_terms_batch, dynamic=None) if fused_loss else None
        # TRiGS's deform as one compiled graph (forward and backward a few fused kernels instead of ~150 elementwise
        # launches); dynamic=True: the active-row count changes every step.
        self._compiled_deform = torch.compile(self._eager_deform, dynamic=True) if compile_deform else None
        self.stride_state: StrideState = StrideState()
        self._strided_of: CameraConfig | None = None
        self._strided: dict[tuple[int, int], CameraConfig] = {}
        self._const_cache: dict[tuple[object, str, torch.dtype], Tensor] = {}
        self._step_memo: dict[tuple[str, int, int], Any] | None = None
        """While a ``SharedTimeGroup`` runs: each (term, frame, camera)'s losses, computed once for a camera drawn twice."""
        self.depth_at_step: Callable[[int], bool] | None = None
        """With a depth config (``render_depth``): whether a train step rasterizes RGB+ED (the MoGe-2 prior's
        ``renders_depth``); None = every step. Renders outside the train loop (no step) always keep the depth."""
        self._rgb_config: tuple[TrigsTrainingConfig, TrigsTrainingConfig] | None = None

    @property
    def camera_loss_hook(self) -> Any:  # pyrefly: ignore[bad-override]
        """The recipe's per-camera loss hook (the MoGe-2 prior), called once per distinct camera of a shared-time step."""
        hook = self._camera_loss_hook
        if hook is None or self._step_memo is None:
            return hook
        memo: dict[tuple[str, int, int], Any] = self._step_memo

        def once(**kwargs: Any) -> Tensor:
            key: tuple[str, int, int] = ("hook", kwargs["frame_idx"], kwargs["cam_idx"])
            if key not in memo:
                memo[key] = hook(**kwargs)
            return memo[key]

        return once

    @camera_loss_hook.setter
    def camera_loss_hook(self, hook: Any) -> None:
        self._camera_loss_hook = hook

    def step_config(self, config: TrigsTrainingConfig, step: int | None) -> TrigsTrainingConfig:
        """``config`` without ``render_depth`` on a train step that ``depth_at_step`` leaves RGB-only, else ``config``."""
        if step is None or not config.render_depth or self.depth_at_step is None or self.depth_at_step(step):
            return config
        if self._rgb_config is None or self._rgb_config[0] is not config:
            self._rgb_config = (config, dataclasses.replace(config, render_depth=False))
        return self._rgb_config[1]

    def _render(  # pyrefly: ignore[bad-override]  # the recipe's signature, typed
        self,
        params: nn.ParameterDict,
        cameras: CameraConfig,
        cam_idx: int,
        t: float | Tensor,
        sh_degree: int,
        config: TrigsTrainingConfig,
        step: int | None = None,
    ) -> RenderOut:
        """The per-camera path's render, RGB-only on steps ``depth_at_step`` leaves without depth."""
        return super()._render(params, cameras, cam_idx, t, sh_degree, self.step_config(config, step), step=step)

    def train_from_params(  # pyrefly: ignore[bad-override]  # keyword-only: every caller (TRiGS's train too) passes keywords
        self,
        *,
        gt_images_per_frame: dict[int, list[Tensor]],
        gt_alphas_per_frame: dict[int, list[Tensor]],
        **kwargs: Any,  # the rest of TrigsTrainer.train_from_params, passed through
    ) -> TrigsTrainingResult:
        """Lever 4: with ``pixel_stride=2`` the train loop reads every frame and mask through ``strided_frames``."""
        if self.pixel_stride > 1:
            gt_images_per_frame = strided_frames(gt_images_per_frame, self.stride_state)
            gt_alphas_per_frame = strided_frames(gt_alphas_per_frame, self.stride_state)
        return super().train_from_params(
            gt_images_per_frame=gt_images_per_frame,
            gt_alphas_per_frame=gt_alphas_per_frame,
            **kwargs,
        )

    def _build_optimizers(self, params: nn.ParameterDict, config: TrigsTrainingConfig, scene_extent: float) -> dict[str, torch.optim.Optimizer]:
        """Lever 5 (HiGS 4.3, PennyGS 3.7): with ``adam_moments="bf16"`` every Adam of TRiGS becomes ``AdamBf16`` with the same
        groups and hyperparameters (bfloat16 moments, stochastic rounding, one fused kernel per parameter)."""
        optimizers: dict[str, torch.optim.Optimizer] = super()._build_optimizers(params, config, scene_extent)
        if self.adam_moments == "fp32":
            return optimizers
        from open4dgs.adam_bf16 import AdamBf16

        return {
            name: AdamBf16(
                [{"params": g["params"], "lr": g["lr"], "betas": g["betas"], "eps": g["eps"]} for g in opt.param_groups],
                seed=i,
            )
            for i, (name, opt) in enumerate(optimizers.items())
        }

    def _optimizer_step(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        """Lever 5 (recipe patch 0012's seam): every ``AdamBf16`` steps in ONE launch (``adam_bf16.fused_step``, bit-identical
        to their own steps); anything else steps as TRiGS does."""
        from open4dgs.adam_bf16 import AdamBf16, fused_step

        opts: list[torch.optim.Optimizer] = list(optimizers.values())
        if opts and all(isinstance(o, AdamBf16) for o in opts):
            fused_step(opts)  # pyrefly: ignore[bad-argument-type]  # checked above
            return
        super()._optimizer_step(optimizers)  # pyrefly: ignore[missing-attribute]  # recipe patch 0012

    def strided_cameras(self, cameras: CameraConfig, offset: tuple[int, int]) -> CameraConfig:
        """Lever 4: ``strided_camera_config(cameras, offset)``, built once per offset of one ``cameras`` object."""
        if cameras is not self._strided_of:
            self._strided_of, self._strided = cameras, {}
        out = self._strided.get(offset)
        if out is None:
            out = self._strided[offset] = strided_camera_config(cameras, offset)
        return out

    def _photometric_losses(
        self,
        params: nn.ParameterDict,
        render_colors: Tensor,
        render_alphas: Tensor,
        gt_rgb: Tensor,
        gt_alphas_per_frame: dict[int, list[Tensor]] | None,
        frame_idx: int,
        cam_idx: int,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Lever 6: the fused masked-composite loss when the configuration is the one it reproduces, else TRiGS's; once per
        distinct camera of a shared-time step."""
        memo = self._step_memo
        if memo is not None:
            key: tuple[str, int, int] = ("photo", frame_idx, cam_idx)
            if key not in memo:
                memo[key] = self._photometric_terms(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)
            return memo[key]
        return self._photometric_terms(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)

    def _photometric_terms(
        self,
        params: nn.ParameterDict,
        render_colors: Tensor,
        render_alphas: Tensor,
        gt_rgb: Tensor,
        gt_alphas_per_frame: dict[int, list[Tensor]] | None,
        frame_idx: int,
        cam_idx: int,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        compiled_terms = self._compiled_terms
        if not (
            compiled_terms is not None
            and config.masked_rgb_loss
            and config.masked_rgb_composite
            and not config.masked_rgb_random_bg
            and config.clamp_render_for_loss
            and "cam_gain" not in params
            and gt_alphas_per_frame is not None
        ):
            return super()._photometric_losses(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)
        mask: Tensor = gt_alphas_per_frame[frame_idx][cam_idx].to(self.device).float()
        alpha: Tensor = render_alphas[0].float()
        if mask.ndim != 2 or alpha.shape[-1] != 1 or tuple(mask.shape) != tuple(render_colors.shape[1:3]):
            return super()._photometric_losses(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)
        bg: Tensor = self._const_tensor(tuple(config.masked_rgb_bg_color), render_colors.device, render_colors.dtype)
        l1, pred_l, gt_l, alpha_out, alpha_mask = compiled_terms(render_colors[0], alpha, gt_rgb, mask, bg, float(config.alpha_loss_eps))
        from hexels_mojo.fused_ssim import fused_ssim

        ssim: Tensor = fused_ssim(pred_l.permute(2, 0, 1)[None], gt_l.permute(2, 0, 1)[None], padding="valid")
        return l1 * (1.0 - config.ssim_lambda) + (1.0 - ssim) * config.ssim_lambda, alpha_out, alpha_mask

    def _eager_deform(self, params: dict, t: Tensor, config: TrigsTrainingConfig) -> tuple[Tensor, Tensor, Tensor]:
        return TrigsTrainer._deform(self, params, t, config)  # pyrefly: ignore[bad-argument-type]

    def _deform(self, params: nn.ParameterDict | dict, t: float | Tensor, config: TrigsTrainingConfig) -> tuple[Tensor, Tensor, Tensor]:
        """With ``compile_deform``: TRiGS's deform compiled, the time always a device tensor (a float would be a constant
        of the graph, one recompile per frame)."""
        compiled = self._compiled_deform
        if compiled is None:
            return super()._deform(params, t, config)  # pyrefly: ignore[bad-argument-type]
        t_dev: Tensor = t if isinstance(t, Tensor) else self._const_tensor(float(t), params["means"].device, params["times"].dtype)
        return compiled(dict(params), t_dev, config)

    def _slice_rowwise_params(self, params: nn.ParameterDict | dict, idx: Tensor) -> dict:
        """TRiGS's per-row gather (``v[idx]`` for every row-wise tensor) as ``index_select``: its backward is one
        ``index_add_`` instead of the sort-based ``index_put_(accumulate=True)`` of advanced indexing. ``idx`` (the active
        rows) has no repeats, so no two adds meet a row and the gradients are the same."""
        n: int = int(params["means"].shape[0])
        return {k: v.index_select(0, idx) if isinstance(v, Tensor) and v.ndim >= 1 and int(v.shape[0]) == n else v for k, v in params.items()}

    def _const_tensor(self, value: object, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        """Lever 1: ``torch.tensor(value)`` on ``device``, built once per (value, device, dtype); never modify it in place."""
        key = (value, str(device), dtype)
        out = self._const_cache.get(key)
        if out is None:
            if len(self._const_cache) >= CONST_CACHE_LIMIT:
                self._const_cache.clear()
            out = self._const_cache[key] = torch.tensor(value, device=device, dtype=dtype)
        return out

    def _camera_group(  # pyrefly: ignore[bad-override]  # the recipe's default seam returns None (untyped)
        self,
        params: nn.ParameterDict,
        cameras: CameraConfig,
        cam_group: Sequence[int],
        frame_idx: int,
        gt_times_per_frame: Mapping[int, Sequence[float | None]] | None,
        offset_active: bool,
        sh_degree: int,
        config: TrigsTrainingConfig,
        step: int,
    ) -> SharedTimeGroup | None:
        """Levers 2 + 3: a ``SharedTimeGroup`` when the step's cameras share one time, else None (per-camera path)."""
        if len(cam_group) < 2 or offset_active or config.rs_enabled():
            return None
        default_t = float(frame_idx) / max(config.fps, 1e-6)
        times: set[float] = set()
        for c in cam_group:
            capture = gt_times_per_frame[frame_idx][c] if gt_times_per_frame is not None else None
            times.add(default_t if capture is None else float(capture))
        if len(times) != 1:
            return None
        return SharedTimeGroup(self, params, cameras, cam_group, times.pop(), sh_degree, config, step)
