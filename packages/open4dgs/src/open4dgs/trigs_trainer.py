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

Four opt-in levers from PennyGS (arXiv 2611.27525) and HiGS (arXiv 2611.31918):

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

from open4dgs.fused_loss import masked_composite_terms
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
        batch_cameras: bool,
    ) -> None:
        self._trainer, self._params, self._config = trainer, params, config
        self._cam_group = list(cam_group)
        self._batch_cameras = batch_cameras
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
        self._batches: list[RenderOut] = []
        self._camera_views: list[tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...] | None, Tensor, dict]] = []
        self._batch_of: dict[int, tuple[int, int]] = {}
        if batch_cameras:
            by_size: dict[tuple[int, int], list[int]] = {}
            for group_i, cam in enumerate(self._cam_group):
                by_size.setdefault((self._cameras.width_for(cam), self._cameras.height_for(cam)), []).append(group_i)
            for members in by_size.values():
                for j, group_i in enumerate(members):
                    self._batch_of[group_i] = (len(self._batches), j)
                cams = [self._cam_group[g] for g in members]
                self._batches.append(trainer._rasterize_inputs(params, self._leaves, self._cameras, cams, config))
            # One unbind per output, not a slice per camera: a slice's backward fills a zero tensor of the WHOLE batch and
            # the cameras' gradients are then summed; unbind's backward stacks the cameras' gradients once.
            self._camera_views = [
                (colors.unbind(0), alphas.unbind(0), info["render_depth"].unbind(0) if "render_depth" in info else None, gamma, info)
                for colors, alphas, gamma, info in self._batches
            ]

    def render(self, group_i: int) -> RenderOut:
        """Camera ``cam_group[group_i]``'s render: its slice of the size batch, or its own rasterize call."""
        if not self._batch_cameras:
            out = self._trainer._rasterize_inputs(self._params, self._leaves, self._cameras, self._cam_group[group_i], self._config)
            out[3]["stride_offset"] = self._trainer.stride_state.offset
            self._last_info = out[3]
            return out
        b, j = self._batch_of[group_i]
        colors, alphas, depths, gamma, info = self._camera_views[b]
        cam_info: dict = {**info, "stride_offset": self._trainer.stride_state.offset}
        if depths is not None:
            cam_info["render_depth"] = depths[j][None]
        return colors[j][None], alphas[j][None], gamma, cam_info

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
        for batch in self._batches:
            relocation.accumulate_2d(reloc_state, batch[3], batch_size=n_group)
        outs = [self._outputs[k] for k in RENDER_INPUT_KEYS if self._leaves[k].grad is not None and self._outputs[k].requires_grad]
        grads = [self._leaves[k].grad for k in RENDER_INPUT_KEYS if self._leaves[k].grad is not None and self._outputs[k].requires_grad]
        if outs:
            torch.autograd.backward(outs, grads)
        self._batches, self._camera_views, self._loss, self._outputs, self._leaves = [], [], None, {}, {}
        self._trainer.stride_state.offset = None


# ---------------------------------------------------------------------------------------------------------------------
# Lever 4 (PennyGS 3.4): strided pixel sampling
# ---------------------------------------------------------------------------------------------------------------------


def strided_camera_config(cameras: CameraConfig, offset: tuple[int, int], stride: int = STRIDE) -> CameraConfig:
    """``cameras`` rendering only full-size pixels ``(offset[0] + stride * i, offset[1] + stride * j)``.

    gsplat samples pixel q at q + 0.5: the strided pixel q is full-size pixel p = stride * q + o, at p + 0.5, so
    ``f' = f / stride`` and ``c' = (c - o - 0.5) / stride + 0.5``. Sizes are the number of full-size pixels sampled.
    Rasterize with ``eps2d = GSPLAT_EPS2D / stride**2`` to keep the full-size dilation.
    """
    oy, ox = offset
    Ks: Tensor = strided_intrinsics(cameras.Ks, offset, stride)
    n: int = int(Ks.shape[0])
    widths: list[int] = [len(range(ox, cameras.width_for(c), stride)) for c in range(n)]
    heights: list[int] = [len(range(oy, cameras.height_for(c), stride)) for c in range(n)]
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
        pixel_stride: int = 1,
        adam_moments: Literal["fp32", "bf16"] = "fp32",
        fused_loss: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        if pixel_stride not in (1, STRIDE):
            raise ValueError(f"pixel_stride must be 1 or {STRIDE}, got {pixel_stride}")
        self.batch_cameras = batch_cameras
        self.pixel_stride: int = pixel_stride
        self.adam_moments: Literal["fp32", "bf16"] = adam_moments
        self.fused_loss: bool = fused_loss
        self._compiled_terms = torch.compile(masked_composite_terms, dynamic=False) if fused_loss else None
        self.stride_state: StrideState = StrideState()
        self._strided: dict[tuple[int, int], CameraConfig] = {}
        self._const_cache: dict[tuple[object, str, torch.dtype], Tensor] = {}

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

    def strided_cameras(self, cameras: CameraConfig, offset: tuple[int, int]) -> CameraConfig:
        """Lever 4: ``strided_camera_config(cameras, offset)``, built once per offset (the run's cameras do not change)."""
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
        """Lever 6: the fused masked-composite loss when the configuration is the one it reproduces, else TRiGS's."""
        if not (
            self.fused_loss
            and config.masked_rgb_loss
            and config.masked_rgb_composite
            and not config.masked_rgb_random_bg
            and getattr(config, "clamp_render_for_loss", True)
            and "cam_gain" not in params
            and gt_alphas_per_frame is not None
        ):
            return super()._photometric_losses(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)
        mask: Tensor = gt_alphas_per_frame[frame_idx][cam_idx].to(self.device).float()
        alpha: Tensor = render_alphas[0].float()
        if mask.ndim != 2 or alpha.shape[-1] != 1 or tuple(mask.shape) != tuple(render_colors.shape[1:3]):
            return super()._photometric_losses(params, render_colors, render_alphas, gt_rgb, gt_alphas_per_frame, frame_idx, cam_idx, config)
        bg: Tensor = self._const_tensor(tuple(config.masked_rgb_bg_color), render_colors.device, render_colors.dtype)
        assert self._compiled_terms is not None
        l1, pred_l, gt_l, alpha_out, alpha_mask = self._compiled_terms(render_colors[0], alpha, gt_rgb, mask, bg, float(config.alpha_loss_eps))
        from hexels_mojo.fused_ssim import fused_ssim

        ssim: Tensor = fused_ssim(pred_l.permute(2, 0, 1)[None], gt_l.permute(2, 0, 1)[None], padding="valid")
        return l1 * (1.0 - config.ssim_lambda) + (1.0 - ssim) * config.ssim_lambda, alpha_out, alpha_mask

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
        return SharedTimeGroup(self, params, cameras, cam_group, times.pop(), sh_degree, config, step, self.batch_cameras)
