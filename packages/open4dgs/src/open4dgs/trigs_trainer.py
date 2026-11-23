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
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

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
        trainer: TrigsTrainer,
        params: nn.ParameterDict,
        cameras: CameraConfig,
        cam_group: Sequence[int],
        t: float,
        sh_degree: int,
        config: TrigsTrainingConfig,
        step: int,
        batch_cameras: bool,
    ) -> None:
        self._trainer, self._params, self._cameras, self._config = trainer, params, cameras, config
        self._cam_group = list(cam_group)
        self._batch_cameras = batch_cameras
        self._outputs: dict = trainer._render_inputs(params, cameras, self._cam_group[0], t, sh_degree, config, step=step)
        self._leaves: dict = dict(self._outputs)
        for key in RENDER_INPUT_KEYS:
            self._leaves[key] = self._outputs[key].detach().requires_grad_(self._outputs[key].requires_grad)
        self._last_info: dict | None = None
        self._loss: Tensor | None = None
        self._batches: list[RenderOut] = []
        self._batch_of: dict[int, tuple[int, int]] = {}
        if batch_cameras:
            by_size: dict[tuple[int, int], list[int]] = {}
            for group_i, cam in enumerate(self._cam_group):
                by_size.setdefault((cameras.width_for(cam), cameras.height_for(cam)), []).append(group_i)
            for members in by_size.values():
                for j, group_i in enumerate(members):
                    self._batch_of[group_i] = (len(self._batches), j)
                cams = [self._cam_group[g] for g in members]
                self._batches.append(trainer._rasterize_inputs(params, self._leaves, cameras, cams, config))

    def render(self, group_i: int) -> RenderOut:
        """Camera ``cam_group[group_i]``'s render: its slice of the size batch, or its own rasterize call."""
        if not self._batch_cameras:
            out = self._trainer._rasterize_inputs(self._params, self._leaves, self._cameras, self._cam_group[group_i], self._config)
            self._last_info = out[3]
            return out
        b, j = self._batch_of[group_i]
        colors, alphas, gamma, info = self._batches[b]
        cam_info = dict(info)
        if "render_depth" in info:
            cam_info["render_depth"] = info["render_depth"][j : j + 1]
        return colors[j : j + 1], alphas[j : j + 1], gamma, cam_info

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
        self._batches, self._loss, self._outputs, self._leaves = [], None, {}, {}


class Open4dgsTrainer(TrigsTrainer):
    """``TrigsTrainer`` with the three speed levers on the recipe's seams (all on by default)."""

    def __init__(self, *args: object, batch_cameras: bool = True, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        self.batch_cameras = batch_cameras
        self._const_cache: dict[tuple[object, str, torch.dtype], Tensor] = {}

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

