"""Train TRiGS (recipes/open4dgs-trigs-src) in memory on a catalog clip.

What the trainer gets:
- cameras: every clip camera, ``CameraConfig.from_matrices`` with viewmats = inverse of ``clip.world_T_cam`` (world ->
  camera, OpenCV RDF), per-camera K and sizes (MAMMA mixes sizes);
- frames: ``gt_images_per_frame[f][c]``, uint8 HWC views of the clip frames, on the GPU when the training cameras'
  frames and masks fit ``RESIDENT_FREE_FRACTION`` of the free GPU memory (``trainer_frames``), else on the CPU (the
  trainer then copies each camera per step). The held-out cameras get a BLACK
  placeholder frame and an all-zero mask, so their pixels never reach the trainer; they are also listed in
  ``config.holdout_views`` so the trainer never samples them, and its in-loop eval is off (tb_stats_every=0);
- masks: ``gt_alphas_per_frame[f][c]`` float32 in [0, 1] (on the same device as the frames) with ``masked_rgb_loss=True`` and ``masked_rgb_composite=True``
  on a BLACK background: the trainer composites its (premultiplied) render over black and supervises the full frame
  against ``gt * mask`` (person matted onto black), plus two alpha terms normalised by the full frame:
  ``lambda_alpha_outside_mask`` (alpha outside the mask -> 0) and ``lambda_alpha_mask`` (BCE of alpha to the mask);
- init: ``InitCloud.to_trigs_mapping()`` through ``build_canonical_from_npz``;
- compensations (all off by default, as in TRiGS): a per-camera appearance affine, learned per-camera time offsets,
  rolling-shutter readout (``CameraConfig.rs_signs`` carries each camera's direction and readout share), and the
  dataset's own per-camera capture times (``gt_times_per_frame``). Held-out cameras get no learned value: their
  affine stays the identity and their offset the gauge value (``TrainedClip.render``).
"""

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray
from torch import Tensor, nn
from trigs.core import losses as trigs_losses
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.init.from_colmap import build_canonical_from_npz

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud
from open4dgs.prior import MonoPriorLoss
from open4dgs.trigs_trainer import HostSizedCameraConfig, Open4dgsTrainer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The trainer settings we expose; everything else keeps the TRiGS default."""

    steps: int = 1500
    """Optimizer steps."""
    cameras_per_step: int = 4
    """Cameras rendered per optimizer step (TRiGS ``batch_cameras``); cost per step scales with it."""
    sh_degree: int = 3
    """Maximum spherical-harmonics degree (TRiGS warms it up from 0 over 3,000 steps)."""
    ssim_lambda: float = 0.2
    """D-SSIM weight in the RGB loss."""
    lambda_alpha_outside_mask: float = 0.1
    """Alpha outside the person mask pushed to 0."""
    lambda_alpha_mask: float = 0.5
    """Alpha-to-mask BCE."""
    relocate_start_step: int = 1000
    """First step of MCMC relocation (TRiGS default 1000)."""
    seed: int = 0
    """Trainer seed (camera/frame sampling)."""
    log_every: int = 25
    """Scalar logging interval (loss/*)."""
    resident_frames: bool = True
    """Keep the training frames and masks on the GPU for the run when they fit (``resident_device``); off = the CPU path
    (the trainer copies each camera's frame and mask every step)."""
    learn_appearance: bool = False
    """Per-camera exposure / white-balance affine (TRiGS ``learn_camera_appearance``): the loss sees ``render * gain + bias``
    per channel; gain clamped to [0.5, 2], bias to [-0.25, 0.25]; L2 pull to the identity (weight 1e-2)."""
    learn_time_offsets: bool = False
    """Learned per-camera time offset added to each camera's capture time (TRiGS ``learn_camera_time_offsets``), from
    ``time_offset_warmup_steps`` on; zero-mean over the cameras (gauge), clamped to ``time_offset_clamp_ms``."""
    time_offset_warmup_steps: int = 8000
    """First step at which the learned offsets enter the render time (TRiGS default 8,000)."""
    time_offset_clamp_ms: float | None = None
    """Bound of each offset, ms; None = half a frame period at the clip fps (TRiGS: 8.3 ms = half a frame at 60 fps)."""
    rs_readout_ms: tuple[float, ...] = ()
    """Rolling-shutter readout, ms over the image: empty = global shutter (off); one value for every camera, or one per
    clip camera (clip order, held-out cameras included). A camera with 0 is global shutter."""
    rs_axis: Literal["row", "column"] = "row"
    """Readout axis in the delivered image: ``row`` (landscape, lines top to bottom) or ``column``."""
    rs_sign: tuple[float, ...] = (1.0,)
    """Readout direction per camera (one value for all, or one per clip camera): +1 = the first row / column is read
    first, -1 = the last. Capture time of line s: t + sign * readout * (s / S - 0.5) (trigs.capture_timing)."""

    def __post_init__(self) -> None:
        if any(r < 0.0 for r in self.rs_readout_ms):
            raise ValueError(f"rs_readout_ms must be >= 0, got {self.rs_readout_ms}")
        if any(s not in (-1.0, 1.0) for s in self.rs_sign):
            raise ValueError(f"rs_sign entries must be +1 or -1, got {self.rs_sign}")
        if not self.rs_sign:
            raise ValueError("rs_sign needs one value or one per camera")

    def rs_signs(self, num_cameras: int) -> tuple[float, ...] | None:
        """``CameraConfig.rs_signs``: sign * readout / max readout per camera (TRiGS takes one readout and scales it by
        the per-camera sign); None when every camera is global shutter."""
        if not self.rs_readout_ms or max(self.rs_readout_ms) == 0.0:
            return None
        readouts: tuple[float, ...] = _per_camera(self.rs_readout_ms, num_cameras, "rs_readout_ms")
        signs: tuple[float, ...] = _per_camera(self.rs_sign, num_cameras, "rs_sign")
        peak: float = max(readouts)
        return tuple(sign * readout / peak for sign, readout in zip(signs, readouts, strict=True))

    def to_trigs(self, fps: float, keyframe_step: int, holdout_positions: tuple[int, ...]) -> TrigsTrainingConfig:
        """The TRiGS config for this run."""
        clamp_ms: float = self.time_offset_clamp_ms if self.time_offset_clamp_ms is not None else 500.0 / fps
        return TrigsTrainingConfig(
            n_steps=self.steps,
            fps=fps,
            keyframe_stride=keyframe_step,
            batch_cameras=self.cameras_per_step,
            ssim_lambda=self.ssim_lambda,
            masked_rgb_loss=True,
            masked_rgb_composite=True,
            masked_rgb_bg_color=(0.0, 0.0, 0.0),
            lambda_alpha_outside_mask=self.lambda_alpha_outside_mask,
            lambda_alpha_mask=self.lambda_alpha_mask,
            relocate_start_iter=self.relocate_start_step,
            holdout_views=holdout_positions,
            train_seed=self.seed,
            tb_log_every=self.log_every,
            tb_stats_every=0,
            viewer_snapshot_every=0,
            verbose=False,
            learn_camera_appearance=self.learn_appearance,
            learn_camera_time_offsets=self.learn_time_offsets,
            camera_time_offset_warmup_steps=self.time_offset_warmup_steps,
            camera_time_offset_clamp_s=clamp_ms / 1000.0,
            rs_readout_ms=max(self.rs_readout_ms, default=0.0),
            rs_axis=self.rs_axis,
        )


def _per_camera(values: tuple[float, ...], num_cameras: int, name: str) -> tuple[float, ...]:
    """One value broadcast to every camera, or exactly one per camera."""
    if len(values) == 1:
        return values * num_cameras
    if len(values) != num_cameras:
        raise ValueError(f"{name} has {len(values)} values for {num_cameras} cameras (give one, or one per camera)")
    return values


@dataclass(frozen=True, slots=True)
class CameraCompensation:
    """One camera's nuisance parameters: ``render * gain + bias`` per channel, rendered at ``capture time + offset_s``."""

    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_s: float = 0.0


@dataclass
class ScalarLog:
    """Duck-typed SummaryWriter: keeps every ``add_scalar``."""

    rows: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.rows.setdefault(tag, []).append((int(step), float(value)))

    def __getattr__(self, name: str) -> Callable[..., None]:
        # add_histogram, flush, ...: ignored.
        return lambda *_args, **_kwargs: None


@dataclass(frozen=True)
class TrainedClip:
    """A trained 4D Gaussian model and what is needed to render it."""

    params: nn.ParameterDict
    trainer: TrigsTrainer
    trigs_config: TrigsTrainingConfig
    cameras: CameraConfig
    """Every clip camera, in clip order."""
    sh_degree: int
    fps: float
    num_frames: int
    scalars: dict[str, list[tuple[int, float]]]
    train_seconds: float
    peak_gpu_bytes: int
    capture_times: dict[int, list[float]] | None = None
    """``gt_times_per_frame`` the trainer was given (seconds per frame and clip camera); None = frame / fps."""

    def capture_time(self, cam: int, frame: int) -> float:
        """Base capture time of clip camera ``cam`` at clip frame ``frame``: the dataset's time, else frame / fps (no offset)."""
        return self.capture_times[frame][cam] if self.capture_times is not None else frame / self.fps

    def render_time(self, cameras: CameraConfig, cam: int, frame: int) -> float:
        """Time the trainer renders ``cam`` at: for the clip cameras the capture time plus the learned (gauge) offset, as
        TRiGS' eval does (``_eval_render_time``); other cameras (orbit) render at frame / fps."""
        if cameras is not self.cameras:
            return frame / self.fps
        return float(self.trainer._eval_render_time(self.params, frame, cam, self.capture_times, self.trigs_config))

    def render(
        self, cameras: CameraConfig, cam: int, frame: int, appearance: bool = False, compensation: CameraCompensation | None = None
    ) -> tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w 1"]]:
        """RGB (premultiplied over black, clamped to [0, 1]) and alpha of camera ``cam`` of ``cameras`` at clip frame ``frame``.

        Raw by default (no appearance affine, as TRiGS evaluates). ``appearance``: apply the camera's learned affine (the
        identity for held-out cameras). ``compensation``: render at capture time + ``offset_s`` with that affine instead
        of the learned values (the held-out test-time fit).
        """
        t: float = self.capture_time(cam, frame) + compensation.offset_s if compensation is not None else self.render_time(cameras, cam, frame)
        with torch.no_grad():
            rgb, alpha, _, _ = self.trainer._render(self.params, cameras, cam, t, self.sh_degree, self.trigs_config)
        out: Tensor = rgb[0]
        if compensation is not None:
            out = out * out.new_tensor(compensation.gain) + out.new_tensor(compensation.bias)
        elif appearance and "cam_gain" in self.params and cameras is self.cameras:
            out = out * self.params["cam_gain"][cam] + self.params["cam_bias"][cam]
        return out.clamp(0.0, 1.0), alpha[0]

    def render_depth(self, cameras: CameraConfig, cam: int, frame: int) -> tuple[Float32[Tensor, "h w"], Float32[Tensor, "h w"]]:
        """Expected (alpha-normalised) camera z-depth (gsplat ``RGB+ED``, recipe patch 0003) and alpha at clip frame ``frame``."""
        config: TrigsTrainingConfig = dataclasses.replace(self.trigs_config, render_depth=True)
        with torch.no_grad():
            _, alpha, _, info = self.trainer._render(self.params, cameras, cam, self.render_time(cameras, cam, frame), self.sh_degree, config)
        return info["render_depth"][0, ..., 0], alpha[0, ..., 0]

    def learned(self) -> list[CameraCompensation]:
        """Each clip camera's learned compensation (identity where a switch is off); ``offset_s`` is the effective offset."""
        n: int = len(self.cameras)
        offsets: list[float] = [0.0] * n
        if "camera_time_offsets" in self.params:
            offsets = [float(self.trainer._eval_render_time(self.params, 0, c, None, self.trigs_config)) for c in range(n)]
        gains: list[list[float]] = self.params["cam_gain"].detach().cpu().tolist() if "cam_gain" in self.params else [[1.0] * 3] * n
        biases: list[list[float]] = self.params["cam_bias"].detach().cpu().tolist() if "cam_bias" in self.params else [[0.0] * 3] * n
        return [CameraCompensation(gain=tuple(g), bias=tuple(b), offset_s=o) for g, b, o in zip(gains, biases, offsets, strict=True)]  # type: ignore[arg-type]

    def save(self, path: Path) -> None:
        """Checkpoint (.pt): the parameters on the CPU plus the scalars needed to render them."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": {name: value.detach().cpu() for name, value in self.params.items()},
                "sh_degree": self.sh_degree,
                "fps": self.fps,
                "num_frames": self.num_frames,
            },
            path,
        )


def camera_config(
    world_T_cam: Float32[Tensor, "c 4 4"],
    K: Float32[Tensor, "c 3 3"],
    sizes: Sequence[tuple[int, int]],
    device: torch.device,
    rs_signs: tuple[float, ...] | None = None,
) -> CameraConfig:
    """TRiGS cameras from camera-to-world poses (OpenCV RDF): viewmats are world -> camera; sizes as Python ints;
    ``rs_signs`` per camera (``TrainConfig.rs_signs``)."""
    viewmats: Float32[Tensor, "c 4 4"] = torch.linalg.inv(world_T_cam.double()).float().to(device)
    return HostSizedCameraConfig.from_matrices(
        viewmats=viewmats,
        Ks=K.float().to(device).contiguous(),
        widths=[w for w, _ in sizes],
        heights=[h for _, h in sizes],
        rs_signs=rs_signs,
    )


def capture_times_per_frame(times_s: Float64[ndarray, "cams frames"]) -> dict[int, list[float]]:
    """``gt_times_per_frame[f][c]`` from each camera's own frame times, relative to camera 0's first frame."""
    rel: Float64[ndarray, "cams frames"] = np.asarray(times_s, dtype=np.float64) - float(times_s[0, 0])
    return {f: [float(rel[c, f]) for c in range(rel.shape[0])] for f in range(rel.shape[1])}


def init_params(init: InitCloud, sh_degree: int, device: torch.device) -> tuple[nn.ParameterDict, int, float]:
    """TRiGS canonical parameters from the init cloud: (params, sh_degree, scene_extent)."""
    return build_canonical_from_npz(
        init.to_trigs_mapping(), fps=init.fps, keyframe_stride=init.keyframe_step, max_sh_degree=sh_degree, device=str(device)
    )


RESIDENT_FREE_FRACTION: float = 0.5
"""Training frames and masks go to the GPU when they need at most this share of the free GPU memory at train start."""


def trainer_frames(
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    holdout_names: tuple[str, ...],
    device: torch.device | None = None,
) -> tuple[dict[int, list[Tensor]], dict[int, list[Tensor]], tuple[int, ...]]:
    """What the trainer sees: ``gt_images[f][c]``, ``gt_alphas[f][c]`` in clip camera order, and the held-out positions.

    Held-out cameras get a black frame and an all-zero mask at every frame; their positions (indices into
    ``clip.camera_names``, the trainer's camera order) go to ``TrigsTrainingConfig.holdout_views``.

    ``device``: where the frames (uint8) and masks (float32, the values the trainer would copy) live. A CUDA device
    keeps them resident for the whole run (one copy per camera here, none per step); ``None`` keeps them where they
    are. Held-out data is never copied.
    """
    holdout_positions: tuple[int, ...] = tuple(clip.camera_names.index(name) for name in holdout_names)
    frames: dict[int, Tensor] = {}
    alphas: dict[int, Tensor] = {}
    for c in range(len(clip.camera_names)):
        if c in holdout_positions:
            frames[c] = torch.zeros_like(clip.frames[c][:1], device=device).expand_as(clip.frames[c])
            alphas[c] = torch.zeros_like(masks[c][:1], dtype=torch.float32, device=device).expand_as(masks[c])
        else:
            frames[c] = clip.frames[c] if device is None else clip.frames[c].to(device)
            alphas[c] = masks[c].to(device=device, dtype=torch.float32)
    num_frames: int = int(clip.frame_indices.shape[0])
    gt_images: dict[int, list[Tensor]] = {f: [frames[c][f] for c in range(len(clip.camera_names))] for f in range(num_frames)}
    gt_alphas: dict[int, list[Tensor]] = {f: [alphas[c][f] for c in range(len(clip.camera_names))] for f in range(num_frames)}
    return gt_images, gt_alphas, holdout_positions


def resident_device(
    clip: MultiViewClip, masks: tuple[Tensor, ...], holdout_names: tuple[str, ...], device: torch.device
) -> torch.device | None:
    """``device`` when the training cameras' frames plus float32 masks fit ``RESIDENT_FREE_FRACTION`` of its free memory."""
    if device.type != "cuda":
        return None
    train: list[int] = [c for c, name in enumerate(clip.camera_names) if name not in holdout_names]
    need: int = sum(
        (0 if clip.frames[c].device == device else clip.frames[c].nbytes) + (0 if masks[c].device == device else masks[c].numel() * 4)
        for c in train
    )
    free, _ = torch.cuda.mem_get_info(device)
    if need <= RESIDENT_FREE_FRACTION * free:
        return device
    print(f"[open4dgs] frames + masks {need / 1e9:.2f} GB > {RESIDENT_FREE_FRACTION} x {free / 1e9:.2f} GB free: kept on the CPU")
    return None


StepCallback: TypeAlias = Callable[[int, "TrainedClip"], None]
"""``on_step(steps_done, live)``: after every optimizer step. ``live`` renders the model as trained so far (it holds the
trainer's own parameters, which TRiGS updates in place; ``train_seconds`` and ``peak_gpu_bytes`` are 0 until the end).
The trainer avoids host syncs, so a callback that renders or reads a tensor should do so only every few hundred steps.
An exception raised here ends the run and propagates out of ``train_clip`` (a Stop)."""


class _StepTicks:
    """TRiGS' ``progress_bar`` (a duck-typed tqdm: ``update`` after each step, ``set_postfix``) that calls ``on_step``."""

    def __init__(self, on_step: StepCallback, live: "TrainedClip") -> None:
        self.on_step: StepCallback = on_step
        self.live: TrainedClip = live
        self.steps: int = 0

    def update(self, n: int = 1) -> None:
        self.steps += n
        self.on_step(self.steps, self.live)

    def set_postfix(self, **_values: object) -> None:
        pass


def train_clip(
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    init: InitCloud,
    config: TrainConfig,
    holdout_names: tuple[str, ...],
    device: torch.device | None = None,
    prior: MonoPriorLoss | None = None,
    capture_times: Float64[ndarray, "cams frames"] | None = None,
    on_step: StepCallback | None = None,
) -> TrainedClip:
    """Train TRiGS on the clip's training cameras (every camera not in ``holdout_names``).

    ``prior`` (optional): the MoGe-2 prior loss holding maps of TRAINING cameras only; the trainer then renders
    ``RGB+ED`` and adds the loss per camera (recipe patch 0003); its ``prior/*`` scalars join ``scalars``.
    ``capture_times`` (optional): each clip camera's own frame times (seconds, clip order), given to the trainer as
    ``gt_times_per_frame`` instead of frame / fps.
    ``on_step`` (optional): progress, previews and Stop (``StepCallback``).
    """
    dev: torch.device = device if device is not None else torch.device("cuda")
    # TRiGS's loss falls back to torchmetrics SSIM (another number, ~10x slower) when hexels_mojo.fused_ssim does not import.
    if trigs_losses._get_ssim_backend() != "fused":
        raise RuntimeError("hexels_mojo.fused_ssim does not import (Mojo kernels not built?): run `pixi run -e open4dgs open4dgs-ssim-mojo-build`")
    num_frames: int = int(clip.frame_indices.shape[0])
    if len(masks) != len(clip.camera_names) or any(m.shape[0] != num_frames for m in masks):
        raise ValueError("masks must be one (frames, h, w) tensor per clip camera")
    gt_images, gt_alphas, holdout_positions = trainer_frames(
        clip, masks, holdout_names, resident_device(clip, masks, holdout_names, dev) if config.resident_frames else None
    )
    trigs_config: TrigsTrainingConfig = config.to_trigs(clip.fps, init.keyframe_step, holdout_positions)
    if prior is not None:
        leaked: set[int] = set(prior.positions) & set(holdout_positions)
        if leaked:
            raise ValueError(f"the prior holds maps of held-out cameras {sorted(clip.camera_names[c] for c in leaked)}")
        trigs_config = dataclasses.replace(trigs_config, render_depth=True)
    torch.manual_seed(config.seed)
    params, sh_degree, scene_extent = init_params(init, config.sh_degree, dev)
    cameras: CameraConfig = camera_config(clip.world_T_cam, clip.K, clip.sizes, dev, config.rs_signs(len(clip.camera_names)))
    gt_times: dict[int, list[float]] | None = None
    if capture_times is not None:
        if capture_times.shape != (len(clip.camera_names), num_frames):
            raise ValueError(f"capture_times {capture_times.shape} must be (cameras, frames) = {(len(clip.camera_names), num_frames)}")
        gt_times = capture_times_per_frame(capture_times)
    writer: ScalarLog = ScalarLog()
    trainer: TrigsTrainer = Open4dgsTrainer(device=str(dev))
    trainer.camera_loss_hook = prior
    ticks: _StepTicks | None = None
    if on_step is not None:
        live: TrainedClip = TrainedClip(
            params=params,
            trainer=trainer,
            trigs_config=trigs_config,
            cameras=cameras,
            sh_degree=sh_degree,
            fps=clip.fps,
            num_frames=num_frames,
            scalars=writer.rows,
            train_seconds=0.0,
            peak_gpu_bytes=0,
            capture_times=gt_times,
        )
        ticks = _StepTicks(on_step, live)
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    start: float = time.perf_counter()
    result = trainer.train_from_params(
        params=params,
        sh_degree=sh_degree,
        scene_extent=scene_extent,
        cameras=cameras,
        n_frames=num_frames,
        gt_images_per_frame=gt_images,
        gt_alphas_per_frame=gt_alphas,
        config=trigs_config,
        gt_times_per_frame=gt_times,
        writer=writer,
        progress_bar=ticks,
    )
    torch.cuda.synchronize(dev)
    train_seconds: float = time.perf_counter() - start
    if prior is not None:
        prior.finish()
        writer.rows.update(prior.rows)
    return TrainedClip(
        params=result.params,
        trainer=trainer,
        trigs_config=trigs_config,
        cameras=cameras,
        sh_degree=sh_degree,
        fps=clip.fps,
        num_frames=num_frames,
        scalars=writer.rows,
        train_seconds=train_seconds,
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated(dev)),
        capture_times=gt_times,
    )


def untrained_clip(
    clip: MultiViewClip, init: InitCloud, config: TrainConfig, holdout_names: tuple[str, ...], device: torch.device | None = None
) -> TrainedClip:
    """The init cloud as a renderable model (no training): the baseline the trained held-out numbers must beat."""
    dev: torch.device = device if device is not None else torch.device("cuda")
    holdout_positions: tuple[int, ...] = tuple(clip.camera_names.index(name) for name in holdout_names)
    params, sh_degree, _ = init_params(init, config.sh_degree, dev)
    trainer: TrigsTrainer = TrigsTrainer(device=str(dev))
    num_frames: int = int(clip.frame_indices.shape[0])
    trainer._sequence_t_max = (num_frames - 1) / clip.fps
    return TrainedClip(
        params=params,
        trainer=trainer,
        trigs_config=config.to_trigs(clip.fps, init.keyframe_step, holdout_positions),
        cameras=camera_config(clip.world_T_cam, clip.K, clip.sizes, dev, config.rs_signs(len(clip.camera_names))),
        sh_degree=sh_degree,
        fps=clip.fps,
        num_frames=num_frames,
        scalars={},
        train_seconds=0.0,
        peak_gpu_bytes=0,
    )
