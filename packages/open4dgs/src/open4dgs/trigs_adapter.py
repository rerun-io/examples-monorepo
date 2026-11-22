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
- init: ``InitCloud.to_trigs_mapping()`` through ``build_canonical_from_npz``.
"""

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float32
from torch import Tensor, nn
from trigs.core import losses as trigs_losses
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.init.from_colmap import build_canonical_from_npz

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud
from open4dgs.prior import MonoPriorLoss
from open4dgs.trigs_trainer import HostSizedCameraConfig, Open4dgsTrainer

StepCallback = Callable[[str, float, int], None]
"""Called with (tag, value, step) for every scalar the trainer reports."""


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

    def to_trigs(self, fps: float, keyframe_step: int, holdout_positions: tuple[int, ...]) -> TrigsTrainingConfig:
        """The TRiGS config for this run."""
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
        )


@dataclass
class ScalarLog:
    """Duck-typed SummaryWriter: keeps every ``add_scalar`` and forwards it to ``on_step``."""

    on_step: StepCallback | None = None
    rows: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.rows.setdefault(tag, []).append((int(step), float(value)))
        if self.on_step is not None:
            self.on_step(tag, float(value), int(step))

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

    def render(self, cameras: CameraConfig, cam: int, frame: int) -> tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w 1"]]:
        """RGB (premultiplied over black, clamped to [0, 1]) and alpha of camera ``cam`` of ``cameras`` at clip frame ``frame``."""
        with torch.no_grad():
            rgb, alpha, _, _ = self.trainer._render(self.params, cameras, cam, frame / self.fps, self.sh_degree, self.trigs_config)
        return rgb[0].clamp(0.0, 1.0), alpha[0]

    def render_depth(self, cameras: CameraConfig, cam: int, frame: int) -> tuple[Float32[Tensor, "h w"], Float32[Tensor, "h w"]]:
        """Expected (alpha-normalised) camera z-depth (gsplat ``RGB+ED``, recipe patch 0003) and alpha at clip frame ``frame``."""
        config: TrigsTrainingConfig = dataclasses.replace(self.trigs_config, render_depth=True)
        with torch.no_grad():
            _, alpha, _, info = self.trainer._render(self.params, cameras, cam, frame / self.fps, self.sh_degree, config)
        return info["render_depth"][0, ..., 0], alpha[0, ..., 0]

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
    world_T_cam: Float32[Tensor, "c 4 4"], K: Float32[Tensor, "c 3 3"], sizes: Sequence[tuple[int, int]], device: torch.device
) -> CameraConfig:
    """TRiGS cameras from camera-to-world poses (OpenCV RDF): viewmats are world -> camera; sizes as Python ints."""
    viewmats: Float32[Tensor, "c 4 4"] = torch.linalg.inv(world_T_cam.double()).float().to(device)
    return HostSizedCameraConfig.from_matrices(
        viewmats=viewmats, Ks=K.float().to(device).contiguous(), widths=[w for w, _ in sizes], heights=[h for _, h in sizes]
    )


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


def train_clip(
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    init: InitCloud,
    config: TrainConfig,
    holdout_names: tuple[str, ...],
    on_step: StepCallback | None = None,
    device: torch.device | None = None,
    prior: MonoPriorLoss | None = None,
) -> TrainedClip:
    """Train TRiGS on the clip's training cameras (every camera not in ``holdout_names``).

    ``prior`` (optional): the MoGe-2 prior loss holding maps of TRAINING cameras only; the trainer then renders
    ``RGB+ED`` and adds the loss per camera (recipe patch 0003); its ``prior/*`` scalars join ``scalars``.
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
    cameras: CameraConfig = camera_config(clip.world_T_cam, clip.K, clip.sizes, dev)
    writer: ScalarLog = ScalarLog(on_step=on_step)
    trainer: TrigsTrainer = Open4dgsTrainer(device=str(dev))
    trainer.camera_loss_hook = prior
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
        writer=writer,
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
        cameras=camera_config(clip.world_T_cam, clip.K, clip.sizes, dev),
        sh_degree=sh_degree,
        fps=clip.fps,
        num_frames=num_frames,
        scalars={},
        train_seconds=0.0,
        peak_gpu_bytes=0,
    )
