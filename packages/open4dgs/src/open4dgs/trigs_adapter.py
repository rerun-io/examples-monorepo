"""Train TRiGS (recipes/open4dgs-trigs-src) in memory on a catalog clip.

What the trainer gets:
- cameras: every clip camera, ``CameraConfig.from_matrices`` with viewmats = inverse of ``clip.world_T_cam`` (world ->
  camera, OpenCV RDF), per-camera K and sizes (MAMMA mixes sizes);
- frames: ``gt_images_per_frame[f][c]``, uint8 HWC CPU views of the clip frames. The held-out cameras get a BLACK
  placeholder frame and an all-zero mask, so their pixels never reach the trainer; they are also listed in
  ``config.holdout_views`` so the trainer never samples them, and its in-loop eval is off (tb_stats_every=0);
- masks: ``gt_alphas_per_frame[f][c]`` float in [0, 1] with ``masked_rgb_loss=True`` and ``masked_rgb_composite=True``
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
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.init.from_colmap import build_canonical_from_npz

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud
from open4dgs.prior import MonoPriorLoss

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
    """TRiGS cameras from camera-to-world poses (OpenCV RDF): viewmats are world -> camera."""
    viewmats: Float32[Tensor, "c 4 4"] = torch.linalg.inv(world_T_cam.double()).float().to(device)
    return CameraConfig.from_matrices(
        viewmats=viewmats, Ks=K.float().to(device).contiguous(), widths=[w for w, _ in sizes], heights=[h for _, h in sizes]
    )


def init_params(init: InitCloud, sh_degree: int, device: torch.device) -> tuple[nn.ParameterDict, int, float]:
    """TRiGS canonical parameters from the init cloud: (params, sh_degree, scene_extent)."""
    return build_canonical_from_npz(
        init.to_trigs_mapping(), fps=init.fps, keyframe_stride=init.keyframe_step, max_sh_degree=sh_degree, device=str(device)
    )


def trainer_frames(
    clip: MultiViewClip, masks: tuple[Float32[Tensor, "frames h w"], ...], holdout_names: tuple[str, ...]
) -> tuple[dict[int, list[Tensor]], dict[int, list[Tensor]], tuple[int, ...]]:
    """What the trainer sees: ``gt_images[f][c]``, ``gt_alphas[f][c]`` in clip camera order, and the held-out positions.

    Held-out cameras get a black frame and an all-zero mask at every frame; their positions (indices into
    ``clip.camera_names``, the trainer's camera order) go to ``TrigsTrainingConfig.holdout_views``.
    """
    holdout_positions: tuple[int, ...] = tuple(clip.camera_names.index(name) for name in holdout_names)
    black: dict[int, Tensor] = {c: torch.zeros_like(clip.frames[c][0]) for c in holdout_positions}
    empty: dict[int, Tensor] = {c: torch.zeros_like(masks[c][0]) for c in holdout_positions}
    num_frames: int = int(clip.frame_indices.shape[0])
    gt_images: dict[int, list[Tensor]] = {
        f: [black[c] if c in black else clip.frames[c][f] for c in range(len(clip.camera_names))] for f in range(num_frames)
    }
    gt_alphas: dict[int, list[Tensor]] = {
        f: [empty[c] if c in empty else masks[c][f] for c in range(len(clip.camera_names))] for f in range(num_frames)
    }
    return gt_images, gt_alphas, holdout_positions


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
    num_frames: int = int(clip.frame_indices.shape[0])
    if len(masks) != len(clip.camera_names) or any(m.shape[0] != num_frames for m in masks):
        raise ValueError("masks must be one (frames, h, w) tensor per clip camera")
    gt_images, gt_alphas, holdout_positions = trainer_frames(clip, masks, holdout_names)
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
    trainer: TrigsTrainer = TrigsTrainer(device=str(dev))
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
