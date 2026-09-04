"""Fine-tune trainable ZipDepth-base weights on catalog PromptDA targets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from math import ceil, isclose, isfinite
from pathlib import Path
from typing import Literal, TypeAlias

import torch
import torch.distributed as dist
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from monopriors.models.zipdepth_checkpoint import (
    load_zipdepth_checkpoint,
    load_zipdepth_state_dict,
    narrow_zipdepth_state_dict,
    range_margin_coverage_max_from_checkpoint,
    range_margin_m_from_checkpoint,
)
from monopriors.third_party.zipdepth.architecture import ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes
from serde import field as serde_field
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor, nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipdepth.catalog.instrument import InstrumentedLoader
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog, split_holdout_segments
from zipdepth.catalog.targets import AugmentPolicy, TargetMode
from zipdepth.catalog.ultrawide import DEFAULT_ULTRAWIDE_PROMPT_SCALE, CameraSelection, UltrawidePolicy
from zipdepth.training.distributed import barrier, cleanup_distributed, fix_state_dict_prefix, print_main, setup_distributed
from zipdepth.training.trainer import ZipDepthTrainer

AmpDtype: TypeAlias = Literal["bfloat16", "float16"]
CompileMode: TypeAlias = Literal["off", "default", "reduce-overhead", "max-autotune"]
ScheduleName: TypeAlias = Literal["onecycle", "constant-cooldown"]
TrainPreset: TypeAlias = Literal["v4", "fast"]


@serde
@dataclass(frozen=True, slots=True)
class UpstreamOptimizerConfig:
    """Upstream optimizer settings used by catalog training."""

    lr: float = 1e-3
    """Base learning rate before the fine-tuning policy is applied."""
    weight_decay: float = 0.01
    """AdamW weight decay."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamSchedulerConfig:
    """Upstream OneCycle scheduler settings used by catalog training."""

    pct_start: float = 0.05
    """Fraction of optimizer steps spent increasing the learning rate."""
    div_factor: float = 25.0
    """Ratio between the peak and initial learning rates."""
    final_div_factor: float = 1000.0
    """Ratio between the initial and final learning rates."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamLossConfig:
    """Upstream relative-depth loss settings used by catalog training."""

    alpha_ssi: float = 1.0
    """Weight of the scale-and-shift-invariant loss term."""
    alpha_grad: float = 2.0
    """Weight of the upstream gradient loss term."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamAmpConfig:
    """Upstream automatic mixed-precision settings."""

    enabled: bool = True
    """Whether automatic mixed precision is enabled."""
    dtype: AmpDtype = "bfloat16"
    """Floating-point format used by automatic mixed precision."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamModelConfig:
    """Upstream model construction settings used by catalog training."""

    upsample_unfold: bool = True
    """Whether ZipDepth uses its unfold-based upsampling path."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamTrainingConfig:
    """Upstream checkpoint-retention settings used by catalog training."""

    max_step_checkpoints: int = 5
    """Maximum number of periodic step checkpoints retained."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamTrainConfig:
    """Typed subset of the upstream training JSON consumed by this command."""

    optimizer: UpstreamOptimizerConfig = field(default_factory=UpstreamOptimizerConfig)
    """Optimizer settings."""
    scheduler: UpstreamSchedulerConfig = field(default_factory=UpstreamSchedulerConfig)
    """OneCycle scheduler settings."""
    loss: UpstreamLossConfig = field(default_factory=UpstreamLossConfig)
    """Relative-depth loss settings."""
    amp: UpstreamAmpConfig = field(default_factory=UpstreamAmpConfig)
    """Automatic mixed-precision settings."""
    model: UpstreamModelConfig = field(default_factory=UpstreamModelConfig)
    """Model construction settings."""
    training: UpstreamTrainingConfig = field(default_factory=UpstreamTrainingConfig)
    """Checkpoint-retention settings."""


@serde
@dataclass(slots=True)
class TrainCatalogConfig:
    """Catalog fine-tuning configuration."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset containing ARKitScenes and PromptDA layers."""
    num_segments: int | None = None
    """Number of training-split segments to use; None uses all of them."""
    segment_ids: list[str] | None = None
    """Explicit training segment identifiers; overrides ``num_segments``."""
    holdout_count: int = 20
    """Number of PromptDA segments reserved from training for evaluation."""
    holdout_seed: int = 0
    """Seed for the deterministic holdout split."""
    height: int = 768
    """Training image height after augmentation."""
    width: int = 1024
    """Training image width after augmentation."""
    batch_size: int = 8
    """Samples per rank and optimizer step."""
    dataloader: Literal["current", "rerun"] = "current"
    """Current threaded dataset or the Rerun-0.37 dataloader A/B path."""
    rerun_fetch_block_size: int = 512
    """Samples requested per catalog fetch by the Rerun dataloader path."""
    target_mode: TargetMode = "metric"
    """Prompted metric distillation by default; ``ssi`` retains the legacy RGB-only lane."""
    cameras: CameraSelection = "wide"
    """Cameras streamed per segment. ``wide`` reproduces the released lane exactly.
    ``ultrawide`` trains on the rectified ultrawide frames with the wide LiDAR prompt
    padded into its footprint. ``both`` interleaves an equal-sized ultrawide subsample
    with the wide frames, giving roughly balanced batches after the shuffle buffer."""
    ultrawide_min_valid_fraction: float = 0.7
    """Reject an ultrawide frame whose in-range target fraction, measured before erosion,
    falls below this. The raycast depth has holes at glazing and outdoors."""
    ultrawide_valid_erosion_px: int = 1
    """Binary-erode the ultrawide target mask by this many pixels; raycast silhouettes
    bleed about one output pixel past the mesh."""
    ultrawide_prompt_scale: float = DEFAULT_ULTRAWIDE_PROMPT_SCALE
    """Prompt block size as a fraction of the prompt canvas per axis. The rectified
    ultrawide camera sees 2.1x the wide field of view on both axes (measured across
    landscape and portrait segments; principal point within 0.7% of centre), so the wide
    LiDAR prompt covers the central 1/2.1 of an ultrawide frame."""
    total_steps: int = 70_000
    """Optimizer steps in the run; the OneCycle schedule spans exactly this many. 70k is about one pass over the 875-segment train split at batch 8 (70.5k steps measured 2026-08-28)."""
    shuffle_buffer_size: int = 256
    """Maximum decoded samples held by the current loader for bounded random shuffling. CUDA samples
    use about 4.7 MB each at 1024x768, so this and the default producer queue
    retain about 2.4 GB of device memory together."""
    seed: int = 0
    """Base seed for dataset segment order and shuffle-buffer eviction."""
    num_producers: int = 6
    """Whole-segment producer threads used by the current loader."""
    prefetch_samples: int = 256
    """Maximum completed samples buffered by the current loader."""
    min_depth_span: float = 1.25
    """Minimum valid-depth p95/p5 ratio; zero disables flat-frame filtering."""
    range_margin_m: float = 0.0
    """Widen the head's output range beyond the prompt's own [min, max] by this many metres
    on each side; needed for ultrawide prompts that cover only the image center. The margin
    is absolute rather than a fraction of the prompt span so that a close-up prompt gains as
    much reach as a wide one. Clipping to the 0.1--4.0 m validity window means 3.9 or more
    opens the full window. The value is recorded in ``resolved_config.json`` and in every
    checkpoint this run writes, so inference rebuilds the same head. Zero reproduces the
    prompt-bounded head exactly."""
    range_margin_coverage_max: float = 1.0
    """Apply ``range_margin_m`` only to images whose valid prompt covers at most this
    fraction of the 192x256 prompt canvas. A wide prompt fills the canvas (coverage ~1.0)
    and does not want a wider head: an unconditional 3.9 m margin cost the wide lane
    AbsRel 0.19 against 0.013 after 600 steps. An ultrawide prompt is the wide LiDAR block
    padded into the canvas centre and covers at most 0.24, so 0.6 separates the two lanes
    cleanly and the ``both`` run widens only the frames that need it. The default 1.0 is
    the largest coverage possible, so the gate admits every image and the head behaves
    exactly as it did before the gate existed. Recorded in ``resolved_config.json`` and in
    every checkpoint, like the margin itself."""
    from_scratch: bool = False
    """Start with random weights instead of the released ZipDepth-base weights."""
    metric_gradient_weight: float = 0.5
    """Weight on the multi-scale gradient term in the metric loss (L1 + w * grad).
    At 0.5 the term supplies only ~9% of the objective; upstream ZipDepth weights its
    equivalent term at 2.0. Raising it targets depth-discontinuity error, where the model
    beats a bilinear prompt upsample by the smallest margin."""
    freeze_bn: bool = True
    """Pin BatchNorm to eval mode while fine-tuning: forwards use the released running
    stats instead of batch statistics, and the stats never drift toward the new data.
    Without this, train-mode forwards rewrite the running stats within ~100 small
    batches and eval-mode loss degrades even at zero learning rate. Ignored from scratch."""
    init_checkpoint: Path | None = None
    """Local trainable ``.pth`` initialization; overrides the released checkpoint."""
    train_config: Path = Path("configs/default.json")
    """Upstream JSON providing loss, optimizer, scheduler, AMP, and model settings."""
    save_dir: Path = Path("data/checkpoints/catalog")
    """Directory for split manifests, TensorBoard events, and checkpoints."""
    save_every_steps: int = 500
    """Write a step checkpoint after this many updates; zero disables step checkpoints."""
    resume: Path | None = None
    """Training checkpoint whose model, optimizer, scheduler, and global step are restored."""
    profile: bool = False
    """Record one torch.profiler trace (CPU+CUDA ops, shapes, memory, stacks) and print the
    trainer's bottleneck summary. Uses the trainer's existing profiler machinery."""
    profile_dir: Path = Path("data/profiler")
    """Output directory for the profiler trace, summary, and TensorBoard events."""
    profile_wait: int = 100
    """Steps to skip before the profiler warms up; must clear the torch.compile warmup so the
    recorded window reflects steady-state kernels, not compilation."""
    profile_warmup: int = 5
    """Profiler schedule — steps the profiler warms up on without recording."""
    profile_active: int = 10
    """Profiler schedule — steps actually recorded in the trace."""
    max_lr: float | None = None
    """OneCycle peak LR; None uses config LR / 100 for fine-tuning and config LR from scratch.
    Measured on the fixed probe set: at config/10 (1e-4, batch 4) fine-tuning diverges once
    warmup ends; at config/100 (1e-5) it improves the probe loss ~29% in 120 steps."""
    preset: TrainPreset = "v4"
    """Named training recipe. ``fast`` intentionally matches ``v4`` until the hill-climb winner lands."""
    compile_mode: CompileMode = "reduce-overhead"
    """``torch.compile`` mode for the student; ``off`` disables it. Beartype dev mode
    disables compilation regardless of this setting."""
    channels_last: bool = False
    """Store the model and four-dimensional training tensors in channels-last format."""
    fused_adamw: bool = False
    """Use the fused CUDA AdamW implementation."""
    amp_dtype: AmpDtype | None = None
    """AMP dtype override; None keeps the upstream JSON setting."""
    schedule: ScheduleName = "onecycle"
    """Learning-rate schedule."""
    warmup_pct: float | None = None
    """OneCycle warmup fraction; None keeps the upstream JSON setting."""
    final_div_factor: float | None = None
    """Ratio between the initial and final learning rates; None keeps the upstream JSON setting."""
    cooldown_pct: float = 0.1
    """Fraction of total steps used by the linear tail of ``constant-cooldown``."""
    stage_schedule: str | None = None
    """Optional ``HEIGHTxWIDTH:FRACTION,...`` progressive-resolution schedule. Fractions
    must sum to one; the catalog loader is rebuilt at each stage boundary."""


@dataclass(frozen=True, slots=True)
class ResolutionStage:
    """One progressive-resolution optimizer-step interval."""

    height: int
    """Training image height during the stage."""
    width: int
    """Training image width during the stage."""
    start_step: int
    """Inclusive global optimizer step at which the stage starts."""
    end_step: int
    """Exclusive global optimizer step at which the stage ends."""


@serde
@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    """Resolved runtime controls selected by a preset and explicit overrides."""

    compile_mode: CompileMode
    """torch.compile mode, or ``off``."""
    channels_last: bool
    """Whether model and batch tensors use channels-last storage."""
    fused_adamw: bool
    """Whether AdamW uses its fused implementation."""
    amp_dtype: AmpDtype
    """Automatic mixed-precision dtype."""
    schedule: ScheduleName
    """Learning-rate schedule name."""
    warmup_pct: float
    """OneCycle warmup fraction."""
    final_div_factor: float
    """Ratio between the initial and final learning rates."""
    cooldown_pct: float
    """Constant-cooldown tail fraction."""
    stage_schedule: str | None
    """Optional progressive-resolution schedule."""


@serde
@dataclass(frozen=True, slots=True)
class ResolvedTrainCatalogConfig:
    """Catalog training configuration plus values resolved at runtime."""

    config: TrainCatalogConfig = serde_field(flatten=True)
    """User-provided catalog training configuration, flattened in the document."""
    rank_count: int
    """Number of distributed ranks participating in training."""
    steps_per_epoch: int
    """Optimizer steps assigned to the single catalog epoch."""
    total_steps: int
    """Total optimizer steps in the run."""
    resolved_max_lr: float
    """Peak learning rate after applying the initialization policy."""
    recipe: TrainingRecipe
    """Fully resolved runtime recipe used by training."""


def resolve_max_lr(config_lr: float, override: float | None, from_scratch: bool, preset: TrainPreset = "v4") -> float:
    """Resolve the OneCycle peak learning rate from initialization policy."""
    # v4 fine-tunes the released weights at config/100 (1e-5). The hill-climb of 2026-09-03 found config/10 (1e-4) reaches
    # the same quick-holdout quality in 2.5-3x fewer steps and keeps improving where v4 stalls; nothing else moved.
    divisor: float = 10.0 if preset == "fast" else 100.0
    max_lr: float = override if override is not None else config_lr if from_scratch else config_lr / divisor
    if max_lr <= 0.0:
        raise ValueError("max_lr must be positive")
    return max_lr


def resolve_amp_dtype(override: AmpDtype | None, configured: AmpDtype) -> AmpDtype:
    """Resolve the CLI AMP dtype override against the upstream JSON setting."""
    return configured if override is None else override


def resolve_training_recipe(
    config: TrainCatalogConfig,
    scheduler_config: UpstreamSchedulerConfig,
    amp_config: UpstreamAmpConfig,
) -> TrainingRecipe:
    """Resolve v4/fast plus explicit hill-climb overrides into runtime values."""
    if config.preset not in ("v4", "fast"):
        raise ValueError(f"unknown training preset {config.preset!r}")
    # `fast` differs from v4 only in the peak learning rate (see resolve_max_lr); every other
    # runtime control is identical, and explicit flags override both presets.
    return TrainingRecipe(
        compile_mode=config.compile_mode,
        channels_last=config.channels_last,
        fused_adamw=config.fused_adamw,
        amp_dtype=resolve_amp_dtype(config.amp_dtype, amp_config.dtype),
        schedule=config.schedule,
        warmup_pct=scheduler_config.pct_start if config.warmup_pct is None else config.warmup_pct,
        final_div_factor=scheduler_config.final_div_factor if config.final_div_factor is None else config.final_div_factor,
        cooldown_pct=config.cooldown_pct,
        stage_schedule=config.stage_schedule,
    )


def parse_stage_schedule(specification: str | None, total_steps: int, height: int, width: int) -> list[ResolutionStage]:
    """Parse progressive resolutions into contiguous global-step intervals."""
    if total_steps <= 0 or height <= 0 or width <= 0:
        raise ValueError("total_steps, height, and width must be positive")
    if specification is None:
        return [ResolutionStage(height=height, width=width, start_step=0, end_step=total_steps)]
    if not specification.strip():
        raise ValueError("stage_schedule must not be empty")

    entries: list[tuple[int, int, float]] = []
    entry: str
    for entry in specification.split(","):
        resolution: str
        fraction_text: str
        resolution, fraction_separator, fraction_text = entry.strip().partition(":")
        height_text: str
        width_text: str
        height_text, resolution_separator, width_text = resolution.partition("x")
        if not fraction_separator or not resolution_separator:
            raise ValueError(f"invalid stage {entry!r}; expected HEIGHTxWIDTH:FRACTION")
        try:
            stage_height: int = int(height_text)
            stage_width: int = int(width_text)
            fraction: float = float(fraction_text)
        except ValueError as error:
            raise ValueError(f"invalid stage {entry!r}; expected HEIGHTxWIDTH:FRACTION") from error
        if stage_height <= 0 or stage_width <= 0 or not isfinite(fraction) or fraction <= 0.0:
            raise ValueError(f"stage dimensions and fraction must be positive: {entry!r}")
        entries.append((stage_height, stage_width, fraction))

    fraction_sum: float = sum(fraction for _, _, fraction in entries)
    if not isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"stage fractions must sum to 1.0, got {fraction_sum}")

    stages: list[ResolutionStage] = []
    cumulative_fraction: float = 0.0
    start_step: int = 0
    index: int
    for index, (stage_height, stage_width, fraction) in enumerate(entries):
        cumulative_fraction += fraction
        end_step: int = total_steps if index == len(entries) - 1 else round(total_steps * cumulative_fraction)
        if end_step <= start_step:
            raise ValueError("each stage must receive at least one optimizer step")
        stages.append(ResolutionStage(stage_height, stage_width, start_step, end_step))
        start_step = end_step
    return stages


def _model_to_device(model: nn.Module, device: torch.device, channels_last: bool) -> nn.Module:
    """Move a model before DDP construction and optionally select channels-last storage."""
    if channels_last:
        # PyTorch supports this documented overload; pyrefly's torch stubs omit it.
        # pyrefly: ignore [no-matching-overload]
        return model.to(device=device, memory_format=torch.channels_last)
    return model.to(device=device)


def _adamw_optimizer(model: nn.Module, lr: float, weight_decay: float, fused: bool) -> optim.AdamW:
    """Build AdamW without changing the established defaults when fusion is disabled."""
    if fused:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _select_train_and_holdout(config: TrainCatalogConfig, catalog: PromptDACatalog) -> tuple[list[str], list[str]]:
    """Apply explicit selection or the deterministic holdout-first train split."""
    if config.holdout_count < 0:
        raise ValueError("holdout_count must be non-negative")
    if config.segment_ids is not None:
        selected_ids: list[str] = list(config.segment_ids)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("explicit segment_ids must be unique")
        catalog.require_segments(selected_ids)
        holdout_candidates: list[str] = [segment_id for segment_id in catalog.segment_ids if segment_id not in set(selected_ids)]
        candidate_split: tuple[list[str], list[str]] = split_holdout_segments(
            holdout_candidates,
            config.holdout_count,
            config.holdout_seed,
        )
        holdout_ids: list[str] = candidate_split[1]
        train_ids: list[str] = selected_ids
    else:
        train_ids, holdout_ids = split_holdout_segments(catalog.segment_ids, config.holdout_count, config.holdout_seed)
        if config.num_segments is not None:
            if config.num_segments <= 0:
                raise ValueError("num_segments must be positive when set")
            train_ids = train_ids[: config.num_segments]
    if not train_ids:
        raise RuntimeError("catalog split selected no training segments")
    return train_ids, holdout_ids


def _load_json_config(path: Path) -> UpstreamTrainConfig:
    """Read the required upstream training JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return from_json(UpstreamTrainConfig, path.read_text(encoding="utf-8"))


def load_trainable_checkpoint(model: ZipDepth, checkpoint: Path) -> None:
    """Strictly load cleaned DDP/compile weights without inference fusion."""
    model.load_state_dict(load_zipdepth_state_dict(checkpoint), strict=True)


class ConstantCooldownLR(optim.lr_scheduler.LRScheduler):
    """Hold the peak learning rate, then linearly cool to its final ratio."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        total_steps: int,
        cooldown_pct: float,
        final_div_factor: float,
        last_epoch: int = -1,
    ) -> None:
        """Configure a step-counted constant schedule with a linear cooldown."""
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < cooldown_pct <= 1.0:
            raise ValueError("cooldown_pct must be in (0, 1]")
        if final_div_factor <= 0.0:
            raise ValueError("final_div_factor must be positive")
        self.total_steps: int = total_steps
        self.cooldown_steps: int = max(1, ceil(total_steps * cooldown_pct))
        self.constant_steps: int = total_steps - self.cooldown_steps
        self.final_factor: float = 1.0 / final_div_factor
        group: dict[str, object]
        for group in optimizer.param_groups:
            group["lr"] = max_lr
            group["initial_lr"] = max_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Return the learning rate for the scheduler's current step."""
        step: int = min(self.last_epoch, self.total_steps)
        if step < self.constant_steps:
            factor: float = 1.0
        else:
            progress: float = min((step - self.constant_steps + 1) / self.cooldown_steps, 1.0)
            factor = 1.0 + (self.final_factor - 1.0) * progress
        return [base_lr * factor for base_lr in self.base_lrs]


def _build_scheduler(
    optimizer: optim.Optimizer,
    *,
    schedule: ScheduleName,
    max_lr: float,
    total_steps: int,
    warmup_pct: float,
    div_factor: float,
    final_div_factor: float,
    cooldown_pct: float,
    last_epoch: int = -1,
) -> optim.lr_scheduler.LRScheduler:
    """Build the selected step-counted learning-rate schedule."""
    if schedule == "onecycle":
        if not 0.0 < warmup_pct <= 1.0:
            raise ValueError("warmup_pct must be in (0, 1]")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            last_epoch=last_epoch,
        )
    return ConstantCooldownLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        cooldown_pct=cooldown_pct,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch,
    )


def _atomic_latest(final_checkpoint: Path) -> Path:
    """Publish a recoverable latest checkpoint with an atomic final rename."""
    latest: Path = final_checkpoint.parent / "latest.pth"
    temporary: Path = final_checkpoint.parent / ".latest.pth.tmp"
    shutil.copy2(final_checkpoint, temporary)
    temporary.replace(latest)
    return latest


def _restore_resume_state(
    resume: Path,
    wrapped_model: nn.Module,
    trainer: ZipDepthTrainer,
    *,
    total_steps: int,
    actual_max_lr: float,
    range_margin_m: float,
    range_margin_coverage_max: float,
    scheduler_config: UpstreamSchedulerConfig,
    schedule: ScheduleName,
    warmup_pct: float,
    final_div_factor: float,
    cooldown_pct: float,
) -> None:
    """Restore model, optimizer, schedule, and trainer step.

    Raises:
        ValueError: If the checkpoint recorded a different output-range margin or
            prompt-coverage gate than this run is configured with. Resuming
            rewrites the same run, so silently changing the head's output range
            mid-run would make the optimizer state and the recorded settings
            describe different models.
    """
    print_main(f"Resuming from: {resume}", trainer.rank)
    checkpoint: dict[str, object] | None
    if trainer.is_distributed:
        loaded_checkpoint: dict[str, object] | None = torch.load(resume, map_location="cpu", weights_only=True) if trainer.is_main else None
        checkpoint = loaded_checkpoint if isinstance(loaded_checkpoint, dict) else None
        barrier()
        objects: list[dict[str, object] | None] = [checkpoint]
        dist.broadcast_object_list(objects, src=0)
        checkpoint = objects[0]
        barrier()
    else:
        loaded_checkpoint = torch.load(resume, map_location=trainer.device, weights_only=True)
        checkpoint = loaded_checkpoint if isinstance(loaded_checkpoint, dict) else None
    if checkpoint is None:
        raise ValueError(f"resume checkpoint must contain a mapping: {resume}")
    resumed_margin_m: float = range_margin_m_from_checkpoint(checkpoint, resume)
    if not isclose(resumed_margin_m, range_margin_m, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"resume checkpoint recorded range_margin_m={resumed_margin_m} but this run is configured with "
            f"range_margin_m={range_margin_m}; resume with the recorded value or start a new run: {resume}"
        )
    resumed_coverage_max: float = range_margin_coverage_max_from_checkpoint(checkpoint, resume)
    if not isclose(resumed_coverage_max, range_margin_coverage_max, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"resume checkpoint recorded range_margin_coverage_max={resumed_coverage_max} but this run is configured "
            f"with range_margin_coverage_max={range_margin_coverage_max}; resume with the recorded value or start a "
            f"new run: {resume}"
        )
    raw_resume_state: object = checkpoint.get("model_state_dict", checkpoint)
    tensor_state: dict[str, Tensor] = narrow_zipdepth_state_dict(raw_resume_state, resume)
    resume_state: dict[str, Tensor] = fix_state_dict_prefix(strip_state_dict_prefixes(tensor_state), trainer.is_distributed)
    wrapped_model.load_state_dict(resume_state, strict=True)
    raw_optimizer_state: object = checkpoint.get("optimizer_state_dict")
    if not isinstance(raw_optimizer_state, dict):
        raise ValueError("resume checkpoint has no optimizer state")
    optimizer_state: dict[str, object] = raw_optimizer_state
    trainer.optimizer.load_state_dict(optimizer_state)
    raw_global_step: object = checkpoint.get("global_step", 0)
    if not isinstance(raw_global_step, int):
        raise ValueError("resume checkpoint global_step must be an integer")
    trainer.global_step = raw_global_step
    raw_training_stage: object = checkpoint.get("training_stage", 0)
    if not isinstance(raw_training_stage, int) or raw_training_stage < 0:
        raise ValueError("resume checkpoint training_stage must be a non-negative integer")
    trainer.training_stage = raw_training_stage
    raw_scheduler_state: object = checkpoint.get("scheduler_state_dict")
    scheduler_state: dict[str, object] | None = raw_scheduler_state if isinstance(raw_scheduler_state, dict) else None
    raw_old_total_steps: object | None = scheduler_state.get("total_steps") if scheduler_state is not None else None
    if raw_old_total_steps is not None and not isinstance(raw_old_total_steps, int):
        raise ValueError("resume checkpoint scheduler total_steps must be an integer")
    old_total_steps: int | None = raw_old_total_steps
    if scheduler_state is not None and old_total_steps == total_steps:
        trainer.scheduler.load_state_dict(scheduler_state)
    else:
        rebased_scheduler: optim.lr_scheduler.LRScheduler = _build_scheduler(
            trainer.optimizer,
            schedule=schedule,
            max_lr=actual_max_lr,
            total_steps=total_steps,
            warmup_pct=warmup_pct,
            div_factor=scheduler_config.div_factor,
            final_div_factor=final_div_factor,
            cooldown_pct=cooldown_pct,
            last_epoch=trainer.global_step - 1,
        )
        trainer.scheduler = rebased_scheduler
        print_main(f"Rebased {schedule} schedule from {old_total_steps} to {total_steps} total steps", trainer.rank)


def main(
    config: TrainCatalogConfig,
    *,
    writer: SummaryWriter | None = None,
) -> Path:
    """Run catalog fine-tuning and return the atomically published latest checkpoint."""
    distributed_result: tuple[int, int, int, bool] = setup_distributed()
    rank: int = distributed_result[0]
    world_size: int = distributed_result[1]
    local_rank: int = distributed_result[2]
    is_distributed: bool = distributed_result[3]
    is_main: bool = rank == 0
    owns_writer: bool = False
    try:
        if config.height <= 0 or config.width <= 0 or config.batch_size <= 0 or config.total_steps <= 0:
            raise ValueError("height, width, batch_size, and total_steps must be positive")
        if config.shuffle_buffer_size <= 0 or config.rerun_fetch_block_size <= 0:
            raise ValueError("shuffle_buffer_size and rerun_fetch_block_size must be positive")
        if config.num_producers <= 0 or config.prefetch_samples <= 0:
            raise ValueError("num_producers and prefetch_samples must be positive")
        if config.save_every_steps < 0:
            raise ValueError("save_every_steps must be non-negative")
        if config.target_mode not in ("ssi", "metric"):
            raise ValueError(f"unknown target mode {config.target_mode!r}")
        if not isfinite(config.range_margin_m) or config.range_margin_m < 0.0:
            raise ValueError("range_margin_m must be finite and non-negative")
        if config.range_margin_m > 0.0 and config.target_mode != "metric":
            raise ValueError("range_margin_m only applies to the prompted metric lane")
        if not isfinite(config.range_margin_coverage_max) or not 0.0 <= config.range_margin_coverage_max <= 1.0:
            raise ValueError("range_margin_coverage_max must be a fraction in [0, 1]")
        if config.cameras not in ("wide", "ultrawide", "both"):
            raise ValueError(f"unknown camera selection {config.cameras!r}")
        if config.cameras != "wide" and config.dataloader == "rerun":
            raise ValueError("the Rerun dataloader A/B path streams the wide camera only")
        # None keeps the wide lane bit-identical: the builders never consult a policy
        # they were not given, and they reject an ultrawide request without one.
        ultrawide_policy: UltrawidePolicy | None = (
            None
            if config.cameras == "wide"
            else UltrawidePolicy(
                min_valid_fraction=config.ultrawide_min_valid_fraction,
                valid_erosion_px=config.ultrawide_valid_erosion_px,
                prompt_scale=config.ultrawide_prompt_scale,
            )
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device: torch.device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
        selected: tuple[list[str], list[str]] = _select_train_and_holdout(config, catalog)
        train_segment_ids: list[str] = selected[0]
        holdout_segment_ids: list[str] = selected[1]

        # Catalog-only imports stay local so zipdepth-dev can run the pure policy tests.
        from rerun.experimental.dataloader import BlockShuffle

        from zipdepth.catalog.builders import CudaSampleBuilder
        from zipdepth.catalog.dataloader_dataset import ExactIndexQueryStats, RerunPromptDepthDataset
        from zipdepth.catalog.dataset import CatalogPromptDepthDataset

        steps_per_epoch: int = config.total_steps
        total_steps: int = config.total_steps

        upstream_config: UpstreamTrainConfig = _load_json_config(config.train_config)
        optimizer_config: UpstreamOptimizerConfig = upstream_config.optimizer
        scheduler_config: UpstreamSchedulerConfig = upstream_config.scheduler
        loss_config: UpstreamLossConfig = upstream_config.loss
        amp_config: UpstreamAmpConfig = upstream_config.amp
        model_config: UpstreamModelConfig = upstream_config.model
        training_config: UpstreamTrainingConfig = upstream_config.training
        config_lr: float = optimizer_config.lr
        actual_max_lr: float = resolve_max_lr(config_lr, config.max_lr, config.from_scratch, config.preset)
        recipe: TrainingRecipe = resolve_training_recipe(config, scheduler_config, amp_config)
        stages: list[ResolutionStage] = parse_stage_schedule(recipe.stage_schedule, config.total_steps, config.height, config.width)

        if is_main:
            config.save_dir.mkdir(parents=True, exist_ok=True)
            with (config.save_dir / "train_segments.json").open("w") as file:
                json.dump(train_segment_ids, file, indent=2)
            with (config.save_dir / "holdout_segments.json").open("w") as file:
                json.dump(holdout_segment_ids, file, indent=2)
            resolved: ResolvedTrainCatalogConfig = ResolvedTrainCatalogConfig(
                config=config,
                rank_count=world_size,
                steps_per_epoch=steps_per_epoch,
                total_steps=total_steps,
                resolved_max_lr=actual_max_lr,
                recipe=recipe,
            )
            (config.save_dir / "resolved_config.json").write_text(to_json(resolved) + "\n", encoding="utf-8")
        barrier()

        if writer is None and is_main:
            writer = SummaryWriter(log_dir=str(config.save_dir / "runs"))
            owns_writer = True

        def build_train_loader(stage: ResolutionStage, stage_steps: int) -> InstrumentedLoader:
            """Build the catalog stream for one resolution stage."""
            dataset: CatalogPromptDepthDataset | RerunPromptDepthDataset
            if config.dataloader == "rerun":
                dataset = RerunPromptDepthDataset(
                    catalog.dataset_entry,
                    train_segment_ids,
                    catalog.row_by_id,
                    device=device,
                    builder_factory=lambda: CudaSampleBuilder(
                        (stage.height, stage.width),
                        AugmentPolicy(),
                        config.min_depth_span,
                        device,
                        target_mode=config.target_mode,
                    ),
                    # Keep fetch-local block shuffling, but do not retain hundreds of
                    # native-resolution decoded samples before ZipDepth resizing.
                    shuffle_strategy=BlockShuffle(),
                    seed=config.seed,
                    fetch_block_size=config.rerun_fetch_block_size,
                    build_batch_size=config.batch_size,
                    load_confidence=False,
                )
            else:
                dataset = CatalogPromptDepthDataset(
                    catalog.dataset_entry,
                    train_segment_ids,
                    catalog.row_by_id,
                    device=device,
                    builder_factory=lambda: CudaSampleBuilder(
                        (stage.height, stage.width),
                        AugmentPolicy(),
                        config.min_depth_span,
                        device,
                        target_mode=config.target_mode,
                        ultrawide_policy=ultrawide_policy,
                    ),
                    shuffle_buffer_size=config.shuffle_buffer_size,
                    seed=config.seed,
                    rank=rank,
                    world_size=world_size,
                    num_producers=config.num_producers,
                    prefetch_samples=config.prefetch_samples,
                    cameras=config.cameras,
                    # Training ignores confidence: the student takes the raw prompt and the model
                    # range-gates internally. Skipping the column drops a column and ~48 KB/frame
                    # off every segment query.
                    load_confidence=False,
                )
            if is_main and isinstance(dataset, RerunPromptDepthDataset):
                index_stats: ExactIndexQueryStats = dataset.index_query_stats
                print_main(
                    f"Rerun dataset setup: {dataset.setup_s:.2f}s "
                    f"(exact-index materialize={index_stats.materialize_s:.2f}s, "
                    f"rows={len(dataset):,}, arrow={index_stats.arrow_bytes / 1_000_000.0:.1f} MB)",
                    rank,
                )
            data_loader: DataLoader[dict[str, Tensor]] = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )
            return InstrumentedLoader(
                dataset,
                data_loader,
                steps_per_epoch=stage_steps,
                writer=writer if is_main else None,
            )

        train_loader: InstrumentedLoader = build_train_loader(stages[0], stages[0].end_step)

        initialization: Path | None = config.init_checkpoint
        if initialization is None and not config.from_scratch and config.resume is None:
            initialization = download_zipdepth_checkpoint()
        if config.target_mode == "metric":
            if initialization is not None:
                print_main(f"Loading prompted trainable initialization: {initialization}", rank)
                # A fine-tune may deliberately change the margin or its coverage gate: v4
                # recorded neither, and the ultrawide run starts from it with the window
                # open on low-coverage prompts. Say so loudly rather than failing, which is
                # what resuming the same run does instead.
                initialization_checkpoint: dict[str, object] = load_zipdepth_checkpoint(initialization)
                initialization_margin_m: float = range_margin_m_from_checkpoint(initialization_checkpoint, initialization)
                if not isclose(initialization_margin_m, config.range_margin_m, rel_tol=0.0, abs_tol=1e-9):
                    print_main(
                        f"OVERRIDE: initialization recorded range_margin_m={initialization_margin_m}, "
                        f"training with range_margin_m={config.range_margin_m}",
                        rank,
                    )
                initialization_coverage_max: float = range_margin_coverage_max_from_checkpoint(
                    initialization_checkpoint, initialization
                )
                if not isclose(initialization_coverage_max, config.range_margin_coverage_max, rel_tol=0.0, abs_tol=1e-9):
                    print_main(
                        f"OVERRIDE: initialization recorded range_margin_coverage_max={initialization_coverage_max}, "
                        f"training with range_margin_coverage_max={config.range_margin_coverage_max}",
                        rank,
                    )
                model: nn.Module = load_zipdepth_prompt(
                    initialization,
                    range_margin_m=config.range_margin_m,
                    range_margin_coverage_max=config.range_margin_coverage_max,
                )
            else:
                model = ZipDepthPrompt(
                    create_model(variant="base", upsample_unfold=model_config.upsample_unfold),
                    range_margin_m=config.range_margin_m,
                    range_margin_coverage_max=config.range_margin_coverage_max,
                )
                print_main("Training ZipDepth-PromptDA from scratch", rank)
        else:
            relative_model: ZipDepth = create_model(variant="base", upsample_unfold=model_config.upsample_unfold)
            if initialization is not None:
                print_main(f"Loading trainable initialization: {initialization}", rank)
                load_trainable_checkpoint(relative_model, initialization)
            else:
                print_main("Training ZipDepth-base from scratch", rank)
            model = relative_model
        pin_batchnorm_eval: bool = config.freeze_bn and not config.from_scratch
        if pin_batchnorm_eval:
            pinned: int = sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
            print_main(f"Pinned {pinned} BatchNorm modules to eval (released running stats)", rank)
        model = _model_to_device(model, device, recipe.channels_last)
        wrapped_model: nn.Module | DDP = (
            DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if is_distributed else model
        )

        optimizer: optim.AdamW = _adamw_optimizer(
            model,
            lr=actual_max_lr,
            weight_decay=optimizer_config.weight_decay,
            fused=recipe.fused_adamw,
        )
        scheduler: optim.lr_scheduler.LRScheduler = _build_scheduler(
            optimizer,
            schedule=recipe.schedule,
            max_lr=actual_max_lr,
            total_steps=total_steps,
            warmup_pct=recipe.warmup_pct,
            div_factor=scheduler_config.div_factor,
            final_div_factor=recipe.final_div_factor,
            cooldown_pct=recipe.cooldown_pct,
        )
        trainer: ZipDepthTrainer = ZipDepthTrainer(
            student=wrapped_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=amp_config.enabled,
            writer=writer if is_main else None,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            amp_dtype=recipe.amp_dtype,
            compile_mode=recipe.compile_mode,
            channels_last=recipe.channels_last,
            alpha_ssi=loss_config.alpha_ssi,
            alpha_grad=loss_config.alpha_grad,
            target_mode=config.target_mode,
            metric_gradient_weight=config.metric_gradient_weight,
            pin_batchnorm_eval=pin_batchnorm_eval,
            range_margin_m=config.range_margin_m,
            range_margin_coverage_max=config.range_margin_coverage_max,
            use_profiler=config.profile,
            profile_dir=str(config.profile_dir),
            profile_wait=config.profile_wait,
            profile_warmup=config.profile_warmup,
            profile_active=config.profile_active,
        )
        if config.resume is not None:
            _restore_resume_state(
                config.resume,
                wrapped_model,
                trainer,
                total_steps=total_steps,
                actual_max_lr=actual_max_lr,
                range_margin_m=config.range_margin_m,
                range_margin_coverage_max=config.range_margin_coverage_max,
                scheduler_config=scheduler_config,
                schedule=recipe.schedule,
                warmup_pct=recipe.warmup_pct,
                final_div_factor=recipe.final_div_factor,
                cooldown_pct=recipe.cooldown_pct,
            )
            train_loader.set_global_step(trainer.global_step)

        print_main(
            f"Catalog train: {len(train_segment_ids)} segments, {total_steps} optimizer steps, "
            f"dataloader={config.dataloader}, max_lr={actual_max_lr:.2e}, device={device}",
            rank,
        )
        stage_index: int
        stage: ResolutionStage
        for stage_index, stage in enumerate(stages):
            if trainer.global_step >= stage.end_step:
                continue
            if stage_index > 0:
                train_loader = build_train_loader(stage, stage.end_step - trainer.global_step)
                trainer.train_loader = train_loader
            train_loader.set_global_step(trainer.global_step)
            trainer.training_stage = stage_index
            print_main(
                f"Resolution stage {stage_index + 1}/{len(stages)}: {stage.height}x{stage.width}, steps {trainer.global_step}:{stage.end_step}",
                rank,
            )
            trainer.train(
                num_epochs=1,
                save_dir=str(config.save_dir),
                start_epoch=0,
                save_every_steps=config.save_every_steps,
                max_step_checkpoints=training_config.max_step_checkpoints,
                max_steps=stage.end_step,
                skip_batches=0,
            )
        barrier()
        latest: Path = config.save_dir / "latest.pth"
        if is_main:
            latest = _atomic_latest(config.save_dir / "final_model.pth")
        barrier()
        return latest
    finally:
        if owns_writer and writer is not None:
            writer.close()
        cleanup_distributed()
