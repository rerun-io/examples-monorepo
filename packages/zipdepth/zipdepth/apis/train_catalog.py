"""Fine-tune trainable ZipDepth-base weights on catalog PromptDA targets."""

from __future__ import annotations

import json
import shutil
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, cast

import torch
import torch.distributed as dist
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from monopriors.third_party.zipdepth.architecture import ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import StateDict, strip_state_dict_prefixes
from torch import Tensor, nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipdepth.catalog.instrument import InstrumentedLoader
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.targets import AugmentPolicy, TargetMode
from zipdepth.training.distributed import barrier, cleanup_distributed, fix_state_dict_prefix, print_main, setup_distributed
from zipdepth.training.trainer import ZipDepthTrainer


@dataclass(slots=True)
class TrainCatalogConfig:
    """Catalog fine-tuning configuration."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
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
    target_mode: TargetMode = "metric"
    """Prompted metric distillation by default; ``ssi`` retains the legacy RGB-only lane."""
    total_steps: int = 70_000
    """Optimizer steps in the run; the OneCycle schedule spans exactly this many. 70k is about one pass over the 875-segment train split at batch 8 (70.5k steps measured 2026-08-28)."""
    shuffle_buffer_size: int = 256
    """Maximum decoded samples held for bounded random shuffling. CUDA samples
    use about 4.7 MB each at 1024x768, so this and the default producer queue
    retain about 2.4 GB of device memory together."""
    seed: int = 0
    """Base seed for dataset segment order and shuffle-buffer eviction."""
    num_producers: int = 6
    """Whole-segment catalog producer threads, each with its own video decoder."""
    prefetch_samples: int = 256
    """Maximum completed samples buffered across catalog producers."""
    min_depth_span: float = 1.25
    """Minimum valid-depth p95/p5 ratio; zero disables flat-frame filtering."""
    from_scratch: bool = False
    """Start with random weights instead of the released ZipDepth-base weights."""
    metric_gradient_weight: float = 0.5
    """Weight on the multi-scale gradient term in the metric loss (L1 + w * grad).
    At 0.5 the term supplies only ~9% of the objective; upstream ZipDepth weights its
    equivalent term at 2.0. Raising it targets depth-discontinuity error, where the model
    beats a bilinear prompt upsample by the smallest margin."""
    metric_grad_scale_weights: tuple[float, float, float, float] | None = None
    """Per-scale gradient weights over k = 1, 2, 4, 8 (normalized to sum to one). None
    weights all scales equally; only the k = 1 scale sees full-resolution edges, so
    fine-heavy allocations move gradient pressure onto real discontinuities."""
    metric_grad_max_depth_m: float = 4.0
    """Gradient-term validity ceiling in metres. The L1 window stays at 4 m; widening
    this keeps gradient pairs straddling the 4 m boundary — the strongest room edges,
    which the default deletes entirely."""
    metric_grad_pool_mask_threshold: float = 0.99
    """Pooled-validity threshold at coarse gradient scales. 0.99 erodes every kxk block
    near an invalid pixel (silhouette rings); 0.5 keeps majority-valid blocks."""
    metric_edge_weight_alpha: float = 0.0
    """Extra L1 weight on teacher depth edges: w = 1 + alpha * clip(g / q90(g), 0, 1).
    Zero keeps plain masked L1."""
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
    max_lr: float | None = None
    """OneCycle peak LR; None uses config LR / 100 for fine-tuning and config LR from scratch.
    Measured on the fixed probe set: at config/10 (1e-4, batch 4) fine-tuning diverges once
    warmup ends; at config/100 (1e-5) it improves the probe loss ~29% in 120 steps."""


class _LossObserver(nn.Module):
    """Observe every loss only when the smoke gate supplies a callback."""

    def __init__(self, criterion: nn.Module, callback: Callable[[float], None]) -> None:
        super().__init__()
        self._criterion: nn.Module = criterion
        self._callback: Callable[[float], None] = callback

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        """Forward the loss and synchronously expose its scalar value to the smoke gate."""
        result: tuple[Tensor, dict[str, float]] = self._criterion(pred=pred, target=target, mask=mask)
        self._callback(float(result[0].detach().item()))
        return result


def split_holdout_segments(segment_ids: list[str], holdout_count: int, seed: int) -> tuple[list[str], list[str]]:
    """Split sorted identifiers into deterministic disjoint train and holdout lists.

    Args:
        segment_ids: Unique segment identifiers in any input order.
        holdout_count: Number of identifiers to reserve.
        seed: Random seed used to select the holdout.

    Returns:
        The training identifiers followed by the holdout identifiers, both sorted.

    Raises:
        ValueError: If identifiers repeat or the holdout count is invalid.
    """
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("segment_ids must be unique")
    if not 0 <= holdout_count <= len(segment_ids):
        raise ValueError(f"holdout_count must be in [0, {len(segment_ids)}]")
    ordered_ids: list[str] = sorted(segment_ids)
    holdout_set: set[str] = set(Random(seed).sample(ordered_ids, holdout_count))
    train_ids: list[str] = [segment_id for segment_id in ordered_ids if segment_id not in holdout_set]
    holdout_ids: list[str] = [segment_id for segment_id in ordered_ids if segment_id in holdout_set]
    return train_ids, holdout_ids


def resolve_max_lr(config_lr: float, override: float | None, from_scratch: bool) -> float:
    """Resolve the OneCycle peak learning rate from initialization policy."""
    max_lr: float = override if override is not None else config_lr if from_scratch else config_lr / 100.0
    if max_lr <= 0.0:
        raise ValueError("max_lr must be positive")
    return max_lr


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


def _load_json_config(path: Path) -> dict[str, Any]:
    """Read the required upstream training JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as file:
        loaded: Any = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"training config must contain a JSON object: {path}")
    return loaded


def load_trainable_checkpoint(model: ZipDepth, checkpoint: Path) -> None:
    """Strictly load cleaned DDP/compile weights without inference fusion."""
    loaded: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint}")
    raw_state: Any = loaded.get("model_state_dict", loaded)
    if not isinstance(raw_state, dict):
        raise ValueError(f"checkpoint model state must contain a mapping: {checkpoint}")
    state_dict: StateDict = strip_state_dict_prefixes(raw_state)
    model.load_state_dict(state_dict, strict=True)


def pin_batchnorm_eval(model: nn.Module) -> int:
    """Pin every BatchNorm module to eval mode for the whole training run.

    Overrides the root model's ``train`` method to apply the requested mode and
    then return every BatchNorm module to eval, so the trainer's per-epoch
    ``model.train()`` cannot flip them back. Affine
    weights still receive gradients; only the normalization statistics are frozen.
    State-dict layout is unchanged, so strict checkpoint round-trips still pass.

    Returns:
        The number of pinned BatchNorm modules.
    """
    def train_with_batchnorm_eval(self: nn.Module, mode: bool = True) -> nn.Module:
        """Apply a root mode change, then restore every BatchNorm to eval."""
        nn.Module.train(self, mode)
        child: nn.Module
        for child in self.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()
        return self

    pinned: int = sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    current_mode: bool = model.training
    model.train = types.MethodType(train_with_batchnorm_eval, model)
    model.train(current_mode)
    return pinned


def _one_cycle_scheduler(
    optimizer: optim.Optimizer,
    max_lr: float,
    total_steps: int,
    scheduler_config: dict[str, Any],
    last_epoch: int = -1,
) -> optim.lr_scheduler.OneCycleLR:
    """Build the upstream OneCycle schedule with resolved total steps."""
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=float(scheduler_config.get("pct_start", 0.05)),
        anneal_strategy="cos",
        div_factor=float(scheduler_config.get("div_factor", 25.0)),
        final_div_factor=float(scheduler_config.get("final_div_factor", 1000.0)),
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
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.OneCycleLR,
    trainer: ZipDepthTrainer,
    *,
    device: torch.device,
    is_distributed: bool,
    is_main: bool,
    rank: int,
    total_steps: int,
    actual_max_lr: float,
    scheduler_config: dict[str, Any],
) -> None:
    """Restore model, optimizer, schedule, and trainer step."""
    print_main(f"Resuming from: {resume}", rank)
    checkpoint: dict[str, Any] | None
    if is_distributed:
        loaded_checkpoint: Any = torch.load(resume, map_location="cpu", weights_only=True) if is_main else None
        checkpoint = loaded_checkpoint if isinstance(loaded_checkpoint, dict) else None
        barrier()
        objects: list[Any] = [checkpoint]
        dist.broadcast_object_list(objects, src=0)
        checkpoint = objects[0]
        barrier()
    else:
        loaded_checkpoint = torch.load(resume, map_location=device, weights_only=True)
        checkpoint = loaded_checkpoint if isinstance(loaded_checkpoint, dict) else None
    if checkpoint is None:
        raise ValueError(f"resume checkpoint must contain a mapping: {resume}")
    raw_resume_state: Any = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_resume_state, dict):
        raise ValueError("resume checkpoint has no model state mapping")
    resume_state: dict[str, Any] = fix_state_dict_prefix(strip_state_dict_prefixes(raw_resume_state), is_distributed)
    wrapped_model.load_state_dict(resume_state, strict=True)
    optimizer_state: Any = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("resume checkpoint has no optimizer state")
    optimizer.load_state_dict(optimizer_state)
    trainer.global_step = int(checkpoint.get("global_step", 0))
    scheduler_state: Any = checkpoint.get("scheduler_state_dict")
    old_total_steps: int | None = int(scheduler_state["total_steps"]) if isinstance(scheduler_state, dict) and "total_steps" in scheduler_state else None
    if isinstance(scheduler_state, dict) and old_total_steps == total_steps:
        scheduler.load_state_dict(scheduler_state)
    else:
        rebased_scheduler: optim.lr_scheduler.OneCycleLR = _one_cycle_scheduler(
            optimizer,
            actual_max_lr,
            total_steps,
            scheduler_config,
            last_epoch=trainer.global_step - 1,
        )
        trainer.scheduler = rebased_scheduler
        print_main(f"Rebased OneCycle schedule from {old_total_steps} to {total_steps} total steps", rank)


def main(
    config: TrainCatalogConfig,
    *,
    writer: SummaryWriter | None = None,
    step_loss_callback: Callable[[float], None] | None = None,
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
        if config.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if config.num_producers <= 0 or config.prefetch_samples <= 0:
            raise ValueError("num_producers and prefetch_samples must be positive")
        if config.save_every_steps < 0:
            raise ValueError("save_every_steps must be non-negative")
        if config.target_mode not in ("ssi", "metric"):
            raise ValueError(f"unknown target mode {config.target_mode!r}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device: torch.device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
        selected: tuple[list[str], list[str]] = _select_train_and_holdout(config, catalog)
        train_segment_ids: list[str] = selected[0]
        holdout_segment_ids: list[str] = selected[1]

        # Catalog-only imports stay local so zipdepth-dev can run the pure policy tests.
        from zipdepth.catalog.builders import CudaSampleBuilder
        from zipdepth.catalog.dataset import CatalogPromptDepthDataset

        steps_per_epoch: int = config.total_steps
        total_steps: int = config.total_steps

        upstream_config: dict[str, Any] = _load_json_config(config.train_config)
        optimizer_config: dict[str, Any] = dict(upstream_config.get("optimizer", {}))
        scheduler_config: dict[str, Any] = dict(upstream_config.get("scheduler", {}))
        loss_config: dict[str, Any] = dict(upstream_config.get("loss", {}))
        amp_config: dict[str, Any] = dict(upstream_config.get("amp", {}))
        model_config: dict[str, Any] = dict(upstream_config.get("model", {}))
        training_config: dict[str, Any] = dict(upstream_config.get("training", {}))
        config_lr: float = float(optimizer_config.get("lr", 1e-3))
        actual_max_lr: float = resolve_max_lr(config_lr, config.max_lr, config.from_scratch)

        if is_main:
            config.save_dir.mkdir(parents=True, exist_ok=True)
            with (config.save_dir / "train_segments.json").open("w") as file:
                json.dump(train_segment_ids, file, indent=2)
            with (config.save_dir / "holdout_segments.json").open("w") as file:
                json.dump(holdout_segment_ids, file, indent=2)
            resolved: dict[str, Any] = {
                **asdict(config),
                "rank_count": world_size,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "resolved_max_lr": actual_max_lr,
            }
            with (config.save_dir / "resolved_config.json").open("w") as file:
                json.dump(resolved, file, indent=2, default=str)
        barrier()

        if writer is None and is_main:
            writer = SummaryWriter(log_dir=str(config.save_dir / "runs"))
            owns_writer = True

        dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
            catalog.dataset_entry,
            train_segment_ids,
            catalog.row_by_id,
            device=device,
            builder_factory=lambda: CudaSampleBuilder(
                (config.height, config.width),
                AugmentPolicy(),
                config.min_depth_span,
                device,
                target_mode=config.target_mode,
            ),
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
            num_producers=config.num_producers,
            prefetch_samples=config.prefetch_samples,
            # Training ignores confidence: the student takes the raw prompt and the model
            # range-gates internally. Skipping the column drops a column and ~48 KB/frame
            # off every segment query.
            load_confidence=False
        )
        data_loader: DataLoader[dict[str, Tensor]] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        train_loader: InstrumentedLoader = InstrumentedLoader(
            dataset,
            data_loader,
            steps_per_epoch=steps_per_epoch,
            writer=writer if is_main else None,
        )

        initialization: Path | None = config.init_checkpoint
        if initialization is None and not config.from_scratch and config.resume is None:
            initialization = download_zipdepth_checkpoint()
        if config.target_mode == "metric":
            if initialization is not None:
                print_main(f"Loading prompted trainable initialization: {initialization}", rank)
                model: nn.Module = load_zipdepth_prompt(initialization)
            else:
                model = ZipDepthPrompt(
                    create_model(variant="base", upsample_unfold=bool(model_config.get("upsample_unfold", True)))
                )
                print_main("Training ZipDepth-PromptDA from scratch", rank)
        else:
            relative_model: ZipDepth = create_model(variant="base", upsample_unfold=bool(model_config.get("upsample_unfold", True)))
            if initialization is not None:
                print_main(f"Loading trainable initialization: {initialization}", rank)
                load_trainable_checkpoint(relative_model, initialization)
            else:
                print_main("Training ZipDepth-base from scratch", rank)
            model = relative_model
        if config.freeze_bn and not config.from_scratch:
            pinned: int = pin_batchnorm_eval(model)
            print_main(f"Pinned {pinned} BatchNorm modules to eval (released running stats)", rank)
        model = model.to(device)
        wrapped_model: nn.Module | DDP = (
            DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if is_distributed else model
        )

        model_parameters: Any = model.parameters()
        optimizer: optim.AdamW = optim.AdamW(
            model_parameters,
            lr=actual_max_lr,
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
        )
        scheduler: optim.lr_scheduler.OneCycleLR = _one_cycle_scheduler(optimizer, actual_max_lr, total_steps, scheduler_config)
        trainer: ZipDepthTrainer = ZipDepthTrainer(
            student=wrapped_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=bool(amp_config.get("enabled", True)),
            writer=writer if is_main else None,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            amp_dtype=str(amp_config.get("dtype", "bfloat16")),
            alpha_ssi=float(loss_config.get("alpha_ssi", 1.0)),
            alpha_grad=float(loss_config.get("alpha_grad", 2.0)),
            target_mode=config.target_mode,
            metric_gradient_weight=config.metric_gradient_weight,
            metric_grad_scale_weights=config.metric_grad_scale_weights,
            metric_grad_max_depth_m=config.metric_grad_max_depth_m,
            metric_grad_pool_mask_threshold=config.metric_grad_pool_mask_threshold,
            metric_edge_weight_alpha=config.metric_edge_weight_alpha,
            use_profiler=config.profile,
            profile_dir=str(config.profile_dir),
            profile_wait=config.profile_wait,
            profile_warmup=5,
            profile_active=10,
        )
        if step_loss_callback is not None:
            trainer.criterion = cast(Any, _LossObserver(trainer.criterion, step_loss_callback))

        if config.resume is not None:
            _restore_resume_state(
                config.resume,
                wrapped_model,
                optimizer,
                scheduler,
                trainer,
                device=device,
                is_distributed=is_distributed,
                is_main=is_main,
                rank=rank,
                total_steps=total_steps,
                actual_max_lr=actual_max_lr,
                scheduler_config=scheduler_config,
            )

        print_main(
            f"Catalog train: {len(train_segment_ids)} segments, {total_steps} optimizer steps, max_lr={actual_max_lr:.2e}, device={device}",
            rank,
        )
        trainer.train(
            num_epochs=1,
            save_dir=str(config.save_dir),
            start_epoch=0,
            save_every_steps=config.save_every_steps,
            max_step_checkpoints=int(training_config.get("max_step_checkpoints", 5)),
            max_steps=total_steps,
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
