"""Live end-to-end smoke gate for catalog train, resume, DDP, and inference."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import UInt8
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from monopriors.third_party.zipdepth.architecture import ZipDepth, create_model
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipdepth.apis.train_catalog import TrainCatalogConfig, load_trainable_checkpoint
from zipdepth.apis.train_catalog import main as train_catalog
from zipdepth.apis.train_smoke import check_checkpoint, example_rgb, publish_latest, run_required
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.targets import build_eval_transform
from zipdepth.data.transforms import AlbumentationsWrapper
from zipdepth.loss import ZipDepthLoss


@dataclass(slots=True)
class TrainCatalogSmokeConfig:
    """Small live catalog smoke configuration."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the live local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
    """Catalog dataset containing ARKitScenes and PromptDA layers."""
    work_dir: Path = Path("data/smoke/catalog-runs")
    """Parent directory whose ``latest`` child is published after every gate passes."""
    train_config: Path = Path("configs/default.json")
    """Upstream training JSON used by every smoke leg."""
    total_steps: int = 120
    """Initial fine-tuning updates; 120 is the shortest horizon that moved the fixed probe reliably (36 was order-sensitive)."""
    resume_steps: int = 4
    """Additional updates after resuming the atomic latest checkpoint."""
    ddp_steps: int = 4
    """Updates in the one-process torchrun leg."""
    probe_batches: int = 12
    """Fixed probe batches (of the training batch size) scored before and after training."""


class _LossWriter(SummaryWriter):
    """TensorBoard writer that also retains smoke-step losses in memory."""

    def __init__(self, log_dir: Path) -> None:
        super().__init__(log_dir=str(log_dir))
        self.losses: list[float] = []

    def record_loss(self, loss: float) -> None:
        """Append and emit one trainer loss using a monotonic one-based step."""
        self.losses.append(loss)
        self.add_scalar("smoke/train_loss", loss, len(self.losses))


def collect_probe_batches(catalog_url: str, dataset_name: str, batch_size: int, num_batches: int) -> list[dict[str, Tensor]]:
    """Materialize a fixed, deterministic probe set from the smoke's two training segments.

    Uses the deterministic eval transform (no flip or color jitter), a stride that
    spreads the probe across each segment's easy and hard sections, and a
    unit shuffle buffer so ordering stays reproducible. The probe measures
    before/after training loss on IDENTICAL data — a windowed mean over the live
    stream confounds learning with scene difficulty (later scene sections score
    ~2.5x worse under the frozen released model).
    """
    # Catalog-only imports stay local so zipdepth-dev can import this module.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    catalog: PromptDACatalog = load_promptda_catalog(catalog_url, dataset_name)
    transform: AlbumentationsWrapper = build_eval_transform(768, 1024)
    dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        catalog.dataset_entry,
        catalog.segment_ids[:2],
        catalog.row_by_id,
        device=torch.device("cuda"),
        builder_factory=lambda: CpuSampleBuilder(transform, min_depth_span=1.25),
        shuffle_buffer_size=1,
        frame_stride=8,
        num_producers=1,
        prefetch_samples=4,
    )
    loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    batches: list[dict[str, Tensor]] = []
    batch: dict[str, Tensor]
    for batch in loader:
        batches.append({key: value.clone() for key, value in batch.items()})
        if len(batches) >= num_batches:
            break
    if len(batches) < num_batches:
        raise RuntimeError(f"probe collection produced {len(batches)} batches, expected {num_batches}")
    return batches


def mean_probe_loss(checkpoint: Path | None, batches: list[dict[str, Tensor]]) -> float:
    """Score one trainable checkpoint (None = released weights) on the fixed probe set.

    Mirrors the trainer's batch arithmetic (image/255, depth/256, float mask) but
    runs eval-mode fp32 so released and trained weights are compared identically.
    """
    device: torch.device = torch.device("cuda")
    model: ZipDepth = create_model(variant="base")
    load_trainable_checkpoint(model, checkpoint if checkpoint is not None else download_zipdepth_checkpoint())
    model = model.to(device).eval()
    criterion: ZipDepthLoss = ZipDepthLoss()
    losses: list[float] = []
    with torch.no_grad():
        batch: dict[str, Tensor]
        for batch in batches:
            image: Tensor = batch["image"].to(device).float() / 255.0
            depth: Tensor = batch["depth"].to(device).float() / 256.0
            mask: Tensor = batch["mask"].to(device).float()
            pred: Tensor = model(image)
            loss: Tensor = criterion(pred=pred, target=depth, mask=mask)[0]
            losses.append(float(loss.item()))
    del model
    torch.cuda.empty_cache()
    return float(np.mean(losses))


def main(config: TrainCatalogSmokeConfig) -> None:
    """Run the live catalog smoke and atomically publish its output directory."""
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.resume_steps <= 0 or config.ddp_steps <= 0:
        raise ValueError("resume_steps and ddp_steps must be positive")
    if config.probe_batches <= 0:
        raise ValueError("probe_batches must be positive")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path = Path(tempfile.mkdtemp(prefix="run-", dir=config.work_dir))
    single_dir: Path = run_dir / "single"
    ddp_dir: Path = run_dir / "ddp"
    writer: _LossWriter = _LossWriter(single_dir / "runs-smoke")
    base: TrainCatalogConfig = TrainCatalogConfig(
        catalog_url=config.catalog_url,
        dataset_name=config.dataset_name,
        num_segments=2,
        holdout_count=0,
        height=768,
        width=1024,
        batch_size=4,
        target_mode="ssi",
        total_steps=config.total_steps,
        shuffle_buffer_size=32,
        num_producers=1,
        prefetch_samples=8,
        train_config=config.train_config,
        save_dir=single_dir,
        save_every_steps=5,
    )
    try:
        probe: list[dict[str, Tensor]] = collect_probe_batches(config.catalog_url, config.dataset_name, base.batch_size, config.probe_batches)
        released_loss: float = mean_probe_loss(None, probe)

        latest_checkpoint: Path = train_catalog(base, writer=writer, step_loss_callback=writer.record_loss)
        if len(writer.losses) != config.total_steps:
            raise RuntimeError(f"expected {config.total_steps} captured losses, got {len(writer.losses)}")

        trained_loss: float = mean_probe_loss(latest_checkpoint, probe)
        if not trained_loss < released_loss:
            raise RuntimeError(f"fixed-probe loss did not fall: released={released_loss:.6f}, trained={trained_loss:.6f}")
        print(f"loss gate passed on the fixed probe set: released={released_loss:.6f}, trained={trained_loss:.6f}")

        resume_config: TrainCatalogConfig = replace(
            base,
            resume=latest_checkpoint,
            total_steps=config.total_steps + config.resume_steps,
        )
        resumed_checkpoint: Path = train_catalog(resume_config, writer=writer, step_loss_callback=writer.record_loss)
        expected_losses: int = config.total_steps + config.resume_steps
        if len(writer.losses) != expected_losses:
            raise RuntimeError(f"resume reached {len(writer.losses)} captured losses, expected {expected_losses}")
    finally:
        writer.close()

    tool_path: Path = Path(__file__).resolve().parents[2] / "tools" / "train_catalog.py"
    ddp: TrainCatalogConfig = replace(
        base,
        total_steps=config.ddp_steps,
        save_dir=ddp_dir,
        save_every_steps=2,
        init_checkpoint=resumed_checkpoint,
    )
    command: list[str] = [
        "torchrun",
        "--nproc_per_node=1",
        str(tool_path),
        "--catalog-url",
        ddp.catalog_url,
        "--dataset-name",
        ddp.dataset_name,
        "--num-segments",
        str(ddp.num_segments),
        "--holdout-count",
        str(ddp.holdout_count),
        "--height",
        str(ddp.height),
        "--width",
        str(ddp.width),
        "--batch-size",
        str(ddp.batch_size),
        # Pin the lane explicitly: the CLI default became "metric", and without this
        # flag the DDP leg silently trained the prompted model while check_checkpoint
        # verifies the bare one (broken since the default changed).
        "--target-mode",
        ddp.target_mode,
        "--total-steps",
        str(ddp.total_steps),
        "--shuffle-buffer-size",
        str(ddp.shuffle_buffer_size),
        "--train-config",
        str(ddp.train_config),
        "--save-dir",
        str(ddp.save_dir),
        "--save-every-steps",
        str(ddp.save_every_steps),
        "--init-checkpoint",
        str(resumed_checkpoint),
    ]
    run_required(command)

    image_path: Path = Path(__file__).resolve().parents[2] / "assets" / "examples" / "im0.jpg"
    rgb_hw3: UInt8[ndarray, "h w 3"] = example_rgb(image_path)
    check_checkpoint(single_dir / "final_model.pth", rgb_hw3)
    check_checkpoint(ddp_dir / "final_model.pth", rgb_hw3)

    latest_dir: Path = publish_latest(config.work_dir, run_dir)
    print(f"catalog smoke passed; outputs in {latest_dir}")
