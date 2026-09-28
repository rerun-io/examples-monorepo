"""Tyro training entry point: DetNet-F, KeyNet-F or both, trained from one catalog stream."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import torch
from serde import serde
from serde.json import to_json

from handtrack.data.batches import BatchSource
from handtrack.data.catalog import DatasetName, SplitName
from handtrack.data.stream import CatalogStream, StreamConfig
from handtrack.train.loop import LoopSettings, Nets, OptimiserSettings, Trainer


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class StreamSettings:
    """Stream settings, mapped onto ``handtrack.data.stream.StreamConfig`` by ``build_source``."""
    datasets: tuple[str, ...] = ('dataforge-umetrack', 'dataforge-show3d')
    """Catalog datasets to sample."""
    train_split: str = 'training'
    """Training split name."""
    val_split: str = 'validation'
    """Validation split name."""
    segment_ids: tuple[str, ...] = ()
    """Fixed segment selection for overfit runs; empty selects the split."""
    producers: int = 4
    """Shared decoding producers for both nets."""
    detnet_batch: int = 256
    """DetNet images per batch."""
    keynet_batch: int = 256
    """KeyNet crops per batch."""
    seed: int = 0
    """Source shuffle/augmentation and model initialization seed."""
    max_val_batches: int = 20
    """Maximum batches per net per validation pass."""
    detnet_buffer: int = 65_536
    """DetNet shuffle pool on the GPU (uint8 pooled frames: 19.6 kB each, 1.29 GB at the default)."""
    keynet_buffer: int = 65_536
    """KeyNet shuffle pool on the GPU (uint8 crops: 9.8 kB each, 0.64 GB at the default)."""
    gpu_memory_gb: float = 0.0
    """Hard cap on this process's GPU memory (torch's allocator gets the cap minus 0.75 GB for the CUDA context and
    NVDEC); 0 leaves it unbounded. At batch 256 the two nets' fp32 activations alone need 7.5 GB, bf16 3.8 GB."""

    def __post_init__(self) -> None:
        if min(self.producers, self.detnet_batch, self.keynet_batch, self.max_val_batches, self.detnet_buffer, self.keynet_buffer) < 1:
            raise ValueError('Stream counts must be positive')
        if self.gpu_memory_gb < 0:
            raise ValueError('gpu_memory_gb must be nonnegative (0 disables the cap)')


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Config:
    """Train DetNet-F, KeyNet-F, or both with one shared source."""
    nets: Nets = 'both'
    """Networks to train."""
    detnet: OptimiserSettings = field(default_factory=lambda: OptimiserSettings(lr=0.001))
    """DetNet SGD settings."""
    keynet: OptimiserSettings = field(default_factory=lambda: OptimiserSettings(lr=0.025))
    """KeyNet SGD settings."""
    loop: LoopSettings = field(default_factory=LoopSettings)
    """Epoch count, logging, validation, and checkpoint cadence."""
    run_dir: Path = Path('runs/handtrack')
    """Checkpoint and JSONL destination."""
    resume: bool = False
    """Load run_dir/last.pt, including optimisers, progress, and metric history."""
    device: str = 'cpu'
    """Torch device, e.g. cpu or cuda:0; CPU is safe for local fake-data tests."""
    stream: StreamSettings = field(default_factory=StreamSettings)
    """Shared stream factory settings."""


SPLITS: dict[str, SplitName] = {"training": "train", "train": "train", "validation": "val", "val": "val", "testing": "test", "test": "test"}
"""Trainer split names onto ``catalog.select_split``'s."""
DATASETS: dict[str, DatasetName] = {name: name for name in get_args(DatasetName)}
"""Every catalog dataset the stream knows, keyed by its CLI spelling."""
VALIDATION_SEGMENTS: int = 32
"""Segments an evaluation set is built from (one seeded shuffle of the split's segments), unless segment_ids fixes them."""


def build_source(settings: StreamSettings, split: str, nets: Nets) -> BatchSource:
    """A ``CatalogStream`` for the split: the training split streams with augmentation and one decode for every
    requested net; any other split is a fixed, unaugmented evaluation set with exact validation metadata
    (``detnet_validation`` / ``keynet_validation``). ``segment_ids`` fixes the segments of both (overfit runs)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    unknown: list[str] = [name for name in settings.datasets if name not in DATASETS]
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; expected some of {sorted(DATASETS)}")
    ours: SplitName = SPLITS[split]
    evaluation: bool = ours != "train"
    return CatalogStream(
        StreamConfig(
            datasets=tuple(DATASETS[name] for name in settings.datasets),
            split=ours,
            segment_ids=settings.segment_ids,
            max_segments=VALIDATION_SEGMENTS if evaluation and not settings.segment_ids else None,
            nets=nets,
            producers=2 if evaluation else settings.producers,
            fetchers=2 if evaluation else 4,
            detnet_buffer=settings.detnet_buffer,
            keynet_buffer=settings.keynet_buffer,
            detnet_batch_size=settings.detnet_batch,
            keynet_batch_size=settings.keynet_batch,
            seed=settings.seed,
            validation=evaluation,
            validation_samples=settings.max_val_batches * max(settings.detnet_batch, settings.keynet_batch),
        )
    )


GPU_OVERHEAD_GB: float = 0.75
"""GPU memory outside torch's allocator: the CUDA context and the NVDEC decoders' surfaces."""


def cap_gpu_memory(gigabytes: float, device: str) -> None:
    """Bound torch's caching allocator so the whole process stays under ``gigabytes`` (it frees cached blocks before failing)."""
    if gigabytes <= 0 or not device.startswith("cuda"):
        return
    total: int = torch.cuda.get_device_properties(torch.device(device)).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, (gigabytes - GPU_OVERHEAD_GB) * 2**30 / total), torch.device(device))


def main(config: Config) -> None:
    """Build exactly one training and one validation source, then train."""
    torch.manual_seed(config.stream.seed)
    cap_gpu_memory(config.stream.gpu_memory_gb, config.device)
    trainer: Trainer = Trainer(config.nets, config.detnet, config.keynet, config.loop, config.run_dir, config.device,
                               resume=config.resume, max_val_batches=config.stream.max_val_batches, config_json=to_json(config))
    source: BatchSource = build_source(config.stream, config.stream.train_split, config.nets)
    try:
        validation: BatchSource = build_source(config.stream, config.stream.val_split, config.nets)
    except BaseException:
        source.close()
        raise
    trainer.run(source, validation)
