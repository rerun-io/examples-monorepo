"""Tyro training entry point: DetNet-F, KeyNet-F or both, trained from one catalog stream."""
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import torch
from serde import serde
from serde.json import from_json, to_json

from handtrack.data.augment import DetNetAugment
from handtrack.data.batches import BatchSource
from handtrack.data.cache import DetNetCache, KeyNetCache
from handtrack.data.catalog import CATALOG_URL, DatasetName, SplitName, list_segments, select_split
from handtrack.data.echo import EchoStream, KeyNetProducer, Producer
from handtrack.data.keynet_holdout import HoldoutSource, load_holdout
from handtrack.data.keynet_rerun import CutStream, DrawAugment, keynet_finish, stream_cuts
from handtrack.data.rerun_source import DATASET_SPECS, build_manifest, nvdec_producer, rerun_producer, source_spec, training_finish
from handtrack.data.stream import CatalogStream, KeyNetAugment, StreamConfig, sample_count
from handtrack.train.loop import LoopSettings, Nets, OptimiserSettings, Trainer


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetectorHoldout:
    """Episodes of the 2D-keypoint datasets held out of DetNet training (configs/detector_holdout.json), frozen before training."""
    note: str
    segments: dict[str, list[str]]
    """Catalog dataset name -> held-out segment ids."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RerunStreamSettings:
    """Train DetNet from the Rerun 0.42 dataloader: one streaming producer per dataset into per-dataset echo pools."""
    datasets: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d", "gsi-six-view", "dataforge-gen-humanego-subset")
    """Training datasets; each gets an equal share of every batch."""
    decode: Literal["nvdec", "cpu"] = "nvdec"
    """nvdec: decode on the GPU in the training process (simplecv's segment-wide decoder); cpu: the 0.42 CPU decoder in spawn workers."""
    workers: tuple[int, ...] = (4, 6, 8, 8)
    """Spawn decode workers per dataset for ``decode=cpu`` (same order as ``datasets``)."""
    pool_capacity: int = 32_768
    """Images per dataset pool on the device (~20 kB each)."""
    min_fill: int = 4_096
    """Images a pool holds before the first batch."""
    max_echo: int = 16
    """Draws per decoded image before it leaves the pool."""
    min_echo: int = 4
    """Draws an image must get before its producer may replace it, so fast datasets leave decode time to slow ones."""
    prefetch: int = 3
    """Batches assembled ahead in a thread, overlapping the training step."""
    switch_interval_s: float = 0.0005
    """The interpreter's thread switch interval while this stream runs (Python's default is 5 ms). The training thread gives up the
    interpreter lock at each kernel launch; with the default a busy producer thread then keeps it up to 5 ms, the GIL convoy that
    left the GPU idle (2026-11-08: 0.5 ms took a 3-minute run from ~7.3k to ~8.2k images/s, GPU 55-60 % to 73 %)."""
    epoch_steps: int = 20_000
    """Batches per trainer epoch (checkpoint cadence); producers keep running across epochs."""
    keynet_cuts: int = 2
    """KeyNet: independently jittered crops cut from each decoded chunk (a cut is a grid sample; a decode is not)."""
    keynet_full_rotation: float = 0.25
    """KeyNet: share of crops rolled by a uniform angle on the whole circle instead of +-30 degrees (RoboCap cuts without a roll)."""
    keynet_draw: DrawAugment = field(default_factory=DrawAugment)
    """KeyNet: augmentation drawn anew every time a stored crop is drawn."""
    holdout: Path = Path("configs/detector_holdout.json")
    """Held-out episodes of the 2D-keypoint datasets (relative to the package directory)."""
    manifest_dir: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-detector/manifests")
    """Where each dataset's frozen sampling manifest is written once and reused."""
    catalog_url: str = CATALOG_URL

    def __post_init__(self) -> None:
        if len(self.workers) != len(self.datasets) or min(self.workers) < 1:
            raise ValueError("rerun.workers needs one positive count per dataset")
        unknown: list[str] = [name for name in self.datasets if name not in DATASET_SPECS]
        if unknown:
            raise ValueError(f"unknown rerun datasets {unknown}; expected some of {sorted(DATASET_SPECS)}")
        if not 0 < self.switch_interval_s <= 0.005:
            raise ValueError("switch_interval_s must be in (0, 0.005] seconds")


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
    detnet_cache: Path | None = None
    """A decoded DetNet cache (``tools/build_detnet_cache.py``) to train from instead of the catalog stream; DetNet only.
    Validation still decodes its fixed evaluation set from the catalog."""
    keynet_crop: Literal["affine", "perspective"] = "affine"
    """KeyNet crops of the streams (training and validation): 'perspective' = UmeTrack's crop cameras on the native frames.
    A KeyNet cache records its own crop type; it must match."""
    keynet_negative_margin: float = 0.0
    """``KeyNetAugment.negative_margin`` of the streams (validation's negatives); a KeyNet cache records its own and must match."""
    keynet_other_hand_negatives: bool = True
    """False: drop every cached negative that shows the other hand (``KeyNetSamples.other_inside``), the ablation of the
    left/right presence task. Validation keeps them, so both variants are scored on the same set."""
    keynet_cache: tuple[Path, ...] = ()
    """KeyNet cache passes (``tools/build_detnet_cache.py --net keynet``) to train from instead of the catalog stream; KeyNet only.
    Epoch e reads pass e mod len(passes). Validation still decodes its fixed evaluation set from the catalog."""
    detnet_augment: DetNetAugment = field(default_factory=DetNetAugment)
    """Geometric, 16:9 and photometric augmentation of cached DetNet batches (``handtrack.data.augment``); off by default."""
    upright: bool = False
    """Turn every camera upright (``catalog.UPRIGHT_TURNS``) in training and validation, frames and labels alike; DetNet only.
    False is the legacy input of repro-1 and the deployed DetNet-F (UmeTrack as the sensors are mounted, Quest 3 turned sideways)."""
    rerun: RerunStreamSettings | None = None
    """Train DetNet from the Rerun 0.42 dataloader (``handtrack.data.echo``); validation keeps ``datasets`` on the catalog stream."""
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
    keynet_init: Path | None = None
    """Start KeyNet from these weights (a ``*.weights.pt`` with its ``.sha256`` sidecar) instead of from scratch."""


SPLITS: dict[str, SplitName] = {"training": "train", "train": "train", "validation": "val", "val": "val", "testing": "test", "test": "test"}
"""Trainer split names onto ``catalog.select_split``'s."""
DATASETS: dict[str, DatasetName] = {name: name for name in get_args(DatasetName)}
"""Every catalog dataset the stream knows, keyed by its CLI spelling."""
VALIDATION_SEGMENTS: int = 32
"""Segments an evaluation set is built from (one seeded shuffle of the split's segments), unless segment_ids fixes them."""


def build_source(settings: StreamSettings, split: str, nets: Nets, device: str = "cuda") -> BatchSource:
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
    if settings.rerun is not None and not evaluation:
        if nets == "both":
            raise ValueError("stream.rerun feeds one net at a time; train with nets=detnet or nets=keynet")
        return rerun_source(settings, settings.rerun, torch.device(device), nets)
    if settings.rerun is not None and nets == "keynet":  # validation scores the frozen holdout (``data.keynet_holdout_cuts``), cut anew
        holdout_source: HoldoutSource = HoldoutSource(load_holdout(settings.rerun.catalog_url, torch.device(device)), settings.keynet_batch, device)
        rows: int = sample_count(holdout_source.samples)
        if rows > settings.max_val_batches * settings.keynet_batch:
            raise ValueError(f"the KeyNet holdout has {rows} rows but validation reads at most {settings.max_val_batches} x {settings.keynet_batch}; "
                             "raise stream.max_val_batches so every model is scored on the whole holdout")
        return holdout_source
    if settings.detnet_cache is not None and not evaluation:
        if nets != "detnet":
            raise ValueError("stream.detnet_cache holds DetNet samples only; train with nets=detnet")
        cache: DetNetCache = DetNetCache(settings.detnet_cache, settings.detnet_batch, device, seed=settings.seed, augment=settings.detnet_augment)
        if cache.dataset_names != tuple(settings.datasets):
            raise ValueError(f"{settings.detnet_cache} holds {cache.dataset_names}, the run asks for {settings.datasets}")
        return cache
    if settings.keynet_cache and not evaluation:
        if nets != "keynet":
            raise ValueError("stream.keynet_cache holds KeyNet samples only; train with nets=keynet")
        crops: KeyNetCache = KeyNetCache(settings.keynet_cache, settings.keynet_batch, device, seed=settings.seed,
                                         other_hand_negatives=settings.keynet_other_hand_negatives)
        if {manifest.negative_margin for manifest in crops.manifests} != {settings.keynet_negative_margin}:
            raise ValueError(f"{settings.keynet_cache} were built with negative margins {sorted({m.negative_margin for m in crops.manifests})}, "
                             f"the run validates with {settings.keynet_negative_margin}")
        if {manifest.keynet_crop for manifest in crops.manifests} != {settings.keynet_crop}:
            raise ValueError(f"{settings.keynet_cache} hold {sorted({m.keynet_crop for m in crops.manifests})} crops; the run validates on {settings.keynet_crop!r}")
        if crops.dataset_names != tuple(settings.datasets):
            raise ValueError(f"{settings.keynet_cache} hold {crops.dataset_names}, the run asks for {settings.datasets}")
        return crops
    return CatalogStream(
        StreamConfig(
            device=device,
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
            keynet_crop=settings.keynet_crop,
            keynet=KeyNetAugment(negative_margin=settings.keynet_negative_margin),
            upright=settings.upright,
        )
    )


def training_segments(name: str, holdout: DetectorHoldout, catalog_url: str) -> list[str]:
    """Base datasets: the catalog stream's training split; 2D-keypoint datasets: every segment not held out."""
    import rerun as rr

    dataset = rr.catalog.CatalogClient(catalog_url).get_dataset(name)
    if name in DATASETS:
        return [info.segment_id for info in select_split(list_segments(dataset, name), "train")]
    held: set[str] = set(holdout.segments.get(name, []))
    if not held:
        raise ValueError(f"{name} has no held-out episodes in the holdout file; freeze them before training")
    return sorted(set(dataset.segment_ids()) - held)


def rerun_source(settings: StreamSettings, rerun: RerunStreamSettings, device: torch.device, nets: Nets = "detnet") -> EchoStream:
    """One endless Rerun-dataloader producer per dataset into an ``EchoStream`` with step-budget epochs: DetNet frames, or KeyNet
    crops (``data.keynet_rerun``; their priors are zeroed at draw time by ``DrawAugment``, so every stored crop keeps its prior)."""
    holdout: DetectorHoldout = from_json(DetectorHoldout, rerun.holdout.read_text())
    if nets == "keynet":
        augment: KeyNetAugment = KeyNetAugment(negative_margin=settings.keynet_negative_margin, full_rotation_probability=rerun.keynet_full_rotation,
                                               zero_input_probability=0.0)
        cuts: list[CutStream] = [stream_cuts(name, index, training_segments(name, holdout, rerun.catalog_url), catalog_url=rerun.catalog_url, device=device,
                                             augments=[augment] * rerun.keynet_cuts, seed=settings.seed + index, endless=True) for index, name in enumerate(rerun.datasets)]
        crops: list[KeyNetProducer] = [lambda epoch, cut=cut: (samples for samples, _, _ in cut(epoch)) for cut in cuts]
        return EchoStream(crops, batch_size=settings.keynet_batch, pool_capacity=rerun.pool_capacity, min_fill=rerun.min_fill, max_echo=rerun.max_echo,
                          device=device, seed=settings.seed, finish=keynet_finish(rerun.keynet_draw), epoch_steps=rerun.epoch_steps, min_echo=rerun.min_echo,
                          prefetch=rerun.prefetch, keynet=True)
    producers: list[Producer] = []
    for index, (name, workers) in enumerate(zip(rerun.datasets, rerun.workers, strict=True)):
        segments: list[str] = training_segments(name, holdout, rerun.catalog_url)
        if rerun.decode == "nvdec":
            producers.append(nvdec_producer(source_spec(name, index, settings.upright), segments, catalog_url=rerun.catalog_url, device=device, endless=True))
            continue
        digest: str = hashlib.sha256("\n".join(segments).encode()).hexdigest()[:10]
        manifest: Path = build_manifest(source_spec(name, index), segments, rerun.catalog_url, rerun.manifest_dir / f"{name}-train-{digest}-w{workers}.parquet", workers)
        producers.append(rerun_producer(source_spec(name, index, settings.upright), segments, catalog_url=rerun.catalog_url, manifest_path=manifest, workers=workers,
                                        device=device, endless=True))
    return EchoStream(producers, batch_size=settings.detnet_batch, pool_capacity=rerun.pool_capacity, min_fill=rerun.min_fill, max_echo=rerun.max_echo,
                      device=device, seed=settings.seed, finish=training_finish(settings.detnet_augment), epoch_steps=rerun.epoch_steps,
                      min_echo=rerun.min_echo, prefetch=rerun.prefetch)


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
    if config.stream.rerun is not None:
        sys.setswitchinterval(config.stream.rerun.switch_interval_s)
    torch.backends.cudnn.benchmark = config.device.startswith("cuda")
    cap_gpu_memory(config.stream.gpu_memory_gb, config.device)
    trainer: Trainer = Trainer(config.nets, config.detnet, config.keynet, config.loop, config.run_dir, config.device,
                               resume=config.resume, max_val_batches=config.stream.max_val_batches, config_json=to_json(config),
                               keynet_init=config.keynet_init)
    source: BatchSource = build_source(config.stream, config.stream.train_split, config.nets, device=config.device)
    try:
        validation: BatchSource = build_source(config.stream, config.stream.val_split, config.nets, device=config.device)
    except BaseException:
        source.close()
        raise
    trainer.run(source, validation)
