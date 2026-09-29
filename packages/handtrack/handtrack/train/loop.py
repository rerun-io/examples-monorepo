"""Single-process training over shared DetNet/KeyNet producer pools."""
import math
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import torch
from einops import rearrange
from jaxtyping import Bool, Float32
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor, nn
from torch.optim import Optimizer

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.eval.metrics import Counts, DetectionMetrics, KeypointMetrics, detection_metrics, keynet_metrics, presence_counts
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.models.detnet import Detections, DetNetF, DetNetLoss, DetNetOutput, decode_detections, detnet_loss
from handtrack.models.keynet import HeatmapReduction, KeyNetF, KeyNetLoss, KeyNetOutput, keynet_loss
from handtrack.train.checkpoint import MetricRecord, TrainingState, export_weights, load_checkpoint, save_checkpoint
from handtrack.train.source import BatchSource, DetNetValidation, DetNetValidationSource, KeyNetValidation, KeyNetValidationSource

Nets: TypeAlias = Literal['detnet', 'keynet', 'both']
Net: TypeAlias = Literal['detnet', 'keynet']


@runtime_checkable
class ReadySource(Protocol):
    """Optional backpressure-aware scheduling contract for shared producer pools."""

    def detnet_ready(self) -> bool: ...

    def keynet_ready(self) -> bool: ...

    def wait_for_batch(self) -> bool:
        """Wait for either pool; False permits terminal next_* probes."""
        ...


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class OptimiserSettings:
    """SGD settings for one network."""
    lr: float = 0.001
    """Initial learning rate; KeyNet defaults to 0.025 in the CLI."""
    momentum: float = 0.9
    """SGD momentum."""
    schedule: Literal['constant', 'cosine'] = 'constant'
    """Cosine decays per epoch; constant matches the paper."""

    def __post_init__(self) -> None:
        if self.lr <= 0 or not 0 <= self.momentum < 1:
            raise ValueError('Require lr > 0 and momentum in [0,1)')


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LoopSettings:
    """Training and persistence cadence."""
    epochs: int = 75
    """Total epoch count, including epochs completed before resume."""
    log_every: int = 100
    """Global optimiser steps between JSONL/console records."""
    validate_every: int = 1000
    """Global steps between validation passes; zero disables step-based passes."""
    checkpoint_every: int = 5
    """Keep epoch_NNN.pt every this many completed epochs."""
    keynet_steps_per_detnet_step: int = 1
    """Minimum KeyNet steps requested per DetNet step, subject to readiness and exhaustion.
    Shared pools drain all ready KeyNet batches, which may exceed this ratio;
    they never block to enforce it. Sources without readiness use the fixed ratio.
    """
    bf16: bool = False
    """Enable bfloat16 autocast in production (models have Float32 dev contracts)."""
    channels_last: bool = False
    """Hold the models and their image inputs in channels-last layout (with bf16, DetNet-F trains about 2x faster)."""
    compile: Literal['none', 'default', 'reduce-overhead'] = 'none'
    """torch.compile mode for DetNet's forward (reduce-overhead adds CUDA graphs); Triton needs TRITON_PTXAS_BLACKWELL_PATH
    set to the environment's ptxas on sm_120."""
    presence_weight: float = 1.0
    """Chosen multiplier for the added KeyNet presence BCE."""
    detnet_circle_weight: float = 100.0
    """Multiplier for DetNet's circle MSE in normalized units (cx/640, cy/480, r/640). The paper's L = MSE + 100 BCE (circle
    weight 1) leaves the circle 0.4 % of the loss at these units: after 150 k steps it stalls near 50 px while presence trains.
    Chosen with detnet_presence_weight from a seven-variant probe on one decode (2026-09-28): at 2 k steps from scratch,
    centre RMS 103 px at (1, 100) vs 74 px at (100, 10); warm-started from the (1, 100) run, 51 px and still falling."""
    detnet_presence_weight: float = 10.0
    """Multiplier for DetNet's presence BCE (the paper's 100 with circle weight 1)."""
    resume_next_epoch: bool = True
    """On resume from a mid-epoch last.pt, skip the rest of that epoch instead of replaying its consumed batches: start
    the next epoch with the global step, optimisers, best scores and history kept. Replay assumes a deterministic source;
    ``CatalogStream`` pools depend on producer timing and cannot replay. Deterministic sources may opt into replay
    with ``--loop.no-resume-next-epoch``."""
    heatmap_reduction: HeatmapReduction = 'mean'
    """KeyNet heatmap MSE over pixels: 'mean' averages every value, 'pixel_sum' sums each keypoint's pixels."""
    heatmap_warmup_epochs: int = 0
    """Epochs that use 'mean' before heatmap_reduction applies: 'pixel_sum' from initialisation drives the output ReLU dead."""

    def __post_init__(self) -> None:
        if min(self.epochs, self.log_every, self.checkpoint_every, self.keynet_steps_per_detnet_step) < 1 or self.validate_every < 0:
            raise ValueError('Cadences must be positive (validate_every may be zero)')
        if min(self.presence_weight, self.detnet_circle_weight, self.detnet_presence_weight) < 0:
            raise ValueError('Loss weights must be nonnegative')
        if self.heatmap_warmup_epochs < 0:
            raise ValueError('heatmap_warmup_epochs must be nonnegative')
        if self.bf16 and os.environ.get('PIXI_DEV_MODE') == '1':
            raise ValueError('bf16 needs the prod environment: existing model internals enforce Float32 in dev')


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrainLog:
    """One interval in the append-only training JSONL."""
    step: int
    """Completed optimiser steps across both networks."""
    epoch: int
    """Zero-based epoch."""
    losses: dict[str, dict[str, float]]
    """Most recent loss terms for each network active in this interval."""
    images_per_second: float
    """Processed frames plus crops / training interval seconds (validation excluded)."""
    data_wait_fraction: float
    """Seconds inside next_* / training interval wall seconds."""
    gpu_memory_bytes: int
    """Current allocated CUDA tensor memory, or zero on CPU."""
    gpu_peak_memory_bytes: int
    """Process peak allocated CUDA tensor memory, or zero on CPU."""


@dataclass(slots=True)
class ValidationTotals:
    """Additive sufficient statistics across bounded validation batches."""
    losses: dict[str, float] = field(default_factory=dict)
    """Sample-weighted loss sums."""
    samples: dict[str, int] = field(default_factory=dict)
    """Images/crops contributing to each loss summary."""
    detections: DetectionMetrics = field(default_factory=lambda: DetectionMetrics(Counts(), {}))
    """DetNet counts, global and per camera and hand."""
    keypoints: KeypointMetrics = field(default_factory=lambda: KeypointMetrics(0.0, 0.0, 0, Counts()))
    """KeyNet error sums and presence counts."""


class Trainer:
    """Own models, SGD, validation, and resumable progress.

    Mid-epoch resume starts the next epoch by default. Explicit replay requires
    start_epoch to recreate deterministic batches; catalog streams cannot do so.
    Sources own their random generators. SIGTERM cancels source waits and saves
    incomplete progress in the main control flow.
    """
    def __init__(self, nets: Nets, detnet: OptimiserSettings, keynet: OptimiserSettings,
                 cadence: LoopSettings, run_dir: Path, device: str, *, resume: bool = False,
                 max_val_batches: int = 20, config_json: str = '{}') -> None:
        if max_val_batches < 1:
            raise ValueError('max_val_batches must be positive')
        self.cadence: LoopSettings = cadence
        self.run_dir: Path = run_dir
        self.device: torch.device = torch.device(device)
        self.max_val_batches: int = max_val_batches
        self.settings: dict[str, OptimiserSettings] = {'detnet': detnet, 'keynet': keynet}
        self.models: dict[str, nn.Module] = {}
        if nets in ('detnet', 'both'):
            self.models['detnet'] = DetNetF().to(self.device)
        if nets in ('keynet', 'both'):
            self.models['keynet'] = KeyNetF().to(self.device)
        if cadence.channels_last:
            for model in self.models.values():
                model.to(memory_format=torch.channels_last)  # pyrefly: ignore[no-matching-overload]  (the stubs omit memory_format)
        detnet_model: nn.Module | None = self.models.get('detnet')
        if cadence.compile != 'none' and isinstance(detnet_model, DetNetF):
            # An instance attribute shadows the method; state_dict and checkpoints are unchanged.
            detnet_model.forward_pooled = torch.compile(detnet_model.forward_pooled, mode=cadence.compile)
        self.optimisers: dict[str, Optimizer] = {name: torch.optim.SGD(model.parameters(), lr=self.settings[name].lr, momentum=self.settings[name].momentum)
                                                for name, model in self.models.items()}
        self.state: TrainingState = TrainingState(config_json=config_json)
        if resume:
            self.state = load_checkpoint(run_dir / 'last.pt', self.models, self.optimisers)
            if config_json != '{}':
                # Import at the CLI boundary to avoid a module initialization cycle.
                from handtrack.apis.train import Config, StreamSettings

                old: Config = from_json(Config, self.state.config_json)
                new: Config = from_json(Config, config_json)
                changes: dict[str, tuple[str, str]] = {}
                for section in ('detnet', 'keynet', 'loop', 'stream'):
                    old_section: OptimiserSettings | LoopSettings | StreamSettings = getattr(old, section)
                    new_section: OptimiserSettings | LoopSettings | StreamSettings = getattr(new, section)
                    for setting in fields(new_section):
                        before: object = getattr(old_section, setting.name)
                        after: object = getattr(new_section, setting.name)
                        if before != after:
                            name: str = f'{section}.{setting.name}'
                            changes[name] = (str(before), str(after))
                            print(f'RECIPE CHANGE on resume at step {self.state.step}: {name}: {before} -> {after}', flush=True)
                if changes:
                    record: MetricRecord = MetricRecord(self.state.step, self.state.epoch, {}, changes)
                    self.state = replace(self.state, history=[*self.state.history, record])
                self.state = replace(self.state, config_json=config_json)
            # Keep momentum buffers, but use the new recipe's LR and momentum.
            # Epoch startup applies the new schedule to this base learning rate.
            for name, optimiser in self.optimisers.items():
                for group in optimiser.param_groups:
                    group['lr'] = self.settings[name].lr
                    group['momentum'] = self.settings[name].momentum
            if cadence.resume_next_epoch and self.state.epoch_steps:
                print(f'resume-next-epoch: skipping the rest of epoch {self.state.epoch} ({self.state.epoch_steps} batches were consumed); '
                      f'starting epoch {self.state.epoch + 1} at step {self.state.step}', flush=True)
                self.state = replace(self.state, epoch=self.state.epoch + 1, epoch_steps={})
        self.sources: tuple[BatchSource, ...] = ()
        self.stopping: bool = False
        self.elapsed: float = 0.0
        self.wait_seconds: float = 0.0
        self.images: int = 0
        self.losses: dict[str, dict[str, float]] = {}

    def handle_sigterm(self, signum: int, frame: FrameType | None) -> None:
        """Signal-safe request; checkpoint writes happen in the main control flow."""
        del signum, frame
        self.stopping = True
        for source in self.sources:
            source.cancel()

    def layout(self, images: Float32[Tensor, "b 1 h w"]) -> Float32[Tensor, "b 1 h w"]:
        return images.contiguous(memory_format=torch.channels_last) if self.cadence.channels_last else images

    def forward_detnet(self, batch: DetNetBatch) -> DetNetOutput:
        """Run pooled images and cast heads to Float32 before the existing loss."""
        model: nn.Module = self.models['detnet']
        assert isinstance(model, DetNetF)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cadence.bf16):
            output: DetNetOutput = model.forward_pooled(self.layout(batch.pooled.to(self.device)))
        return DetNetOutput(output.center.float(), output.radius.float(), output.presence_logit.float())

    def forward_keynet(self, batch: KeyNetBatch) -> KeyNetOutput:
        """Run crop and keypoint branches and cast heads before loss/decoding."""
        model: nn.Module = self.models['keynet']
        assert isinstance(model, KeyNetF)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cadence.bf16):
            output: KeyNetOutput = model(self.layout(batch.crops.to(self.device)), batch.keypoints.to(self.device))
        return KeyNetOutput(output.heatmaps.float(), output.distance.float(), output.presence_logit.float())

    def detnet_objective(self, batch: DetNetBatch, output: DetNetOutput) -> DetNetLoss:
        return detnet_loss(output, batch.circle.to(self.device), batch.presence.to(self.device), batch.presence_mask.to(self.device), batch.circle_mask.to(self.device),
                           circle_weight=self.cadence.detnet_circle_weight, presence_weight=self.cadence.detnet_presence_weight)

    def keynet_objective(self, batch: KeyNetBatch, output: KeyNetOutput) -> KeyNetLoss:
        reduction: HeatmapReduction = 'mean' if self.state.epoch < self.cadence.heatmap_warmup_epochs else self.cadence.heatmap_reduction
        return keynet_loss(output, batch.heatmaps.to(self.device), batch.distance.to(self.device), batch.presence.to(self.device), batch.positive.to(self.device),
                           batch.presence_mask.to(self.device), self.cadence.presence_weight, reduction)

    def train_batch(self, batch: DetNetBatch | KeyNetBatch) -> None:
        """Apply one SGD step and retain detached scalar loss terms."""
        name: Net = 'detnet' if isinstance(batch, DetNetBatch) else 'keynet'
        self.models[name].train()
        self.optimisers[name].zero_grad(set_to_none=True)
        if isinstance(batch, DetNetBatch):
            det_loss: DetNetLoss = self.detnet_objective(batch, self.forward_detnet(batch))
            loss: DetNetLoss | KeyNetLoss = det_loss
            terms: dict[str, float] = {'total': float(det_loss.total.detach()), 'circle': float(det_loss.circle), 'presence': float(det_loss.presence)}
            count: int = batch.pooled.shape[0]
        else:
            key_loss: KeyNetLoss = self.keynet_objective(batch, self.forward_keynet(batch))
            loss = key_loss
            terms: dict[str, float] = {'total': float(key_loss.total.detach()), 'heatmap': float(key_loss.heatmap), 'distance': float(key_loss.distance), 'presence': float(key_loss.presence)}
            count = batch.crops.shape[0]
        if not math.isfinite(terms['total']):
            raise ValueError(f'Nonfinite {name} training loss at step {self.state.step}')
        loss.total.backward()
        self.optimisers[name].step()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.images += count
        self.losses[name] = terms
        progress: dict[str, int] = dict(self.state.epoch_steps)
        progress[name] = progress.get(name, 0) + 1
        self.state = replace(self.state, step=self.state.step + 1, epoch_steps=progress)

    def log(self) -> None:
        """Append and print a training interval, including final partial intervals."""
        if not self.images:
            return
        record: TrainLog = TrainLog(self.state.step, self.state.epoch, self.losses, self.images / max(self.elapsed, 1e-9),
                                    self.wait_seconds / max(self.elapsed, 1e-9),
                                    torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                                    torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0)
        with (self.run_dir / 'train.jsonl').open('a') as stream:
            stream.write(to_json(record) + '\n')
            stream.flush()
        print(f'epoch={record.epoch} step={record.step} losses={record.losses} images/s={record.images_per_second:.1f} wait={record.data_wait_fraction:.3f}', flush=True)
        self.elapsed = 0.0
        self.wait_seconds = 0.0
        self.images = 0
        self.losses = {}

    @torch.no_grad()
    def validate(self, source: BatchSource) -> MetricRecord:
        """Bound each net to max_val_batches; preserve train/eval modes."""
        totals: ValidationTotals = ValidationTotals()
        modes: dict[str, bool] = {name: model.training for name, model in self.models.items()}
        try:
            source.start_epoch(self.state.epoch)
            for model in self.models.values():
                model.eval()
            active: set[str] = set(self.models)
            for _ in range(self.max_val_batches):
                for name in tuple(active):
                    if self.stopping:
                        break
                    batch: DetNetBatch | KeyNetBatch | None = source.next_detnet_batch() if name == 'detnet' else source.next_keynet_batch()
                    if batch is None:
                        active.remove(name)
                        continue
                    if isinstance(batch, DetNetBatch):
                        output: DetNetOutput = self.forward_detnet(batch)
                        loss: DetNetLoss | KeyNetLoss = self.detnet_objective(batch, output)
                        count: int = batch.pooled.shape[0]
                        if isinstance(source, DetNetValidationSource):
                            metadata: DetNetValidation = source.detnet_validation()
                            detections: Detections = decode_detections(output)
                            result: DetectionMetrics = detection_metrics(rearrange(detections.box, 'b h c -> (b h) c'), detections.probability.flatten(),
                                rearrange(metadata.points.to(self.device), 'b h k c -> (b h) k c'), rearrange(metadata.in_front.to(self.device), 'b h k -> (b h) k'),
                                metadata.camera.to(self.device).repeat_interleave(2), torch.arange(2, device=self.device).repeat(count), metadata.eligible.to(self.device).flatten())
                            totals.detections = totals.detections + result
                    else:
                        key_output: KeyNetOutput = self.forward_keynet(batch)
                        loss = self.keynet_objective(batch, key_output)
                        count = batch.crops.shape[0]
                        probability: Float32[Tensor, "b"] = key_output.presence_logit.sigmoid()
                        presence: Float32[Tensor, "b"] = batch.presence.to(self.device)
                        presence_mask: Bool[Tensor, "b"] = batch.presence_mask.to(self.device)
                        if isinstance(source, KeyNetValidationSource):
                            key_metadata: KeyNetValidation = source.keynet_validation()
                            key_result: KeypointMetrics = keynet_metrics(decode_heatmaps(key_output.heatmaps)[0], key_metadata.points_crop.to(self.device),
                                key_metadata.crop_from_net.to(self.device), decode_distance(key_output.distance), key_metadata.distance_mm.to(self.device),
                                probability, presence, batch.positive.to(self.device), presence_mask)
                        else:  # Presence needs no metadata; the geometric scores stay unscored.
                            key_result = KeypointMetrics(0.0, 0.0, 0, presence_counts(probability, presence, presence_mask))
                        totals.keypoints = totals.keypoints + key_result
                    if not torch.isfinite(loss.total):
                        raise ValueError(f'Nonfinite {name} validation loss')
                    totals.losses[name] = totals.losses.get(name, 0.0) + float(loss.total) * count
                    totals.samples[name] = totals.samples.get(name, 0) + count
        finally:
            for name, model in self.models.items():
                model.train(modes[name])
        values: dict[str, float | None] = {f'{name}/loss': loss / totals.samples[name] for name, loss in totals.losses.items()}
        if 'detnet' in self.models:
            values.update({'detnet/precision': totals.detections.total.precision, 'detnet/recall': totals.detections.total.recall})
            for (camera, hand), counts in totals.detections.by_camera_hand.items():
                for label, value in (('tp', counts.true_positive), ('predicted', counts.predicted), ('gt', counts.ground_truth),
                                     ('precision', counts.precision), ('recall', counts.recall)):
                    values[f'detnet/camera_{camera}/hand_{hand}/{label}'] = float(value) if value is not None else None
        if 'keynet' in self.models:
            values.update({'keynet/error_px': totals.keypoints.error_px, 'keynet/d_rel_mm': totals.keypoints.distance_mm,
                           'keynet/precision': totals.keypoints.presence.precision, 'keynet/recall': totals.keypoints.presence.recall})
        record: MetricRecord = MetricRecord(self.state.step, self.state.epoch, values)
        if self.stopping:
            return record  # Partial SIGTERM validation must not select best weights.
        best: dict[str, float] = dict(self.state.best)
        improved: list[str] = []
        for name in self.models:
            if name == 'detnet':
                precision: float | None = values.get('detnet/precision')
                recall: float | None = values.get('detnet/recall')
                score: float | None = (precision + recall) / 2 if precision is not None and recall is not None else None
                better: bool = score is not None and score > best.get(name, -math.inf)
            else:
                score = values.get('keynet/error_px')
                better: bool = score is not None and score < best.get(name, math.inf)
            if better and score is not None and math.isfinite(score):
                best[name] = score
                improved.append(name)
        self.state = replace(self.state, history=[*self.state.history, record], best=best)
        with (self.run_dir / 'validation.jsonl').open('a') as stream:
            stream.write(to_json(record) + '\n')
        for name in improved:
            directory: Path = self.run_dir / name if len(self.models) > 1 else self.run_dir
            save_checkpoint(directory / 'best.pt', self.models, self.optimisers, self.state)
            export_weights(directory / 'best.weights.pt', self.models[name])
        return record

    def run(self, source: BatchSource, validation: BatchSource) -> TrainingState:
        """Train to exhaustion of BOTH pools, close sources, and retain checkpoints."""
        self.sources = (source, validation)
        if self.stopping:
            for stream in self.sources:
                stream.cancel()
        schedule: list[Net] = []
        if 'detnet' in self.models:
            schedule.append('detnet')
        if 'keynet' in self.models:
            schedule.extend(['keynet'] * self.cadence.keynet_steps_per_detnet_step)
        previous: signal.Handlers | int | Callable[[int, FrameType | None], None] | None = signal.signal(signal.SIGTERM, self.handle_sigterm)
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            while self.state.epoch < self.cadence.epochs and not self.stopping:
                source.start_epoch(self.state.epoch)
                replay: dict[str, int] = dict(self.state.epoch_steps)
                active: set[str] = set(self.models)
                for name, optimiser in self.optimisers.items():
                    settings: OptimiserSettings = self.settings[name]
                    factor: float = (1 + math.cos(math.pi * self.state.epoch / self.cadence.epochs)) / 2 if settings.schedule == 'cosine' else 1.0
                    for group in optimiser.param_groups:
                        group['lr'] = settings.lr * factor
                drain_keynet: bool = False
                while active and not self.stopping:
                    draws: list[Net] = schedule
                    if len(self.models) == 2 and isinstance(source, ReadySource):
                        det_ready: bool = 'detnet' in active and source.detnet_ready()
                        key_ready: bool = 'keynet' in active and source.keynet_ready()
                        if key_ready and (drain_keynet or not det_ready):
                            draws = ['keynet']
                        elif det_ready:
                            draws = ['detnet']
                            drain_keynet = True
                        else:
                            start_wait: float = time.perf_counter()
                            available: bool = source.wait_for_batch()
                            waited_for_either: float = time.perf_counter() - start_wait
                            self.wait_seconds += waited_for_either
                            self.elapsed += waited_for_either
                            if available:
                                continue
                            # Both pools are empty and production ended: retain next_*'s
                            # None contract, including resume-cursor checks below.
                    for name in draws:
                        if name not in active or self.stopping:
                            continue
                        start: float = time.perf_counter()
                        batch: DetNetBatch | KeyNetBatch | None = source.next_detnet_batch() if name == 'detnet' else source.next_keynet_batch()
                        waited: float = time.perf_counter() - start
                        if batch is None:
                            if replay.get(name, 0) and not self.stopping:
                                raise ValueError(f'{name} source exhausted before the resume cursor')
                            self.wait_seconds += waited
                            self.elapsed += time.perf_counter() - start
                            active.remove(name)
                            continue
                        if replay.get(name, 0):
                            replay[name] -= 1
                            continue
                        self.train_batch(batch)
                        self.elapsed += time.perf_counter() - start
                        self.wait_seconds += waited
                        if self.state.step % self.cadence.log_every == 0:
                            self.log()
                        if self.cadence.validate_every and self.state.step % self.cadence.validate_every == 0 and not self.stopping:
                            self.validate(validation)
                            save_checkpoint(self.run_dir / 'last.pt', self.models, self.optimisers, self.state)
                self.log()
                if self.stopping:
                    break
                self.validate(validation)
                if self.stopping:
                    break
                self.state = replace(self.state, epoch=self.state.epoch + 1, epoch_steps={})
                save_checkpoint(self.run_dir / 'last.pt', self.models, self.optimisers, self.state)
                if self.state.epoch % self.cadence.checkpoint_every == 0:
                    save_checkpoint(self.run_dir / f'epoch_{self.state.epoch:03d}.pt', self.models, self.optimisers, self.state)
            self.log()
            save_checkpoint(self.run_dir / 'last.pt', self.models, self.optimisers, self.state)
            return self.state
        finally:
            self.sources = ()
            signal.signal(signal.SIGTERM, previous)
            try:
                source.close()
            finally:
                validation.close()
