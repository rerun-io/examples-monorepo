"""A decode-once DetNet cache: one catalog pass writes the unaugmented pooled frames and targets to disk, and training
draws its batches from host RAM instead of NVDEC.

The stream is decode-bound (NVDEC near 100 %, SM near 5 %), while DetNet-F trains at about 14 k images/s. The cache holds
``CatalogStream``'s DetNet samples exactly as the pool stores them (uint8 160x120 frames, 19.6 kB each with labels), so the
pixels, labels and validity rules stay the stream's; only the intensity scaling is drawn again at every batch.
"""
import dataclasses
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.data.stream import CatalogStream, DetNetSamples, StreamStats, sample_count
from handtrack.labels.crops import scale_intensity

CACHE_VERSION: int = 1
TRAINING_FIELDS: tuple[str, ...] = ("pooled", "circle", "presence", "circle_mask", "presence_mask", "dataset")
"""The DetNetSamples fields a training batch needs; the rest (validation metadata) stay on disk."""
TRAINING_SPLITS: tuple[str, ...] = ("train", "training")
MAX_CIRCLE_RADIUS: float = 0.5
"""Largest usable target radius, in image widths. A few SHOW3D hands right at a fisheye lens project to circles tens of image
widths wide (radius up to 33); in the squared circle loss one of them outweighs millions of ordinary targets."""
CENTRE_MARGIN: float = 0.5
"""How far outside the image, in image widths/heights, a target centre may lie."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CacheManifest:
    """What a cache directory holds; written last, so a directory without one is incomplete."""

    version: int
    datasets: tuple[str, ...]
    """Names behind the ``dataset`` index, in the stream's order."""
    split: str
    samples: int
    segments: int
    """Segments the pass completed (the stream's count, partial failures included)."""
    skipped_segments: int
    failures: tuple[str, ...]
    images_considered: int
    images_missing: int
    images_invalid: int
    build_seconds: float


def write_cache(stream: CatalogStream, directory: Path, split: str, draw: int = 8192) -> CacheManifest:
    """Drain one epoch of ``stream`` (built with nets='detnet', training mode) into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    start: float = time.perf_counter()
    parts: dict[str, list[Tensor]] = {f.name: [] for f in dataclasses.fields(DetNetSamples)}
    stream.start_epoch(0)
    total: int = 0
    last_report: float = start
    while True:
        samples: DetNetSamples | None = stream.next_detnet_samples(draw)
        if samples is None:
            break
        for name, values in parts.items():
            values.append(getattr(samples, name).cpu())
        total += sample_count(samples)
        now: float = time.perf_counter()
        if now - last_report > 30.0:
            stats: StreamStats = stream.stats
            print(f"[cache] {total} samples, {stats.segments}/{len(stream.segments)} segments, {total / (now - start):.0f} samples/s", flush=True)
            last_report = now
    for name, values in parts.items():
        np.save(directory / f"{name}.npy", torch.cat(values).numpy())
    stats = stream.stats
    manifest: CacheManifest = CacheManifest(
        version=CACHE_VERSION,
        datasets=stream.dataset_names,
        split=split,
        samples=total,
        segments=stats.segments,
        skipped_segments=stats.skipped_segments,
        failures=tuple(stats.failures),
        images_considered=stats.images_considered,
        images_missing=stats.images_missing,
        images_invalid=stats.images_invalid,
        build_seconds=time.perf_counter() - start,
    )
    (directory / "manifest.json").write_text(to_json(manifest))
    return manifest


def read_manifest(directory: Path) -> CacheManifest:
    path: Path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(f"{directory} has no manifest.json: the cache is missing or its build did not finish")
    manifest: CacheManifest = from_json(CacheManifest, path.read_text())
    if manifest.version != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {manifest.version}, this code reads {CACHE_VERSION}")
    return manifest


@dataclass(frozen=True, slots=True)
class _HostBatch:
    """One batch gathered on the host into pinned memory."""

    pooled: UInt8[Tensor, "b 120 160"]
    circle: Float32[Tensor, "b 2 3"]
    presence: Float32[Tensor, "b 2"]
    circle_mask: Bool[Tensor, "b 2"]
    presence_mask: Bool[Tensor, "b 2"]
    dataset: Int64[Tensor, "b"]


class DetNetCache:
    """DetNet training batches from a cache held in host RAM (a ``batches.BatchSource``).

    Every epoch visits each cached sample once, in a seeded order; a background thread gathers the next batches into
    pinned memory, and the main thread copies them to the device and scales their intensity there.
    """

    def __init__(self, directory: Path, batch_size: int, device: str, seed: int = 0,
                 intensity_range: tuple[float, float] = (0.6, 1.4), prefetch: int = 8) -> None:
        if batch_size < 1 or prefetch < 1:
            raise ValueError("batch_size and prefetch must be positive")
        self.manifest: CacheManifest = read_manifest(directory)
        self.arrays: dict[str, Tensor] = {name: torch.from_numpy(np.load(directory / f"{name}.npy")) for name in TRAINING_FIELDS}
        counts: set[int] = {int(array.shape[0]) for array in self.arrays.values()}
        if counts != {self.manifest.samples}:
            raise ValueError(f"{directory}: arrays hold {sorted(counts)} rows, the manifest says {self.manifest.samples}")
        if self.manifest.split not in TRAINING_SPLITS:
            raise ValueError(f"{directory} caches the {self.manifest.split!r} split; train only from a training cache")
        circle: Float32[Tensor, "n 2 3"] = self.arrays["circle"]
        centre: Float32[Tensor, "n 2 2"] = circle[..., :2]
        sane: Bool[Tensor, "n 2"] = (circle[..., 2] <= MAX_CIRCLE_RADIUS) & (centre >= -CENTRE_MARGIN).all(-1) & (centre <= 1 + CENTRE_MARGIN).all(-1)
        self.dropped_circles: int = int((self.arrays["circle_mask"] & ~sane).sum())
        """Circle targets masked as unusable; their hands still count as present."""
        self.arrays["circle_mask"] = self.arrays["circle_mask"] & sane
        self.dataset_names: tuple[str, ...] = self.manifest.datasets
        self.batch_size: int = batch_size
        self.device: torch.device = torch.device(device)
        self.seed: int = seed
        self.intensity_range: tuple[float, float] = intensity_range
        self._pin: bool = self.device.type == "cuda"
        self._generator: torch.Generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._queue: queue.Queue[_HostBatch | None] = queue.Queue(maxsize=prefetch)
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start_epoch(self, epoch: int) -> None:
        self._halt()
        self._raise_if_failed()
        self._stop.clear()
        order: Int64[Tensor, "n"] = torch.randperm(self.manifest.samples, generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch))
        self._thread = threading.Thread(target=self._gather, args=(order,), name="detnet-cache-gather", daemon=True)
        self._thread.start()

    def _gather(self, order: Int64[Tensor, "n"]) -> None:
        try:
            for begin in range(0, len(order), self.batch_size):
                index: Int64[Tensor, "b"] = order[begin : begin + self.batch_size].sort().values
                rows: dict[str, Tensor] = {name: array.index_select(0, index) for name, array in self.arrays.items()}
                batch: _HostBatch = _HostBatch(**{name: value.pin_memory() if self._pin else value for name, value in rows.items()})
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
            self._queue.put(None)
        except BaseException as error:  # re-raised on the main thread, beartype's included
            self._error = error
            self._queue.put(None)

    def next_detnet_batch(self) -> DetNetBatch | None:
        self._raise_if_failed()
        if self._thread is None:
            raise RuntimeError("call start_epoch() before drawing batches")
        host: _HostBatch | None = None
        while not self._stop.is_set():
            try:
                host = self._queue.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        self._raise_if_failed()
        if host is None:
            return None
        moved: dict[str, Tensor] = {f.name: getattr(host, f.name).to(self.device, non_blocking=True) for f in dataclasses.fields(host)}
        pooled: Float32[Tensor, "b 1 120 160"] = scale_intensity(moved["pooled"][:, None].float() / 255.0, self._generator, *self.intensity_range)
        return DetNetBatch(pooled=pooled, circle=moved["circle"], presence=moved["presence"], circle_mask=moved["circle_mask"],
                           presence_mask=moved["presence_mask"], dataset=moved["dataset"])

    def next_keynet_batch(self) -> KeyNetBatch | None:
        raise RuntimeError("a DetNet cache has no KeyNet batches")

    def cancel(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._halt()

    def _halt(self) -> None:
        self._stop.set()
        if self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    self._thread.join(timeout=0.05)
            self._thread = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"the DetNet cache gather thread failed: {self._error!r}") from self._error
