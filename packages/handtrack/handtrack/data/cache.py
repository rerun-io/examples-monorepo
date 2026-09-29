"""Decode-once caches: one catalog pass writes a network's samples to disk, and training draws its batches from the host
instead of NVDEC.

The stream is decode-bound (NVDEC near 100 %, SM near 5 %), while DetNet-F trains at about 14 k images/s. A cache holds
``CatalogStream``'s samples exactly as its pool stores them, so the pixels, labels and validity rules stay the stream's.
DetNet: uint8 160x120 frames (19.6 kB each with labels); only the intensity scaling and the optional augmentation are drawn
again at every batch. KeyNet: uint8 96x96 crops with the crop jitter, keypoint input and presence negatives already applied
(9.8 kB each); the border occlusion and intensity scaling are drawn again, and each epoch reads one pass, so passes built with
different seeds and ``row_phase`` give each epoch other frames and other jitter.
"""
import dataclasses
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

from handtrack.data.augment import DetNetAugment, augment_detnet
from handtrack.data.batches import CropKind, DetNetBatch, KeyNetBatch
from handtrack.data.stream import CatalogStream, DetNetSamples, KeyNetAugment, KeyNetSamples, StreamStats, sample_count
from handtrack.labels.crops import boundary_occlusion, scale_intensity
from handtrack.labels.heatmaps import render_distance, render_heatmaps

CACHE_VERSION: int = 1
TRAINING_FIELDS: tuple[str, ...] = ("pooled", "circle", "presence", "circle_mask", "presence_mask", "dataset", "points", "in_front")
"""The DetNetSamples fields a training batch needs (keypoints for the augmentation's visibility recount); the camera id stays on disk."""
KEYNET_TRAINING_FIELDS: tuple[str, ...] = ("crops", "points_crop", "d_rel_mm", "keypoints", "presence", "kind", "dataset")
"""The KeyNetSamples fields a training batch needs; the crop affine stays on disk."""
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
    net: str = "detnet"
    """Whose samples: 'detnet' (``DetNetSamples``) or 'keynet' (``KeyNetSamples``)."""
    seed: int = 0
    row_phase: float = 0.0
    row_density: int = 1
    keynet_crop: str = "affine"
    """KeyNet caches: 'affine' or 'perspective' crops (``StreamConfig.keynet_crop``)."""


def write_cache(stream: CatalogStream, directory: Path, split: str, draw: int = 8192, net: str = "detnet") -> CacheManifest:
    """Drain one epoch of ``stream`` (built with nets=``net``, training mode) into ``directory``."""
    if net not in ("detnet", "keynet"):
        raise ValueError(f"net must be 'detnet' or 'keynet', not {net!r}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    start: float = time.perf_counter()
    parts: dict[str, list[Tensor]] = {f.name: [] for f in dataclasses.fields(DetNetSamples if net == "detnet" else KeyNetSamples)}
    stream.start_epoch(0)
    total: int = 0
    last_report: float = start
    while True:
        samples: DetNetSamples | KeyNetSamples | None = stream.next_detnet_samples(draw) if net == "detnet" else stream.next_keynet_samples(draw)
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
        net=net,
        seed=stream.config.seed,
        row_phase=stream.config.row_phase,
        row_density=stream.config.row_density,
        keynet_crop=stream.config.keynet_crop,
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


class _PinnedEpochs:
    """Seeded epochs over host-side rows in full batches only (the few left over differ every epoch): a background thread
    gathers the next batches into pinned memory, and the main thread moves them to the device."""

    def __init__(self, batch_size: int, device: str, seed: int, prefetch: int, name: str) -> None:
        if batch_size < 1 or prefetch < 1:
            raise ValueError("batch_size and prefetch must be positive")
        self.batch_size: int = batch_size
        self.device: torch.device = torch.device(device)
        self.seed: int = seed
        self._name: str = name
        self._pin: bool = self.device.type == "cuda"
        self._generator: torch.Generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._queue: queue.Queue[dict[str, Tensor] | None] = queue.Queue(maxsize=prefetch)
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _epoch_arrays(self, epoch: int) -> dict[str, Tensor]:
        """The rows epoch ``epoch`` visits, one array per field (host tensors, memory-mapped or in RAM)."""
        raise NotImplementedError

    def start_epoch(self, epoch: int) -> None:
        self._halt()
        self._raise_if_failed()
        self._stop.clear()
        arrays: dict[str, Tensor] = self._epoch_arrays(epoch)
        count: int = int(next(iter(arrays.values())).shape[0])
        order: Int64[Tensor, "n"] = torch.randperm(count, generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch))
        self._thread = threading.Thread(target=self._gather, args=(arrays, order), name=f"{self._name}-gather", daemon=True)
        self._thread.start()

    def _gather(self, arrays: dict[str, Tensor], order: Int64[Tensor, "n"]) -> None:
        try:
            full: int = len(order) - len(order) % self.batch_size  # drop the partial batch: a new shape makes torch.compile recompile
            for begin in range(0, full, self.batch_size):
                index: Int64[Tensor, "b"] = order[begin : begin + self.batch_size].sort().values
                rows: dict[str, Tensor] = {name: array.index_select(0, index) for name, array in arrays.items()}
                batch: dict[str, Tensor] = {name: value.pin_memory() if self._pin else value for name, value in rows.items()}
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

    def _next_moved(self) -> dict[str, Tensor] | None:
        """The next gathered batch on the device; None once the epoch is drained or the draw was cancelled."""
        self._raise_if_failed()
        if self._thread is None:
            raise RuntimeError("call start_epoch() before drawing batches")
        host: dict[str, Tensor] | None = None
        while not self._stop.is_set():
            try:
                host = self._queue.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        self._raise_if_failed()
        if host is None:
            return None
        return {name: value.to(self.device, non_blocking=True) for name, value in host.items()}

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
            raise RuntimeError(f"the {self._name} gather thread failed: {self._error!r}") from self._error


def _training_manifest(directory: Path, net: str) -> CacheManifest:
    manifest: CacheManifest = read_manifest(directory)
    if manifest.net != net:
        raise ValueError(f"{directory} caches {manifest.net} samples, not {net}")
    if manifest.split not in TRAINING_SPLITS:
        raise ValueError(f"{directory} caches the {manifest.split!r} split; train only from a training cache")
    return manifest


def _check_rows(directory: Path, arrays: dict[str, Tensor], samples: int) -> None:
    counts: set[int] = {int(array.shape[0]) for array in arrays.values()}
    if counts != {samples}:
        raise ValueError(f"{directory}: arrays hold {sorted(counts)} rows, the manifest says {samples}")


class DetNetCache(_PinnedEpochs):
    """DetNet training batches from a cache held in host RAM (a ``batches.BatchSource``); the device scales their intensity
    and applies the optional augmentation."""

    def __init__(self, directory: Path, batch_size: int, device: str, seed: int = 0,
                 intensity_range: tuple[float, float] = (0.6, 1.4), prefetch: int = 8, augment: DetNetAugment | None = None) -> None:
        super().__init__(batch_size, device, seed, prefetch, "detnet-cache")
        self.manifest: CacheManifest = _training_manifest(directory, "detnet")
        self.arrays: dict[str, Tensor] = {name: torch.from_numpy(np.load(directory / f"{name}.npy")) for name in TRAINING_FIELDS}
        _check_rows(directory, self.arrays, self.manifest.samples)
        circle: Float32[Tensor, "n 2 3"] = self.arrays["circle"]
        centre: Float32[Tensor, "n 2 2"] = circle[..., :2]
        sane: Bool[Tensor, "n 2"] = (circle[..., 2] <= MAX_CIRCLE_RADIUS) & (centre >= -CENTRE_MARGIN).all(-1) & (centre <= 1 + CENTRE_MARGIN).all(-1)
        self.dropped_circles: int = int((self.arrays["circle_mask"] & ~sane).sum())
        """Circle targets masked as unusable; their hands still count as present."""
        self.arrays["circle_mask"] = self.arrays["circle_mask"] & sane
        self.dataset_names: tuple[str, ...] = self.manifest.datasets
        self.intensity_range: tuple[float, float] = intensity_range
        self.augment: DetNetAugment | None = augment

    def _epoch_arrays(self, epoch: int) -> dict[str, Tensor]:
        del epoch
        return self.arrays

    def next_detnet_batch(self) -> DetNetBatch | None:
        moved: dict[str, Tensor] | None = self._next_moved()
        if moved is None:
            return None
        pooled: Float32[Tensor, "b 1 120 160"] = scale_intensity(moved["pooled"][:, None].float() / 255.0, self._generator, *self.intensity_range)
        batch: DetNetBatch = DetNetBatch(pooled=pooled, circle=moved["circle"], presence=moved["presence"], circle_mask=moved["circle_mask"],
                                         presence_mask=moved["presence_mask"], dataset=moved["dataset"])
        if self.augment is not None and self.augment.enabled:
            batch = augment_detnet(batch, moved["points"], moved["in_front"], self.augment, self._generator)
        return batch

    def next_keynet_batch(self) -> KeyNetBatch | None:
        raise RuntimeError("a DetNet cache has no KeyNet batches")


class KeyNetCache(_PinnedEpochs):
    """KeyNet training batches from one or more cache passes (a ``batches.BatchSource``): epoch e reads pass e mod len(passes),
    memory-mapped, and the device draws the stream's border occlusion and intensity scaling and renders the targets."""

    def __init__(self, directories: tuple[Path, ...], batch_size: int, device: str, seed: int = 0,
                 augment: KeyNetAugment | None = None, prefetch: int = 8) -> None:
        super().__init__(batch_size, device, seed, prefetch, "keynet-cache")
        if not directories:
            raise ValueError("a KeyNet cache needs at least one pass directory")
        self.manifests: tuple[CacheManifest, ...] = tuple(_training_manifest(directory, "keynet") for directory in directories)
        names: set[tuple[str, ...]] = {manifest.datasets for manifest in self.manifests}
        if len(names) != 1:
            raise ValueError(f"the KeyNet cache passes hold different datasets: {sorted(names)}")
        self.passes: list[dict[str, Tensor]] = []
        for directory, manifest in zip(directories, self.manifests, strict=True):
            with warnings.catch_warnings():  # read-only memory maps: torch warns that writes would be undefined; nothing writes them
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                arrays: dict[str, Tensor] = {name: torch.from_numpy(np.load(directory / f"{name}.npy", mmap_mode="r")) for name in KEYNET_TRAINING_FIELDS}
            _check_rows(directory, arrays, manifest.samples)
            self.passes.append(arrays)
        self.dataset_names: tuple[str, ...] = self.manifests[0].datasets
        self.augment: KeyNetAugment = KeyNetAugment() if augment is None else augment

    def _epoch_arrays(self, epoch: int) -> dict[str, Tensor]:
        return self.passes[epoch % len(self.passes)]

    def next_detnet_batch(self) -> DetNetBatch | None:
        raise RuntimeError("a KeyNet cache has no DetNet batches")

    def next_keynet_batch(self) -> KeyNetBatch | None:
        moved: dict[str, Tensor] | None = self._next_moved()
        if moved is None:
            return None
        count: int = int(moved["crops"].shape[0])
        occluded: Bool[Tensor, "b 96 96"] = boundary_occlusion(count, self._generator, self.device, self.augment.occlusion_probability,
                                                               self.augment.occlusion_max_fraction)
        crops: Float32[Tensor, "b 1 96 96"] = scale_intensity((moved["crops"].float() / 255.0)[:, None].masked_fill(occluded[:, None], 0.0),
                                                              self._generator, *self.augment.intensity_range)
        positive: Bool[Tensor, "b"] = moved["kind"] == int(CropKind.POSITIVE)
        return KeyNetBatch(
            crops=crops,
            keypoints=moved["keypoints"],
            heatmaps=render_heatmaps(moved["points_crop"]) * positive[:, None, None, None],
            distance=render_distance(moved["d_rel_mm"]) * positive[:, None, None],
            presence=moved["presence"],
            positive=positive,
            presence_mask=torch.ones_like(positive),
            kind=moved["kind"],
            dataset=moved["dataset"],
        )
