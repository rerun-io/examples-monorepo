"""A DetNet or KeyNet batch source that keeps the GPU busy: per-dataset echo pools fed by streaming producers.

Each dataset's producer (a Rerun-dataloader stream) runs on its own thread and adds decoded images to that dataset's
pool on the training device. Every batch takes an equal share from each pool. A decoded image is drawn up to
``max_echo`` times, never twice in one batch, so a slow decoder costs reuse instead of GPU idle time. A producer may
only evict images that were drawn at least once; when none are, it waits (decoding throttles to the GPU's pace).
"""

import dataclasses
import queue
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import cast

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.data.stream import DetNetSamples, KeyNetSamples, empty_detnet_samples, empty_keynet_samples, sample_count, select_samples

Samples = DetNetSamples | KeyNetSamples
Producer = Callable[[int], Iterable[DetNetSamples]]
"""One dataset's decoded images for an epoch (CPU or device tensors, any chunk size)."""
KeyNetProducer = Callable[[int], Iterable[KeyNetSamples]]
"""One dataset's KeyNet crops for an epoch."""
Finish = Callable[[DetNetSamples, torch.Generator], DetNetBatch]
"""Turns drawn samples into a training batch (augmentation lives here)."""
KeyNetFinish = Callable[[KeyNetSamples, torch.Generator], KeyNetBatch]


def warm_linalg(device: torch.device) -> None:
    """torch initialises its CUDA linalg backend lazily, and the first call must not race between producer threads ("lazy wrapper
    should be called at most once"; it killed a KeyNet run in ``local_crop_from_net``): make that first call before they start."""
    if device.type == "cuda":
        torch.linalg.inv(torch.eye(3, device=device)[None])


def _select(samples: Samples, index: Int64[Tensor, "m"]) -> Samples:
    """``select_samples`` on either sample type (its type variable takes one type at a time)."""
    return select_samples(samples, index) if isinstance(samples, DetNetSamples) else select_samples(samples, index)


def plain_batch(samples: DetNetSamples, generator: torch.Generator) -> DetNetBatch:
    """No augmentation: pooled bytes to [0, 1] and the stored targets."""
    del generator
    return DetNetBatch(pooled=samples.pooled[:, None].float() / 255.0, circle=samples.circle, presence=samples.presence,
                       circle_mask=samples.circle_mask, presence_mask=samples.presence_mask, dataset=samples.dataset)


class _EchoPool:
    """Fixed slots on the device; ``alive`` rows can still be drawn, ``uses`` counts draws, ``arrival`` orders eviction."""

    def __init__(self, capacity: int, max_echo: int, device: torch.device, generator: torch.Generator, min_echo: int = 1, keynet: bool = False) -> None:
        self.storage: Samples = empty_keynet_samples(capacity, device) if keynet else empty_detnet_samples(capacity, device)
        # Bookkeeping stays on the host: reading counts from device tensors would sync the GPU on every producer check.
        self.alive: Bool[Tensor, "capacity"] = torch.zeros(capacity, dtype=torch.bool)
        self.uses: Int64[Tensor, "capacity"] = torch.zeros(capacity, dtype=torch.int64)
        self.arrival: Int64[Tensor, "capacity"] = torch.zeros(capacity, dtype=torch.int64)
        self.device: torch.device = device
        self.capacity: int = capacity
        self.max_echo: int = max_echo
        self.min_echo: int = min_echo
        """A producer may replace an image only after this many draws (fast datasets then leave decode time to slow ones)."""
        self.generator: torch.Generator = generator
        self.sequence: int = 0
        self.lock: threading.Condition = threading.Condition()
        self.exhausted: bool = False
        self.decoded: int = 0
        self.drawn: int = 0

    def drawable(self) -> int:
        return int(self.alive.sum())

    def room(self) -> int:
        """Slots a producer may fill now: free ones plus alive rows drawn at least ``min_echo`` times."""
        return int((~self.alive).sum() + (self.alive & (self.uses >= self.min_echo)).sum())

    def add(self, samples: Samples) -> None:
        """Write samples into free slots, evicting the oldest rows drawn ``min_echo`` times if needed (caller checked ``room``)."""
        incoming: int = sample_count(samples)
        free: Int64[Tensor, "f"] = (~self.alive).nonzero().flatten()
        if free.numel() < incoming:
            used: Bool[Tensor, "capacity"] = self.alive & (self.uses >= self.min_echo)
            age: Int64[Tensor, "capacity"] = torch.where(used, self.arrival, torch.full_like(self.arrival, torch.iinfo(torch.int64).max))
            oldest: Int64[Tensor, "e"] = age.argsort()[: incoming - free.numel()]
            self.alive[oldest] = False
            free = torch.cat((free, oldest))
        slots: Int64[Tensor, "m"] = free[:incoming]
        on_device: Int64[Tensor, "m"] = slots.to(self.device, non_blocking=True)
        for field in dataclasses.fields(self.storage):
            getattr(self.storage, field.name)[on_device] = getattr(samples, field.name)
        self.alive[slots] = True
        self.uses[slots] = 0
        self.arrival[slots] = torch.arange(self.sequence, self.sequence + incoming)
        self.sequence += incoming
        self.decoded += incoming

    def draw(self, n: int) -> Samples:
        """n distinct alive rows, the less used ones more likely; rows reaching ``max_echo`` uses leave the pool."""
        weight: Float32[Tensor, "capacity"] = torch.where(self.alive, (self.max_echo - self.uses).float(), torch.zeros_like(self.uses, dtype=torch.float32))
        chosen: Int64[Tensor, "n"] = torch.multinomial(weight, n, replacement=False, generator=self.generator)
        drawn: Samples = _select(self.storage, chosen.to(self.device, non_blocking=True))
        self.uses[chosen] += 1
        self.alive[chosen] = self.uses[chosen] < self.max_echo
        self.drawn += n
        return drawn


class EchoStream:
    """``BatchSource`` for DetNet (or KeyNet, ``keynet=True``): equal per-dataset shares from echo pools fed by one producer thread per dataset."""

    def __init__(
        self,
        producers: Sequence[Producer] | Sequence[KeyNetProducer],
        *,
        batch_size: int,
        pool_capacity: int,
        min_fill: int,
        max_echo: int,
        device: torch.device,
        seed: int,
        finish: Finish | KeyNetFinish = plain_batch,
        epoch_steps: int | None = None,
        min_echo: int = 1,
        prefetch: int = 0,
        keynet: bool = False,
    ) -> None:
        """``epoch_steps`` set: an epoch is that many batches, and later epochs keep the running producers and their pools
        (producers then loop over their data themselves). None: an epoch ends when a dataset runs out, and each epoch restarts.
        ``min_echo``: draws an image must get before its producer may replace it. ``prefetch`` > 0: a thread assembles that many
        batches ahead, so drawing, merging and augmentation overlap the training step. ``keynet``: the pools hold ``KeyNetSamples`` and
        the stream serves ``next_keynet_batch`` instead of ``next_detnet_batch``."""
        if not 1 <= min_echo <= max_echo:
            raise ValueError("need 1 <= min_echo <= max_echo")
        if batch_size % len(producers):
            raise ValueError(f"batch size {batch_size} does not split evenly across {len(producers)} datasets")
        self.producers: list[Producer | KeyNetProducer] = list(producers)
        self.share: int = batch_size // len(producers)
        if not self.share <= min_fill <= pool_capacity:
            raise ValueError("need share <= min_fill <= pool_capacity")
        self.min_fill: int = min_fill
        self.finish: Finish | KeyNetFinish = finish
        self.device: torch.device = device
        self.generator: torch.Generator = torch.Generator(device=device).manual_seed(seed)
        """Augmentation randomness on the device (``finish``)."""
        self.draws: torch.Generator = torch.Generator().manual_seed(seed)
        """Draw and eviction randomness on the host (``_EchoPool``)."""
        self.keynet: bool = keynet
        self.pools: list[_EchoPool] = [_EchoPool(pool_capacity, max_echo, device, self.draws, min_echo, keynet) for _ in producers]
        self.prefetch: int = prefetch
        self.ready: queue.Queue[DetNetBatch | KeyNetBatch | BaseException | None] = queue.Queue(maxsize=max(prefetch, 1))
        self.assembler: threading.Thread | None = None
        self.threads: list[threading.Thread] = []
        self.stop: threading.Event = threading.Event()
        self.error: BaseException | None = None
        self.started: bool = False
        self.epoch_steps: int | None = epoch_steps
        self.steps: int = 0

    def _produce(self, index: int, epoch: int) -> None:
        pool: _EchoPool = self.pools[index]
        try:
            for chunk in self.producers[index](epoch):
                samples: Samples = type(chunk)(**{f.name: getattr(chunk, f.name).to(self.device, non_blocking=True) for f in dataclasses.fields(chunk)})
                for start in range(0, sample_count(samples), pool.capacity):
                    part: Samples = _select(samples, torch.arange(start, min(start + pool.capacity, sample_count(samples)), device=self.device))
                    with pool.lock:
                        pool.lock.wait_for(lambda part=part: self.stop.is_set() or pool.room() >= sample_count(part))
                        if self.stop.is_set():
                            return
                        pool.add(part)
                        pool.lock.notify_all()
        except BaseException as error:  # handed to the trainer thread, which re-raises it
            self.error = error
        finally:
            with pool.lock:
                pool.exhausted = True
                pool.lock.notify_all()

    def start_epoch(self, epoch: int) -> None:
        self.steps = 0
        if self.epoch_steps is not None and self.threads and all(thread.is_alive() for thread in self.threads):
            return
        self.close()
        self.stop.clear()
        self.error = None
        for pool in self.pools:
            pool.alive.zero_()
            pool.exhausted = False
        warm_linalg(self.device)
        self.threads = [threading.Thread(target=self._produce, args=(i, epoch), daemon=True, name=f"echo-producer-{i}") for i in range(len(self.pools))]
        for thread in self.threads:
            thread.start()
        self.started = False
        if self.prefetch:
            self.ready = queue.Queue(maxsize=self.prefetch)
            self.assembler = threading.Thread(target=self._assemble_ahead, daemon=True, name="echo-assembler")
            self.assembler.start()

    def _assemble_ahead(self) -> None:
        """Keep ``prefetch`` batches ready; the end of the data (None) or an error travels through the same queue."""
        while not self.stop.is_set():
            try:
                item: DetNetBatch | KeyNetBatch | BaseException | None = self._assemble()
            except BaseException as error:  # re-raised in the trainer thread
                item = error
            while not self.stop.is_set():
                try:
                    self.ready.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
            if not isinstance(item, DetNetBatch | KeyNetBatch):
                return

    def _ready(self, pool: _EchoPool) -> bool:
        need: int = self.share if self.started else self.min_fill
        return self.error is not None or self.stop.is_set() or pool.drawable() >= need or pool.exhausted

    def next_detnet_batch(self) -> DetNetBatch | None:
        if self.keynet:
            raise RuntimeError("this EchoStream feeds KeyNet")
        return cast(DetNetBatch | None, self._next())

    def next_keynet_batch(self) -> KeyNetBatch | None:
        if not self.keynet:
            raise RuntimeError("this EchoStream feeds DetNet")
        return cast(KeyNetBatch | None, self._next())

    def _next(self) -> DetNetBatch | KeyNetBatch | None:
        if self.epoch_steps is not None and self.steps >= self.epoch_steps:
            return None
        if self.prefetch:
            item: DetNetBatch | KeyNetBatch | BaseException | None = None
            while not self.stop.is_set():
                try:
                    item = self.ready.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            if isinstance(item, BaseException):
                raise item
            batch: DetNetBatch | KeyNetBatch | None = item
        else:
            batch = self._assemble()
        if batch is not None:
            self.steps += 1
        return batch

    def _assemble(self) -> DetNetBatch | KeyNetBatch | None:
        parts: list[Samples] = []
        for pool in self.pools:
            with pool.lock:
                pool.lock.wait_for(lambda pool=pool: self._ready(pool))
                if self.error is not None:
                    raise self.error
                if self.stop.is_set() or pool.drawable() < self.share:
                    return None
                parts.append(pool.draw(self.share))
                pool.lock.notify_all()
        self.started = True
        merged: Samples = type(parts[0])(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(parts[0])})
        # The constructor pairs the pools' sample type with its finish (keynet=True: KeyNetSamples and a KeyNet finish).
        if isinstance(merged, KeyNetSamples):
            return cast(KeyNetFinish, self.finish)(merged, self.generator)
        return cast(Finish, self.finish)(merged, self.generator)

    def cancel(self) -> None:
        self.stop.set()
        for pool in self.pools:
            with pool.lock:
                pool.lock.notify_all()

    def close(self) -> None:
        self.cancel()
        for thread in [*self.threads, *([self.assembler] if self.assembler is not None else [])]:
            thread.join()
        self.threads = []
        self.assembler = None
