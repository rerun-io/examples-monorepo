"""Measure catalog data stalls separately from trainer compute time."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from time import perf_counter
from typing import Any

import torch
from beartype.roar import BeartypeException
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipdepth.catalog.stats import STAGE_FIELDS, CatalogDatasetStats


def _stage_totals(stats: CatalogDatasetStats) -> dict[str, float]:
    """Read the cumulative floating-point duration for every catalog stage."""
    return {
        "segment_query": stats.segment_query,
        "video_decode": stats.video_decode,
        "blob_decode": stats.blob_decode,
        "augment": stats.augment,
    }


@dataclass(slots=True)
class _Interval:
    """Mutable wait, compute, and frame totals for one logging interval."""

    data_wait_s: float = 0.0
    """Seconds blocked on the next batch."""
    data_wait_count: int = 0
    """Number of measured batch waits."""
    compute_s: float = 0.0
    """Seconds spent by the consumer between yielded batches."""
    compute_count: int = 0
    """Number of measured consumer intervals."""
    frames: int = 0
    """Frames yielded during the interval."""

    def reset(self) -> None:
        """Clear every interval counter in place."""
        self.data_wait_s = 0.0
        self.data_wait_count = 0
        self.compute_s = 0.0
        self.compute_count = 0
        self.frames = 0


class InstrumentedLoader:
    """Wrap a catalog DataLoader with fixed pacing and IO timing."""

    def __init__(
        self,
        dataset: Any,
        loader: DataLoader[dict[str, Tensor]],
        steps_per_epoch: int,
        writer: SummaryWriter | None,
        log_every: int = 50,
    ) -> None:
        """Configure fixed-length iteration and periodic instrumentation.

        Args:
            dataset: Catalog dataset whose epoch and stage counters are exposed.
            loader: In-process DataLoader; it must use ``num_workers=0``.
            steps_per_epoch: Batches yielded to the trainer in one epoch.
            writer: Optional TensorBoard scalar sink.
            log_every: Number of yielded batches between reports.

        Raises:
            ValueError: If a count is not positive.
        """
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self._dataset: Any = dataset
        self._loader: DataLoader[dict[str, Tensor]] = loader
        self._steps_per_epoch: int = steps_per_epoch
        self.sampler: None = None
        """The vendored trainer probes ``loader.sampler.set_epoch``; pass seeding lives in ``__iter__``, so there is none."""
        self._writer: SummaryWriter | None = writer
        self._log_every: int = log_every
        self._global_step: int = 0
        self._stage_snapshot: dict[str, float] = _stage_totals(dataset.stats)
        self._samples_built_snapshot: int = dataset.stats.samples_built

    def __len__(self) -> int:
        """Return the fixed optimizer steps assigned to each epoch."""
        return self._steps_per_epoch

    def _log_interval(
        self,
        *,
        data_wait_s: float,
        data_wait_count: int,
        compute_s: float,
        compute_count: int,
        frames: int,
    ) -> None:
        """Report one interval to the console and optional TensorBoard writer."""
        data_wait_ms: float = data_wait_s * 1000.0 / max(data_wait_count, 1)
        compute_ms: float = compute_s * 1000.0 / max(compute_count, 1)
        frames_per_s: float = frames / max(data_wait_s + compute_s, 1e-12)
        current_stages: dict[str, float] = _stage_totals(self._dataset.stats)
        current_samples_built: int = self._dataset.stats.samples_built
        built_delta: int = current_samples_built - self._samples_built_snapshot
        stage_ms: dict[str, float] = {
            stage: (current_stages[stage] - self._stage_snapshot[stage]) * 1000.0 / max(built_delta, 1.0)
            for stage in STAGE_FIELDS
        }
        self._stage_snapshot = current_stages
        self._samples_built_snapshot = current_samples_built

        gpu_util_pct: float | None = None
        try:
            if torch.cuda.is_available():
                gpu_util_pct = float(torch.cuda.utilization())
        except BeartypeException:
            raise
        except Exception:
            gpu_util_pct = None

        gpu_text: str = "n/a" if gpu_util_pct is None else f"{gpu_util_pct:.0f}%"
        stage_text: str = " ".join(f"{name}={stage_ms[name]:.1f}ms" for name in STAGE_FIELDS)
        print(
            f"[catalog io step {self._global_step}] wait={data_wait_ms:.1f}ms compute={compute_ms:.1f}ms "
            f"throughput={frames_per_s:.2f} frames/s gpu={gpu_text} "
            f"{stage_text} skipped={self._dataset.skipped_frames}"
        )
        if self._writer is None:
            return
        scalar_values: dict[str, float] = {
            "io/data_wait_ms": data_wait_ms,
            "io/compute_ms": compute_ms,
            "io/frames_per_s": frames_per_s,
            "io/skipped_frames": float(self._dataset.skipped_frames),
        }
        name: str
        for name in STAGE_FIELDS:
            scalar_values[f"io/{name}_ms"] = stage_ms[name]
        if gpu_util_pct is not None:
            scalar_values["io/gpu_util_pct"] = gpu_util_pct
        tag: str
        value: float
        for tag, value in scalar_values.items():
            self._writer.add_scalar(tag, value, self._global_step)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield the fixed run length across reshuffled finite dataset passes.

        A pass exhausts the filtered stream once. It is not an epoch: the
        schedule has only an explicit optimizer-step count. Each new pass gets
        a distinct dataset seed through :meth:`set_epoch`.
        """
        interval: _Interval = _Interval()
        yield_started: float | None = None
        yielded_batches: int = 0
        pass_index: int
        for pass_index in count():
            self._dataset.set_epoch(pass_index)
            batches: Iterator[dict[str, Tensor]] = iter(self._loader)
            pass_batches: int = 0
            while yielded_batches < self._steps_per_epoch:
                resumed: float = perf_counter()
                if yield_started is not None:
                    interval.compute_s += resumed - yield_started
                    interval.compute_count += 1

                wait_started: float = perf_counter()
                try:
                    batch: dict[str, Tensor] = next(batches)
                except StopIteration:
                    break
                interval.data_wait_s += perf_counter() - wait_started
                interval.data_wait_count += 1
                pass_batches += 1
                yielded_batches += 1
                image: Tensor = batch["image"]
                interval.frames += int(image.shape[0]) if image.ndim > 0 else 1
                self._global_step += 1

                if self._global_step % self._log_every == 0:
                    self._log_interval(
                        data_wait_s=interval.data_wait_s,
                        data_wait_count=interval.data_wait_count,
                        compute_s=interval.compute_s,
                        compute_count=interval.compute_count,
                        frames=interval.frames,
                    )
                    interval.reset()

                yield_started = perf_counter()
                yield batch
            if yielded_batches >= self._steps_per_epoch:
                return
            if pass_batches == 0:
                raise RuntimeError("catalog dataset pass yielded no batches")
