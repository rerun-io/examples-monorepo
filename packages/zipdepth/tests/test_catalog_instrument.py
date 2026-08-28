"""Behavioral tests for catalog loader instrumentation."""

from collections.abc import Iterator
from pathlib import Path
from threading import get_ident
from time import sleep
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from zipdepth.catalog.instrument import InstrumentedLoader
from zipdepth.catalog.stats import CatalogDatasetStats


class _FakeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Small iterable with the catalog dataset's public control surface."""

    def __init__(self) -> None:
        self.stats: CatalogDatasetStats = CatalogDatasetStats()
        self.skipped_frames: int = 0
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        """Record the forwarded epoch."""
        self.epochs.append(epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield two one-frame batches."""
        for value in range(2):
            yield {"image": torch.tensor([value])}


def test_fixed_schedule_cycles_over_reshuffled_dataset_passes() -> None:
    """Expose the exact run length and reseed every finite dataset pass."""
    dataset: _FakeDataset = _FakeDataset()
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=1, num_workers=0)
    instrumented: InstrumentedLoader = InstrumentedLoader(dataset, loader, steps_per_epoch=5, writer=None)

    values: list[int] = [int(batch["image"].item()) for batch in instrumented]

    assert len(instrumented) == 5
    assert values == [0, 1, 0, 1, 0]
    assert dataset.epochs == [0, 1, 2]
    assert instrumented.sampler is None  # compatibility attribute for the vendored trainer probe; there is no set_epoch to call


class _RecordingWriter(SummaryWriter):
    """SummaryWriter that keeps scalar calls visible to tests."""

    def __init__(self, log_dir: Path) -> None:
        super().__init__(log_dir=str(log_dir))
        self.scalars: dict[str, list[tuple[float, int]]] = {}

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        """Record one scalar without writing it to an event file."""
        del walltime, new_style, double_precision
        self.scalars.setdefault(tag, []).append((float(scalar_value), int(global_step or 0)))


class _DelayedDataset(_FakeDataset):
    """Yield delayed values and optionally fail from the producing thread."""

    def __init__(self, values: list[int], delay_s: float = 0.0, fail_after: int | None = None) -> None:
        super().__init__()
        self._values: list[int] = values
        self._delay_s: float = delay_s
        self._fail_after: int | None = fail_after
        self.producer_thread_ids: set[int] = set()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield in order, updating catalog-like cumulative stage counters."""
        value: int
        for index, value in enumerate(self._values):
            if self._fail_after is not None and index == self._fail_after:
                raise RuntimeError("producer failed")
            self.producer_thread_ids.add(get_ident())
            sleep(self._delay_s)
            self.stats.blob_decode += 0.002
            self.stats.samples_built += 1
            yield {"image": torch.tensor([[value]])}


def test_data_wait_and_compute_are_attributed_separately(tmp_path: Path) -> None:
    """Measure blocking ``next`` time apart from work done by the consumer."""
    dataset: _DelayedDataset = _DelayedDataset([0, 1], delay_s=0.015)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=None, num_workers=0)
    writer: _RecordingWriter = _RecordingWriter(tmp_path)
    instrumented: InstrumentedLoader = InstrumentedLoader(
        dataset,
        loader,
        steps_per_epoch=2,
        writer=writer,
        log_every=2,
    )

    iterator: Iterator[dict[str, torch.Tensor]] = iter(instrumented)
    next(iterator)
    sleep(0.025)
    next(iterator)

    data_wait_ms: float = writer.scalars["io/data_wait_ms"][-1][0]
    compute_ms: float = writer.scalars["io/compute_ms"][-1][0]
    blob_decode_ms: float = writer.scalars["io/blob_decode_ms"][-1][0]
    assert data_wait_ms == pytest.approx(15.0, abs=10.0)
    assert compute_ms == pytest.approx(25.0, abs=10.0)
    assert blob_decode_ms == pytest.approx(2.0, abs=0.2)
    assert writer.scalars["io/frames_per_s"][-1][0] > 0.0


def test_logging_global_step_persists_across_epochs(tmp_path: Path) -> None:
    """Keep TensorBoard steps monotonic when the trainer recreates the iterator."""
    dataset: _DelayedDataset = _DelayedDataset([0])
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=None, num_workers=0)
    writer: _RecordingWriter = _RecordingWriter(tmp_path)
    instrumented: InstrumentedLoader = InstrumentedLoader(dataset, loader, steps_per_epoch=1, writer=writer, log_every=1)

    list(instrumented)
    list(instrumented)

    assert [step for _, step in writer.scalars["io/data_wait_ms"]] == [1, 2]
