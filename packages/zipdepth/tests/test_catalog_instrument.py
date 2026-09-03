"""Behavioral tests for catalog loader instrumentation."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from time import sleep

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from zipdepth.catalog.instrument import InstrumentedDataset, InstrumentedLoader
from zipdepth.catalog.stats import CatalogDatasetStats


class _FakeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Small iterable with the catalog dataset's public control surface."""

    def __init__(self) -> None:
        self._stats: CatalogDatasetStats = CatalogDatasetStats()
        self.stats_reads: int = 0
        self.epochs: list[int] = []

    @property
    def stats(self) -> CatalogDatasetStats:
        """Return the cumulative stats and count snapshot reads."""
        self.stats_reads += 1
        return replace(self._stats)

    def set_epoch(self, epoch: int) -> None:
        """Record the forwarded epoch."""
        self.epochs.append(epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield two one-frame batches."""
        for value in range(2):
            yield {"image": torch.tensor([value])}


def test_fake_dataset_satisfies_instrumentation_protocol() -> None:
    """Keep loader instrumentation structural and usable with lightweight datasets."""
    assert isinstance(_FakeDataset(), InstrumentedDataset)


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
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        """Record one scalar without writing it to an event file."""
        del walltime, new_style, double_precision
        self.scalars.setdefault(tag, []).append((float(scalar_value), int(global_step or 0)))


class _DelayedDataset(_FakeDataset):
    """Yield delayed values while updating catalog-like counters."""

    def __init__(self, values: list[int], delay_s: float = 0.0) -> None:
        super().__init__()
        self._values: list[int] = values
        self._delay_s: float = delay_s

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield in order, updating catalog-like cumulative stage counters."""
        value: int
        for value in self._values:
            sleep(self._delay_s)
            self._stats.blob_decode += 0.002
            self._stats.samples_built += 1
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
    assert dataset.stats_reads == 2  # constructor baseline plus one report snapshot


def test_compute_time_is_not_counted_twice_at_a_pass_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Account consumer work once when the next finite dataset pass starts."""
    times: Iterator[float] = iter([0.0, 0.0, 0.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0, 13.0])
    monkeypatch.setattr("zipdepth.catalog.instrument.perf_counter", lambda: next(times))
    dataset: _DelayedDataset = _DelayedDataset([0])
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=None, num_workers=0)
    writer: _RecordingWriter = _RecordingWriter(tmp_path)
    instrumented: InstrumentedLoader = InstrumentedLoader(dataset, loader, steps_per_epoch=2, writer=writer, log_every=2)

    list(instrumented)

    assert writer.scalars["io/compute_ms"] == [(1000.0, 2)]


def test_gpu_utilization_failure_warns_once_and_disables_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not retry a GPU utilization backend that raised an exception."""
    probe_calls: int = 0

    def fail_probe() -> int:
        nonlocal probe_calls
        probe_calls += 1
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "utilization", fail_probe)
    dataset: _DelayedDataset = _DelayedDataset([0, 1])
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=None, num_workers=0)
    writer: _RecordingWriter = _RecordingWriter(tmp_path)
    instrumented: InstrumentedLoader = InstrumentedLoader(dataset, loader, steps_per_epoch=2, writer=writer, log_every=1)

    with pytest.warns(RuntimeWarning, match="NVML unavailable"):
        list(instrumented)

    assert probe_calls == 1


def test_logging_global_step_starts_after_restored_training_step(tmp_path: Path) -> None:
    """Keep resumed TensorBoard steps monotonic when the trainer recreates the iterator."""
    dataset: _DelayedDataset = _DelayedDataset([0])
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=None, num_workers=0)
    writer: _RecordingWriter = _RecordingWriter(tmp_path)
    instrumented: InstrumentedLoader = InstrumentedLoader(dataset, loader, steps_per_epoch=1, writer=writer, log_every=1)
    instrumented.set_global_step(40)

    list(instrumented)
    list(instrumented)

    assert [step for _, step in writer.scalars["io/data_wait_ms"]] == [41, 42]
