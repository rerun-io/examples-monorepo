"""Audit shared-GPU timing against measured idle process activity."""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Self

from serde import serde
from serde.json import to_json


def require_idle_gpu(device: int = 0, *, allow_idle_residents: bool = False) -> None:
    """Require an empty device, or explicitly authorized residents below 5% utilization."""
    if allow_idle_residents:
        utilization: int = int(
            subprocess.check_output(
                ["nvidia-smi", "-i", str(device), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
            ).strip()
        )
        if utilization > 5:
            raise RuntimeError(f"GPU utilization is {utilization}%, above the 5% pre-run limit.")
        return
    output: str = subprocess.check_output(["nvidia-smi", "-i", str(device), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    others: set[int] = {int(row.strip()) for row in output.splitlines() if row.strip()} - {os.getpid()}
    if others:
        raise RuntimeError(f"GPU {device} has other compute processes {sorted(others)}; latency measurement is forbidden.")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ProcessUsage:
    """One nvidia-smi pmon row, sampled once per second."""

    utc_seconds: float
    """Host UTC wall clock when the row was received."""
    pid: int
    """GPU context owner."""
    name: str
    """Process name reported by the driver."""
    utilization: tuple[int, int, int, int, int, int]
    """Percent SM, memory bandwidth, encoder, decoder, JPEG and optical-flow activity."""
    memory_mb: int
    """Resident framebuffer memory; allocation alone is not GPU utilization."""


def external_spikes(baseline: list[ProcessUsage], samples: list[ProcessUsage], *, own_pid: int) -> list[ProcessUsage]:
    """Find any external engine usage above that process's observed idle envelope."""
    envelope: dict[int, tuple[int, ...]] = {}
    for sample in baseline:
        prior: tuple[int, ...] = envelope.get(sample.pid, (0,) * 6)
        envelope[sample.pid] = tuple(max(a, b) for a, b in zip(prior, sample.utilization, strict=True))
    return [
        sample
        for sample in samples
        if sample.pid != own_pid and any(a > b for a, b in zip(sample.utilization, envelope.get(sample.pid, (0,) * 6), strict=True))
    ]


def _usage(line: str) -> ProcessUsage | None:
    """Parse the driver format verified on both target machines; '-' means no sample activity."""
    fields: list[str] = line.split()
    if not fields or fields[0].startswith("#"):
        return None
    if len(fields) < 12:
        raise ValueError(f"Unrecognized nvidia-smi pmon row: {line}")
    if fields[1] == "-":
        return None
    values: tuple[int, ...] = tuple(0 if x == "-" else int(x) for x in fields[3:9])
    return ProcessUsage(time.time(), int(fields[1]), fields[-1], (values[0], values[1], values[2], values[3], values[4], values[5]), int(fields[9]))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TimingAudit:
    """Acceptance evidence for one timing process; raw pmon and device samples are adjacent."""

    accepted: bool
    """True only if inference succeeded and no external engine use increased."""
    own_pid: int
    """The benchmark process whose work is excluded from contention checks."""
    baseline_utilization_percent: int
    """Whole-GPU activity just before the monitored run; must be <=5%."""
    baseline: list[ProcessUsage]
    """Thirty idle per-process samples before the run; captures periodic resident activity."""
    samples: list[ProcessUsage]
    """Per-process samples throughout the run."""
    spikes: list[ProcessUsage]
    """External rows above their idle envelope; any row invalidates the run."""


class GpuActivityMonitor:
    """Monitor an owned timing run without touching resident applications."""

    def __init__(self, output: Path, *, allow_idle_residents: bool = False, device: int = 0) -> None:
        self.output: Path = output
        self.allow_idle_residents: bool = allow_idle_residents
        self.device: int = device
        self.baseline: list[ProcessUsage] = []
        self.samples: list[ProcessUsage] = []
        self.utilization: int = -1
        self.reader_error: str | None = None
        self.pmon: subprocess.Popen[str] | None = None
        self.device_monitor: subprocess.Popen[str] | None = None
        self.reader: Thread | None = None

    def __enter__(self) -> Self:
        self.output.mkdir(parents=True, exist_ok=True)
        # Model setup can leave one stale utilization sample. Allow that sample to drain.
        deadline: float = time.monotonic() + 60.0
        while True:
            self.utilization = int(
                subprocess.check_output(
                    ["nvidia-smi", "-i", str(self.device), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
                ).strip()
            )
            if self.utilization <= 5:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"GPU utilization stayed above 5% before timing ({self.utilization}%).")
            time.sleep(1.0)
        require_idle_gpu(self.device, allow_idle_residents=self.allow_idle_residents)
        baseline_text: str = subprocess.check_output(["nvidia-smi", "pmon", "-i", str(self.device), "-s", "um", "-c", "30"], text=True)
        (self.output / "gpu-idle-pmon.txt").write_text(baseline_text)
        self.baseline = [sample for line in baseline_text.splitlines() if (sample := _usage(line)) is not None]
        require_idle_gpu(self.device, allow_idle_residents=self.allow_idle_residents)
        self.pmon = subprocess.Popen(
            ["stdbuf", "-oL", "nvidia-smi", "pmon", "-i", str(self.device), "-s", "um", "-d", "1"], stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()
        with (self.output / "gpu-device.csv").open("w") as output:
            self.device_monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.device),
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.current.sm,power.draw",
                    "--format=csv",
                    "-lms",
                    "500",
                ],
                stdout=output,
                text=True,
            )
        return self

    def _read(self) -> None:
        assert self.pmon is not None and self.pmon.stdout is not None
        with (self.output / "gpu-run-pmon.txt").open("w") as output:
            for line in self.pmon.stdout:
                output.write(line)
                output.flush()
                try:
                    sample: ProcessUsage | None = _usage(line)
                except ValueError as error:
                    self.reader_error = str(error)
                    continue
                if sample is not None:
                    self.samples.append(sample)

    def __exit__(self, exception_type: type[BaseException] | None, exception: BaseException | None, traceback: TracebackType | None) -> None:
        # Capture the driver's trailing one-second utilization window before stopping our monitors.
        time.sleep(1.1)
        for process in (self.pmon, self.device_monitor):
            if process is not None:
                if process.poll() is not None:
                    self.reader_error = f"GPU monitor exited before the benchmark: status {process.returncode}."
                process.terminate()
                process.wait(timeout=5.0)
        if self.reader is not None:
            self.reader.join(timeout=5.0)
        spikes: list[ProcessUsage] = external_spikes(self.baseline, self.samples, own_pid=os.getpid())
        accepted: bool = exception_type is None and not spikes and self.reader_error is None and bool(self.samples)
        audit: TimingAudit = TimingAudit(accepted, os.getpid(), self.utilization, self.baseline, self.samples, spikes)
        (self.output / "gpu-audit.json").write_text(to_json(audit) + "\n")
        if exception_type is None and not accepted:
            raise RuntimeError(f"GPU timing rejected: {len(spikes)} external activity samples; monitor error={self.reader_error}. Re-time the run.")
