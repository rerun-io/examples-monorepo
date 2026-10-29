"""The trainer sampler and runner: peak GPU memory, loud failures."""

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pynvml
import pytest

from hexels_bench.gpu_monitor import Measured, PeakVramSampler, run_measured

GIB: int = 2**30


def test_peak_vram_ignores_samples_after_training() -> None:
    """Brush renders the test views after training in the same process; they must not count as training memory."""
    sampler = PeakVramSampler(pid=0, changes=[(10.0, 100 * 2**20), (11.0, 400 * 2**20), (12.0, 300 * 2**20), (12.5, 2400 * 2**20)])
    assert sampler.peak_gib(until_unix_s=12.0) == 400 / 1024 and sampler.peak_gib() == 2400 / 1024
    assert sampler.peak_gib(until_unix_s=9.0) is None, "no usage reported by the cutoff is unknown, not zero"
    assert PeakVramSampler(pid=0, available=False).peak_gib() is None, "without NVML the peak is unmeasured, not zero"


def test_run_measured_logs_the_command_then_the_output(tmp_path: Path) -> None:
    measured: Measured = run_measured([sys.executable, "-c", "print('hello')"], None, tmp_path / "log.txt")
    lines: list[str] = (tmp_path / "log.txt").read_text().splitlines()
    assert measured.seconds > 0.0 and lines == [f"{sys.executable} -c print('hello')", "hello"]
    assert measured.vram.peak_gib() is None, "NVML never lists a CPU-only child, so its peak is unknown"


def test_run_measured_raises_on_a_nonzero_exit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=f"exited with 3; see {tmp_path / 'log.txt'}"):
        run_measured([sys.executable, "-c", "print('broken'); raise SystemExit(3)"], None, tmp_path / "log.txt")
    assert "broken" in (tmp_path / "log.txt").read_text()


@dataclass(frozen=True, slots=True)
class FakeProcess:
    """One row of nvmlDeviceGetComputeRunningProcesses."""

    pid: int
    """Process id."""
    usedGpuMemory: int | None
    """Bytes NVML reports for it; None when the driver does not say."""


class FakeNvml:
    """One GPU whose process list follows `script`, one list per sample; past the end it raises (`fail_after`) or repeats the last list."""

    def __init__(self, script: list[list[FakeProcess]], fail_after: bool) -> None:
        self.script: list[list[FakeProcess]] = script
        self.fail_after: bool = fail_after
        self.calls: int = 0
        self.done: threading.Event = threading.Event()

    def processes(self, handle: int) -> list[FakeProcess]:
        assert handle == 0
        index: int = self.calls
        self.calls += 1
        if index < len(self.script):
            return self.script[index]
        self.done.set()
        if self.fail_after:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_GPU_IS_LOST)
        return self.script[-1]


@pytest.mark.parametrize(
    ("script", "fail_after", "peak", "failed"),
    [
        ([[FakeProcess(42, None)], [FakeProcess(42, GIB), FakeProcess(7, 5 * GIB)], [FakeProcess(42, 2 * GIB)], []], False, 2.0, False),
        ([[FakeProcess(42, GIB)], [FakeProcess(42, 3 * GIB)]], True, None, True),
        ([[FakeProcess(7, 2 * GIB)], [FakeProcess(42, None)]], False, None, False),
    ],
    ids=["measured", "nvml-raised-mid-run", "pid-never-reported"],
)
def test_peak_is_unknown_when_sampling_failed_or_never_saw_the_pid(
    monkeypatch: pytest.MonkeyPatch, script: list[list[FakeProcess]], fail_after: bool, peak: float | None, failed: bool
) -> None:
    fake = FakeNvml(script, fail_after)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", fake.processes)
    with PeakVramSampler(pid=42, interval_s=0.0) as sampler:
        assert fake.done.wait(timeout=5.0)
    assert sampler.peak_gib() == peak
    assert (sampler.error is not None) == failed
