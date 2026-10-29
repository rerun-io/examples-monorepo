"""The trainer sampler and runner: peak GPU memory, loud failures."""

import sys
from pathlib import Path

import pytest

from hexels_bench.gpu_monitor import Measured, PeakVramSampler, run_measured


def test_peak_vram_ignores_samples_after_training() -> None:
    """Brush renders the test views after training in the same process; they must not count as training memory."""
    sampler = PeakVramSampler(pid=0, changes=[(10.0, 100 * 2**20), (11.0, 400 * 2**20), (12.0, 300 * 2**20), (12.5, 2400 * 2**20)])
    assert sampler.peak_gib(until_unix_s=12.0) == 400 / 1024 and sampler.peak_gib() == 2400 / 1024
    assert sampler.peak_gib(until_unix_s=9.0) == 0.0
    assert PeakVramSampler(pid=0, available=False).peak_gib() is None, "without NVML the peak is unmeasured, not zero"


def test_run_measured_logs_the_command_then_the_output(tmp_path: Path) -> None:
    measured: Measured = run_measured([sys.executable, "-c", "print('hello')"], None, tmp_path / "log.txt")
    lines: list[str] = (tmp_path / "log.txt").read_text().splitlines()
    assert measured.seconds > 0.0 and lines == [f"{sys.executable} -c print('hello')", "hello"]
    assert measured.vram.peak_gib() in (0.0, None), "a CPU-only child holds no GPU memory"


def test_run_measured_raises_on_a_nonzero_exit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=f"exited with 3; see {tmp_path / 'log.txt'}"):
        run_measured([sys.executable, "-c", "print('broken'); raise SystemExit(3)"], None, tmp_path / "log.txt")
    assert "broken" in (tmp_path / "log.txt").read_text()
