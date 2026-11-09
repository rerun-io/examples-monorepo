"""Shared devices must not produce accepted latency measurements."""

import os
import subprocess

import pytest

from posekit.eval.gpu_timing import require_idle_gpu


def test_shared_compute_blocks_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: f"{os.getpid()}\n987654\n")
    with pytest.raises(RuntimeError, match="987654"):
        require_idle_gpu()


def test_only_this_process_can_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: f"{os.getpid()}\n")
    require_idle_gpu()


def test_idle_resident_processes_are_authorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "1\n")
    require_idle_gpu(allow_idle_residents=True)


def test_busy_resident_gpu_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "6\n")
    with pytest.raises(RuntimeError, match="utilization"):
        require_idle_gpu(allow_idle_residents=True)


def test_per_process_spike_cannot_be_hidden_by_own_gpu_work() -> None:
    from posekit.eval.gpu_timing import ProcessUsage, external_spikes

    idle: list[ProcessUsage] = [ProcessUsage(0.0, 5, "rerun", (2, 0, 0, 0, 0, 0), 581)]
    samples: list[ProcessUsage] = [ProcessUsage(1.0, 10, "python", (99, 30, 0, 0, 0, 0), 5000), ProcessUsage(1.0, 5, "rerun", (2, 0, 0, 0, 0, 0), 581)]
    assert external_spikes(idle, samples, own_pid=10) == []
    samples.append(ProcessUsage(2.0, 5, "rerun", (8, 0, 0, 0, 0, 0), 581))
    assert external_spikes(idle, samples, own_pid=10)


def test_malformed_process_sample_is_explicit() -> None:
    from posekit.eval.gpu_timing import _usage

    with pytest.raises(ValueError, match="Unrecognized"):
        _usage("0")
