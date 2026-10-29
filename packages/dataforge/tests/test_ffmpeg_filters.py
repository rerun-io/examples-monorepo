"""One ffmpeg capability probe across concurrent camera encodes."""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dataforge import video_encoding as video


def test_filter_probe_shared_by_concurrent_encodes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        time.sleep(0.02)
        return subprocess.CompletedProcess(command, 0, " ... scale_cuda V->V Scale CUDA frames\n", "")

    monkeypatch.setattr(video.subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(video.has_scale_cuda, [Path("/fake/filter-probe")] * 8))
    assert len(calls) == 1
