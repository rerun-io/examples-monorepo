"""ffmpeg CUDA scaling capability detection."""

import subprocess
from pathlib import Path

import pytest

from dataforge import video_encoding as video


@pytest.mark.parametrize(("filters", "expected"), [
    (" ... scale_cuda V->V Scale CUDA frames\n", True),
    (" ... scale V->V Scale frames\n", False),
    (" ... scale_cuda_extra V->V Other filter\n", False),
])
def test_filter_probe_answer(filters: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, filters, "")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.has_scale_cuda.cache_clear()
    try:
        assert video.has_scale_cuda(Path("/fake/filter-probe")) is expected
    finally:
        video.has_scale_cuda.cache_clear()
