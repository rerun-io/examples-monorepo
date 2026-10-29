"""Camera scheduling contracts at the video boundary."""

import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from dataforge.datasets import aria_gen2_pilot_layers as gen2
from dataforge.datasets.aria_gen2_pilot_source import Camera, Scene
from dataforge.timing import SequenceTimer


def test_local_gen2_cameras_start_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    barrier: threading.Barrier = threading.Barrier(3, timeout=2.0)
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "3")
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **kwargs: "ext4\n")
    scene = cast(Scene, Mock(spec=Scene, vrs=SimpleNamespace(path=tmp_path / "video.vrs"), cameras=[SimpleNamespace(stream_id=str(i)) for i in range(3)]))

    def encode(scene: Scene, camera: Camera, clip: Path) -> int:
        barrier.wait()
        return 1

    monkeypatch.setattr(gen2, "encode_camera", encode, raising=False)
    gen2.encode_cameras(scene, tmp_path, SequenceTimer())


@pytest.mark.parametrize("filesystem,slots", [("nfs4", "3"), ("cifs", "3"), ("unknown", "3"), ("ext4", "1")])
def test_network_sources_and_one_slot_stay_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filesystem: str, slots: str) -> None:
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", slots)
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **kwargs: filesystem)
    scene = cast(Scene, Mock(spec=Scene, vrs=SimpleNamespace(path=tmp_path / "video.vrs"), cameras=[SimpleNamespace(stream_id=str(i)) for i in range(3)]))
    order: list[str] = []
    active: int = 0
    peak: int = 0
    lock: threading.Lock = threading.Lock()

    def encode(scene: Scene, camera: Camera, clip: Path) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            order.append(camera.stream_id)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 1

    monkeypatch.setattr(gen2, "encode_camera", encode)
    gen2.encode_cameras(scene, tmp_path, SequenceTimer())
    assert order == ["0", "1", "2"]
    assert peak == 1
