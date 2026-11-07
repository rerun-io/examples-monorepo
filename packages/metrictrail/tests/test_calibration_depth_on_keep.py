"""SharedK calibration's depth on keep (recipes/vipe-src patch 0006): a keyframe gets its metric-depth forward only once
the frontend keeps it, a dropped candidate never gets one, and a backend run sees every keyframe with depth."""

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

engine_module = pytest.importorskip("vipe.metrictrail.calibration.engine", reason="needs vipe-src (metrictrail envs)")


@dataclass
class FakeBuffer:
    tstamp: torch.Tensor = field(default_factory=lambda: torch.zeros(16))
    disps_sens: torch.Tensor = field(default_factory=lambda: torch.ones(16, 1, 2, 2))
    n_frames: int = 0
    depth_for: list[int] = field(default_factory=list)  # source frame of every depth forward

    def update_disps_sens(self, _depth_model: Any, frame_idx: int) -> None:
        self.depth_for.append(int(self.tstamp[frame_idx].item()))
        self.disps_sens[frame_idx] = 1.0

    def drop_second_newest(self) -> None:
        """What the frontend does to a redundant candidate (``rm_second_newest_keyframe``)."""
        self.tstamp[self.n_frames - 2] = self.tstamp[self.n_frames - 1]
        self.disps_sens[self.n_frames - 2] = self.disps_sens[self.n_frames - 1]
        self.n_frames -= 1


class FakeFrontend:
    def __init__(self) -> None:
        self.is_initialized = False
        self.drop_next = False
        self.buffer: FakeBuffer | None = None

    def run(self) -> None:
        assert self.buffer is not None
        if self.drop_next:
            self.buffer.drop_second_newest()
            self.drop_next = False


class FakeBackend:
    def __init__(self, buffer: FakeBuffer) -> None:
        self.buffer = buffer
        self.runs: list[tuple[bool, bool]] = []  # (update_depth, a keyframe without depth was present)

    def run(self, steps: int, *, update_depth: bool) -> None:
        self.runs.append((update_depth, bool((self.buffer.disps_sens[: self.buffer.n_frames] == 0).any())))


class FakeSystem:
    def __init__(self) -> None:
        self.buffer = FakeBuffer()
        self.frontend = FakeFrontend()
        self.frontend.buffer = self.buffer
        self.backend = FakeBackend(self.buffer)
        self.metric_depth: object | None = None

    def _add_keyframe(self, frame_index: int, _inputs: Any, _masks: Any, _frames: Any, phase: int) -> None:
        slot = self.buffer.n_frames
        self.buffer.tstamp[slot] = frame_index
        if phase == 1 and self.metric_depth is not None:
            self.buffer.update_disps_sens(self.metric_depth, frame_idx=slot)
        self.buffer.n_frames += 1


def make_engine() -> tuple[Any, FakeSystem]:
    engine = object.__new__(engine_module.ViPESharedFocalEngine)
    system = FakeSystem()
    engine.system = system
    engine.depth_model = system.metric_depth = object()
    engine._pending = None
    engine._depth_pending = set()
    return engine, system


def add(engine: Any, frame_index: int) -> None:
    engine._pending = (frame_index, None, None, [])
    engine.add_keyframe(frame_index)


def test_only_kept_keyframes_get_a_depth_forward() -> None:
    engine, system = make_engine()
    for frame in (0, 1):  # before initialisation: eager, as upstream
        add(engine, frame)
        engine.run_frontend()
    system.frontend.is_initialized = True
    add(engine, 5)
    engine.run_frontend()  # 5 is the newest: not judged yet
    assert system.buffer.depth_for == [0, 1]
    add(engine, 7)
    system.frontend.drop_next = True  # 5 is redundant
    engine.run_frontend()
    add(engine, 9)
    engine.run_frontend()  # 7 kept, 9 pending
    assert system.buffer.depth_for == [0, 1, 7]
    engine.run_backend(5, update_depth=True)
    assert system.buffer.depth_for == [0, 1, 7, 9]
    assert system.backend.runs == [(True, False)]
