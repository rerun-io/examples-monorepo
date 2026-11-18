"""Callers that score only the raw VIO never run the global BA (k0 review), so a GBA failure cannot fail them.

The gate (:func:`slam_rs.tracking.run_segment`) and the RoboCap fleet (:func:`slam_rs.tracking.run_robocap`) ask
:func:`slam_rs.tracking.drive` for no GBA; the RoboCap probe and ``bench_track`` drive the estimator themselves and never
reach ``finish_map``. RoboCap runs on Mali, where no GBA runs until a probe-backed run exists (DESIGN Q7).
"""

import ast
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from fixture_types import CameraFactory

from slam_rs.catalog_calibration import ImuCalib
from slam_rs.config import SlamConfig
from slam_rs.reference import Benchmarks

EUROC_SEGMENT: str = "vslam-lab-euroc__MH_01_easy"
"""Any reference segment: the stand-in feed stops the run before a frame."""


class _Driven(Exception):
    """Raised by the stand-in ``drive`` once it has seen the caller's arguments."""


def _capture_drive(monkeypatch: pytest.MonkeyPatch, camera: CameraFactory, imu: ImuCalib) -> list[bool]:
    """Replace the feed with two real cameras and ``drive`` with a recorder of ``run_gba``."""
    from slam_rs import tracking

    asked: list[bool] = []

    @contextmanager
    def opened(*_args: object, **_kwargs: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(cameras=(camera(0, 0.0), camera(1, 0.1)), imu=imu)

    def driven(*_args: object, run_gba: bool, **_kwargs: object) -> None:
        asked.append(run_gba)
        raise _Driven

    monkeypatch.setattr(tracking, "open_segment", opened)
    monkeypatch.setattr(tracking, "drive", driven)
    return asked


def test_the_gate_never_runs_the_gba(
    benchmarks: Benchmarks, settings: SlamConfig, camera: CameraFactory, imu: ImuCalib, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slam_rs import tracking

    asked: list[bool] = _capture_drive(monkeypatch, camera, imu)
    with pytest.raises(_Driven):
        tracking.run_segment(settings, benchmarks.by_id(EUROC_SEGMENT), backend="cpu")
    assert asked == [False]


def test_the_robocap_fleet_never_runs_the_gba(
    benchmarks: Benchmarks, settings: SlamConfig, camera: CameraFactory, imu: ImuCalib, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slam_rs import tracking

    asked: list[bool] = _capture_drive(monkeypatch, camera, imu)
    monkeypatch.setattr(tracking, "check_calibration_matches_recording", lambda *_args: None)
    with pytest.raises(_Driven):
        tracking.run_robocap(settings, benchmarks.robocap.session("s00000015"), backend="cpu")
    assert asked == [False]


def _called_names(module: ModuleType) -> set[str]:
    """Every function or method name the module calls."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_the_robocap_probe_and_bench_track_never_reach_the_gba() -> None:
    from slam_rs.apis import bench_track, robocap_probe

    for module in (robocap_probe, bench_track):
        assert _called_names(module).isdisjoint({"finish_map", "final_trajectory", "drive", "run_segment", "run_robocap"}), module.__name__
