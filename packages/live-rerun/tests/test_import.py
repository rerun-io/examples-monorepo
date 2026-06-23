"""Smoke test: every module imports (incl. the depthai-backed source in-env)."""

from __future__ import annotations

import importlib


def test_modules_import() -> None:
    for name in (
        "live_rerun",
        "live_rerun.rig",
        "live_rerun.calibration",
        "live_rerun.rerun_video_logger",
        "live_rerun.blueprint",
        "live_rerun.sources.depthai",
        "live_rerun.apis.oak_live_rerun",
    ):
        importlib.import_module(name)
