"""StreamLogger.log_tick_fit: meshes go to the fit-log worker, each fitted frame once (CPU, no Rerun sink)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from mamma.fitting.window_fitter import FitResult


def _fit(frame_idx: int) -> FitResult:
    from mamma.fitting.window_fitter import FitResult

    return FitResult(
        frame_idx=frame_idx,
        vertices=np.zeros((4, 3), np.float32),
        joints=np.zeros((2, 3), np.float32),
        pose=np.zeros(165, np.float32),
        betas=np.zeros(16, np.float32),
        trans=np.zeros(3, np.float32),
        rest_joints=np.zeros((55, 3), np.float32),
    )


def test_a_refit_of_the_same_frame_is_logged_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Between emits a fitter re-returns its previous fit; the logger must not log that mesh again."""
    from mamma.viz.stream_logger import StreamLogger

    logged: list[tuple[int, list[int]]] = []
    logger = StreamLogger.__new__(StreamLogger)  # no sequence/encoders needed for the fit path
    logger._fit_worker = ThreadPoolExecutor(max_workers=1)
    logger._fit_pending = []
    logger._last_fit_frame = {}
    monkeypatch.setattr(logger, "_log_tick_fit_now", lambda tick, fits, _tri, _faces: logged.append((tick, sorted(f.frame_idx for f in fits.values()))))
    previous = _fit(6)
    for tick, fit in ((10, previous), (11, previous), (12, _fit(8))):
        logger.log_tick_fit(tick, {0: fit}, {}, None)
    for pending in logger._fit_pending:
        pending.result()
    logger._fit_worker.shutdown()
    # Every tick still logs (its triangulated cloud); the mesh of frame 6 only once.
    assert logged == [(10, [6]), (11, []), (12, [8])]
