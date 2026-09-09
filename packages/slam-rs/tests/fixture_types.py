"""Types, sizes, periods and the two stand-ins the suites share.

:mod:`conftest` injects the fixtures themselves, but a ``TypeAlias`` is not
something pytest can inject and every suite that annotates a factory has to
name its type. They live here once, so ``PipelineFactory`` means one thing
across the suite; the synthetic rig's frame size and the two sample periods come
with them, because a test that builds a batch by hand needs the same numbers the
fixtures were built for. This is not a test module, so conftest's rule that no
test module sits on another one's import path still holds, and it is where
:func:`gravity_batch` and :func:`never` sit for the same reason: both are called
from module-level helpers a fixture cannot reach.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib

FRAME: int = 200
"""Side of the synthetic frame, which is the size the fixture rig is calibrated for: four whole 50-pixel detection cells."""
IMU_PERIOD_NS: int = 1_000_000
"""Synthetic IMU period: 1 kHz, the Index device's own rate."""
FRAME_PERIOD_NS: int = 33_000_000
"""Synthetic frame period: about 30 Hz."""

CameraFactory: TypeAlias = Callable[[int, float], CameraCalib]
"""One camera of the synthetic rig, by rig index and baseline in metres."""
RigFactory: TypeAlias = Callable[[int], _core.Calibration]
"""A calibration for a rig of the given camera count."""
FrontendFactory: TypeAlias = Callable[[int], _core.OpticalFlow]
"""A frontend on a rig of the given camera count."""
PipelineFactory: TypeAlias = Callable[[int], _core.Vio]
"""The whole pipeline on a rig of the given camera count."""
TextureFactory: TypeAlias = Callable[[int, int], UInt8[ndarray, "h w"]]
"""The synthetic scene, shifted by whole pixels in x and y."""


class Row(NamedTuple):
    """One logged row of one entity, as :func:`conftest.read_rows` hands it back."""

    t_ns: int
    """Where on ``video_time`` the row sits, in nanoseconds."""
    values: dict[str, list]
    """The components this row set, by their short name (``Points2D:positions`` and such)."""


Rows: TypeAlias = dict[str, list[Row]]
"""Per entity path, its non-static rows in ``video_time`` order."""
RowsReader: TypeAlias = Callable[[Path], Rows]
"""The :mod:`conftest` fixture that reads a recording back."""


def never(message: str) -> Callable[..., object]:
    """A stand-in for a call that must not happen, which fails the test if it does.

    What a monkeypatched function a case is asserting is *never* reached becomes.
    The message names the rule the call would have broken, so the failure reads
    as the rule rather than as a traceback through the tool.

    Args:
        message: What calling it would mean, in the failure's own words.

    Returns:
        A callable accepting anything and raising :class:`AssertionError`.
    """

    def called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(message)

    return called


def gravity_batch(t_ns: Int64[ndarray, " n_samples"]) -> tuple[Float64[ndarray, "n_samples 3"], Float64[ndarray, "n_samples 3"]]:
    """A still rig: no rotation, and gravity along the rig's z axis.

    Args:
        t_ns: Sample timestamps, which only fix the batch's length here.

    Returns:
        The gyroscope and accelerometer blocks, C-contiguous.
    """
    return np.zeros((len(t_ns), 3), dtype=np.float64), np.tile(np.array([0.0, 0.0, 9.81]), (len(t_ns), 1))
