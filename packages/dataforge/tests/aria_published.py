"""The published LaMAria calibration, read only to cross-check the one read out of the VRS (three test modules use it).

A test helper module, not ``conftest.py``: the serde fields name ``dataforge`` types, and conftest must not import
``dataforge`` (pytest-beartype instruments it only if conftest has not imported it).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from serde import field, from_dict, serde

from dataforge.datasets.lamaria_source import PublishedTransform


@serde
@dataclass(frozen=True)
class PublishedResolution:
    """A published camera's image size."""

    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""


@serde
@dataclass(frozen=True)
class PublishedCamera:
    """One ``cam0``/``cam1`` entry of an ``aria_calibrations/<split>/<seq>.json``.

    This is the *published* calibration, kept only to cross-check the one read
    out of the VRS: it covers the two SLAM cameras and no RGB, and its 16-float
    ``params`` repeat the single Aria focal length as ``fx, fy``.
    """

    model: str
    """Always ``RAD_TAN_THIN_PRISM_FISHEYE``, Aria's FISHEYE624 under COLMAP's name."""
    resolution: PublishedResolution
    """The camera's image size."""
    params: list[float]
    """``[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`` — 16 floats."""
    rig_T_cam: PublishedTransform = field(rename="T_b_s")
    """Published as ``T_b_s``; the body frame is imu-right, so this *is* ``rig_T_cam``."""


def read_calibration_json(path: Path) -> dict[str, PublishedCamera]:
    """Read an ``aria_calibrations/<split>/<seq>.json`` file's camera entries.

    The file's third entry, ``imu0``, is skipped: it is the body frame itself, so
    its transform is the identity by definition, and the rest of it is noise
    densities rather than a calibration.

    Args:
        path: The published calibration JSON.

    Returns:
        The camera entries, keyed by their published names (``cam0``, ``cam1``).
    """
    if not path.is_file():
        pytest.skip(f"published Aria calibration is absent: {path}")
    document: dict = json.loads(path.read_text())
    return {name: from_dict(PublishedCamera, entry) for name, entry in document.items() if name.startswith("cam")}
