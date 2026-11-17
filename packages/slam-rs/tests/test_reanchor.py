"""The final two-sided re-anchoring (``_core.reanchor``, PR f1): its boundary, and RootSLAM's own merge on its run CSVs.

The golden reads the reproduction runs on the NAS: each run's raw causal output (``rt_tracking.<seq>.csv``), its map's
optimized keyframes (``map/kf_trajectory.csv``) and what ``merge_frames_keyframes.py`` made of the two
(``eval/lc.csv``, the "RS +LC" trajectory). Ours must give the same poses within 1e-6 m and 1e-6 rad.
"""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs import _core

ROOTSLAM_RUNS: Path = Path("/mnt/nas/ssd-8tb/scratch/rootslam/runs")
"""RootSLAM reproduction runs, one directory per sequence."""
TOLERANCE: float = 1e-6
"""Metres and radians."""


def _run_sequences() -> list[str]:
    """Sequences whose run has the three CSVs the golden compares; empty when the NAS asset is absent."""
    if not ROOTSLAM_RUNS.is_dir():
        return []
    return sorted(
        run.name
        for run in ROOTSLAM_RUNS.iterdir()
        if (run / f"rt_tracking.{run.name}.csv").is_file() and (run / "map" / "kf_trajectory.csv").is_file() and (run / "eval" / "lc.csv").is_file()
    )


def _read_euroc(path: Path) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 7"]]:
    """An EuRoC pose CSV as nanosecond stamps (read as integers: EuRoC's 1.4e18 ns are not exact in float64) and ``[tx, ty, tz, qx, qy, qz, qw]``."""
    t_ns: Int64[ndarray, " n"] = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    columns: Float64[ndarray, "n 7"] = np.loadtxt(path, delimiter=",", comments="#", usecols=range(1, 8), dtype=np.float64, ndmin=2)
    return t_ns, columns[:, [0, 1, 2, 4, 5, 6, 3]]


def _rotation_angle(a: Float64[ndarray, "n 4"], b: Float64[ndarray, "n 4"]) -> Float64[ndarray, " n"]:
    """Angle between unit quaternions ``xyzw`` row by row, sign-agnostic."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0))


def _pose(x: float, yaw: float) -> Float64[ndarray, " 7"]:
    return np.array([x, 0.1 * x, 0.0, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def test_keyframes_take_their_pose_and_frames_their_keyframes_correction() -> None:
    frame_t_ns: Int64[ndarray, " n"] = np.arange(0, 50, 10, dtype=np.int64)
    frames: Float64[ndarray, "n 7"] = np.stack([_pose(float(i), 0.1 * i) for i in range(5)])
    keyframe_t_ns: Int64[ndarray, " k"] = np.array([10, 30], dtype=np.int64)
    shifted: Float64[ndarray, "k 7"] = frames[[1, 3]].copy()
    shifted[:, 2] += 1.0
    anchored: Float64[ndarray, "n 7"] = _core.reanchor(frame_t_ns, frames, keyframe_t_ns, shifted)
    assert anchored.shape == (5, 7)
    assert anchored.dtype == np.float64
    # Both keyframes moved up by 1 m: every frame moves up by 1 m and keeps its own x, y and rotation.
    expected: Float64[ndarray, "n 7"] = frames.copy()
    expected[:, 2] += 1.0
    np.testing.assert_allclose(anchored[:, 0:3], expected[:, 0:3], atol=1e-12)
    assert _rotation_angle(anchored[:, 3:], expected[:, 3:]).max() < 1e-12


def test_bad_inputs_are_refused_with_a_reason() -> None:
    frame_t_ns: Int64[ndarray, " n"] = np.arange(0, 30, 10, dtype=np.int64)
    frames: Float64[ndarray, "n 7"] = np.stack([_pose(float(i), 0.0) for i in range(3)])
    with pytest.raises(ValueError, match="not one of the frames"):
        _core.reanchor(frame_t_ns, frames, np.array([15], dtype=np.int64), frames[:1])
    with pytest.raises(ValueError, match="increase strictly"):
        _core.reanchor(frame_t_ns[::-1].copy(), frames, np.array([], dtype=np.int64), np.zeros((0, 7)))
    with pytest.raises(ValueError, match="x 7 poses"):
        _core.reanchor(frame_t_ns, frames[:, :6].copy(), np.array([], dtype=np.int64), np.zeros((0, 7)))


@pytest.mark.golden
@pytest.mark.parametrize("sequence", _run_sequences() or [pytest.param("absent", marks=pytest.mark.skip(reason=f"NAS asset {ROOTSLAM_RUNS} is absent"))])
def test_reanchor_matches_rootslam_merge_frames_keyframes(sequence: str) -> None:
    run: Path = ROOTSLAM_RUNS / sequence
    frame_t_ns, frames = _read_euroc(run / f"rt_tracking.{sequence}.csv")
    keyframe_t_ns, keyframes = _read_euroc(run / "map" / "kf_trajectory.csv")
    merged_t_ns, merged = _read_euroc(run / "eval" / "lc.csv")
    np.testing.assert_array_equal(merged_t_ns, frame_t_ns)
    ours: Float64[ndarray, "n 7"] = _core.reanchor(frame_t_ns, frames, keyframe_t_ns, keyframes)
    position_error: float = float(np.linalg.norm(ours[:, 0:3] - merged[:, 0:3], axis=1).max())
    rotation_error: float = float(_rotation_angle(ours[:, 3:], merged[:, 3:]).max())
    assert position_error <= TOLERANCE, f"{sequence}: {position_error:.3g} m"
    assert rotation_error <= TOLERANCE, f"{sequence}: {rotation_error:.3g} rad"
