"""Golden-artifact comparison against the frozen original-pipeline `baseline` run.

Slimmed port of the original ``scripts/validate_golden.py``: the streaming
rewrite intentionally changes operating point (720p crops, forward-only
tracking, sliding-window fitting), so the PRIMARY gate is the final SMPL-X
fit (MPJPE/PVE vs golden ``ma_3d``); 2D landmark deltas vs golden ``ma_2d``
are diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

GOLDEN_FRAME_START: int = 60
"""First source-video frame of the golden run's processed slice."""
GOLDEN_FRAME_END: int = 90
"""One-past-last source-video frame of the golden slice."""


@dataclass(frozen=True, slots=True)
class Landmark2DDiagnostics:
    """Per-camera 2D landmark agreement vs golden ma_2d (report-only)."""

    camera: str
    """Camera name."""
    px_p50: float
    """Median per-coordinate abs diff in golden pixel space (4K)."""
    px_p99: float
    """99th-percentile per-coordinate abs diff in golden pixel space (4K)."""
    vis_p99: float
    """99th-percentile abs diff of visibility probabilities."""
    frames_compared: int
    """Frames with predictions on both sides."""


def landmarks_diagnostics(
    golden_ma2d_dir: Path,
    camera_names: list[str],
    predicted: dict[str, Float32[ndarray, "f j 3"]],
    predicted_vis: dict[str, Float32[ndarray, "f j"]],
    pred_scale_to_golden: float,
) -> list[Landmark2DDiagnostics]:
    """Compare streaming landmark predictions against golden ma_2d NPZs.

    Args:
        golden_ma2d_dir: Directory holding per-camera ``<cam>.npz`` files with
            ``landmarks (F, P, J, 3)`` in golden (4K) pixel coords.
        camera_names: Cameras to compare.
        predicted: Per-camera ``(F, J, 3)`` landmarks in engine pixel coords.
        predicted_vis: Per-camera ``(F, J)`` visibility probabilities.
        pred_scale_to_golden: Multiplier mapping engine pixels to golden pixels
            (e.g. 3.0 for 1280x720 -> 3840x2160).
    """
    results: list[Landmark2DDiagnostics] = []
    for cam in camera_names:
        golden = np.load(golden_ma2d_dir / f"{cam}.npz", allow_pickle=True)
        golden_lm: Float32[ndarray, "f j 2"] = golden["landmarks"][:, 0, :, :2]
        golden_vis: Float32[ndarray, "f j"] = golden["visibilities"][:, 0]
        pred_lm: Float32[ndarray, "f j 2"] = predicted[cam][:, :, :2] * pred_scale_to_golden
        n_frames: int = min(golden_lm.shape[0], pred_lm.shape[0])
        diff: Float32[ndarray, "f j 2"] = np.abs(pred_lm[:n_frames] - golden_lm[:n_frames])
        vis_diff: Float32[ndarray, "f j"] = np.abs(predicted_vis[cam][:n_frames] - golden_vis[:n_frames])
        results.append(
            Landmark2DDiagnostics(
                camera=cam,
                px_p50=float(np.percentile(diff, 50)),
                px_p99=float(np.percentile(diff, 99)),
                vis_p99=float(np.percentile(vis_diff, 99)),
                frames_compared=n_frames,
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class Fit3DComparison:
    """Final-3D agreement vs golden ma_3d — the primary gate inputs."""

    mpjpe_mm: float
    """Mean per-joint position error vs golden pred_joints, millimeters."""
    pve_mm: float
    """Mean per-vertex error vs golden pred_vertices, millimeters."""
    frames_compared: int
    """Frames compared."""


def fit3d_comparison(
    golden_ma3d_npz: Path,
    pred_joints: Float32[ndarray, "f j 3"],
    pred_vertices: Float32[ndarray, "f v 3"] | None,
) -> Fit3DComparison:
    """MPJPE/PVE of streaming SMPL-X outputs vs the golden fit (meters in, mm out)."""
    golden = np.load(golden_ma3d_npz, allow_pickle=True)
    golden_joints: Float32[ndarray, "f j 3"] = golden["pred_joints"]
    n_frames: int = min(golden_joints.shape[0], pred_joints.shape[0])
    mpjpe_m: float = float(np.linalg.norm(pred_joints[:n_frames] - golden_joints[:n_frames], axis=-1).mean())
    pve_m: float = float("nan")
    if pred_vertices is not None:
        golden_vertices: Float32[ndarray, "f v 3"] = golden["pred_vertices"]
        pve_m = float(np.linalg.norm(pred_vertices[:n_frames] - golden_vertices[:n_frames], axis=-1).mean())
    return Fit3DComparison(mpjpe_mm=mpjpe_m * 1000.0, pve_mm=pve_m * 1000.0, frames_compared=n_frames)
