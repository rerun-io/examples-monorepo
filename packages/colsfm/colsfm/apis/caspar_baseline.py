"""Time CASPAR's global bundle adjustment of a COLMAP model with colsfm's options: the baseline the Mojo port must meet.

Each timed run solves a fresh copy of the model, so every run starts from the same state. The summary pycolmap binds
carries only `num_residuals` and the termination type, so one more, untimed run turns on COLMAP's VLOG(2), which
makes CASPAR print its per-iteration progress (`print_progress`), and reads from it the Levenberg-Marquardt and PCG
iteration counts, the scores and `dt_tot`: CASPAR's own solver time, from its first linearization to its last
iteration, without the problem setup, the upload and the write-back that `adjuster.solve()` also pays.
"""

from __future__ import annotations

import ctypes
import os
import re
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pycolmap

from colsfm.ba_backend import BUNDLE_ADJUSTMENT_BACKEND

_PROGRESS_LINE: Final[re.Pattern[str]] = re.compile(
    r"solver_iter:\s*(\d+)\s+pcg_iter:\s*(\d+)\s+score_current:\s*(\S+)\s+score_best:\s*(\S+).*dt_tot:\s*(\S+)"
)
_SCORE_INIT: Final[re.Pattern[str]] = re.compile(r"score_init:\s*(\S+)")
_EXIT_LINE: Final[re.Pattern[str]] = re.compile(r"Caspar: (\w+)(?: after (\d+) iters| \((\d+)\))")


@dataclass(frozen=True, slots=True)
class CasparTrace:
    """What CASPAR's progress output says about one solve."""

    exit_reason: str
    """CASPAR's `ExitReason` (COLMAP maps two of them to NO_CONVERGENCE)."""
    iterations: int
    """CASPAR's `iteration_count`: Levenberg-Marquardt trials."""
    pcg_iterations: int
    """PCG iterations summed over the printed trials; a trial that ends the solve on the damping limit is not printed."""
    initial_score: float
    """Half the sum of squared residuals at the start."""
    final_score: float
    solver_seconds: float
    """`dt_tot` of the last printed trial: the solver loop alone."""


@dataclass(frozen=True, slots=True)
class CasparBaseline:
    """N timed CASPAR solves of one model from the same start, and one traced solve."""

    model_dir: Path
    seconds: tuple[float, ...]
    """`adjuster.solve()` wall time per run."""
    num_residuals: int
    termination: str
    reprojection_before_px: float
    reprojection_after_px: float
    """After the last timed run (all runs agree to the printed precision)."""
    trace: CasparTrace

    @property
    def best_seconds(self) -> float:
        """Fastest run.

        Returns:
            Seconds.
        """
        return min(self.seconds)

    @property
    def median_seconds(self) -> float:
        """Median run.

        Returns:
            Seconds.
        """
        return statistics.median(self.seconds)


def caspar_options(refine_intrinsics: bool = False, refine_sensor_from_rig: bool = False) -> pycolmap.BundleAdjustmentOptions:
    """colsfm's CASPAR options (`colsfm.mapping.bundle_adjustment_options`) for a standalone solve.

    Args:
        refine_intrinsics: Refine focal length and distortion together.
        refine_sensor_from_rig: Refine the non-reference rig extrinsics.

    Returns:
        Options on the CASPAR backend, reduced camera system on, GPU 0.
    """
    options = pycolmap.BundleAdjustmentOptions()
    options.backend = BUNDLE_ADJUSTMENT_BACKEND["caspar"]
    options.print_summary = False
    options.refine_focal_length = refine_intrinsics
    options.refine_extra_params = refine_intrinsics
    options.refine_principal_point = False
    options.refine_rig_from_world = True
    options.refine_sensor_from_rig = refine_sensor_from_rig
    options.refine_points3D = True
    options.caspar.gpu_index = "0"
    options.caspar.reduced_camera_system = True
    return options


def gauge_config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    """Every registered image, the smallest registered frame constant: colsfm's global solve.

    Args:
        reconstruction: The model.

    Returns:
        The config.
    """
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.reg_image_ids()):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.reg_frame_ids()))
    return config


def _captured_output(run: Callable[[], None]) -> str:
    """Run `run` with the process's stdout and stderr (C's included) sent to a file, and return what was written."""
    libc = ctypes.CDLL(None)
    saved: list[int] = [os.dup(1), os.dup(2)]
    with tempfile.TemporaryFile(mode="w+b") as sink:
        libc.fflush(None)
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            run()
        finally:
            libc.fflush(None)
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)
        sink.seek(0)
        return sink.read().decode(errors="replace")


def parse_trace(text: str) -> CasparTrace:
    """Read CASPAR's progress lines and COLMAP's exit line.

    Args:
        text: The solve's stdout and stderr with VLOG(2).

    Returns:
        The trace.

    Raises:
        ValueError: When the text holds no progress line or no exit line.
    """
    rows: list[tuple[str, ...]] = _PROGRESS_LINE.findall(text)
    exit_match: re.Match[str] | None = _EXIT_LINE.search(text)
    init_match: re.Match[str] | None = _SCORE_INIT.search(text)
    if not rows or exit_match is None or init_match is None:
        raise ValueError(f"no CASPAR progress or exit line in the solve's output:\n{text[-2000:]}")
    return CasparTrace(
        exit_reason=exit_match.group(1),
        iterations=int(exit_match.group(2) or exit_match.group(3)),
        pcg_iterations=sum(int(row[1]) + 1 for row in rows),
        initial_score=float(init_match.group(1)),
        final_score=float(rows[-1][3]),
        solver_seconds=float(rows[-1][4]),
    )


def trace_caspar(reconstruction: pycolmap.Reconstruction, options: pycolmap.BundleAdjustmentOptions) -> CasparTrace:
    """Solve a copy once with COLMAP's VLOG(2) on and read CASPAR's progress.

    Args:
        reconstruction: The model; it is not changed.
        options: The solve's options.

    Returns:
        The trace.
    """
    copy = pycolmap.Reconstruction(reconstruction)
    config: pycolmap.BundleAdjustmentConfig = gauge_config(copy)
    saved: tuple[int, bool] = (pycolmap.logging.verbose_level, pycolmap.logging.logtostderr)
    pycolmap.logging.verbose_level = 2
    pycolmap.logging.logtostderr = True
    try:
        text: str = _captured_output(lambda: pycolmap.create_default_bundle_adjuster(options, config, copy).solve())
    finally:
        pycolmap.logging.verbose_level, pycolmap.logging.logtostderr = saved
    return parse_trace(text)


def run_baseline(model_dir: Path, runs: int, options: pycolmap.BundleAdjustmentOptions) -> CasparBaseline:
    """Time `runs` CASPAR solves of fresh copies of a model, then trace one more.

    Args:
        model_dir: COLMAP model directory.
        runs: Timed solves.
        options: The solve's options (`caspar_options`).

    Returns:
        The baseline.
    """
    original = pycolmap.Reconstruction(model_dir)
    original.update_point_3d_errors()
    before: float = original.compute_mean_reprojection_error()
    seconds: list[float] = []
    summary: pycolmap.BundleAdjustmentSummary | None = None
    after: float = float("nan")
    for _ in range(runs):
        copy = pycolmap.Reconstruction(original)
        adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(options, gauge_config(copy), copy)
        started: float = time.perf_counter()
        summary = adjuster.solve()
        seconds.append(time.perf_counter() - started)
        copy.update_point_3d_errors()
        after = copy.compute_mean_reprojection_error()
    assert summary is not None
    return CasparBaseline(
        model_dir=model_dir,
        seconds=tuple(seconds),
        num_residuals=summary.num_residuals,
        termination=summary.termination_type.name,
        reprojection_before_px=before,
        reprojection_after_px=after,
        trace=trace_caspar(original, options),
    )


@dataclass(frozen=True)
class Config:
    """Which model, how often, which refinements."""

    model_dir: Path
    """COLMAP model directory."""
    runs: int = 5
    refine_intrinsics: bool = False
    refine_sensor_from_rig: bool = False
    gpu_label: str = ""
    """Names the GPU in the printed table row."""


def main(config: Config) -> None:
    """Print one markdown table row and a readable summary.

    Args:
        config: Model, run count and refinements.
    """
    baseline: CasparBaseline = run_baseline(config.model_dir, config.runs, caspar_options(config.refine_intrinsics, config.refine_sensor_from_rig))
    trace: CasparTrace = baseline.trace
    print(f"runs (s): {', '.join(f'{s:.3f}' for s in baseline.seconds)}")
    print("| model | GPU | best s | median s | solver-only s | px before | px after | LM iters | PCG iters | residuals | termination (CASPAR exit) |")
    print(
        f"| {config.model_dir.name}{' +intrinsics' if config.refine_intrinsics else ''}{' +rig' if config.refine_sensor_from_rig else ''} | {config.gpu_label} | {baseline.best_seconds:.3f} | {baseline.median_seconds:.3f} | "
        f"{trace.solver_seconds:.3f} | {baseline.reprojection_before_px:.4f} | {baseline.reprojection_after_px:.4f} | "
        f"{trace.iterations} | {trace.pcg_iterations} | {baseline.num_residuals} | {baseline.termination} ({trace.exit_reason}) |"
    )
    print(f"score {trace.initial_score:.6e} -> {trace.final_score:.6e}")
