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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

import pycolmap

from colsfm.mapping import default_gauge_frame_id, global_ba_config, refinement_options, select_backend

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
    """The mapper's CASPAR options (`colsfm.mapping.bundle_adjustment_options` without the Ceres settings) on GPU 0.

    Args:
        refine_intrinsics: Refine focal length and distortion together.
        refine_sensor_from_rig: Refine the non-reference rig extrinsics.

    Returns:
        Options on the CASPAR backend, reduced camera system on, GPU 0.
    """
    options: pycolmap.BundleAdjustmentOptions = refinement_options(refine_intrinsics, refine_sensor_from_rig)
    select_backend(options, "caspar", use_gpu=True, caspar_options={})
    return options


def gauge_config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    """The mapper's global solve: every registered image, its gauge frame constant (`colsfm.mapping.default_gauge_frame_id`).

    Args:
        reconstruction: The model.

    Returns:
        The config.
    """
    return global_ba_config(reconstruction, default_gauge_frame_id(reconstruction))


@contextmanager
def _native_output_into(sink: IO[bytes]) -> Iterator[None]:
    """COLMAP's VLOG(2) on and the process's stdout and stderr (C's included) sent to `sink` for the block.

    Each restoration is registered before the change it undoes, so a failure anywhere in the block or in the setup
    leaves logging and both descriptors as they were.
    """
    libc = ctypes.CDLL(None)
    with ExitStack() as restore:
        saved_logging: tuple[int, bool] = (pycolmap.logging.verbose_level, pycolmap.logging.logtostderr)
        restore.callback(setattr, pycolmap.logging, "logtostderr", saved_logging[1])
        restore.callback(setattr, pycolmap.logging, "verbose_level", saved_logging[0])
        pycolmap.logging.verbose_level = 2
        pycolmap.logging.logtostderr = True
        libc.fflush(None)
        for fd in (1, 2):
            saved: int = os.dup(fd)
            restore.callback(os.close, saved)
            restore.callback(os.dup2, saved, fd)
            os.dup2(sink.fileno(), fd)
        # Last registered, so it runs first: what the solver buffered lands in the sink, not on the restored terminal.
        restore.callback(libc.fflush, None)
        yield


def _timed_solve(adjuster: pycolmap.BundleAdjuster) -> tuple[pycolmap.BundleAdjustmentSummary, float]:
    """`adjuster.solve()` and its wall time: the one definition of a timed CASPAR solve in the benchmarks."""
    started: float = time.perf_counter()
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    return summary, time.perf_counter() - started


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


def traced_solve(
    reconstruction: pycolmap.Reconstruction, config: pycolmap.BundleAdjustmentConfig, options: pycolmap.BundleAdjustmentOptions
) -> tuple[CasparTrace, float]:
    """Solve in place with COLMAP's VLOG(2) on and read CASPAR's progress.

    Args:
        reconstruction: The model, adjusted in place.
        config: The solve's config.
        options: The solve's options.

    Returns:
        The trace and the wall time of `adjuster.solve()`.
    """
    with tempfile.TemporaryFile(mode="w+b") as sink:
        with _native_output_into(sink):
            adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
            _, seconds = _timed_solve(adjuster)
        sink.seek(0)
        text: str = sink.read().decode(errors="replace")
    return parse_trace(text), seconds


def trace_caspar(reconstruction: pycolmap.Reconstruction, options: pycolmap.BundleAdjustmentOptions) -> CasparTrace:
    """Solve a copy once with COLMAP's VLOG(2) on and read CASPAR's progress.

    Args:
        reconstruction: The model; it is not changed.
        options: The solve's options.

    Returns:
        The trace.
    """
    copy = pycolmap.Reconstruction(reconstruction)
    return traced_solve(copy, gauge_config(copy), options)[0]


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
    summaries: list[pycolmap.BundleAdjustmentSummary] = []
    copy: pycolmap.Reconstruction = original
    for _ in range(runs):
        copy = pycolmap.Reconstruction(original)
        summary, elapsed = _timed_solve(pycolmap.create_default_bundle_adjuster(options, gauge_config(copy), copy))
        summaries.append(summary)
        seconds.append(elapsed)
    copy.update_point_3d_errors()
    return CasparBaseline(
        model_dir=model_dir,
        seconds=tuple(seconds),
        num_residuals=summaries[-1].num_residuals,
        termination=summaries[-1].termination_type.name,
        reprojection_before_px=before,
        reprojection_after_px=copy.compute_mean_reprojection_error(),
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

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError(f"runs must be at least 1, got {self.runs}")


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
