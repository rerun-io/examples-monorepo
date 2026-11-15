"""Render a `Comparison` as markdown or as pyserde JSON.

Nothing here computes anything: every number already sits on the `Comparison`
that `colsfm.benchmark.compare_runs` returned. Only the *presentation* lives
here — the stage labels, the `n/a` for a measurement that does not exist, and the
tall-rather-than-wide pose-delta table the Rerun blueprint's report pane needs. The
JSON dump is `colsfm.benchmark.write_json_report`, beside the model it serialises.

The dependency runs one way, `bench_report` -> `benchmark`, so that
`colsfm.benchmark` stays importable without pulling the renderer in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from colsfm.benchmark import AcceptanceCheck, Comparison, ReconstructionMetrics, RunMetrics, StageComparison, TrajectoryMetrics
from colsfm.runtime import Stage

STAGE_LABELS: Final[dict[Stage, str]] = {
    "extraction": "1 feature extraction + keyframe selection",
    "retrieval": "2 BoW vocabulary + index",
    "association": "3 loop-closure association",
    "pose_graph": "4 pose graph (blob: loop verification + optimisation)",
    "pair_selection": "5 match pair selection",
    "matching": "6 feature matching",
    "mapping": "7 triangulation + bundle adjustment",
    "export": "8 COLMAP + TUM export",
}
"""Human labels for the report, numbered as the plan numbers the stages.

Looked up at render time rather than stored on `StageComparison`: the label is
prose about a fixed stage name, and freezing it into the JSON would let a
report's wording and this table drift apart."""


def number(value: float | None, decimals: int = 2) -> str:
    """Format an optional measurement for a report cell.

    The two absent cases are told apart on purpose: a run that reported nothing
    at all for a stage reads as an em dash, while a quantity that exists but
    could not be measured — a NaN ATE over fewer than three matched samples —
    reads as `n/a`. The audit tools print the same two states in their stdout
    tables, which is why this is public.

    Args:
        value: The number, or None when nothing was reported at all.
        decimals: Digits after the point.

    Returns:
        `—` for None, `n/a` for a non-finite value, else the fixed-point number.
    """
    if value is None:
        return "—"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _trajectory_row(label: str, metrics: TrajectoryMetrics | None) -> str:
    """One markdown row of the trajectory table.

    Args:
        label: Row heading.
        metrics: The alignment, or None when the dataset ships no such reference.

    Returns:
        The markdown row.
    """
    if metrics is None:
        return f"| {label} | — | — | — | — |"
    return (
        f"| {label} | {metrics.num_matched} | {number(metrics.rmse_millimeters)} | "
        f"{number(metrics.max_millimeters)} | {number(metrics.would_be_scale, 5)} |"
    )


def _acceptance_result(check: AcceptanceCheck) -> str:
    """The `Result` cell of one acceptance row.

    Args:
        check: The evaluated bound.

    Returns:
        `PASS`, `FAIL`, or `n/a` when the bound could not be measured.
    """
    if check.passed is None:
        return "n/a"
    return "PASS" if check.passed else "FAIL"


def render_markdown_report(comparison: Comparison) -> str:
    """Render the comparison as a markdown document.

    Args:
        comparison: The result of `compare_runs`.

    Returns:
        The markdown text, ending in a newline.
    """
    run_a: RunMetrics = comparison.run_a
    run_b: RunMetrics = comparison.run_b
    model_a: ReconstructionMetrics = run_a.reconstruction
    model_b: ReconstructionMetrics = run_b.reconstruction
    lines: list[str] = [
        f"# cuSFM benchmark — {comparison.dataset}",
        "",
        f"- **A** `{run_a.name}` — `{run_a.run_dir}`",
        f"- **B** `{run_b.name}` — `{run_b.run_dir}`",
        "",
        "## Reconstruction",
        "",
        "| Metric | A | B |",
        "|---|---|---|",
        (
            f"| registered / total images | {model_a.registered_images} / {run_a.total_images} "
            f"| {model_b.registered_images} / {run_b.total_images} |"
        ),
        f"| 3D points | {model_a.num_points3D} | {model_b.num_points3D} |",
        f"| observations | {model_a.num_observations} | {model_b.num_observations} |",
        (
            f"| mean reprojection error (px, recomputed) | {number(model_a.mean_reprojection_error_px, 3)} "
            f"| {number(model_b.mean_reprojection_error_px, 3)} |"
        ),
        (
            f"| track length mean / median | {number(model_a.track_length_mean)} / {number(model_a.track_length_median)} "
            f"| {number(model_b.track_length_mean)} / {number(model_b.track_length_median)} |"
        ),
        (
            f"| rig rigidity spread (mm) | {number(run_a.rig_rigidity_spread_millimeters, 3)} "
            f"| {number(run_b.rig_rigidity_spread_millimeters, 3)} |"
        ),
        f"| rig frames registered | {run_a.num_rig_frames} | {run_b.num_rig_frames} |",
        f"| point colours present | {'no' if run_a.points_are_black else 'yes'} | {'no' if run_b.points_are_black else 'yes'} |",
        "",
        "## Trajectory (rigid alignment, scale held at 1.0)",
        "",
        "| Comparison | matched | RMSE (mm) | max (mm) | would-be scale |",
        "|---|---|---|---|---|",
        _trajectory_row(f"A `{run_a.name}` vs input", run_a.vs_input),
        _trajectory_row(f"B `{run_b.name}` vs input", run_b.vs_input),
        _trajectory_row(f"A `{run_a.name}` vs ground truth", run_a.vs_ground_truth),
        _trajectory_row(f"B `{run_b.name}` vs ground truth", run_b.vs_ground_truth),
        "",
        # Laid out tall rather than wide: six columns of a single row overflow
        # the report pane in the Rerun blueprint and clip the last heading.
        "| A vs B rig poses (B aligned onto A) | Value |",
        "|---|---|",
        f"| samples matched | {comparison.pose_delta.num_matched} |",
        f"| position RMSE (mm) | {number(comparison.pose_delta.position_rmse_millimeters)} |",
        f"| position max (mm) | {number(comparison.pose_delta.position_max_millimeters)} |",
        f"| rotation RMSE (deg) | {number(comparison.pose_delta.rotation_rmse_degrees, 3)} |",
        f"| rotation max (deg) | {number(comparison.pose_delta.rotation_max_degrees, 3)} |",
        "",
        "## Runtime",
        "",
        "| Stage | A (s) | B (s) | B/A |",
        "|---|---|---|---|",
    ]
    row: StageComparison
    for row in comparison.stages:
        label: str = STAGE_LABELS[row.stage]
        lines.append(f"| {label} | {number(row.seconds_a)} | {number(row.seconds_b)} | {number(row.ratio_b_over_a)} |")
    lines.append(
        f"| **total** | **{number(run_a.total_runtime_seconds)}** | **{number(run_b.total_runtime_seconds)}** "
        f"| **{number(comparison.total_runtime_ratio)}** |"
    )

    if comparison.acceptance:
        lines += [
            "",
            "## Acceptance (plan bounds, B relative to A)",
            "",
            "| Check | Bound | Value | Result |",
            "|---|---|---|---|",
        ]
        for check in comparison.acceptance:
            lines.append(f"| {check.name} | {check.bound} | {number(check.value, 3)} | {_acceptance_result(check)} |")
        measurable: int = sum(1 for check in comparison.acceptance if check.passed is not None)
        passed: int = sum(1 for check in comparison.acceptance if check.passed)
        unmeasured: int = len(comparison.acceptance) - measurable
        tally: str = f"**{passed}/{measurable} bounds met.**"
        lines += ["", tally if unmeasured == 0 else f"{tally} {unmeasured} not measurable."]
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, comparison: Comparison) -> Path:
    """Write the markdown report.

    Args:
        path: Destination file; parent directories are created.
        comparison: The result of `compare_runs`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(comparison))
    return path
