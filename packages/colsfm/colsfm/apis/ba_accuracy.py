"""CASPAR against the Mojo solver from the identical start: the accuracy matrix of the port (the 2 % bar).

One row = one model and one option set. The model is loaded once; one copy is solved by CASPAR (pycolmap), another
by the Mojo solver through `colsfm.ba_mojo`, and `colsfm.ba_compare` measures the two solutions against each other.
Timed runs alternate between the solvers, so a GPU shared with other jobs slows both alike. Each CASPAR run is traced
(COLMAP's VLOG(2)), which gives its solver-only seconds and its LM and PCG counts; the Mojo problem is bridged once
and solved `runs` times, because the bridge's extraction is not the solver. The solution compared is the last run of
each: after convergence float32 noise decides the tail of an LM run on both solvers, so the counts of every run are
kept next to the times.

A converged model (clip 007's final reconstruction) gives the solver nothing to do; `perturb` knocks the free poses
and points off it with a seed, so the two solvers have the same real work.
"""

from __future__ import annotations

import dataclasses
import datetime
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from caspar_mojo.problem import BaOptions, BaResult
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from colsfm.apis.caspar_baseline import CasparTrace, caspar_options, gauge_config, traced_solve
from colsfm.ba_compare import SolveComparison, compare_solutions, problem_cost
from colsfm.ba_mojo import BridgedProblem, apply_result, mojo_options, problem_from_reconstruction

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AccuracyRow:
    """One model and option set solved by both solvers from the same start.

    The CASPAR fields and the differences are None on a Mojo-only row (FULL_OPENCV with free intrinsics, which CASPAR
    cannot refine)."""

    row: str
    """Model directory name plus the refinements and the perturbation."""
    gpu: str
    commit: str
    """`git rev-parse HEAD` of the checkout that ran (it carries caspar-mojo/port merged)."""
    date: str
    observations: int
    reprojection_before_px: float
    reprojection_caspar_px: float | None
    reprojection_mojo_px: float
    reprojection_relative_difference: float | None
    """(Mojo - CASPAR) / CASPAR: the 2 % bar."""
    cost_before: float
    cost_caspar: float | None
    cost_mojo: float
    cost_relative_difference: float | None
    rotation_difference_max_deg: float | None
    rotation_difference_median_deg: float | None
    center_difference_max: float | None
    """Relative to the scene extent (diagonal of the camera centres' bounding box)."""
    center_difference_median: float | None
    point_difference_median: float | None
    """Relative to the scene extent."""
    point_difference_max: float | None
    registered_images_caspar: int | None
    registered_images_mojo: int
    caspar_lm_trials: list[int]
    """Per timed run, in run order; the last run is the solution compared."""
    caspar_pcg_iterations: list[int]
    caspar_exit: str | None
    caspar_solver_seconds: list[float]
    """CASPAR's `dt_tot`: the solver loop alone."""
    caspar_solve_call_seconds: list[float]
    """`adjuster.solve()`, setup and upload included."""
    mojo_lm_trials: list[int]
    mojo_pcg_iterations: list[int]
    mojo_exit: str
    mojo_solve_seconds: list[float]
    """`BaResult.solve_seconds`: the solver loop on the GPU."""
    mojo_call_seconds: list[float]
    """`BaResult.total_seconds`: the `caspar_mojo.solver.solve` call, host arrays in and out."""
    mojo_solver_rel_decrease_min: float = 1.0
    """The Mojo side's accept factor; 1 - 1e-6 stands in for the lead's 1e-6 accept floor until the solver has it
    (NOTES-mojo.md, ruling 22:40), 1.0 is "no floor". CASPAR always runs with its own 1.0."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AccuracyTable:
    """The rows of `accuracy.json`."""

    rows: list[AccuracyRow]


def perturb(
    reconstruction: pycolmap.Reconstruction, config: pycolmap.BundleAdjustmentConfig, seed: int, translation: float, rotation_deg: float, point: float
) -> None:
    """Knock every non-constant frame pose and every point off its value, by fixed magnitudes in random directions.

    Args:
        reconstruction: Changed in place.
        config: The solve's config; its constant frames stay where they are.
        seed: Seed of the directions.
        translation: Distance each pose's translation moves, in model units.
        rotation_deg: Angle each pose turns.
        point: Distance each point moves, in model units.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    for frame_id in sorted(reconstruction.reg_frame_ids()):
        if config.has_constant_rig_from_world_pose(frame_id):
            continue
        frame: pycolmap.Frame = reconstruction.frame(frame_id)
        axis: Float64[ndarray, " 3"] = rng.normal(size=3)
        direction: Float64[ndarray, " 3"] = rng.normal(size=3)
        turn = pycolmap.Rotation3d(np.deg2rad(rotation_deg) * axis / np.linalg.norm(axis))
        pose: pycolmap.Rigid3d | None = frame.rig_from_world
        if pose is None:
            raise ValueError(f"registered frame {frame_id} has no pose")
        frame.rig_from_world = pycolmap.Rigid3d(
            turn * pose.rotation, np.asarray(pose.translation) + translation * direction / np.linalg.norm(direction)
        )
    for point3D_id in sorted(reconstruction.point3D_ids()):
        direction = rng.normal(size=3)
        reconstruction.point3D(point3D_id).xyz = np.asarray(reconstruction.point3D(point3D_id).xyz) + point * direction / np.linalg.norm(direction)


def _row_name(config: Config) -> str:
    return (
        config.model_dir.name
        + (" +intrinsics" if config.refine_intrinsics else "")
        + (" +rig" if config.refine_sensor_from_rig else "")
        + (f" perturbed(seed {config.perturb_seed})" if config.perturb_seed is not None else "")
        + (" [floor]" if config.mojo_solver_rel_decrease_min != 1.0 else "")
        + (" [Mojo only]" if config.mojo_only else "")
    )


@dataclass(frozen=True)
class Config:
    """One row of the matrix: a model, its refinements, an optional perturbation, where the row goes."""

    model_dir: Path
    """COLMAP model directory."""
    gpu: str
    """Names the GPU in the row."""
    output: Path
    """JSON table (`AccuracyTable`) the row is added to; a row with the same name and GPU is replaced."""
    runs: int = 5
    refine_intrinsics: bool = False
    refine_sensor_from_rig: bool = False
    perturb_seed: int | None = None
    """Perturb the start (`perturb`) with this seed; None solves the model as stored."""
    perturb_translation: float = 0.02
    perturb_rotation_deg: float = 0.1
    perturb_point: float = 0.02
    mojo_solver_rel_decrease_min: float = 1.0
    """Accept factor of the Mojo side (`AccuracyRow.mojo_solver_rel_decrease_min`)."""
    mojo_only: bool = False
    """Solve with Mojo alone: CASPAR cannot refine FULL_OPENCV intrinsics."""


def measure(config: Config) -> AccuracyRow:
    """Solve one model with both solvers from the same start and compare.

    Args:
        config: The row.

    Returns:
        The row.
    """
    from caspar_mojo.solver import solve

    start = pycolmap.Reconstruction(config.model_dir)
    ba_config: pycolmap.BundleAdjustmentConfig = gauge_config(start)
    if config.perturb_seed is not None:
        perturb(start, ba_config, config.perturb_seed, config.perturb_translation, config.perturb_rotation_deg, config.perturb_point)
    options: pycolmap.BundleAdjustmentOptions = caspar_options(config.refine_intrinsics, config.refine_sensor_from_rig)
    start.update_point_3d_errors()
    reprojection_before: float = start.compute_mean_reprojection_error()

    mojo = pycolmap.Reconstruction(start)
    bridged: BridgedProblem = problem_from_reconstruction(mojo, ba_config, options)
    cost_before: float = problem_cost(start, bridged)
    solver_options: BaOptions = dataclasses.replace(mojo_options(options), solver_rel_decrease_min=config.mojo_solver_rel_decrease_min)
    traces: list[CasparTrace] = []
    call_seconds: list[float] = []
    mojo_results: list[BaResult] = []
    caspar = start
    for _ in range(config.runs):
        if not config.mojo_only:
            caspar = pycolmap.Reconstruction(start)
            trace, seconds = traced_solve(caspar, gauge_config(caspar), options)
            traces.append(trace)
            call_seconds.append(seconds)
        mojo_results.append(solve(bridged.problem, solver_options))
    apply_result(mojo, bridged, mojo_results[-1])
    comparison: SolveComparison | None = None if config.mojo_only else compare_solutions(caspar, mojo, ba_config, options)
    mojo.update_point_3d_errors()
    head: str = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_DIR, capture_output=True, text=True, check=True).stdout.strip()
    return AccuracyRow(
        row=_row_name(config),
        gpu=config.gpu,
        commit=head,
        date=datetime.datetime.now().isoformat(timespec="minutes"),
        observations=len(bridged.problem.pixels),
        reprojection_before_px=reprojection_before,
        reprojection_caspar_px=None if comparison is None else comparison.reprojection_a_px,
        reprojection_mojo_px=mojo.compute_mean_reprojection_error(),
        reprojection_relative_difference=None if comparison is None else comparison.reprojection_relative_difference,
        cost_before=cost_before,
        cost_caspar=None if comparison is None else comparison.cost_a,
        cost_mojo=problem_cost(mojo, problem_from_reconstruction(mojo, ba_config, options)),
        cost_relative_difference=None if comparison is None else comparison.cost_relative_difference,
        rotation_difference_max_deg=None if comparison is None else comparison.rotation_difference_max_deg,
        rotation_difference_median_deg=None if comparison is None else comparison.rotation_difference_median_deg,
        center_difference_max=None if comparison is None else comparison.center_difference_max,
        center_difference_median=None if comparison is None else comparison.center_difference_median,
        point_difference_median=None if comparison is None else comparison.point_difference_median,
        point_difference_max=None if comparison is None else comparison.point_difference_max,
        registered_images_caspar=None if comparison is None else comparison.registered_images_a,
        registered_images_mojo=mojo.num_reg_images(),
        caspar_lm_trials=[t.iterations for t in traces],
        caspar_pcg_iterations=[t.pcg_iterations for t in traces],
        caspar_exit=traces[-1].exit_reason if traces else None,
        caspar_solver_seconds=[t.solver_seconds for t in traces],
        caspar_solve_call_seconds=call_seconds,
        mojo_lm_trials=[r.iterations for r in mojo_results],
        mojo_pcg_iterations=[r.pcg_iterations for r in mojo_results],
        mojo_exit=mojo_results[-1].exit_reason,
        mojo_solve_seconds=[r.solve_seconds for r in mojo_results],
        mojo_call_seconds=[r.total_seconds for r in mojo_results],
        mojo_solver_rel_decrease_min=config.mojo_solver_rel_decrease_min,
    )


def read_table(path: Path) -> AccuracyTable:
    """The table at `path`, empty when there is no file.

    Args:
        path: JSON written by `write_table`.

    Returns:
        The table.
    """
    return from_json(AccuracyTable, path.read_text()) if path.exists() else AccuracyTable(rows=[])


def merged(table: AccuracyTable, rows: list[AccuracyRow]) -> AccuracyTable:
    """The table with `rows` added, each replacing a row of the same name and GPU.

    Args:
        table: Existing rows.
        rows: New rows.

    Returns:
        The merged table, sorted by GPU then row.
    """
    keyed: dict[tuple[str, str], AccuracyRow] = {(r.gpu, r.row): r for r in table.rows}
    keyed.update({(r.gpu, r.row): r for r in rows})
    return AccuracyTable(rows=[keyed[key] for key in sorted(keyed)])


def _times(seconds: list[float]) -> str:
    return f"{min(seconds):.3f} / {statistics.median(seconds):.3f}" if seconds else "n/a"


def _number(value: float | int | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _counts(lm: list[int], pcg: list[int]) -> str:
    return ", ".join(f"{a}/{b}" for a, b in zip(lm, pcg, strict=True))


def markdown(table: AccuracyTable) -> str:
    """The readable table: accuracy first, then speed.

    Args:
        table: The rows.

    Returns:
        Markdown.
    """
    lines: list[str] = [
        "# CASPAR vs Mojo from the identical start",
        "",
        "rel = (Mojo - CASPAR) / CASPAR. Differences of centres and points are over the scene extent. "
        "Times: best / median of the runs, seconds. LM/PCG per run in run order (the last run is the compared solution). "
        "[floor]: the Mojo side ran with solver_rel_decrease_min = 1 - 1e-6 (the lead's accept floor, NOTES-mojo.md 22:40); "
        "rows without it ran with no floor. [Mojo only]: CASPAR cannot refine these intrinsics; px and cost are Mojo's alone.",
        "",
        "| row | GPU | obs | px before | px CASPAR | px Mojo | px rel | cost rel | rot max / med deg | centre max / med | point med | reg C / M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        rel: float | None = r.reprojection_relative_difference
        flag: str = " **>2 %**" if rel is not None and abs(rel) > 0.02 else ""
        lines.append(
            f"| {r.row} | {r.gpu} | {r.observations:,} | {r.reprojection_before_px:.4f} | {_number(r.reprojection_caspar_px, '.6f')} | {r.reprojection_mojo_px:.6f} | "
            f"{_number(rel, '+.2e')}{flag} | {_number(r.cost_relative_difference, '+.2e')} | "
            f"{_number(r.rotation_difference_max_deg, '.2e')} / {_number(r.rotation_difference_median_deg, '.2e')} | "
            f"{_number(r.center_difference_max, '.2e')} / {_number(r.center_difference_median, '.2e')} | {_number(r.point_difference_median, '.2e')} | "
            f"{_number(r.registered_images_caspar, 'd')} / {r.registered_images_mojo} |"
        )
    lines += [
        "",
        "| row | GPU | CASPAR solver-only s | CASPAR solve() s | CASPAR LM/PCG (exit) | Mojo solve_seconds | Mojo call s | Mojo LM/PCG (exit) | commit | date |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        lines.append(
            f"| {r.row} | {r.gpu} | {_times(r.caspar_solver_seconds)} | {_times(r.caspar_solve_call_seconds)} | {_counts(r.caspar_lm_trials, r.caspar_pcg_iterations)} ({r.caspar_exit}) | "
            f"{_times(r.mojo_solve_seconds)} | {_times(r.mojo_call_seconds)} | {_counts(r.mojo_lm_trials, r.mojo_pcg_iterations)} ({r.mojo_exit}) | {r.commit} | {r.date} |"
        )
    return "\n".join(lines) + "\n"


def write_table(path: Path, table: AccuracyTable) -> None:
    """Write the JSON table and its markdown beside it (same stem, `.md`).

    Args:
        path: The `.json` path.
        table: The rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(table))
    path.with_suffix(".md").write_text(markdown(table))


def main(config: Config) -> None:
    """Measure one row, add it to the table at `config.output`, print it.

    Args:
        config: The row.
    """
    row: AccuracyRow = measure(config)
    write_table(config.output, merged(read_table(config.output), [row]))
    print(markdown(AccuracyTable(rows=[row])))


@dataclass(frozen=True)
class MergeConfig:
    """Fold tables measured on other hosts into one."""

    output: Path
    sources: tuple[Path, ...]


def merge_main(config: MergeConfig) -> None:
    """Add every source table's rows to the output table.

    Args:
        config: Output and sources.
    """
    table: AccuracyTable = read_table(config.output)
    for source in config.sources:
        table = merged(table, read_table(source).rows)
    write_table(config.output, table)
    print(markdown(table))
