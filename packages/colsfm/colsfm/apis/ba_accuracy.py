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

import datetime
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from caspar_mojo.problem import BaResult
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from colsfm.apis.caspar_baseline import CasparTrace, caspar_options, gauge_config, traced_solve
from colsfm.ba_compare import SolveComparison, compare_solutions, problem_cost
from colsfm.ba_mojo import BridgedProblem, apply_result, mojo_options, problem_from_reconstruction

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]

PARITY_BAR: float = 0.02
"""PLAN §5: final cost and reprojection error within 2 % of CASPAR's, with the same registered images."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AccuracyRow:
    """One model and option set solved by both solvers from the same start."""

    row: str
    """Display name: model directory name plus the refinements and the perturbation (`_row_name`)."""
    gpu: str
    commit: str
    """`git rev-parse HEAD` of the checkout that ran (it carries caspar-mojo/port merged)."""
    date: str
    observations: int
    reprojection_before_px: float
    reprojection_caspar_px: float
    reprojection_mojo_px: float
    reprojection_relative_difference: float | None
    """(Mojo - CASPAR) / CASPAR: the 2 % bar; None when CASPAR's is zero and Mojo's is not."""
    cost_before: float
    cost_caspar: float
    cost_mojo: float
    cost_relative_difference: float | None
    rotation_difference_max_deg: float
    rotation_difference_median_deg: float
    center_difference_max: float
    """Relative to the scene extent (diagonal of the camera centres' bounding box)."""
    center_difference_median: float
    point_difference_median: float
    """Relative to the scene extent."""
    point_difference_max: float
    registered_images_caspar: int
    registered_images_mojo: int
    caspar_lm_trials: list[int]
    """Per timed run, in run order; the last run is the solution compared."""
    caspar_pcg_iterations: list[int]
    caspar_exit: str
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
    model_dir: str = ""
    """Resolved model directory; empty in rows written before it was recorded, which `case_key` keys by `row`."""
    refine_intrinsics: bool = False
    refine_sensor_from_rig: bool = False
    perturb_seed: int | None = None
    perturb_translation: float | None = None
    """None when the start was not perturbed."""
    perturb_rotation_deg: float | None = None
    perturb_point: float | None = None

    def case_key(self) -> tuple[object, ...]:
        """What makes two rows the same experiment: the GPU and every resolved input; a later row of the same case replaces it.

        Returns:
            The key; rows from before the inputs were recorded are keyed by their display name.
        """
        if not self.model_dir:
            return (self.gpu, self.row)
        return (
            self.gpu,
            self.model_dir,
            self.refine_intrinsics,
            self.refine_sensor_from_rig,
            self.perturb_seed,
            self.perturb_translation,
            self.perturb_rotation_deg,
            self.perturb_point,
        )

    def within_parity(self) -> bool:
        """Whether the row meets the 2 % bar on cost and reprojection and registers the same images (`PARITY_BAR`).

        Returns:
            False when a relative difference is undefined (`colsfm.ba_compare.relative_difference`).
        """
        relatives: tuple[float | None, float | None] = (self.reprojection_relative_difference, self.cost_relative_difference)
        return all(r is not None and abs(r) <= PARITY_BAR for r in relatives) and self.registered_images_caspar == self.registered_images_mojo


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
    perturbation: str = ""
    if config.perturb_seed is not None:
        magnitudes: tuple[float, float, float] = (config.perturb_translation, config.perturb_rotation_deg, config.perturb_point)
        # The default magnitudes keep the short name of the rows measured before the magnitudes were recorded.
        shown: str = "" if magnitudes == DEFAULT_PERTURBATION else f"; {magnitudes[0]:g} m, {magnitudes[1]:g} deg, {magnitudes[2]:g} m"
        perturbation = f" perturbed(seed {config.perturb_seed}{shown})"
    return (
        config.model_dir.name
        + (" +intrinsics" if config.refine_intrinsics else "")
        + (" +rig" if config.refine_sensor_from_rig else "")
        + perturbation
    )


DEFAULT_PERTURBATION: tuple[float, float, float] = (0.02, 0.1, 0.02)
"""`Config`'s translation, rotation (deg) and point magnitudes."""


@dataclass(frozen=True)
class Config:
    """One row of the matrix: a model, its refinements, an optional perturbation, where the row goes."""

    model_dir: Path
    """COLMAP model directory."""
    gpu: str
    """Names the GPU in the row."""
    output: Path
    """JSON table (`AccuracyTable`) the row is added to; a row of the same case (`AccuracyRow.case_key`) is replaced."""
    runs: int = 5
    refine_intrinsics: bool = False
    refine_sensor_from_rig: bool = False
    perturb_seed: int | None = None
    """Perturb the start (`perturb`) with this seed; None solves the model as stored."""
    perturb_translation: float = DEFAULT_PERTURBATION[0]
    perturb_rotation_deg: float = DEFAULT_PERTURBATION[1]
    perturb_point: float = DEFAULT_PERTURBATION[2]

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError(f"runs must be at least 1, got {self.runs}")


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
    traces: list[CasparTrace] = []
    call_seconds: list[float] = []
    mojo_lm: list[int] = []
    mojo_pcg: list[int] = []
    mojo_solve: list[float] = []
    mojo_call: list[float] = []
    caspar = start
    result: BaResult | None = None
    for _ in range(config.runs):
        caspar = pycolmap.Reconstruction(start)
        trace, seconds = traced_solve(caspar, gauge_config(caspar), options)
        traces.append(trace)
        call_seconds.append(seconds)
        result = solve(bridged.problem, mojo_options(options))
        mojo_lm.append(result.iterations)
        mojo_pcg.append(result.pcg_iterations)
        mojo_solve.append(result.solve_seconds)
        mojo_call.append(result.total_seconds)
    assert result is not None, "Config refuses runs < 1"
    apply_result(mojo, bridged, result)
    comparison: SolveComparison = compare_solutions(caspar, mojo, bridged)
    head: str = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_DIR, capture_output=True, text=True, check=True).stdout.strip()
    perturbed: bool = config.perturb_seed is not None
    return AccuracyRow(
        row=_row_name(config),
        gpu=config.gpu,
        commit=head,
        date=datetime.datetime.now().isoformat(timespec="minutes"),
        observations=len(bridged.problem.pixels),
        reprojection_before_px=reprojection_before,
        reprojection_caspar_px=comparison.reprojection_a_px,
        reprojection_mojo_px=comparison.reprojection_b_px,
        reprojection_relative_difference=comparison.reprojection_relative_difference,
        cost_before=cost_before,
        cost_caspar=comparison.cost_a,
        cost_mojo=comparison.cost_b,
        cost_relative_difference=comparison.cost_relative_difference,
        rotation_difference_max_deg=comparison.rotation_difference_max_deg,
        rotation_difference_median_deg=comparison.rotation_difference_median_deg,
        center_difference_max=comparison.center_difference_max,
        center_difference_median=comparison.center_difference_median,
        point_difference_median=comparison.point_difference_median,
        point_difference_max=comparison.point_difference_max,
        registered_images_caspar=comparison.registered_images_a,
        registered_images_mojo=comparison.registered_images_b,
        caspar_lm_trials=[t.iterations for t in traces],
        caspar_pcg_iterations=[t.pcg_iterations for t in traces],
        caspar_exit=traces[-1].exit_reason,
        caspar_solver_seconds=[t.solver_seconds for t in traces],
        caspar_solve_call_seconds=call_seconds,
        mojo_lm_trials=mojo_lm,
        mojo_pcg_iterations=mojo_pcg,
        mojo_exit=result.exit_reason,
        mojo_solve_seconds=mojo_solve,
        mojo_call_seconds=mojo_call,
        model_dir=str(config.model_dir.resolve()),
        refine_intrinsics=config.refine_intrinsics,
        refine_sensor_from_rig=config.refine_sensor_from_rig,
        perturb_seed=config.perturb_seed,
        perturb_translation=config.perturb_translation if perturbed else None,
        perturb_rotation_deg=config.perturb_rotation_deg if perturbed else None,
        perturb_point=config.perturb_point if perturbed else None,
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
    """The table with `rows` added, each replacing a row of the same case (`AccuracyRow.case_key`).

    Args:
        table: Existing rows.
        rows: New rows.

    Returns:
        The merged table, sorted by GPU then display name.
    """
    keyed: dict[tuple[object, ...], AccuracyRow] = {r.case_key(): r for r in table.rows}
    keyed.update({r.case_key(): r for r in rows})
    return AccuracyTable(rows=sorted(keyed.values(), key=lambda r: (r.gpu, r.row)))


def _times(seconds: list[float]) -> str:
    return f"{min(seconds):.3f} / {statistics.median(seconds):.3f}"


def _relative(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.2e}"


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
        "A bold >2 % marks a row outside the bar: cost or reprojection more than 2 % apart, or different registered images.",
        "",
        "| row | GPU | obs | px before | px CASPAR | px Mojo | px rel | cost rel | rot max / med deg | centre max / med | point med | reg C / M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        flag: str = "" if r.within_parity() else " **>2 %**"
        lines.append(
            f"| {r.row} | {r.gpu} | {r.observations:,} | {r.reprojection_before_px:.4f} | {r.reprojection_caspar_px:.6f} | {r.reprojection_mojo_px:.6f} | "
            f"{_relative(r.reprojection_relative_difference)}{flag} | {_relative(r.cost_relative_difference)} | {r.rotation_difference_max_deg:.2e} / {r.rotation_difference_median_deg:.2e} | "
            f"{r.center_difference_max:.2e} / {r.center_difference_median:.2e} | {r.point_difference_median:.2e} | {r.registered_images_caspar} / {r.registered_images_mojo} |"
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
