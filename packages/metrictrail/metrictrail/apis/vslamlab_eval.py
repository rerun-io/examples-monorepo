"""Evaluate MetricTrail runs under VSLAM-LAB's protocol and under the paper's, side by side.

For each sequence and each run directory (one per seed; ``exp_it`` = its position in ``runs``):

1. read ``pose/<name>.npz`` (``source_frame_ids`` index the sorted frame directory ``<frame_root>/<name>``);
2. give every tracked frame its VSLAM-LAB timestamp (rgb.csv, or the file name for TUM frames rgb.csv leaves out);
3. write a VSLAM-LAB experiment folder (``<out>/<exp_name>/<DATASET>/<sequence>/`` with
   ``<exp_it:05d>_KeyFrameTrajectory.csv``, ``groundtruth.csv``, ``rgb_exp.csv`` and ``vslamlab_exp_log.csv``);
4. run VSLAM-LAB's own ``Evaluate.evaluate_functions.evaluate_sequence`` in its pixi env and read its ``ate.csv``;
5. compute the paper's number (``metrictrail.eval.score.paper_score``) and an in-process replica of VSLAM-LAB's;
6. write ``<out>/report.json`` and print a table.
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Int64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from metrictrail.eval.protocols import PROTOCOLS, PaperProtocol, VslamlabDataset, units_per_metre
from metrictrail.eval.score import FrameTimes, PaperContractError, PaperScore, frame_times, paper_score, vslamlab_replica_ate, vslamlab_trajectory
from metrictrail.eval.trajectory import AteResult
from metrictrail.eval.vipe_output import MetricTrailTrajectory, RunRecord, read_run_record, read_trajectory
from metrictrail.eval.vslamlab_files import (
    AteCsvRow,
    ExpLogRow,
    PoseTable,
    RgbCsv,
    read_ate_csv,
    read_pose_csv,
    read_rgb_csv,
    trajectory_csv_name,
    write_exp_log,
    write_pose_csv,
    write_rgb_exp_csv,
)
from metrictrail.frames import list_frames

Naming: TypeAlias = Literal["upstream", "vslamlab"]
"""Name of the frame directory and of ``pose/<name>.npz``: upstream's eval case (``desk``) or VSLAM-LAB's (``rgbd_dataset_freiburg1_desk``)."""

DRIVER_SCRIPT: Path = Path(__file__).resolve().parents[2] / "vslamlab_driver" / "evaluate_sequences.py"
"""VSLAM-LAB-side shim (runs in VSLAM-LAB's env, imports only VSLAM-LAB); lives beside the package, outside its import tree."""


@dataclass
class Config:
    """Score MetricTrail output directories with VSLAM-LAB's evaluation and with the paper's protocol."""

    runs: list[Path]
    """MetricTrail output directories (``pipeline.output.path``), one per seed; the order gives ``exp_it``."""
    dataset: VslamlabDataset
    """VSLAM-LAB dataset name."""
    out_dir: Path
    """Where the VSLAM-LAB experiment folder and ``report.json`` go (must not exist yet, or be empty)."""
    sequences: list[str] = field(default_factory=list)
    """VSLAM-LAB sequence names; empty means the paper's list for the dataset."""
    naming: Naming = "upstream"
    """How the frame directory and the pose artifact are named."""
    frame_root: Path | None = None
    """Parent of the frame directories; default: the protocol's MetricTrail data root."""
    benchmark_root: Path = Path("/mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark")
    """VSLAM-LAB benchmark root (``<root>/<DATASET>/<sequence>/{rgb.csv,groundtruth.csv}``)."""
    vslamlab_dir: Path = Path("/tmp/metrictrail-scratch/vslamlab/VSLAM-LAB")
    """VSLAM-LAB checkout with an installed ``vslamlab`` pixi env."""
    exp_name: str = "exp_metrictrail"
    """Experiment folder name."""
    method_name: str = "metrictrail"
    """``method_name`` written to the experiment log."""
    run_vslamlab: bool = True
    """Run VSLAM-LAB's own evaluation; off scores the paper protocol and the in-process replica only."""
    vslamlab_env: str = "vslamlab"
    """Pixi environment of the VSLAM-LAB checkout that runs its evaluation."""
    evo_home: Path | None = None
    """HOME of the VSLAM-LAB subprocess, so evo reads and rewrites ``<evo_home>/.evo/settings.json`` instead of the
    user's ``~/.evo``; None: ``<out_dir>/evo_home``."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunScore:
    """Scores of one run on one sequence (metres)."""

    sequence: str
    """VSLAM-LAB sequence name."""
    name: str
    """Frame directory / pose artifact name."""
    exp_it: int
    """Run index."""
    run_dir: str
    """MetricTrail output directory."""
    num_input_frames: int
    """Frames in the input directory."""
    num_tracked: int
    """Poses in the prediction."""
    num_timestamps_from_filename: int
    """Input frames rgb.csv does not list (timestamp from the file name)."""
    paper_ate_m: float | None
    """The paper's metric: Sim(3) ATE RMSE; null when the paper's contract fails."""
    paper_scale: float | None
    """Sim(3) scale of the paper alignment."""
    paper_se3_ate_m: float | None
    """Diagnostic: SE(3) (no scale) ATE RMSE with the paper's pairing; the paper reports no such number."""
    paper_num_pairs: int | None
    """Pairs the paper metric used."""
    paper_num_without_gt: int | None
    """Tracked frames with no GT under the paper's rule."""
    paper_error: str | None
    """Why the paper metric was not computed."""
    vslamlab_ate_m: float | None
    """VSLAM-LAB's ``ate.csv`` rmse (its own code, its env); null when not run."""
    vslamlab_num_frames: int | None
    """VSLAM-LAB ``num_frames`` (rows of rgb_exp.csv)."""
    vslamlab_num_tracked_frames: int | None
    """VSLAM-LAB ``num_tracked_frames``."""
    vslamlab_num_evaluated_frames: int | None
    """VSLAM-LAB ``num_evaluated_frames`` (pairs evo associated)."""
    vslamlab_replica_ate_m: float
    """In-process replica of VSLAM-LAB's number (cross-check)."""
    vslamlab_replica_num_pairs: int
    """Pairs the replica associated."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Dataset means in the paper's units."""

    dataset: str
    """VSLAM-LAB dataset name."""
    units: str
    """``m`` or ``cm``."""
    paper_table: str
    """Table and page of the headline number."""
    paper_value: float
    """The paper's dataset mean."""
    num_sequences: int
    """Sequences scored."""
    num_runs: int
    """Runs per sequence."""
    paper_protocol_mean: float | None
    """Mean over sequences of the per-sequence mean over runs (pre-registered rule); null if any run failed the contract."""
    vslamlab_mean: float | None
    """Mean over sequences of VSLAM-LAB's per-sequence median over runs (its own tables use the median); null when not run."""
    vslamlab_replica_mean: float
    """The same from the in-process replica."""
    se3_diagnostic_mean: float | None
    """Mean over sequences of the SE(3) diagnostic (not in the paper)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Report:
    """``report.json``."""

    dataset: str
    """VSLAM-LAB dataset name."""
    exp_folder: str
    """VSLAM-LAB experiment folder."""
    runs: list[RunScore]
    """Every run on every sequence."""
    summary: DatasetSummary
    """Dataset means."""


@dataclass(frozen=True, slots=True)
class SequenceInputs:
    """What one sequence contributes to every run's score."""

    sequence: str
    """VSLAM-LAB sequence name."""
    name: str
    """Frame directory / pose artifact name."""
    frames: FrameTimes
    """Frame names and VSLAM-LAB timestamps."""
    rgb: RgbCsv
    """The sequence's rgb.csv."""
    gt: PoseTable
    """The sequence's groundtruth.csv."""
    gt_path: Path
    """Path of groundtruth.csv."""


def load_sequence(config: Config, protocol: PaperProtocol, sequence: str) -> SequenceInputs:
    """Read the frame directory, rgb.csv and groundtruth.csv of one sequence."""
    name: str = protocol.upstream_cases.get(sequence, sequence) if config.naming == "upstream" else sequence
    frame_root: Path = config.frame_root if config.frame_root is not None else Path(protocol.frame_root)
    sequence_dir: Path = config.benchmark_root / protocol.benchmark_folder / sequence
    rgb: RgbCsv = read_rgb_csv(sequence_dir / "rgb.csv")
    rgb_by_name: dict[str, int] = {Path(row.path).name: row.timestamp_ns for row in rgb.rows}
    frames: FrameTimes = frame_times(list_frames(frame_root / name), rgb_by_name, protocol.timestamp_fallback)
    gt_path: Path = sequence_dir / "groundtruth.csv"
    return SequenceInputs(sequence=sequence, name=name, frames=frames, rgb=rgb, gt=read_pose_csv(gt_path), gt_path=gt_path)


def rgb_exp_rows(inputs: SequenceInputs) -> list[tuple[str, ...]]:
    """rgb.csv rows of the frames MetricTrail was given (rows synthesised, depth cells empty, for frames rgb.csv lacks)."""
    by_name: dict[str, tuple[str, ...]] = {Path(row.path).name: row.cells for row in inputs.rgb.rows}
    width: int = len(inputs.rgb.header)
    rows: list[tuple[str, ...]] = []
    for name, stamp in zip(inputs.frames.names, inputs.frames.timestamps_ns.tolist(), strict=True):
        rows.append(by_name.get(name, (str(stamp), f"rgb_0/{name}", *([""] * (width - 2)))))
    return rows


TrajectoryLoader: TypeAlias = Callable[[SequenceInputs, int], MetricTrailTrajectory]
"""Gives the trajectory of run ``exp_it`` on a sequence (``main`` reads ``pose/<name>.npz`` of ``Config.runs[exp_it]``)."""


def read_run(run_dir: Path, inputs: SequenceInputs) -> MetricTrailTrajectory:
    """Read ``pose/<name>.npz`` of one MetricTrail output directory; refuse a run record that did not succeed."""
    record_path: Path = run_dir / "vipe" / f"{inputs.name}_metrictrail.json"
    if record_path.is_file():
        record: RunRecord = read_run_record(record_path)
        if record.status != "success":
            raise ValueError(f"{record_path}: status {record.status!r}")
    return read_trajectory(run_dir, inputs.name)


def score_run(
    inputs: SequenceInputs, trajectory: MetricTrailTrajectory, run_label: str, exp_it: int, protocol: PaperProtocol, sequence_exp_dir: Path
) -> RunScore:
    """Paper score + in-process replica for one run; writes its VSLAM-LAB trajectory file."""
    if trajectory.source_frame_ids.size and trajectory.source_frame_ids[-1] >= len(inputs.frames.names):
        raise ValueError(
            f"{run_label}: source frame id {int(trajectory.source_frame_ids[-1])} is past the {len(inputs.frames.names)} frames of {inputs.name}"
        )
    tracked_ns: Int64[ndarray, "n"] = inputs.frames.timestamps_ns[trajectory.source_frame_ids]
    vslamlab_poses: PoseTable = vslamlab_trajectory(trajectory.world_T_cam, tracked_ns)
    write_pose_csv(sequence_exp_dir / trajectory_csv_name(exp_it), vslamlab_poses)
    replica: AteResult = vslamlab_replica_ate(vslamlab_poses, inputs.gt, protocol.rgb_hz)
    paper: PaperScore | None = None
    paper_error: str | None = None
    try:
        paper = paper_score(trajectory.world_T_cam, trajectory.source_frame_ids, inputs.frames, inputs.gt, protocol)
    except PaperContractError as error:
        paper_error = str(error)
    return RunScore(
        sequence=inputs.sequence,
        name=inputs.name,
        exp_it=exp_it,
        run_dir=run_label,
        num_input_frames=len(inputs.frames.names),
        num_tracked=int(trajectory.source_frame_ids.shape[0]),
        num_timestamps_from_filename=inputs.frames.num_from_filename,
        paper_ate_m=paper.sim3.rmse_m if paper else None,
        paper_scale=paper.sim3.scale if paper else None,
        paper_se3_ate_m=paper.se3.rmse_m if paper else None,
        paper_num_pairs=paper.sim3.num_pairs if paper else None,
        paper_num_without_gt=paper.num_without_gt if paper else None,
        paper_error=paper_error,
        vslamlab_ate_m=None,
        vslamlab_num_frames=None,
        vslamlab_num_tracked_frames=None,
        vslamlab_num_evaluated_frames=None,
        vslamlab_replica_ate_m=replica.rmse_m,
        vslamlab_replica_num_pairs=replica.num_pairs,
    )


def vslamlab_command(config: Config, exp_dir: Path, sequences: list[str]) -> list[str]:
    """The ``pixi run`` command line of VSLAM-LAB's own evaluation for ``sequences`` of ``exp_dir``."""
    return [
        "pixi", "run", "--frozen", "--manifest-path", str(config.vslamlab_dir.resolve() / "pixi.toml"), "-e", config.vslamlab_env, "python",
        str(DRIVER_SCRIPT), "--exp-folder", str(exp_dir), "--dataset", config.dataset, "--sequences", *sequences,
    ]  # fmt: skip


def vslamlab_environment(config: Config, parent: dict[str, str], evo_home: Path) -> dict[str, str]:
    """Environment of the VSLAM-LAB subprocess.

    This process's pixi/conda activation is dropped so the child pixi resolves VSLAM-LAB's own manifest. ``HOME`` is
    ``evo_home``: VSLAM-LAB runs ``evo_config set save_traj_in_zip true`` and evo rewrites an outdated
    ``~/.evo/settings.json``, both under ``Path.home()``. The package cache stays the user's, so pixi needs no download.
    """
    env: dict[str, str] = {key: value for key, value in parent.items() if not key.startswith(("PIXI_", "CONDA_"))}
    real_home: str = parent.get("HOME", str(Path.home()))
    env.setdefault("RATTLER_CACHE_DIR", parent.get("RATTLER_CACHE_DIR", str(Path(real_home) / ".cache" / "rattler")))
    env.setdefault("XDG_CACHE_HOME", parent.get("XDG_CACHE_HOME", str(Path(real_home) / ".cache")))
    env["HOME"] = str(evo_home)
    env["PYTHONPATH"] = str(config.vslamlab_dir.resolve())
    return env


def run_vslamlab_evaluation(config: Config, exp_dir: Path, sequences: list[str], log_path: Path) -> None:
    """Run VSLAM-LAB's own ``evaluate_sequence`` in its pixi env (output in ``log_path``).

    Raises:
        RuntimeError: If the subprocess fails.
    """
    evo_home: Path = (config.evo_home if config.evo_home is not None else config.out_dir / "evo_home").resolve()
    evo_home.mkdir(parents=True, exist_ok=True)
    command: list[str] = vslamlab_command(config, exp_dir, sequences)
    env: dict[str, str] = vslamlab_environment(config, dict(os.environ), evo_home)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=config.vslamlab_dir.resolve(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"VSLAM-LAB evaluation failed (exit {completed.returncode}); see {log_path}")


def summarise(scores: list[RunScore], protocol: PaperProtocol, num_runs: int, ran_vslamlab: bool) -> DatasetSummary:
    """Dataset means in the paper's units (per sequence: mean over runs for the paper rule, median for VSLAM-LAB's)."""
    factor: float = units_per_metre(protocol.units)
    sequences: list[str] = sorted({score.sequence for score in scores})

    def per_sequence(values_of: str, reduce: Literal["mean", "median"]) -> float | None:
        means: list[float] = []
        for sequence in sequences:
            values: list[float | None] = [getattr(score, values_of) for score in scores if score.sequence == sequence]
            if any(value is None for value in values):
                return None
            finite: list[float] = [float(value) for value in values if value is not None]
            means.append(float(np.mean(finite)) if reduce == "mean" else float(np.median(finite)))
        return factor * float(np.mean(means))

    replica: float | None = per_sequence("vslamlab_replica_ate_m", "median")
    if replica is None:
        raise ValueError("no replica scores")
    return DatasetSummary(
        dataset=protocol.dataset,
        units=protocol.units,
        paper_table=protocol.paper_table,
        paper_value=protocol.paper_value,
        num_sequences=len(sequences),
        num_runs=num_runs,
        paper_protocol_mean=per_sequence("paper_ate_m", "mean"),
        vslamlab_mean=per_sequence("vslamlab_ate_m", "median") if ran_vslamlab else None,
        vslamlab_replica_mean=replica,
        se3_diagnostic_mean=per_sequence("paper_se3_ate_m", "mean"),
    )


def print_table(report: Report, protocol: PaperProtocol) -> None:
    """Print one line per run and the dataset summary, in the paper's units."""
    factor: float = units_per_metre(protocol.units)

    def text(value: float | None) -> str:
        return f"{factor * value:9.3f}" if value is not None else "      n/a"

    print(f"{'sequence':34s} it  frames tracked |    paper  vslamlab   replica   se3diag ({protocol.units})")
    for score in report.runs:
        print(
            f"{score.sequence:34s} {score.exp_it:2d} {score.num_input_frames:7d} {score.num_tracked:7d} | "
            f"{text(score.paper_ate_m)} {text(score.vslamlab_ate_m)} {text(score.vslamlab_replica_ate_m)} {text(score.paper_se3_ate_m)}"
        )
        if score.paper_error:
            print(f"    paper contract failed: {score.paper_error}")
    summary: DatasetSummary = report.summary
    print(
        f"dataset {summary.dataset}: {summary.num_sequences} sequences x {summary.num_runs} runs | paper protocol mean "
        f"{summary.paper_protocol_mean} {summary.units} | VSLAM-LAB mean {summary.vslamlab_mean} | replica {summary.vslamlab_replica_mean:.4f} | "
        f"paper {summary.paper_value} ({summary.paper_table})"
    )


def evaluate(config: Config, load: TrajectoryLoader, run_labels: list[str]) -> Report:
    """Score run ``exp_it`` (labelled ``run_labels[exp_it]``) on every sequence, run VSLAM-LAB, write ``report.json``."""
    protocol: PaperProtocol = PROTOCOLS[config.dataset]
    sequences: list[str] = config.sequences or list(protocol.upstream_cases)
    if config.out_dir.exists() and any(config.out_dir.iterdir()):
        raise FileExistsError(f"{config.out_dir} is not empty; pick a new output directory")
    exp_dir: Path = config.out_dir.resolve() / config.exp_name
    scores: list[RunScore] = []
    log_rows: list[ExpLogRow] = []
    for sequence in sequences:
        inputs: SequenceInputs = load_sequence(config, protocol, sequence)
        sequence_exp_dir: Path = exp_dir / protocol.benchmark_folder / sequence
        sequence_exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs.gt_path, sequence_exp_dir / "groundtruth.csv")
        write_rgb_exp_csv(sequence_exp_dir / "rgb_exp.csv", inputs.rgb.header, rgb_exp_rows(inputs))
        for exp_it, run_label in enumerate(run_labels):
            trajectory: MetricTrailTrajectory = load(inputs, exp_it)
            scores.append(score_run(inputs, trajectory, run_label, exp_it, protocol, sequence_exp_dir))
            runtime_s: float = trajectory.runtime_s if np.isfinite(trajectory.runtime_s) else 0.0
            log_rows.append(ExpLogRow(config.method_name, config.dataset, sequence, exp_it, runtime_s))
    write_exp_log(exp_dir / "vslamlab_exp_log.csv", log_rows)
    if config.run_vslamlab:
        run_vslamlab_evaluation(config, exp_dir, sequences, config.out_dir / "vslamlab_evaluate.log")
        merged: list[RunScore] = []
        for score in scores:
            ate_rows: list[AteCsvRow] = read_ate_csv(exp_dir / protocol.benchmark_folder / score.sequence / "vslamlab_evaluation" / "ate.csv")
            row: AteCsvRow = next(row for row in ate_rows if row.exp_it == score.exp_it)
            merged.append(
                replace(
                    score,
                    vslamlab_ate_m=row.rmse_m,
                    vslamlab_num_frames=row.num_frames,
                    vslamlab_num_tracked_frames=row.num_tracked_frames,
                    vslamlab_num_evaluated_frames=row.num_evaluated_frames,
                )
            )
        scores = merged
    report = Report(
        dataset=config.dataset,
        exp_folder=str(exp_dir),
        runs=scores,
        summary=summarise(scores, protocol, len(run_labels), config.run_vslamlab),
    )
    (config.out_dir / "report.json").write_text(to_json(report), encoding="utf-8")
    print_table(report, protocol)
    return report


def main(config: Config) -> Report:
    """Score the MetricTrail output directories ``config.runs`` (one per seed)."""
    return evaluate(config, lambda inputs, exp_it: read_run(config.runs[exp_it], inputs), [str(run) for run in config.runs])
