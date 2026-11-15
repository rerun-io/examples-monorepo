"""One run's stages, its ledger and the boundary at which its artifacts become real.

A cuSFM run used to update its destination in place. `run_pipeline` deleted
`runtime.csv` and left everything else; `create_database` deleted the previous
database before building its replacement; the export and the summary then
overwrote whatever was there, file by file. A run that failed halfway therefore
left the destination holding *some* of this run's artifacts and *some* of the
last one's, with nothing on disk saying so — and `colsfm.benchmark` checked that
the files existed, not that they came from one run.

Three things fix that, and they live here:

* **One ledger.** `StageLedger` is the only record of what ran and how long it
  took. `runtime.csv` is its rows written out as they complete, and
  `PipelineSummary.stage_seconds` is the same rows at the end. A stage that
  raises records nothing, because the ledger is evidence of what *completed*.
* **A staging workspace.** `open_workspace` gives the run a private directory
  beside its destination and every stage writes there — the database included,
  since the run's own options are rebased onto it. Nothing in the destination
  changes while the run is in progress.
* **A publication boundary.** `RunWorkspace.publish` marks the state succeeded
  and swaps the staging directory into place with `os.replace`, so the
  destination goes from the previous run to this one without ever holding a
  mixture. A run that fails marks its staging directory `failed` and leaves it
  for inspection; the destination still holds the last run that worked, whole.

The published layout is byte-for-byte what a `pycusfm.cusfm_runner` run leaves,
plus `summary.json` and `run_state.json`. Readers that care whether a directory
is a finished colsfm run read the latter (`colsfm.benchmark.read_run`); a blob
reference run has neither file and is accepted as it always was.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias, get_args

from serde import serde
from serde.json import from_json, to_json

from colsfm import PACKAGE_DIR
from colsfm.export import RuntimeRecord, append_runtime_record

StageName: TypeAlias = Literal[
    "keyframe_selection",
    "feature_extraction",
    "pair_selection",
    "matching",
    "loop_closure",
    "pose_graph",
    "reconstruction",
    "extrinsic_refinement",
    "export",
]
"""The stages, in the order `CusfmRunner.run_all` runs their blob equivalents."""

EXTRINSIC_REFINEMENT_STAGE: Final[StageName] = "extrinsic_refinement"
"""The second mapping pass `--optimize-extrinsics` adds, mirroring the blob.

`CusfmRunner` runs `keypoints_mapper_main` twice: once with the extrinsics fixed
and once with `--optimize_extrinsics=True`, both over the same matches and the
same `pose_graph/frames_meta.json` poses (`pycusfm/cusfm_runner.py`,
`run_mapping` then `refine_extrinsics`). The second pass overwrites `kpmap/`."""

ALL_STAGE_NAMES: Final[tuple[StageName, ...]] = get_args(StageName)
"""Every stage, in `StageName`'s own order; what an `--optimize-extrinsics` run records."""

STAGE_NAMES: Final[tuple[StageName, ...]] = tuple(
    stage for stage in ALL_STAGE_NAMES if stage != EXTRINSIC_REFINEMENT_STAGE
)
"""The stages a default run records: every stage but the second mapping pass."""

CheapStageName: TypeAlias = Literal["keyframe_selection", "pair_selection"]
"""The stages the `stage` subcommand can run on their own: metadata only, no GPU."""

KEYFRAME_DIR_NAME: Final[str] = "keyframes"
"""`pycusfm.constants.kKEYFRAME_DIR`: where the selected metadata lands."""

POSE_GRAPH_DIR_NAME: Final[str] = "pose_graph"
"""`pycusfm.constants.kPOSE_GRAPH_DIR`: where the optimised rig trajectory lands."""

DATABASE_NAME: Final[str] = "database.db"
"""The COLMAP database, which has no cuSFM equivalent (the blob writes keyframe protos)."""

VEHICLE_POSE_TUM_NAME: Final[str] = "vehicle_pose.tum"
"""`pose_graph_main`'s own rig trajectory file name."""

LOOP_EDGES_NAME: Final[str] = "loop_edges.json"
"""The gated loop constraints, written beside the rig trajectory.

No cuSFM equivalent — the blob keeps its edges inside `vehicle_pose_graph.pb.txt`
— but a viewer that wants to draw the loops needs them as data rather than as a
log line. See `colsfm.pipeline.write_loop_edges`."""

SUMMARY_NAME: Final[str] = "summary.json"
"""Machine-readable run report, written by this pipeline and by nothing in cuSFM."""

RUN_STATE_NAME: Final[str] = "run_state.json"
"""Whether the run that wrote this directory finished; see `RunState`.

Also written by nothing in cuSFM, which is why its *absence* means "not a colsfm
run" rather than "an unfinished one"."""

STAGING_SUFFIX: Final[str] = ".colsfm-staging"
"""Marks a run in progress: `<output_dir>.colsfm-staging-<pid>`, a sibling of the
destination so the closing `os.replace` stays on one filesystem."""

REPLACED_SUFFIX: Final[str] = ".colsfm-replaced"
"""Where the previous run is parked for the microsecond between the two renames."""


def stage_names(optimize_extrinsics: bool) -> tuple[StageName, ...]:
    """The stages a run will record, in order.

    Args:
        optimize_extrinsics: Whether the run makes the second mapping pass.

    Returns:
        `ALL_STAGE_NAMES` when the flag is set, `STAGE_NAMES` otherwise.
    """
    return ALL_STAGE_NAMES if optimize_extrinsics else STAGE_NAMES


def git_sha(repo_root: Path = PACKAGE_DIR) -> str:
    """Read the working tree's commit, for the summary's provenance field.

    Args:
        repo_root: Directory inside the repository.

    Returns:
        The 40-character SHA, or `"unknown"` when git is unavailable or the
        directory is not a checkout.
    """
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


# ══════════════════════════════════════════════════════════════════════════════════════
# the ledger
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One completed stage: what ran, and for how long."""

    stage: StageName
    """The stage that finished."""
    seconds: float
    """Its wall-clock duration."""


@dataclass(slots=True)
class StageLedger:
    """The one record of what a run completed, and the only source of its timings.

    `runtime.csv` is these rows written out as they land, and
    `PipelineSummary.stage_seconds` is the same rows at the end — so the two can
    disagree only if this class does, which it cannot. A stage that raises appends
    nothing: the ledger is evidence of what completed, and a half-run stage
    completed nothing.
    """

    output_dir: Path
    """Workspace holding `runtime.csv`; the run's staging directory while it runs."""
    stages: tuple[StageName, ...] = STAGE_NAMES
    """The stages this run will record, so the progress line counts the right total."""
    records: list[StageRecord] = field(default_factory=list)
    """Completed stages, in completion order."""

    def record(self, stage: StageName, seconds: float) -> None:
        """Store one stage's runtime and append it to the log.

        Args:
            stage: The stage that just finished.
            seconds: Its wall-clock duration.
        """
        self.records.append(StageRecord(stage=stage, seconds=seconds))
        append_runtime_record(self.output_dir, RuntimeRecord(command=stage, runtime_seconds=seconds))

    @property
    def seconds_by_stage(self) -> dict[StageName, float]:
        """Wall-clock seconds per completed stage, in completion order.

        Returns:
            The same rows `runtime.csv` holds, which is what the summary reports.
        """
        return {record.stage: record.seconds for record in self.records}


@contextlib.contextmanager
def timed_stage(ledger: StageLedger, stage: StageName) -> Iterator[None]:
    """Time the enclosed block and record it as one stage, only when it succeeds.

    A stage that raises records nothing: `runtime.csv` and the summary's
    `stage_seconds` are the run's evidence of what completed, and a half-run stage
    completed nothing. The exception propagates unchanged.

    Args:
        ledger: The ledger to record into.
        stage: The stage the block implements.

    Yields:
        Nothing; the block runs inside the timing window.
    """
    started: float = time.perf_counter()
    print(f"[colsfm] stage {ledger.stages.index(stage) + 1}/{len(ledger.stages)}: {stage}")
    yield
    elapsed: float = time.perf_counter() - started
    ledger.record(stage, elapsed)
    print(f"[colsfm] {stage} finished in {elapsed:.2f}s")


# ══════════════════════════════════════════════════════════════════════════════════════
# the publication boundary
# ══════════════════════════════════════════════════════════════════════════════════════


RunStatus: TypeAlias = Literal["running", "succeeded", "failed"]
"""Where a run got to. Only `succeeded` may be published, so a published run is complete."""


@serde
@dataclass(frozen=True, slots=True)
class RunState:
    """`run_state.json`: whether the directory holding it is a finished run."""

    status: RunStatus
    """`running` inside a staging directory, `succeeded` in a published one, `failed`
    in a staging directory a run left behind."""
    git_sha: str
    """`git rev-parse HEAD` of the tree that ran, so a mixed directory is impossible
    to mistake for a coherent one even by hand."""
    stages_planned: tuple[StageName, ...]
    """The stages the run set out to record."""
    stages_completed: tuple[StageName, ...]
    """The stages it actually completed, in completion order."""
    failure: str | None = None
    """The exception that stopped the run, or None."""


def read_run_state(run_dir: Path) -> RunState | None:
    """Read a directory's run state, if it has one.

    Args:
        run_dir: The directory to inspect.

    Returns:
        The state, or None when the directory holds no `run_state.json` — which is
        what a `pycusfm` reference run looks like, and is not an error.
    """
    # Read it and handle its absence, rather than asking first: between an `is_file`
    # and the read the file can be renamed away, which is exactly what `publish` does
    # to a directory somebody else is reading.
    try:
        return from_json(RunState, (run_dir / RUN_STATE_NAME).read_text())
    except FileNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """Where a run writes while it runs, and what it does with that when it ends."""

    output_dir: Path
    """The destination the caller asked for; untouched until `publish`."""
    staging_dir: Path
    """`<output_dir>.colsfm-staging-<pid>`, a sibling so the closing rename is atomic."""
    ledger: StageLedger
    """The run's ledger, writing `runtime.csv` into the staging directory.

    Owned here rather than built beside the workspace, because the two used to be
    given the same stage tuple separately -- `stages_planned` and `StageLedger.stages`
    -- and `publish` and `fail` then had to be *handed* the ledger whose completed
    stages they were about to record. The workspace is what a run has; the ledger is
    part of it."""

    @property
    def stages_planned(self) -> tuple[StageName, ...]:
        """The stages the run set out to record, written into every state.

        Returns:
            `ledger.stages`, which is the same tuple the progress lines count against.
        """
        return self.ledger.stages

    def record_state(self, status: RunStatus, completed: tuple[StageName, ...], failure: str | None = None) -> None:
        """Overwrite the staging directory's `run_state.json`.

        Public because `open_workspace` writes the opening `running` state through
        it; a caller has no reason to.

        Args:
            status: Where the run has got to.
            completed: Stages finished so far, in completion order.
            failure: The exception that stopped the run, when one did.
        """
        state: RunState = RunState(
            status=status,
            git_sha=git_sha(),
            stages_planned=tuple(self.stages_planned),
            stages_completed=completed,
            failure=failure,
        )
        (self.staging_dir / RUN_STATE_NAME).write_text(to_json(state) + "\n")

    def publish(self) -> None:
        """Mark the run succeeded and swap its directory into the destination.

        Two renames with the previous run parked in between, so the destination is
        never a directory being filled in: it holds the previous run, then this one.
        """
        self.record_state("succeeded", tuple(record.stage for record in self.ledger.records))
        replaced: Path = self.output_dir.with_name(f"{self.output_dir.name}{REPLACED_SUFFIX}-{os.getpid()}")
        if self.output_dir.exists():
            os.replace(self.output_dir, replaced)
        os.replace(self.staging_dir, self.output_dir)
        shutil.rmtree(replaced, ignore_errors=True)

    def fail(self, reason: str) -> None:
        """Mark the run failed and leave its staging directory for inspection.

        Nothing is published, so the destination still holds the last run that
        worked, whole. The next run over the same destination clears this away.

        Args:
            reason: What stopped it.
        """
        self.record_state("failed", tuple(record.stage for record in self.ledger.records), failure=reason)
        print(f"[colsfm] run failed; its partial artifacts are in {self.staging_dir}, {self.output_dir} is unchanged")


def _owning_pid(directory: Path) -> int | None:
    """The process id a staging or parked directory names itself after.

    Args:
        directory: A `<name>.colsfm-staging-<pid>` or `<name>.colsfm-replaced-<pid>` path.

    Returns:
        The pid, or None when the trailing field is not a number — a directory
        somebody made by hand, which is not this function's to interpret.
    """
    tail: str = directory.name.rpartition("-")[2]
    return int(tail) if tail.isdigit() else None


def _is_running(pid: int) -> bool:
    """Whether a process with this id exists.

    `os.kill(pid, 0)` delivers no signal and only asks the kernel the question.

    Args:
        pid: The process id to ask about.

    Returns:
        True when the process exists, including when it belongs to another user —
        `PermissionError` is proof that it is there.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_abandoned_directories(output_dir: Path) -> None:
    """Remove the staging and parked directories of runs that are no longer running.

    A directory names the pid that made it, and this reaps only the ones whose pid
    is gone. It used to `rmtree` every match, which meant two runs writing the same
    output directory at once — a sweep, a retry started before the first finished —
    deleted each other's live staging directory, silently, because the removal ran
    under `ignore_errors`.

    This process's own leftovers are always reaped: `publish` and `fail` are the only
    ways a run ends, both leave at most one directory, and the next run in this
    process takes the same name.

    Args:
        output_dir: The destination whose siblings to sweep.
    """
    patterns: tuple[str, ...] = (
        f"{output_dir.name}{STAGING_SUFFIX}-*",
        f"{output_dir.name}{REPLACED_SUFFIX}-*",
    )
    for pattern in patterns:
        for stale in sorted(output_dir.parent.glob(pattern)):
            pid: int | None = _owning_pid(stale)
            if pid is not None and pid != os.getpid() and _is_running(pid):
                print(f"[colsfm] leaving {stale.name} alone; process {pid} is still writing it")
                continue
            shutil.rmtree(stale, ignore_errors=True)


def open_workspace(output_dir: Path, stages: tuple[StageName, ...]) -> RunWorkspace:
    """Start a run: give it a private directory beside its destination.

    Any staging or parked directory an *abandoned* run of this destination left
    behind is removed first, so one failure leaves one directory rather than a pile;
    a directory another live process is writing is left alone.

    Args:
        output_dir: The destination the caller asked for. Not created, not touched.
        stages: The stages the run will record.

    Returns:
        The workspace, with its ledger already pointed at the staging directory.
        Everything the run writes goes into `staging_dir`.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reap_abandoned_directories(output_dir)
    staging_dir: Path = output_dir.with_name(f"{output_dir.name}{STAGING_SUFFIX}-{os.getpid()}")
    staging_dir.mkdir(parents=True)
    workspace: RunWorkspace = RunWorkspace(
        output_dir=output_dir,
        staging_dir=staging_dir,
        ledger=StageLedger(output_dir=staging_dir, stages=stages),
    )
    workspace.record_state("running", ())
    return workspace
