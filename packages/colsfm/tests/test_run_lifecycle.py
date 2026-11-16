"""The run's ledger, its staging workspace and the boundary it publishes across.

Pure contracts against a fake filesystem layout — nothing here runs a stage. What
is pinned is the review's defect: a run used to update its destination in place,
so a rerun that failed halfway left that directory holding some of this run's
artifacts and some of the last one's, with nothing on disk saying which.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from serde.json import from_json

from colsfm.export import RUNTIME_CSV_NAME, RuntimeRecord, read_runtime_records
from colsfm.run_lifecycle import (
    ALL_STAGE_NAMES,
    REPLACED_SUFFIX,
    RUN_STATE_NAME,
    STAGE_NAMES,
    STAGING_SUFFIX,
    RunState,
    RunWorkspace,
    StageLedger,
    open_workspace,
    read_run_state,
    stage_names,
    timed_stage,
)

SMOKE_STAGES: tuple[str, ...] = ("matching", "export")
"""Two stages, enough to see the ledger order without running anything."""

UNUSED_PID: int = 0x7FFFFFFF
"""A pid above every Linux `pid_max`, so no process can ever hold it.

The reaper asks whether the owning process is alive; this is how a test says "it
is not" without racing a real one.""" 


# ─────────────────────────────────────────────────────────────────────────────
# The ledger
# ─────────────────────────────────────────────────────────────────────────────


def test_the_ledger_feeds_the_log_and_the_summary_from_one_list(tmp_path: Path) -> None:
    """`runtime.csv` and `stage_seconds` are the same rows; they cannot disagree."""
    ledger: StageLedger = StageLedger(output_dir=tmp_path, stages=STAGE_NAMES)
    with timed_stage(ledger, "matching"):
        pass
    with timed_stage(ledger, "export"):
        pass

    assert [record.stage for record in ledger.records] == ["matching", "export"]
    assert list(ledger.seconds_by_stage) == ["matching", "export"]
    rows: list[RuntimeRecord] = read_runtime_records(tmp_path / RUNTIME_CSV_NAME)
    assert [row.command for row in rows] == ["matching", "export"]
    assert [row.runtime_seconds for row in rows] == list(ledger.seconds_by_stage.values())


def test_a_stage_that_raises_is_absent_from_both(tmp_path: Path) -> None:
    """The ledger is evidence of what completed, so a half-run stage is nowhere in it."""
    ledger: StageLedger = StageLedger(output_dir=tmp_path, stages=STAGE_NAMES)
    with pytest.raises(RuntimeError, match="stage body failed"), timed_stage(ledger, "matching"):
        raise RuntimeError("stage body failed")

    assert ledger.records == []
    assert ledger.seconds_by_stage == {}
    assert not (tmp_path / RUNTIME_CSV_NAME).exists()


def test_the_extrinsics_run_records_one_more_stage() -> None:
    """`--optimize-extrinsics` adds the second mapping pass, and only that."""
    assert stage_names(optimize_extrinsics=False) == STAGE_NAMES
    assert stage_names(optimize_extrinsics=True) == ALL_STAGE_NAMES
    assert set(ALL_STAGE_NAMES) - set(STAGE_NAMES) == {"extrinsic_refinement"}


# ─────────────────────────────────────────────────────────────────────────────
# The publication boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_a_run_in_progress_leaves_the_destination_alone(tmp_path: Path) -> None:
    """Nothing in the destination changes until the run publishes."""
    output_dir: Path = tmp_path / "cusfm"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("the previous run\n")

    workspace: RunWorkspace = open_workspace(output_dir, STAGE_NAMES)
    (workspace.staging_dir / "summary.json").write_text("this run\n")

    assert (output_dir / "summary.json").read_text() == "the previous run\n"
    assert workspace.staging_dir != output_dir
    assert workspace.staging_dir.parent == output_dir.parent, "renaming across filesystems is not atomic"
    assert read_run_state(workspace.staging_dir) is not None
    state: RunState = from_json(RunState, (workspace.staging_dir / RUN_STATE_NAME).read_text())
    assert state.status == "running"


def test_publishing_swaps_the_whole_directory_at_once(tmp_path: Path) -> None:
    """The destination goes from the previous run to this one, never to a mixture."""
    output_dir: Path = tmp_path / "cusfm"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("the previous run\n")
    (output_dir / "only_in_the_previous_run.txt").write_text("stale\n")

    workspace: RunWorkspace = open_workspace(output_dir, STAGE_NAMES)
    (workspace.staging_dir / "summary.json").write_text("this run\n")
    workspace.publish()

    assert (output_dir / "summary.json").read_text() == "this run\n"
    assert not (output_dir / "only_in_the_previous_run.txt").exists(), "the previous run must not survive in pieces"
    assert not workspace.staging_dir.exists()
    assert not list(output_dir.parent.glob(f"*{REPLACED_SUFFIX}-*")), "the parked directory is cleaned up"
    published: RunState | None = read_run_state(output_dir)
    assert published is not None
    assert published.status == "succeeded"


def test_publishing_into_a_destination_that_does_not_exist_yet(tmp_path: Path) -> None:
    """A first run has nothing to replace and still lands in one rename."""
    output_dir: Path = tmp_path / "runs" / "cusfm"
    workspace: RunWorkspace = open_workspace(output_dir, STAGE_NAMES)
    (workspace.staging_dir / "summary.json").write_text("this run\n")
    workspace.publish()

    assert (output_dir / "summary.json").read_text() == "this run\n"


def test_a_failed_run_publishes_nothing_and_says_where_its_pieces_are(tmp_path: Path) -> None:
    """The destination keeps the last run that worked, whole."""
    output_dir: Path = tmp_path / "cusfm"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("the previous run\n")

    workspace: RunWorkspace = open_workspace(output_dir, STAGE_NAMES)
    with timed_stage(workspace.ledger, "keyframe_selection"):
        pass
    (workspace.staging_dir / "summary.json").write_text("half of this run\n")
    workspace.fail("ValueError: no keyframe survived selection")

    assert (output_dir / "summary.json").read_text() == "the previous run\n"
    failed: RunState | None = read_run_state(workspace.staging_dir)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.stages_completed == ("keyframe_selection",)
    assert failed.stages_planned == tuple(STAGE_NAMES)
    assert failed.failure == "ValueError: no keyframe survived selection"


def test_the_workspace_owns_its_ledger(tmp_path: Path) -> None:
    """One stage tuple, one `runtime.csv`, one object that has both.

    The workspace used to be given the planned stages and the caller then built a
    `StageLedger` from the same tuple and the same directory, and handed it back to
    `publish` and `fail` so they could read the stages it had recorded.
    """
    workspace: RunWorkspace = open_workspace(tmp_path / "cusfm", STAGE_NAMES)

    assert workspace.ledger.stages == STAGE_NAMES
    assert workspace.stages_planned == workspace.ledger.stages
    assert workspace.ledger.output_dir == workspace.staging_dir, "`runtime.csv` lands in the staging directory"

    with timed_stage(workspace.ledger, "keyframe_selection"):
        pass
    workspace.publish()
    published: RunState | None = read_run_state(tmp_path / "cusfm")
    assert published is not None
    assert published.stages_completed == ("keyframe_selection",)
    assert published.stages_planned == tuple(STAGE_NAMES)


def test_a_directory_a_live_process_is_writing_is_not_reaped(tmp_path: Path) -> None:
    """Two runs over one destination must not delete each other's staging directory.

    The reaper used to `rmtree` every `<name>.colsfm-staging-*` sibling under
    `ignore_errors=True`, so a second run started while the first was still going
    removed the first's live workspace and said nothing. The directory names the pid
    that owns it, so the question can simply be asked.

    Pid 1 stands in for the live process: it always exists, it is never this test,
    and asking about it is answered by the kernel rather than by a fixture.
    """
    output_dir: Path = tmp_path / "cusfm"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    live: Path = output_dir.with_name(f"{output_dir.name}{STAGING_SUFFIX}-1")
    live.mkdir(parents=True)
    (live / "database.db").write_text("the other run's work\n")
    abandoned: Path = output_dir.with_name(f"{output_dir.name}{STAGING_SUFFIX}-{UNUSED_PID}")
    abandoned.mkdir(parents=True)

    open_workspace(output_dir, STAGE_NAMES)

    assert (live / "database.db").read_text() == "the other run's work\n"
    assert not abandoned.exists(), "a run that is gone leaves nothing behind"


def test_the_next_run_clears_the_last_failure_away(tmp_path: Path) -> None:
    """One failure leaves one directory behind, not a pile of them."""
    output_dir: Path = tmp_path / "cusfm"
    first: RunWorkspace = open_workspace(output_dir, STAGE_NAMES)
    first.fail("boom")
    assert first.staging_dir.exists()

    open_workspace(output_dir, STAGE_NAMES)
    leftovers: list[Path] = sorted(output_dir.parent.glob(f"{output_dir.name}{STAGING_SUFFIX}-*"))
    assert len(leftovers) == 1


def test_a_directory_without_a_state_file_is_not_an_unfinished_run(tmp_path: Path) -> None:
    """A `pycusfm` reference run has no `run_state.json`, and that is not a defect."""
    reference: Path = tmp_path / "blob_run"
    reference.mkdir()
    assert read_run_state(reference) is None
