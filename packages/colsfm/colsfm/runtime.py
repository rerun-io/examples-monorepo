"""Read `runtime.csv` back and fold its rows onto the plan's eight stages.

`colsfm.export` writes the file one row at a time (`append_runtime_record`); this
module is the reader that sits beside it. Two producers write into the same
layout and neither one's command names are the plan's stage names:

- the **blob** logs binary names (`feature_extractor_main`), so those are matched
  by an ordered substring table;
- **`colsfm.pipeline`** logs its own `StageName` verbatim
  (`RuntimeRecord(command=stage, ...)`), a closed Literal, so those are matched by
  an exhaustive dict. A dict rather than more substrings because the substring
  table silently dropped `extrinsic_refinement`: no rule contained it, so the
  stage vanished from the runtime table, from the total, and from the runtime
  acceptance bound.

`runtime.csv` is also cut down to its last run. The blob's runner *appends*, so a
directory reused across runs holds one block of rows per run
(`data/cusfm_runs/galileo/cusfm/runtime.csv` holds three). Summing the file would
report the history rather than the run, so `latest_run_records` keeps only the
rows after the last repeat of a command name.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypeAlias

from colsfm.export import RuntimeRecord, read_runtime_records

Stage: TypeAlias = Literal[
    "extraction",
    "retrieval",
    "association",
    "pose_graph",
    "pair_selection",
    "matching",
    "mapping",
    "export",
]
"""The eight pipeline stages of `docs/open-pipeline-plan.md` §"Stages and their spec owners"."""

STAGES: Final[tuple[Stage, ...]] = (
    "extraction",
    "retrieval",
    "association",
    "pose_graph",
    "pair_selection",
    "matching",
    "mapping",
    "export",
)
"""Stages in pipeline order; also the row order of the runtime table."""

STAGE_BY_PIPELINE_STAGE: Final[dict[str, Stage]] = {
    "keyframe_selection": "extraction",
    "feature_extraction": "extraction",
    "pair_selection": "pair_selection",
    "matching": "matching",
    "loop_closure": "association",
    "pose_graph": "pose_graph",
    "reconstruction": "mapping",
    "extrinsic_refinement": "mapping",
    "export": "export",
}
"""Every `colsfm.pipeline` stage name, mapped onto the plan's row it belongs to.

The keys are `colsfm.pipeline.StageName` members, which is what that module writes
into the `command` column verbatim; they are typed `str` here rather than
`StageName` so that this reader — which the benchmark and the audit tools import
to make sense of a *finished* run — does not have to import the pipeline, and with
it the whole matcher, retrieval and loop-closure stack. `colsfm.pipeline` stays the
source of truth: `test_benchmark` imports both for real and asserts that every
member of `StageName` is a key here, so a new stage cannot be added without a
row. `colsfm` folds two stages
onto `mapping` (`reconstruction` triangulates and bundle-adjusts,
`extrinsic_refinement` is the blob's second `keypoints_mapper_main` pass) and two
onto `extraction` (keyframe selection is part of the blob's extractor binary). It
has no `retrieval` row: BoW runs inside loop closure rather than as a timed stage."""

STAGE_PATTERNS: Final[tuple[tuple[str, Stage], ...]] = (
    # Ordered, first match wins. Order is load-bearing: `extract_pose_from_map_main`
    # contains "extract", `feature_matcher_task_builder_main` contains "matcher",
    # and `pose_graph_association_main` contains "association".
    ("kpmap_to_colmap", "export"),
    ("extract_pose", "export"),
    ("update_keyframe_pose", "export"),
    ("export", "export"),
    ("task_builder", "pair_selection"),
    ("pair", "pair_selection"),
    ("pose_graph", "pose_graph"),
    ("association", "association"),
    ("loop_closure", "association"),
    ("bow", "retrieval"),
    ("vocabulary", "retrieval"),
    ("retrieval", "retrieval"),
    ("matcher", "matching"),
    ("matching", "matching"),
    ("mapper", "mapping"),
    ("mapping", "mapping"),
    ("reconstruct", "mapping"),
    ("triangulat", "mapping"),
    ("bundle", "mapping"),
    ("extractor", "extraction"),
    ("feature", "extraction"),
    ("keyframe", "extraction"),
    ("extract", "extraction"),
)
"""Substring rules for the blob's binary names, first match wins.

Only the blob needs these: its rows are binaries (`feature_extractor_main`) that
no closed vocabulary describes. `colsfm.pipeline`'s own rows go through
`STAGE_BY_PIPELINE_STAGE` instead, and a few `colsfm.<module>` command lines that
older runs wrote still land here by substring."""


def classify_stage(command: str) -> Stage | None:
    """Map one `runtime.csv` command onto a canonical pipeline stage.

    The command's first token is looked up in `STAGE_BY_PIPELINE_STAGE` first, so
    every `colsfm.pipeline` stage is matched exactly; the blob's binary names fall
    through to `STAGE_PATTERNS`.

    Args:
        command: The `command` column, `"<name> <args...>"`.

    Returns:
        The stage, or None when the command matches neither table.
    """
    head: str = command.split()[0].lower() if command.split() else ""
    for stage_name, mapped in STAGE_BY_PIPELINE_STAGE.items():
        if head == stage_name:
            return mapped
    for needle, stage in STAGE_PATTERNS:
        if needle in head:
            return stage
    return None


def latest_run_records(records: Sequence[RuntimeRecord]) -> list[RuntimeRecord]:
    """Keep only the rows belonging to the most recent run in an appended log.

    `pycusfm.command_runner` *appends* to `runtime.csv`, so a workspace reused
    across runs holds one block of rows per run
    (`data/cusfm_runs/galileo/cusfm/runtime.csv` holds three). A run invokes each
    command **once**, so a repeated command name starts a new block.

    Two refinements the shipped logs force:

    - Compare the command *name*, not the whole row. `kpmap_to_colmap` takes no
      config path, so its full row is byte-identical between two runs that
      differ everywhere else; keying on the row would cut the block in the
      middle of a run.
    - A repeat of the *immediately preceding* name is a sharded stage, not a new
      run — several `feature_matcher_main` task files land as consecutive rows.

    Stage order is deliberately not used: `colsfm` runs matching before loop
    closure, so "the stage index went backwards" is not a run boundary there.

    Args:
        records: Rows in file order.

    Returns:
        The rows of the last block, in file order.
    """
    boundary: int = 0
    seen_names: set[str] = set()
    previous_name: str = ""
    for index, record in enumerate(records):
        name: str = record.command.split()[0] if record.command.split() else ""
        if name in seen_names and name != previous_name:
            boundary = index
            seen_names = set()
        seen_names.add(name)
        previous_name = name
    return list(records[boundary:])


def stage_runtime_seconds(records: Sequence[RuntimeRecord]) -> dict[Stage, float]:
    """Total each stage's wall clock over the most recent run in the log.

    Args:
        records: Rows in file order, possibly spanning several appended runs.

    Returns:
        Seconds per stage; stages with no matching row are absent.
    """
    totals: dict[Stage, float] = {}
    for record in latest_run_records(records):
        stage: Stage | None = classify_stage(record.command)
        if stage is None:
            continue
        totals[stage] = totals.get(stage, 0.0) + record.runtime_seconds
    return totals


def read_stage_runtimes(path: Path) -> dict[Stage, float]:
    """Read a `runtime.csv` and total its most recent run per stage.

    Args:
        path: The run's `runtime.csv`; a missing file reports nothing.

    Returns:
        Seconds per stage; empty when the file does not exist.
    """
    if not path.is_file():
        return {}
    records: list[RuntimeRecord] = read_runtime_records(path)
    return stage_runtime_seconds(records)
