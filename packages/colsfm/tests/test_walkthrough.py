"""The stage-by-stage Rerun walkthrough, checked by reading a saved `.rrd` back.

The seam is `colsfm.walkthrough.run_walkthrough`: one config in, one `.rrd` out
plus a cuSFM-layout workspace on disk. Every assertion goes through
`rerun.experimental.RrdReader`, so what is checked is what landed in the file,
not what the logging calls were handed. The expected entity paths are written
out as literals rather than imported from the module under test — importing them
would make each assertion agree with the code by construction.

The run uses the stock `feature_extractor_main` distance gate (0.5 m), which
keeps 34 of Galileo's 226 keyframes: the same smoke-sized selection
`test_pipeline` uses, and enough for every stage to produce real data.

NOTES.md gotcha 9 applies here as it does in `test_rerun_log`: chunk record
batches carry **bare** component names, so column lookups are unprefixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rerun.experimental as rrx
from serde.json import from_json

from colsfm.export import RUNTIME_CSV_NAME, RuntimeRecord, read_runtime_records
from colsfm.frames_meta import FRAMES_META_NAME
from colsfm.run_lifecycle import KEYFRAME_DIR_NAME, LOOP_EDGES_NAME, SUMMARY_NAME
from colsfm.run_report import PipelineSummary
from colsfm.walkthrough import WalkthroughConfig, WalkthroughResult, run_walkthrough

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes."""

EXPECTED_STAGE_NAMES: tuple[str, ...] = (
    "keyframe_selection",
    "feature_extraction",
    "pair_selection",
    "matching",
    "loop_closure",
    "pose_graph",
    "reconstruction",
    "export",
)
"""The eight stages a default (no `--optimize-extrinsics`) walkthrough records."""

EXPECTED_ENTITY_PATHS: tuple[str, ...] = (
    "/walkthrough/stage_01/notes",
    "/walkthrough/stage_01/input/frusta",
    "/walkthrough/stage_01/input/centres",
    "/walkthrough/stage_01/selected/frusta",
    "/walkthrough/stage_01/selected/rig_track",
    "/walkthrough/stage_02/notes",
    "/walkthrough/stage_02/keypoint_counts",
    "/walkthrough/stage_03/notes",
    "/walkthrough/stage_03/centres",
    "/walkthrough/stage_03/pairs/consecutive",
    "/walkthrough/stage_03/pairs/stereo",
    "/walkthrough/stage_04/notes",
    "/walkthrough/stage_04/inlier_histogram",
    "/walkthrough/stage_04/inliers_per_pair",
    "/walkthrough/stage_04/pair/stacked/image",
    "/walkthrough/stage_04/pair/stacked/matches",
    "/walkthrough/stage_04/pair/left/image",
    "/walkthrough/stage_04/pair/left/keypoints",
    "/walkthrough/stage_04/pair/right/image",
    "/walkthrough/stage_04/pair/right/keypoints",
    "/walkthrough/stage_05/notes",
    "/walkthrough/stage_06/notes",
    "/walkthrough/stage_06/before/nodes",
    "/walkthrough/stage_06/before/edges",
    "/walkthrough/stage_06/after/nodes",
    "/walkthrough/stage_06/after/edges",
    "/walkthrough/stage_07/notes",
    "/walkthrough/stage_07/points",
    "/walkthrough/stage_07/rounds/mean_reprojection_error_px",
    "/walkthrough/stage_07/rounds/num_observations",
    "/walkthrough/stage_08/notes",
    "/walkthrough/stage_08/points",
    "/walkthrough/stage_08/report",
)
"""Every entity path a default walkthrough must produce, spelled out."""


@pytest.fixture(scope="module")
def walkthrough(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> WalkthroughResult:
    """Run the whole walkthrough once on the 34-frame Galileo selection.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        What the run produced, including the saved recording's path.
    """
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    root: Path = tmp_path_factory.mktemp("walkthrough")
    config: WalkthroughConfig = WalkthroughConfig(
        input_dir=galileo_input_dir,
        output=root / "galileo_walkthrough.rrd",
        work_dir=root / "cusfm",
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
    )
    return run_walkthrough(config)


@pytest.fixture(scope="module")
def chunks(walkthrough: WalkthroughResult) -> list[rrx.Chunk]:
    """Every chunk of the saved recording."""
    reader: rrx.RrdReader = rrx.RrdReader(str(walkthrough.rrd_path))
    return list(reader.stream(store=reader.recordings()[0]).to_chunks())


def entity_paths(chunks: list[rrx.Chunk]) -> set[str]:
    """Every entity path present in a recording."""
    return {str(chunk.entity_path) for chunk in chunks}


def stage_indices(chunks: list[rrx.Chunk]) -> set[int]:
    """Every value the `stage` timeline takes across the recording."""
    values: set[int] = set()
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if "stage" in batch.schema.names:
            values.update(int(value) for value in batch.column("stage").to_pylist())
    return values


def test_the_run_records_one_stage_per_pipeline_stage(walkthrough: WalkthroughResult) -> None:
    """The walkthrough walks exactly the stages `colsfm.pipeline` runs, in order."""
    assert tuple(stage.name for stage in walkthrough.stages) == EXPECTED_STAGE_NAMES
    assert tuple(stage.index for stage in walkthrough.stages) == tuple(range(1, len(EXPECTED_STAGE_NAMES) + 1))


def test_the_recording_holds_every_stage_entity(chunks: list[rrx.Chunk]) -> None:
    """Each stage logged the spatial, tabular and textual entities its tab needs."""
    missing: list[str] = [path for path in EXPECTED_ENTITY_PATHS if path not in entity_paths(chunks)]
    assert not missing, f"the walkthrough did not log {missing}"


def test_every_stage_has_a_notes_document(chunks: list[rrx.Chunk], walkthrough: WalkthroughResult) -> None:
    """`/walkthrough/stage_NN/notes` exists for every stage that ran."""
    paths: set[str] = entity_paths(chunks)
    missing: list[str] = [f"{stage.entity_root}/notes" for stage in walkthrough.stages if f"{stage.entity_root}/notes" not in paths]
    assert not missing, f"stages without notes: {missing}"


def test_the_stage_timeline_spans_one_to_the_last_stage(chunks: list[rrx.Chunk], walkthrough: WalkthroughResult) -> None:
    """Scrubbing `stage` from 1 to N walks every stage; no index is skipped."""
    indices: set[int] = stage_indices(chunks)
    assert indices == set(range(1, len(walkthrough.stages) + 1))


def test_the_feature_stage_shows_sample_keyframes_with_keypoints(chunks: list[rrx.Chunk]) -> None:
    """At least one sample keyframe carries both an image and its ALIKED keypoints."""
    paths: set[str] = entity_paths(chunks)
    images: set[str] = {path for path in paths if path.startswith("/walkthrough/stage_02/samples/") and path.endswith("/image")}
    assert images, "stage 2 logged no sample imagery"
    for image_path in images:
        assert image_path.removesuffix("/image") + "/keypoints" in paths


def test_the_recording_ships_a_blueprint(walkthrough: WalkthroughResult) -> None:
    """The `.rrd` carries its own blueprint, so it opens laid out by stage."""
    reader: rrx.RrdReader = rrx.RrdReader(str(walkthrough.rrd_path))
    assert reader.blueprints(), "the recording carries no blueprint"


# ─────────────────────────────────────────────────────────────────────────────
# The `--optimize-extrinsics` variant
#
# Its own run rather than a parametrisation: it adds a ninth stage, renumbers
# export, and is the only path that draws `Arrows3D`. Nothing in the eight-stage
# run exercises any of that, and ruff cannot help — the repo ignores F821 so
# jaxtyping's string annotations parse, which also hides an undefined name in a
# branch no test enters.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def extrinsics_walkthrough(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> WalkthroughResult:
    """The same walkthrough with the rig extrinsics free.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        What the run produced.
    """
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    root: Path = tmp_path_factory.mktemp("walkthrough_extrinsics")
    config: WalkthroughConfig = WalkthroughConfig(
        input_dir=galileo_input_dir,
        output=root / "galileo_walkthrough_extrinsics.rrd",
        work_dir=root / "cusfm",
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
        optimize_extrinsics=True,
    )
    return run_walkthrough(config)


def test_the_extrinsics_run_inserts_stage_7b_and_moves_export_last(extrinsics_walkthrough: WalkthroughResult) -> None:
    """Nine stages, with `extrinsic_refinement` between reconstruction and export."""
    names: tuple[str, ...] = tuple(stage.name for stage in extrinsics_walkthrough.stages)
    assert names == (*EXPECTED_STAGE_NAMES[:-1], "extrinsic_refinement", "export")
    assert tuple(stage.index for stage in extrinsics_walkthrough.stages) == tuple(range(1, len(names) + 1))


def test_the_extrinsics_run_logs_the_per_camera_shift_arrows(extrinsics_walkthrough: WalkthroughResult) -> None:
    """Stage 7b draws where each camera moved, and export moves to `stage_09`."""
    reader: rrx.RrdReader = rrx.RrdReader(str(extrinsics_walkthrough.rrd_path))
    paths: set[str] = entity_paths(list(reader.stream(store=reader.recordings()[0]).to_chunks()))
    expected: tuple[str, ...] = (
        "/walkthrough/stage_08/notes",
        "/walkthrough/stage_08/extrinsics/before",
        "/walkthrough/stage_08/extrinsics/after",
        "/walkthrough/stage_08/extrinsics/shift",
        "/walkthrough/stage_09/notes",
        "/walkthrough/stage_09/points",
        "/walkthrough/stage_09/report",
    )
    assert not [path for path in expected if path not in paths]


def test_the_two_variants_do_not_share_a_recording_id(
    walkthrough: WalkthroughResult, extrinsics_walkthrough: WalkthroughResult
) -> None:
    """Two recordings with one id are one store to the viewer, and would merge."""
    default_id: str = rrx.RrdReader(str(walkthrough.rrd_path)).recordings()[0].recording_id
    extrinsics_id: str = rrx.RrdReader(str(extrinsics_walkthrough.rrd_path)).recordings()[0].recording_id
    assert default_id != extrinsics_id


def test_the_workspace_holds_a_readable_cusfm_run(walkthrough: WalkthroughResult) -> None:
    """The walkthrough leaves *every* artifact a `run_pipeline` run leaves.

    `summary.json` is in the list because it is what the old split runner never
    wrote: the walkthrough sequenced the stages itself and skipped the summary,
    and this test excluded it, so the two executors could drift unnoticed. The
    walkthrough is now an observer of `run_pipeline`, so the full set applies.
    """
    expected: list[Path] = [
        walkthrough.work_dir / "database.db",
        walkthrough.work_dir / KEYFRAME_DIR_NAME / FRAMES_META_NAME,
        walkthrough.work_dir / "pose_graph" / FRAMES_META_NAME,
        walkthrough.work_dir / "pose_graph" / "vehicle_pose.tum",
        walkthrough.work_dir / "pose_graph" / LOOP_EDGES_NAME,
        walkthrough.work_dir / "sparse" / "images.txt",
        walkthrough.work_dir / "sparse" / "cameras.txt",
        walkthrough.work_dir / "sparse" / "points3D.txt",
        walkthrough.work_dir / "kpmap" / "keyframes" / FRAMES_META_NAME,
        walkthrough.work_dir / RUNTIME_CSV_NAME,
        walkthrough.work_dir / SUMMARY_NAME,
    ]
    missing: list[str] = [str(path) for path in expected if not path.exists()]
    assert not missing, f"the walkthrough did not write {missing}"


def test_the_workspace_summary_reports_the_stages_the_recording_shows(walkthrough: WalkthroughResult) -> None:
    """`summary.json` and the recording agree on which stages ran and how long they took.

    The equivalence check the split runner could not have: one ledger feeds
    `runtime.csv`, `summary.json` and the observer's own timings.
    """
    summary: PipelineSummary = from_json(
        PipelineSummary, (walkthrough.work_dir / SUMMARY_NAME).read_text()
    )
    assert tuple(summary.stage_seconds) == EXPECTED_STAGE_NAMES
    assert summary.stage_seconds == walkthrough.seconds_by_stage
    rows: list[RuntimeRecord] = read_runtime_records(walkthrough.work_dir / RUNTIME_CSV_NAME)
    assert tuple(row.command for row in rows) == EXPECTED_STAGE_NAMES
