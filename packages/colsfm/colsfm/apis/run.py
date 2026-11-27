"""The colsfm CLI: `run` (a cuSFM input folder), `catalog` (a Rerun catalog segment) and `stage` (one cheap stage).

`run` and `catalog` with no other flag run the fast full pipeline: RaCo-ALIKED features, FlexGlue matching (both on
TensorRT), loop closure, CASPAR bundle adjustment (no closing Ceres solve; `--caspar-ceres-polish` adds one), and the regularised rig-extrinsic refinement. Every one
of those is a flag with a `--no-` or a `pycolmap`/`ceres` spelling (`colsfm.run_config.PipelineOptions`). The RaCo graphs
come from `pixi run -e colsfm colsfm-raco-export` and FlexGlue's from `colsfm-download-flexglue`; a missing graph stops the
run and names those tasks.

`catalog` reads the rig, the video and the initial trajectory from a catalog segment (`colsfm.catalog`), runs the
pipeline on the decoded keyframes, and writes the run as a `/world/runs/colsfm` layer of that segment
(`colsfm.rerun_layer`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TypeAlias

import tyro
from simplecv.run_record import capture_run_record

from colsfm import DATA_DIR
from colsfm.catalog import CatalogInputConfig, PreparedInput, prepare_catalog_input, serve_catalog_input
from colsfm.frame_manifest import FRAME_SOURCE_FILE, read_manifest
from colsfm.pipeline import StageResult, run_pipeline, run_stage
from colsfm.rerun_layer import DEFAULT_SOURCE, LayerSummary, write_run_layer
from colsfm.run_config import PipelineOptions, SelectionOptions
from colsfm.run_lifecycle import SUMMARY_NAME, CheapStageName
from colsfm.run_report import PipelineSummary

UNSET_DIR: Path = Path("<from-catalog>")
"""Placeholder for the two directories `catalog` fills in itself."""


@dataclass(frozen=True, slots=True)
class RunCommand:
    """Run all eight stages over one cuSFM input folder (`frames_meta.json` plus images)."""

    options: tyro.conf.OmitArgPrefixes[PipelineOptions]
    """Input, output, config directory and the per-stage knobs."""

    def execute(self) -> None:
        """Run the pipeline and print where the summary landed."""
        summary: PipelineSummary = run_pipeline(self.options)
        print(f"[colsfm] summary written to {summary.options.output_dir / SUMMARY_NAME}")


@dataclass(frozen=True)
class CatalogCommand:
    """Read a catalog segment, run all eight stages on it, and write the run as a layer of that segment."""

    source: CatalogInputConfig
    """Segment (on a catalog, or a local base rrd plus layer rrds), cameras, initial trajectory and sampling."""
    options: PipelineOptions = field(default_factory=lambda: PipelineOptions(input_dir=UNSET_DIR, output_dir=UNSET_DIR))
    """The per-stage knobs; `input_dir` and `output_dir` are set from the segment unless given."""
    layer_path: Path | None = None
    """Where the layer `.rrd` goes; `packages/colsfm/data/rrd/<segment>.colsfm.rrd` by default."""
    view_rig: bool = True
    """Also log `rig_view` (`colsfm.rerun_layer.log_view_rig`) when the segment has viewing faces (`pinhole/video_view`)."""

    def execute(self) -> None:
        """Prepare the input, run the pipeline and write the layer (a local input is served for the whole run)."""
        started: datetime = datetime.now(UTC)
        with serve_catalog_input(self.source) as source:
            self._run(source, started)

    def _run(self, source: CatalogInputConfig, started: datetime) -> None:
        prepared: PreparedInput = prepare_catalog_input(source, view_rig=self.view_rig)
        output_dir: Path = (
            DATA_DIR / "runs" / prepared.input_dir.name if self.options.output_dir == UNSET_DIR else self.options.output_dir
        )
        options: PipelineOptions = dataclasses.replace(self.options, input_dir=prepared.input_dir, output_dir=output_dir)
        summary: PipelineSummary = run_pipeline(options)
        print(f"[colsfm] summary written to {summary.options.output_dir / SUMMARY_NAME}")
        layer_path: Path = self.layer_path or DATA_DIR / "rrd" / f"{prepared.input_dir.name}.colsfm.rrd"
        layer: LayerSummary = write_run_layer(
            output_dir,
            layer_path,
            segment=source.segment,
            timeline=source.timeline,
            frames=read_manifest(prepared.input_dir / FRAME_SOURCE_FILE),
            view=prepared.view,
            record=capture_run_record(DEFAULT_SOURCE, source.segment, config=self, started=started),
        )
        print(
            f"[colsfm] layer {layer.path}: {layer.num_rig_poses} rig poses, {layer.num_points} points, "
            f"{layer.metrics.registered_images} registered images, {layer.metrics.mean_reprojection_error_px:.3f} px"
        )


@dataclass(frozen=True, slots=True)
class StageCommand:
    """Run one metadata-only stage, without touching images or a GPU."""

    options: tyro.conf.OmitArgPrefixes[SelectionOptions]
    """The input directory, the config profile and the three selection gates."""
    stage: CheapStageName = "keyframe_selection"
    """Which stage to run on its own."""

    def execute(self) -> None:
        """Run the stage and print its counts."""
        result: StageResult = run_stage(self.options, self.stage)
        print(f"[colsfm] {result.stage}: {result.num_selected_keyframes} keyframes, {result.num_pairs} pairs")


Command: TypeAlias = (
    Annotated[RunCommand, tyro.conf.subcommand("run", description="Run the whole pipeline on a cuSFM input folder")]
    | Annotated[CatalogCommand, tyro.conf.subcommand("catalog", description="Run the whole pipeline on a catalog segment")]
    | Annotated[StageCommand, tyro.conf.subcommand("stage", description="Run one metadata-only stage")]
)
"""The subcommands the CLI offers."""


def main(command: Command) -> None:
    """Run the chosen subcommand.

    Args:
        command: The parsed subcommand.
    """
    command.execute()
