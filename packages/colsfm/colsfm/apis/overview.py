"""Add the Overview tab to a catalog walkthrough recording and write the combined `.rrd` (the viewer page's file).

Inputs: the walkthrough recording of a catalog run (`tools/walkthrough.py` on a `colsfm.catalog` input directory) and
its run directory. The overview is logged into a second file with the walkthrough's own application and recording id,
with one blueprint (Overview first, then the stage tabs); the walkthrough's data, without its own blueprint, and that file
are merged with `rerun rrd merge`, so the result carries exactly one blueprint.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.chunk import RrdReader

from colsfm.benchmark import read_run
from colsfm.catalog import CatalogInputConfig, read_catalog_cameras, read_catalog_trajectory, serve_catalog_input
from colsfm.overview import OverviewNumbers, log_overview, notes_markdown, overview_eye, overview_tab
from colsfm.walkthrough import APPLICATION_ID
from colsfm.walkthrough_blueprint import stage_tab
from colsfm.walkthrough_stages import SamplePane, walkthrough_stages


@dataclass
class Config:
    """The catalog run to summarise and where the combined recording goes."""

    source: CatalogInputConfig
    """The same source the run read (dataset, segment, cameras, trajectory entity, ground-truth entity)."""
    run_dir: Path
    """The run's output directory (the walkthrough's `--work-dir`)."""
    walkthrough_rrd: Path
    """The walkthrough recording."""
    output: Path
    """The combined recording."""
    recording_id: str
    """The walkthrough's recording id (`<dataset>_walkthrough[_extrinsics]`)."""
    optimize_extrinsics: bool = True
    """Whether the run made stage 7b (the stage tabs follow the run)."""
    overlay: Path = field(default_factory=lambda: Path("/tmp/colsfm-overview-overlay.rrd"))
    """Scratch file for the overview half before the merge."""


def sample_panes(walkthrough_rrd: Path) -> list[SamplePane]:
    """The feature-extraction sample panes the walkthrough logged, read back from its entity paths.

    Args:
        walkthrough_rrd: The walkthrough recording.

    Returns:
        One pane per `<stage>/samples/<name>` entity.
    """
    names: set[str] = set()
    for chunk in RrdReader(str(walkthrough_rrd)).store().stream():
        path: str = str(chunk.entity_path)
        if "/samples/" in path and path.endswith("/image"):
            names.add(path.removesuffix("/image"))
    return [SamplePane(entity_path=path, name=path.rsplit("/", 1)[-1]) for path in sorted(names)]


def main(config: Config) -> OverviewNumbers:
    """Write the combined recording and print the numbers table (a local input is served while it reads).

    Args:
        config: The run and the destination.

    Returns:
        The overview's numbers.
    """
    with serve_catalog_input(config.source) as source:
        return _overview(dataclasses.replace(config, source=source))


def _overview(config: Config) -> OverviewNumbers:
    dataset: DatasetEntry = CatalogClient(config.source.catalog_url).get_dataset(name=config.source.dataset)
    if config.source.ground_truth_entity is None:
        raise ValueError("--source.ground-truth-entity is required: the overview scores against ground truth")
    cameras = read_catalog_cameras(dataset, config.source.segment, config.source.cameras, config.source.calibration_entity)
    start = read_catalog_trajectory(dataset, config.source.segment, config.source.trajectory_entity, config.source.timeline)
    ground_truth = read_catalog_trajectory(dataset, config.source.segment, config.source.ground_truth_entity, config.source.timeline)
    recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=config.recording_id, send_properties=False)
    recording.save(config.overlay)
    numbers: OverviewNumbers = log_overview(
        dataset=dataset,
        segment=config.source.segment,
        cameras=cameras,
        artifacts=read_run(config.run_dir, "colsfm"),
        input_times_ns=start.t_ns,
        input_poses=start.at(start.t_ns),
        ground_truth=ground_truth,
        timeline=config.source.timeline,
        recording=recording,
    )
    stages = walkthrough_stages(config.optimize_extrinsics)
    panes: list[SamplePane] = sample_panes(config.walkthrough_rrd)
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Tabs(overview_tab([camera.name for camera in cameras], overview_eye(ground_truth)), *[stage_tab(stage, panes) for stage in stages], active_tab=0),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.TimePanel(state="expanded", timeline=config.source.timeline),
        ),
        make_active=True,
        make_default=True,
    )
    recording.flush()
    # The walkthrough's data without its own blueprint: with two blueprints in the merged file, which one the viewer
    # activates depends on merge order (one page recording opened on stage 1 with no Overview tab).
    walkthrough_data: Path = config.overlay.with_name(f"{config.overlay.stem}-walkthrough-data.rrd")
    RrdReader(str(config.walkthrough_rrd)).store().stream().write_rrd(walkthrough_data, application_id=APPLICATION_ID, recording_id=config.recording_id)
    rerun_cli: str = shutil.which("rerun") or "rerun"
    config.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([rerun_cli, "rrd", "merge", str(walkthrough_data), str(config.overlay), "-o", str(config.output)], check=True)
    print(notes_markdown(config.source.segment, numbers))
    print(f"[colsfm] wrote {config.output} ({config.output.stat().st_size / 2**20:.1f} MiB)")
    return numbers
