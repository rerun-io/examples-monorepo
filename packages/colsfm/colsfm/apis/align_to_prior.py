"""Align a FINISHED run to its pose prior (`colsfm.prior_alignment`) and write it as a new run directory with its layer.

For runs made without `--options.align-to-prior`: the new directory gets the aligned `sparse/`, `kpmap/`, TUM files and
`summary.json` (`prior_alignment`), a link to the original `database.db`, and a fresh `colsfm.rrd` layer when the run's
input is a catalog snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from serde.json import from_json

from colsfm.frame_manifest import FRAME_SOURCE_FILE, read_manifest
from colsfm.prior_alignment import align_run_directory
from colsfm.rerun_layer import write_run_layer
from colsfm.run_lifecycle import SUMMARY_NAME
from colsfm.run_report import PipelineSummary


@dataclass(frozen=True)
class Config:
    """Which run, where the aligned copy goes."""

    run_dir: Path
    """The finished run directory."""
    output_dir: Path
    """The new, aligned run directory (must not exist)."""
    prior_frames_meta: Path | None = None
    """The prior's `frames_meta.json`; the run's input snapshot (summary.json options.input_dir) when None."""


def main(config: Config) -> None:
    """Align, write, and rewrite the layer.

    Args:
        config: Run and destination.
    """
    align_run_directory(config.run_dir, config.output_dir, config.prior_frames_meta)
    input_dir: Path = from_json(PipelineSummary, (config.run_dir / SUMMARY_NAME).read_text()).options.input_dir
    if (input_dir / FRAME_SOURCE_FILE).is_file():
        manifest = read_manifest(input_dir / FRAME_SOURCE_FILE)
        layer = write_run_layer(config.output_dir, config.output_dir / "colsfm.rrd", segment=manifest.segment, timeline=manifest.timeline, frames=manifest)
        print(f"[colsfm] layer {layer.path}: {layer.num_rig_poses} rig poses, {layer.num_points} points")
