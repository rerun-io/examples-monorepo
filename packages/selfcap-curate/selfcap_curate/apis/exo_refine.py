"""Fit and report a joint optical calibration from a named catalog dataset."""

from dataclasses import dataclass, replace
from pathlib import Path

from rerun.catalog import CatalogClient
from serde.json import from_json, to_json

from selfcap_curate import catalog_io
from selfcap_curate.apis import session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.exo_landmarks import read_landmarks
from selfcap_curate.exo_refine import FocalSource
from selfcap_curate.exo_refine_catalog import evaluate, load_inputs, read_focals
from selfcap_curate.quest_fit import read_session_episode


@dataclass
class Config:
    segment: str
    """Explicit segment id with an exo_landmarks layer."""
    output_dir: Path
    """New output directory for the model, gate and optional calibration layer."""
    dataset: str = "selfcap-post-dev-exofit"
    """Read-only source dataset."""
    landmark_dataset: str | None = None
    """Optional separate catalog dataset carrying the same segment's label layer."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog endpoint."""
    focal_source: FocalSource = "fit"
    """Table, fitted, or fixed focal source."""
    focal_file: Path | None = None
    """Typed fixed-focal JSON record."""
    calib_record: Path | None = None
    """Existing CalibSession JSON; when supplied, also write its refined calib RRD."""


def main(config: Config) -> None:
    """Read catalog inputs, evaluate holdouts, and write a replayable candidate."""
    dataset = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
    segment = catalog_io.open_segment(dataset, config.segment)
    labels = (None if config.landmark_dataset is None else
              read_landmarks(CatalogClient(config.catalog_url).get_dataset(name=config.landmark_dataset), config.segment))
    inputs = load_inputs(segment, landmarks=labels)
    fixed, digest = read_focals(config.focal_file, inputs.phones)
    with session_output(config.output_dir) as output:
        model, gate = evaluate(inputs, focal_source=config.focal_source, fixed_focals=fixed, focal_record=digest)
        (output / "refinement.json").write_text(to_json(model))
        (output / "gate.json").write_text(to_json(gate))
        if config.calib_record is not None:
            state = from_json(CalibSession, config.calib_record.read_text())
            state = replace(state, exo_refinements={**state.exo_refinements, segment.id: model})
            motion = read_session_episode(segment, inputs.sync)
            write_calib_layer(output / f"{segment.id}.rrd", state, motion)
            (output / "calib.json").write_text(to_json(state))
        for metric, candidate in gate.candidate.items():
            baseline = gate.baseline[metric]
            print(metric, baseline.median, baseline.p90, "->", candidate.median, candidate.p90, flush=True)
        print("GATE", "pass" if gate.passes else "FAIL: candidate only", flush=True)
