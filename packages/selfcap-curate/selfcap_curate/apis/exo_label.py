"""One-session bounded optical labeling CLI; never registers catalog data."""

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient
from serde.json import from_json, to_json

from selfcap_curate import catalog_io
from selfcap_curate.calib_layer import CalibSession
from selfcap_curate.calibration_objective import ObjectiveShares
from selfcap_curate.exo_label import GateThresholds, LabelFocalSource, label_session, run_session


@dataclass
class Config:
    segment: str
    """One explicit ep001 segment ID."""
    calib_record: Path
    """Initial unrefined CalibSession JSON."""
    output_dir: Path
    """Persistent packet, labels, gate and accepted refinement directory."""
    dataset: str = 'selfcap-post-dev'
    """Read-only dataset carrying base, initial calibration, Sapiens and RTMW."""
    catalog_url: str = 'rerun+http://dgx-spark:51235'
    """Read-only catalog URL."""
    timeout_s: float = 360.0
    """Hard wall-clock deadline including all child Sol calls."""
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    """Explicit optical and independent hand acceptance thresholds."""
    focal_source: LabelFocalSource = "device-vp"
    """Fixed device VP focal, or the initial calibration for unknown devices; never fit focal."""
    objective: ObjectiveShares = field(default_factory=ObjectiveShares)
    """Observation-count invariant headset, phone, hand and body loss shares."""
    cached_labels: Path | None = None
    """Use these existing landmarks only; never build crops or invoke Sol."""
    rtmw_rrd: Path | None = None
    """Explicit original S1 RTMW recording; required for cached redecisions."""
    worker: bool = False
    """Internal isolated process entry point."""

    def __post_init__(self) -> None:
        if self.cached_labels is not None and self.rtmw_rrd is None:
            raise ValueError('cached_labels requires an original S1 rtmw_rrd')


def main(config: Config) -> None:
    """Run one isolated session and retain the initial calibration on failure."""
    if config.worker:
        segment = catalog_io.open_segment(CatalogClient(config.catalog_url).get_dataset(name=config.dataset), config.segment)
        state = from_json(CalibSession, config.calib_record.read_text())
        result = run_session(segment, state, config.output_dir, thresholds=config.thresholds, focal_source=config.focal_source,
                             objective=config.objective, cached_labels=config.cached_labels, rtmw_rrd=config.rtmw_rrd)
    else:
        result = label_session(config.segment, config.calib_record, config.output_dir, dataset=config.dataset,
                               catalog_url=config.catalog_url, timeout_s=config.timeout_s, thresholds=config.thresholds, focal_source=config.focal_source,
                               objective=config.objective, cached_labels=config.cached_labels, rtmw_rrd=config.rtmw_rrd)
    print(to_json(result), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Config))
