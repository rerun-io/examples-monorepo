"""Replay a RoboCap segment from the catalog through the VIO graph and draw it as ``replay --stage vio`` does (the world in 3D,
the cameras, eight plots); then write the trajectory and report its agreement with the session's regression trajectory. RoboCap
has no ground truth: the agreement is reported, never gated. No layer is written; the "pipeline (debug)" tab shows the graph."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app, with_pipeline
from dagkit.doc import HEALTH_ROOT
from dagkit.graph import Graph
from dagkit.node import Mode
from dagkit.runners import RunReport
from dagkit.sources import SegmentRef
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs.catalog_feed import CameraCalib
from slam_rs.config import load_slam_config
from slam_rs.graphs.vio import VioSource, vio_graph
from slam_rs.nodes.viewer import VioViewer
from slam_rs.nodes.vio import VioNode, VioParams
from slam_rs.reference import load_benchmarks
from slam_rs.trajectory import ate, coverage, read_trajectory, write_trajectory
from slam_rs.types import Trajectory
from slam_rs.vio_log import vio_blueprint


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef = SegmentRef("robocap", "robocap__f408193e6447b3b0__s00000015")
    """A RoboCap segment (default: session 15, the regression session); a window ends the replay at its stop."""
    vio: VioParams = VioParams(backend="cpu", decode_device="cpu")
    reference_csv: Path | None = None
    """Regression trajectory (device clock); default: the session's in benchmarks.toml, if it lists the segment."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written (device clock); default ``data/<segment>/slam_rs.csv``."""
    log_frames: bool = True
    """Log the camera images; off measures the estimator without the JPEG encode."""
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def blueprint(cameras: tuple[CameraCalib, ...]) -> rrb.Blueprint:
    """``replay --stage vio``'s blueprint (the world, the cameras, eight plots), then the "pipeline (debug)" tab."""
    return with_pipeline(vio_blueprint(cameras), [rrb.TimeSeriesView(origin=HEALTH_ROOT + "vio", name="2 vio · step ms and queue")])


def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
    """The VIO graph without a layer, its tracked framesets drawn; once it ran, the CSV and the agreement."""
    vio = VioNode(config.segment, config.vio, None, views=True)
    viewer = VioViewer(recording, blueprint, "jpeg" if config.log_frames else "off")
    yield vio_graph(VioSource(config.segment, config.vio), vio, {"viewer": viewer})
    assert vio.run is not None, "the graph ran without a flush"
    estimate: Trajectory = vio.run.estimate  # device clock
    output_csv: Path = config.output_csv or Path("data") / config.segment.segment / "slam_rs.csv"
    write_trajectory(output_csv, estimate)
    print(f"{len(estimate)} tracked poses -> {output_csv} (absolute device clock)")
    settings = load_slam_config()
    listed: str | None = next((str(s.reference_csv) for s in load_benchmarks(settings).robocap.sessions if s.segment_id == config.segment.segment and s.reference_csv), None)
    reference_csv: Path | None = config.reference_csv or (settings.package_root / listed if listed else None)
    if len(estimate) == 0 or reference_csv is None:
        print("ground truth absent and no regression trajectory: not scored")
        return
    reference: Trajectory = read_trajectory(reference_csv)
    print(f"vs regression reference, {coverage(reference, estimate):.1%} of its span covered (not gated)")
    print(ate(estimate, reference).summary())
    print(f"first estimated pose at {int(estimate.t_ns[0])} ns, position {estimate.position_m[0].round(4).tolist()}")


def main(config: Config) -> RunReport:
    return run_app(config, build)
