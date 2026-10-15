"""Graph-tier check of slam-rs's catalog VIO, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

Both blocks share ``run``. ``vio`` (robocap, no ground truth): ``score`` compares a golden with the same case's ``pre``
golden (``pre`` itself scores 0, ``pre2`` gives the run-to-run noise). ``vio-gt`` (msd-index): ``score_gt`` is the slam-rs
gate's ATE against the catalog's ``gt`` layer. Both read only golden rrds (plus GT), never the pipeline.

Golden contract (on the segment's ``video_time``): the ``slam_rs`` layer exactly as :func:`~slam_rs.catalog_layer.write_layer`
writes it, plus ``/verify/vio/run``: static ``AnyValues`` with the run's counts.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import rerun as rr
from dagkit.catalog import served
from dagkit.runners import RunReport, run_sequential
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase, pre_golden, run_counts
from rerun.catalog import DatasetEntry

from slam_rs.apis.catalog_layer import vio_publish
from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE, CatalogSegment, read_ground_truth, resolve_catalog_segments, rig_trajectory
from slam_rs.config import load_slam_config
from slam_rs.nodes.vio import VioNode, VioParams
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, ate

RUN: str = "/verify/vio/run"
"""Static ``AnyValues`` of a golden: the run's counts, settings and wall time."""


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The VIO graph (sequential, lossless) on the case's segment, its layer (unregistered, in a temporary directory) copied into
    ``recording``. VIO initialises from rest, so it starts at the segment's first frameset; a window ends it at ``window_s[1]``."""
    segment: SegmentRef = case.segment_ref(load_slam_config().catalog_url)
    params: VioParams = case.configure(VioParams)
    with TemporaryDirectory() as tmp:
        graph, publish = vio_publish(segment, params, Path(tmp), "off", rr.RecordingStream("verify-vio", default_enabled=False))
        report: RunReport = run_sequential(graph)
        assert publish.published is not None, f"{case.name}: the run published no layer ({report.status})"
        recording.send_chunks(rr.chunk.RrdReader(publish.published).stream())
    result: SegmentRun | None = cast(VioNode, graph.nodes["vio"].obj).run
    assert result is not None
    recording.log(RUN, rr.AnyValues(framesets=result.framesets, poses=len(result.estimate), lost=result.lost, wall_s=result.wall_s,
                                    median_tracker_ms=result.median_tracker_ms, config_sha256=result.config_sha256), static=True)  # fmt: skip
    print(f"{case.name} {phase}: {result.framesets} framesets, {len(result.estimate)} poses, {result.lost} lost, {result.wall_s:.1f} s")


def _read(golden: Path) -> tuple[Trajectory, float]:
    """A golden's rig poses on ``video_time`` and its run's lost framesets."""
    with served(golden) as dataset:
        poses: Trajectory = rig_trajectory(dataset.filter_contents([RIG_ENTITY]).reader(index=TIMELINE).to_arrow_table(), golden.name)
        return poses, run_counts(dataset, "vio")["lost"]


def score(case: Case, golden: Path) -> dict[str, float]:
    """Compare a golden's poses with the case's ``pre`` golden (pre-port reference; ``pre`` itself scores 0)."""
    reference: Path = pre_golden(case, golden)
    ours: Trajectory = _read(golden)[0]
    theirs: Trajectory = ours if reference == golden else _read(reference)[0]
    common, i_ours, i_ref = np.intersect1d(ours.t_ns, theirs.t_ns, assume_unique=True, return_indices=True)
    return {
        "ate_vs_pre_cm": 100.0 * ate(ours, theirs).rmse_m,
        "max_pos_vs_pre_cm": 100.0 * float(np.linalg.norm(ours.position_m[i_ours] - theirs.position_m[i_ref], axis=1).max())
        if common.size
        else float("nan"),
        "poses_delta_pct": 100.0 * (len(ours) + len(theirs) - 2 * common.size) / len(theirs),
        "pose_count": float(len(ours)),
    }


def score_gt(case: Case, golden: Path) -> dict[str, float]:
    """The slam-rs gate's ATE of a golden against the segment's ``gt`` layer, and the framesets that produced no pose."""
    segment: SegmentRef = case.segment_ref(load_slam_config().catalog_url)
    ours, lost = _read(golden)
    source: CatalogSegment = resolve_catalog_segments(
        (CatalogSegment(segment.catalog_url, segment.dataset, segment.segment),), require_ground_truth=True
    )[0]
    assert source.ground_truth_uri is not None
    with rr.server.Server(host="127.0.0.1") as server:
        truth: DatasetEntry = server.client().create_dataset("gt")
        truth.register([source.ground_truth_uri]).wait()
        gt: Trajectory = read_ground_truth(truth, segment.segment, int(ours.t_ns[0]), int(ours.t_ns[-1]))
    return {"gt_ate_cm": 100.0 * ate(ours, gt).rmse_m, "lost_framesets": lost, "pose_count": float(len(ours))}
