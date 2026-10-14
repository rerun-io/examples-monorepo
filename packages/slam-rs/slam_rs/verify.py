"""Graph-tier check of slam-rs's catalog VIO, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

Both blocks share ``run``. ``vio`` (robocap, no ground truth): ``score`` compares a golden with the same case's ``pre``
golden (``pre`` itself scores 0, ``pre2`` gives the run-to-run noise). ``vio-gt`` (msd-index): ``score_gt`` is the slam-rs
gate's ATE against the catalog's ``gt`` layer. Both read only golden rrds (plus GT), never the pipeline.

Golden contract (the ported graph writes the same entities on the segment's ``video_time``): the ``slam_rs`` layer exactly as
:func:`~slam_rs.catalog_layer.write_layer` writes it, plus ``/verify/vio/run``: static ``AnyValues`` with the run's counts.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.verify import Case, Phase
from rerun.catalog import DatasetEntry

from slam_rs import _core
from slam_rs.catalog_feed import (
    RIG_ENTITY,
    ROBOCAP_RIG,
    TIMELINE,
    CatalogSegment,
    RigProfile,
    SegmentFeed,
    _read_ground_truth,
    _rig_trajectory,
    open_segment,
    resolve_catalog_segments,
)
from slam_rs.catalog_layer import write_layer
from slam_rs.config import SlamConfig, config_text_sha256, load_slam_config
from slam_rs.tracking import Lockstep, SegmentRun, drive
from slam_rs.trajectory import Trajectory, ate

RUN: str = "/verify/vio/run"
"""Static ``AnyValues`` of a golden: the run's counts, settings and wall time."""


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """Today's catalog-layer tool (``apis/catalog_layer.py``) without its register step; its layer file goes into ``recording``.

    VIO initialises from rest, so it starts at the segment's first frameset; a window ends the run at ``window_s[1]``.
    """
    assert case.dataset is not None and case.segment is not None, "slam-rs cases are catalog segments"
    settings: SlamConfig = load_slam_config()
    robocap: bool = case.dataset == "robocap"
    segment: CatalogSegment = resolve_catalog_segments((CatalogSegment(settings.catalog_url, case.dataset, case.segment),))[0]
    assert case.params["profile"] in ("fast", "reference") and case.params["decode_device"] in ("cpu", "cuda"), f"{case.name}: {case.params}"
    profile: Literal["fast", "reference"] = "fast" if case.params["profile"] == "fast" else "reference"
    decoder: Literal["cpu", "cuda"] = "cpu" if case.params["decode_device"] == "cpu" else "cuda"
    feed: SegmentFeed
    with (
        open_segment(
            segment, profile=ROBOCAP_RIG if robocap else RigProfile(), cache_video=True, decode_device=decoder, include_ground_truth=False
        ) as feed,
        TemporaryDirectory() as tmp,
    ):
        config_text: str = settings.robocap_config_text(profile) if robocap else settings.vio_config_text(case.dataset, profile)
        vio = _core.Vio(
            _core.Calibration.from_catalog(feed.cameras, feed.imu), _core.VioConfig.from_json(config_text), gpu=case.params["backend"] == "gpu"
        )
        stop_ns: int | None = None if case.window_s is None else round(case.window_s[1] * 1e9) + feed.imu.cam_time_offset_ns  # feed clock
        result: SegmentRun = drive(feed, Lockstep(vio), stop_ns, config_sha256=config_text_sha256(config_text))
        backend: str = str(_core.gpu_backend) if vio.gpu else "cpu"
        layer: Path = Path(tmp) / f"{case.segment}.rrd"
        write_layer(layer, case.segment, result, clock_offset_ns=feed.export_offset_ns + feed.imu.cam_time_offset_ns,
                    profile=profile, backend=backend, decoder=decoder)  # fmt: skip
        recording.send_chunks(rr.chunk.RrdReader(layer).stream())
    recording.log(RUN, rr.AnyValues(framesets=result.framesets, poses=len(result.estimate), lost=result.lost, wall_s=result.wall_s,
                                    median_tracker_ms=result.median_tracker_ms, backend=backend, decoder=decoder,
                                    config_sha256=result.config_sha256), static=True)  # fmt: skip
    print(f"{case.name} {phase}: {result.framesets} framesets, {len(result.estimate)} poses, {result.lost} lost, {result.wall_s:.1f} s ({backend})")


def _read(golden: Path) -> tuple[Trajectory, float]:
    """A golden's rig poses on ``video_time`` and its run's lost framesets."""
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        poses: Trajectory = _rig_trajectory(dataset.filter_contents([RIG_ENTITY]).reader(index=TIMELINE).to_arrow_table(), golden.name)
        statics: pa.Table = dataset.filter_contents([RUN]).reader(index=None).to_arrow_table()
    return poses, float(np.asarray(statics[f"{RUN}:lost"].to_pylist()[0]).reshape(-1)[0])


def score(case: Case, golden: Path) -> dict[str, float]:
    """Compare a golden's poses with the case's ``pre`` golden (pre-port reference; ``pre`` itself scores 0)."""
    reference: Path = golden if golden.with_suffix("").name == f"{case.name}.pre" else golden.with_name(f"{case.name}.pre.rrd")
    if not reference.exists():
        raise FileNotFoundError(f"{reference}: the pre-port reference golden of case {case.name!r} is missing (run --phase pre first)")
    (ours, _), (theirs, _) = _read(golden), _read(reference)
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
    assert case.dataset is not None and case.segment is not None, "slam-rs cases are catalog segments"
    ours, lost = _read(golden)
    source: CatalogSegment = resolve_catalog_segments(
        (CatalogSegment(load_slam_config().catalog_url, case.dataset, case.segment),), require_ground_truth=True
    )[0]
    assert source.ground_truth_uri is not None
    with rr.server.Server(host="127.0.0.1") as server:
        truth: DatasetEntry = server.client().create_dataset("gt")
        truth.register([source.ground_truth_uri]).wait()
        gt: Trajectory = _read_ground_truth(truth, case.segment, int(ours.t_ns[0]), int(ours.t_ns[-1]))
    return {"gt_ate_cm": 100.0 * ate(ours, gt).rmse_m, "lost_framesets": lost, "pose_count": float(len(ours))}
