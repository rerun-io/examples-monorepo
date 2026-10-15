"""Graph-tier check of mast3r-slam, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

``run`` writes a golden rrd; ``score`` reads only that golden plus the segment's camera and GT from the catalog.

Golden contract (the ported graph writes the same paths on the segment's ``video_time``):

- ``/mast3r_slam/cam_00``: the FINAL corrected keyframe camera poses, one ``Transform3D`` per keyframe at the time of its
  frame (``translation`` + ``quaternion`` xyzw, parent-from-child = world_from_cam; the Sim3 scale only scales the
  camera's own frame, so it is left out).
- ``/mast3r_slam/tracked/cam_00``: one ``Transform3D`` (same encoding) per input frame that the frontend left in
  TRACKING (the INIT frame included; frames that triggered or failed relocalisation are absent).
- ``/verify/slam/run``: static ``AnyValues`` run counts (``input_frames``, ``tracked_frames``, ``keyframes``, ``wall_s``).

Input frames are the clip's cam_00 frames at indices ``0, s, 2s, ...`` (``n // s`` of them, ``s`` = ``subsample``),
lossless (the frontend waits for the backend after every keyframe and relocalisation request); case params choose
calibrated tracking (``calib``) and undistortion by the catalog source (``undistort``).

Block ``slam`` (msd-odyssey): its pre/pre2 goldens came from the old pipeline (dagkit.verify never overwrites them). Block
``paper`` (TUM RGB-D fr1, EuRoC; ``score_paper``) recorded every phase with this graph and associates poses within evo's
default 10 ms, as the MASt3R-SLAM paper's ``evo_ape -as``.
"""

import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.node import Emit, Source, Tick
from dagkit.runners import RunReport, run_threaded
from dagkit.sinks import RerunSink
from dagkit.sources import SegmentRef
from dagkit.types import flat, served
from dagkit.verify import Case, Phase, load_card, run_counts
from datafusion import col, lit
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import catalog_keyframes
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.types.rig import RigCameras

from mast3r_slam.apis.slam import PACKAGE_DIR, Config, frame_hw, outputs, slam_config, video_source
from mast3r_slam.graphs.slam import slam_graph
from mast3r_slam.types import SlamConfig

TIMELINE: str = "video_time"
CAMERA: str = "/world/rig_00/cam_00"
VIDEO: str = f"{CAMERA}/pinhole/video"
KEYFRAME_FLAG: str = f"{VIDEO}:VideoStream:is_keyframe"
GT_RIG: str = "/world/rig_00"
KEYFRAMES: str = "/mast3r_slam/cam_00"
TRACKED: str = "/mast3r_slam/tracked/cam_00"
RUN: str = "/verify/slam/run"
MAX_ASSOCIATION_NS: int = 5_000_000
"""An estimated pose is scored against the nearest GT pose only within this time gap (block ``slam``)."""
EVO_MAX_DIFF_NS: int = 10_000_000
"""evo's default ``t_max_diff`` (0.01 s), the association of the paper's ``evo_ape tum gt est -as`` (block ``paper``). The
paper's EuRoC script hands evo nanosecond stamps, so there 0.01 matched only stamps equal as float64 (about half of the
keyframes); this block keeps 10 ms on both benchmarks."""


def case_config(case: Case) -> Config:
    """The App config of a case: its clip of cam_00 at ``img_size`` and ``subsample``, the calib preset when ``calib``,
    undistorted by the source when ``undistort``; lossless and threaded (the App's defaults)."""
    p = case.params
    return Config(RerunTyroConfig(headless=True), case.segment_ref(), undistort=p.get("undistort", False), preset="calib" if p["calib"] else None, img_size=p["img_size"], subsample=p["subsample"])


def examples() -> list[Config]:
    """The card's short cases as App configs: the gradio app's examples, one click from a banded run."""
    return [case_config(case) for block in load_card(PACKAGE_DIR / "VERIFY.toml").graph for case in block.cases if case.clip == "short"]


def _input_frames(case: Case) -> int:
    """Input frames of the case: ``n // s`` of the clip's ``n`` cam_00 packets (``window_s`` [start, stop) or the whole
    segment; the first must be a keyframe). Reads the packets' keyframe flags only, not the packets."""
    segment: SegmentRef = case.segment_ref()
    dataset: DatasetEntry = segment.entry()  # the entry must outlive its queries
    reader = dataset.filter_segments(segment.segment).filter_contents([VIDEO]).reader(index=TIMELINE)
    if segment.window_ns is not None:
        time = col(TIMELINE).cast(pa.int64())
        reader = reader.filter((time >= lit(segment.window_ns[0])) & (time < lit(segment.window_ns[1])))
    flags: pa.ChunkedArray = reader.select(TIMELINE, KEYFRAME_FLAG).to_arrow_table()[KEYFRAME_FLAG]
    keyframes: list[bool] = catalog_keyframes(flags)  # one row per packet; a null flag (sparse logging) is false
    if not keyframes[0]:
        raise ValueError(f"{segment.segment}: the window {case.window_s} does not start on a keyframe")
    return len(keyframes) // case.params["subsample"]


class _GoldenSink(RerunSink):
    """The graph's final keyframes and tracked poses into the golden, counted per port."""

    def __init__(self, recording: rr.RecordingStream, camera: str) -> None:
        ports, roots = outputs(camera)
        super().__init__(recording, {p: ports[p] for p in ("keyframes", "tracked")}, roots)
        self.counts: Counter[str] = Counter()

    def step(self, tick: Tick, out: Emit) -> None:
        self.counts[tick.port] += 1
        super().step(tick, out)


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The App's graph for ``case_config(case)``, a golden sink in place of the App's viewer and layer."""
    config: Config = case_config(case)
    cfg: SlamConfig = slam_config(config)
    video: Source = video_source(config, cfg)
    golden = _GoldenSink(recording, CAMERA)
    start: float = time.perf_counter()
    report: RunReport = run_threaded(slam_graph(video, cfg, frame_hw(video, cfg.img_size), {"golden": golden}), mode=config.mode)
    wall_s: float = time.perf_counter() - start
    if report.status != "complete":
        raise RuntimeError(f"{case.name}: the graph ended {report.status}")
    inputs: int = report.nodes["frontend"].ticks
    counts = rr.AnyValues(input_frames=float(inputs), tracked_frames=float(golden.counts["tracked"]), keyframes=float(golden.counts["keyframes"]), wall_s=wall_s)
    recording.log(RUN, counts, static=True)
    print(f"{case.name} {phase}: {inputs} input frames, {golden.counts['tracked']} tracked, {golden.counts['keyframes']} keyframes, {wall_s:.1f} s")


def _poses(table: pa.Table, path: str) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 3"], Float64[ndarray, "n 4"]]:
    """Times, translations and xyzw quaternions of the rows of ``path`` that carry a pose."""
    translation: str = f"{path}:Transform3D:translation"
    if translation not in table.column_names:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 4))
    rows: pa.Table = table.filter(table[translation].is_valid())
    t_ns: Int64[ndarray, " n"] = rows[TIMELINE].combine_chunks().to_numpy(zero_copy_only=False).astype("timedelta64[ns]").astype(np.int64)
    xyz: Float64[ndarray, "n 3"] = flat(rows[translation].combine_chunks(), np.float64).reshape(-1, 3)
    xyzw: Float64[ndarray, "n 4"] = flat(rows[f"{path}:Transform3D:quaternion"].combine_chunks(), np.float64).reshape(-1, 4)
    return t_ns, xyz, xyzw


def _gt_camera_centers(case: Case) -> tuple[Int64[ndarray, " g"], Float64[ndarray, "g 3"]]:
    """GT cam_00 centres = GT rig pose (world_T_rig) applied to the cam_00 centre in the rig frame (static extrinsic)."""
    segment: SegmentRef = case.segment_ref()
    dataset: DatasetEntry = segment.entry()  # the entry must outlive its queries
    static: pa.Table = dataset.filter_segments(segment.segment).filter_contents([CAMERA, f"{CAMERA}/pinhole"]).reader(index=None).to_arrow_table()
    _, rig = next(RigCameras.from_layer(static, GT_RIG, TIMELINE))  # cam_T_world of a camera under GT_RIG is cam_T_rig
    rig_center: Float64[ndarray, " 3"] = np.linalg.inv(rig.select([CAMERA.removeprefix(f"{GT_RIG}/")]).cam_T_world[0])[:3, 3]
    t_ns, xyz, xyzw = _poses(dataset.filter_segments(segment.segment).filter_contents([GT_RIG]).reader(index=TIMELINE).to_arrow_table(), GT_RIG)
    return t_ns, Rotation.from_quat(xyzw).apply(rig_center) + xyz


def _ate(
    t_ns: Int64[ndarray, " n"],
    xyz: Float64[ndarray, "n 3"],
    gt_t_ns: Int64[ndarray, " g"],
    gt_xyz: Float64[ndarray, "g 3"],
    max_association_ns: int = MAX_ASSOCIATION_NS,
) -> tuple[float, float, int]:
    """Sim3-aligned (Umeyama) RMSE of the estimated camera centres against the nearest GT centres (within the gap).

    Returns (rmse in m, Sim3 scale, number of associated poses); NaN when fewer than three poses associate.
    """
    right: Int64[ndarray, " n"] = np.clip(np.searchsorted(gt_t_ns, t_ns), 1, len(gt_t_ns) - 1)
    nearest: Int64[ndarray, " n"] = np.where(np.abs(gt_t_ns[right - 1] - t_ns) <= np.abs(gt_t_ns[right] - t_ns), right - 1, right)
    keep: np.ndarray = np.abs(gt_t_ns[nearest] - t_ns) <= max_association_ns
    if keep.sum() < 3:
        return float("nan"), float("nan"), int(keep.sum())
    gt_T_est: SimilarityTransform = umeyama_alignment(xyz[keep], gt_xyz[nearest[keep]], allow_scaling=True)
    error: Float64[ndarray, " m"] = np.linalg.norm(gt_T_est.apply(xyz[keep]) - gt_xyz[nearest[keep]], axis=1)
    return float(np.sqrt(np.mean(error**2))), gt_T_est.scale, int(keep.sum())


def score(case: Case, golden: Path, max_association_ns: int = MAX_ASSOCIATION_NS) -> dict[str, float]:
    """Banded: ``ate_rmse_m`` (final keyframes vs GT cam_00, Sim3) and ``tracked_fraction``; the rest is reported."""
    with served(golden) as dataset:
        kf_t, kf_xyz, _ = _poses(dataset.filter_contents([KEYFRAMES]).reader(index=TIMELINE).to_arrow_table(), KEYFRAMES)
        tr_t, tr_xyz, _ = _poses(dataset.filter_contents([TRACKED]).reader(index=TIMELINE).to_arrow_table(), TRACKED)
        reported: dict[str, float] = {f"run_{k}": v for k, v in run_counts(dataset, "slam").items()}  # RUN, in both blocks
    gt_t, gt_xyz = _gt_camera_centers(case)
    ate, scale, associated = _ate(kf_t, kf_xyz, gt_t, gt_xyz, max_association_ns)
    tracked_ate, _, _ = _ate(tr_t, tr_xyz, gt_t, gt_xyz, max_association_ns)
    return {
        "ate_rmse_m": ate,
        "tracked_fraction": len(tr_t) / _input_frames(case),  # the denominator comes from the catalog, not the run
        "keyframes": float(len(kf_t)),
        "keyframes_associated": float(associated),
        "sim3_scale": scale,
        "tracked_ate_rmse_m": tracked_ate,
    } | reported


score_paper = partial(score, max_association_ns=EVO_MAX_DIFF_NS)
