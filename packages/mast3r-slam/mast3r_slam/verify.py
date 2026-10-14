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

Block ``slam`` (msd-odyssey): its pre/pre2 goldens came from the old pipeline, so this tree only runs ``post``. Block
``paper`` (TUM RGB-D fr1, EuRoC; ``run_paper``/``score_paper``) records every phase with this graph and associates poses
within evo's default 10 ms, as the MASt3R-SLAM paper's ``evo_ape -as``.
"""

import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.node import Emit, Tick
from dagkit.runners import RunReport, run_threaded
from dagkit.sinks import RerunSink
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment
from simplecv.sources.catalog_video import CatalogVideoSource

from mast3r_slam.apis.slam import frame_hw, outputs
from mast3r_slam.graphs.slam import slam_graph
from mast3r_slam.types import SlamConfig

TIMELINE: str = "video_time"
CAMERA: str = "/world/rig_00/cam_00"
VIDEO: str = f"{CAMERA}/pinhole/video"
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
CHILD_FROM_PARENT: int = 2
"""``Transform3D:relation`` of a node logged as child_from_parent (the catalog's camera extrinsics)."""


def _window_ns(case: Case) -> tuple[int, int] | None:
    return None if case.window_s is None else (round(case.window_s[0] * 1e9), round(case.window_s[1] * 1e9))


def _clip(case: Case) -> CatalogVideo:
    """The case's cam_00 packets (``window_s`` [start, stop) or the whole segment); the first must be a keyframe."""
    assert case.dataset is not None and case.segment is not None, "mast3r-slam cases are catalog segments"
    dataset: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()  # the entry must outlive its queries
    video: CatalogVideo = read_catalog_videos(dataset, case.segment, [VIDEO], TIMELINE, _window_ns(case))[0]
    if not video.keyframes[0]:
        raise ValueError(f"{case.segment}: the window {case.window_s} does not start on a keyframe")
    return video


def _input_times(case: Case, video: CatalogVideo) -> Int64[ndarray, " n"]:
    """``video_time`` of every input frame of the case (clip indices 0, s, 2s, ...; ``n // s`` of them)."""
    stride: int = int(case.params["subsample"])
    return video.t_ns[: len(video.t_ns) // stride * stride : stride]


class _GoldenSink(RerunSink):
    """The graph's final keyframes and tracked poses into the golden, counted per port."""

    def __init__(self, recording: rr.RecordingStream, camera: str) -> None:
        ports, roots = outputs(camera)
        super().__init__(recording, {p: ports[p] for p in ("keyframes", "tracked")}, roots)
        self.counts: Counter[str] = Counter()

    def step(self, tick: Tick, out: Emit) -> None:
        self.counts[tick.port] += 1
        super().step(tick, out)


def run(case: Case, phase: Phase, recording: rr.RecordingStream, *, phases: tuple[Phase, ...] = ("post",)) -> None:
    """The graph on the case's clip (cam_00, every ``subsample``-th frame), lossless, threaded.

    ``phases`` this tree may record: block ``slam`` only ``post`` (its ``pre``/``pre2`` goldens come from the card commit,
    which drove the old pipeline: git history); block ``paper`` every phase (``run_paper``).
    """
    if phase not in phases:
        raise ValueError(f"phase {phase}: pre-port goldens come from the card commit (the old pipeline); this tree runs the port")
    assert case.dataset is not None and case.segment is not None, "mast3r-slam cases are catalog segments"
    cfg = SlamConfig(use_calib=case.params["calib"] == "true", img_size=int(case.params["img_size"]), subsample=int(case.params["subsample"]))  # type: ignore[arg-type]
    segment = SegmentRef(case.dataset, case.segment, window_ns=_window_ns(case))
    video = CatalogVideoSource(segment, (CAMERA.removeprefix("/world/"),), decode="cpu_rgb", stride=cfg.subsample, undistort=case.params.get("undistort") == "true")
    golden = _GoldenSink(recording, CAMERA)
    start: float = time.perf_counter()
    report: RunReport = run_threaded(slam_graph(video, cfg, frame_hw(video, cfg.img_size), {"golden": golden}), mode="lossless")
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
    xyz: Float64[ndarray, "n 3"] = np.array([v[0] for v in rows[translation].to_pylist()], dtype=np.float64).reshape(-1, 3)
    quaternion: str = f"{path}:Transform3D:quaternion"
    xyzw: Float64[ndarray, "n 4"] = np.array([v[0] for v in rows[quaternion].to_pylist()], dtype=np.float64).reshape(-1, 4)
    return t_ns, xyz, xyzw


def _gt_camera_centers(case: Case) -> tuple[Int64[ndarray, " g"], Float64[ndarray, "g 3"]]:
    """GT cam_00 centres = GT rig pose (world_T_rig) applied to the cam_00 centre in the rig frame (static extrinsic)."""
    assert case.dataset is not None and case.segment is not None
    dataset: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()
    static: dict = dataset.filter_segments(case.segment).filter_contents([CAMERA]).reader(index=None).to_arrow_table().to_pylist()[0]
    cam_T_rig: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_rig[:3, :3] = np.asarray(static[f"{CAMERA}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T  # column-major
    cam_T_rig[:3, 3] = static[f"{CAMERA}:Transform3D:translation"][0]
    if static[f"{CAMERA}:Transform3D:relation"][0] != CHILD_FROM_PARENT:
        cam_T_rig = np.linalg.inv(cam_T_rig)
    rig_center: Float64[ndarray, " 3"] = np.linalg.inv(cam_T_rig)[:3, 3]
    t_ns, xyz, xyzw = _poses(dataset.filter_segments(case.segment).filter_contents([GT_RIG]).reader(index=TIMELINE).to_arrow_table(), GT_RIG)
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
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        kf_t, kf_xyz, _ = _poses(dataset.filter_contents([KEYFRAMES]).reader(index=TIMELINE).to_arrow_table(), KEYFRAMES)
        tr_t, tr_xyz, _ = _poses(dataset.filter_contents([TRACKED]).reader(index=TIMELINE).to_arrow_table(), TRACKED)
        run_row: dict = dataset.filter_contents([RUN]).reader(index=None).to_arrow_table().to_pylist()[0]
    gt_t, gt_xyz = _gt_camera_centers(case)
    ate, scale, associated = _ate(kf_t, kf_xyz, gt_t, gt_xyz, max_association_ns)
    tracked_ate, _, _ = _ate(tr_t, tr_xyz, gt_t, gt_xyz, max_association_ns)
    reported: dict[str, float] = {f"run_{k.split(':')[-1]}": float(v[0]) for k, v in run_row.items() if k.startswith(f"{RUN}:") and v is not None}
    return {
        "ate_rmse_m": ate,
        "tracked_fraction": len(tr_t) / len(_input_times(case, _clip(case))),  # the denominator comes from the catalog, not the run
        "keyframes": float(len(kf_t)),
        "keyframes_associated": float(associated),
        "sim3_scale": scale,
        "tracked_ate_rmse_m": tracked_ate,
    } | reported


run_paper = partial(run, phases=("pre", "pre2", "post"))
score_paper = partial(score, max_association_ns=EVO_MAX_DIFF_NS)
