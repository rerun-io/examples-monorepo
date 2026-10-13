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
uncalibrated, lossless (the frontend waits for the backend after every keyframe and relocalisation request).
"""

import contextlib
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from unittest import mock

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment
from simplecv.sources.catalog_video import MP4_NOMINAL_FPS

TIMELINE: str = "video_time"
CAMERA: str = "/world/rig_00/cam_00"
VIDEO: str = f"{CAMERA}/pinhole/video"
GT_RIG: str = "/world/rig_00"
KEYFRAMES: str = "/mast3r_slam/cam_00"
TRACKED: str = "/mast3r_slam/tracked/cam_00"
RUN: str = "/verify/slam/run"
MAX_ASSOCIATION_NS: int = 5_000_000
"""An estimated pose is scored against the nearest GT pose only within this time gap."""
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


def _log_poses(recording: rr.RecordingStream, path: str, t_ns: Int64[ndarray, " n"], sim3: Float32[ndarray, "n 8"]) -> None:
    """lietorch Sim3 rows ``[t, q_xyzw, s]`` (world_from_cam) as one ``Transform3D`` per time."""
    for t, row in zip(t_ns.tolist(), sim3.astype(np.float64), strict=True):
        recording.set_time(TIMELINE, duration=np.timedelta64(t, "ns"))
        recording.log(path, rr.Transform3D(translation=row[:3], quaternion=rr.Quaternion(xyzw=row[3:7])))


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """Pre-port: ``run_slam_pipeline`` on the catalog clip wrapped as an mp4 (no re-encode), lossless, uncalibrated.

    Per-frame poses are read after each frame; the final keyframe poses at the last frame, when (lossless) the backend
    has applied every task.
    """
    import torch
    import torch.multiprocessing as mp

    from mast3r_slam.api import inference
    from mast3r_slam.api.inference import SlamPipelineHandle, run_slam_pipeline
    from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates
    from mast3r_slam.mast3r_utils import load_mast3r

    if case.params["calib"] != "false":
        raise NotImplementedError("the pre-port feed (mp4) has no calibration path: calibrated cases need the port")
    video: CatalogVideo = _clip(case)
    times: Int64[ndarray, " n"] = _input_times(case, video)
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    model = load_mast3r(device="cuda:0")
    model.share_memory()
    # The old pipeline logs to the global stream (blueprint, logger thread): an enabled one that saves to /dev/null. It
    # stays global after the run (rerun cannot unset it); a previous global is restored.
    discard = rr.RecordingStream("mast3r-slam-preport", send_properties=False)
    discard.save(os.devnull)
    handle: SlamPipelineHandle = SlamPipelineHandle()
    tracked: list[tuple[int, Float32[ndarray, " 8"]]] = []
    keyframes: tuple[Int64[ndarray, " k"], Float32[ndarray, "k 8"]] | None = None
    previous: rr.RecordingStream | None = rr.set_global_data_recording(discard)
    start: float = time.perf_counter()
    # Rerun's Mp4Reader rejects AV1 packets muxed without an av1C box, and the old pipeline logs the source video
    # through it: that logging (and the logging timestamps it returns) is skipped; the SLAM inputs are untouched.
    no_video_log = mock.patch.object(inference, "log_video_for_dataset", return_value=None)
    try:
        with TemporaryDirectory() as tmp, no_video_log:
            clip: Path = Path(tmp) / "clip.mp4"
            clip.write_bytes(wrap_mp4(video.samples, video.keyframes, MP4_NOMINAL_FPS, codec=video.codec))
            pipeline = run_slam_pipeline(
                model=model,
                dataset_path=str(clip),
                config_path="config/eval_no_calib.yaml",  # base + uncalibrated + single_thread (lossless)
                device="cuda:0",
                img_size=cast(Literal[224, 512], int(case.params["img_size"])),
                subsample=int(case.params["subsample"]),
                handle=handle,
                recording=discard,
            )
            for i, _ in enumerate(pipeline):
                states: SharedStates | None = handle.states
                shared: SharedKeyframes | None = handle.keyframes
                assert states is not None and shared is not None
                if states.get_mode() == Mode.TRACKING:
                    tracked.append((i, states.get_frame().world_sim3_cam.data[0].cpu().numpy()))
                if i == len(times) - 1:  # the last frame: the map is final (the generator then terminates the backend)
                    with shared.lock:
                        n_kf: int = len(shared)
                        frame_index: Int64[ndarray, " k"] = shared.dataset_idx[:n_kf].cpu().numpy().astype(np.int64)
                        keyframes = (times[frame_index], shared.world_sim3_cam[:n_kf, 0].cpu().numpy())
    finally:
        if previous is not None:
            rr.set_global_data_recording(previous)
    wall_s: float = time.perf_counter() - start
    if keyframes is None or handle.stopped_early:
        raise RuntimeError(f"{case.name}: the old pipeline stopped before its last frame ({len(times)} input frames)")
    _log_poses(recording, KEYFRAMES, *keyframes)
    if tracked:
        index: list[int] = [i for i, _ in tracked]
        _log_poses(recording, TRACKED, times[index], np.stack([pose for _, pose in tracked]))
    counts = rr.AnyValues(input_frames=float(len(times)), tracked_frames=float(len(tracked)), keyframes=float(len(keyframes[0])), wall_s=wall_s)
    recording.log(RUN, counts, static=True)
    print(f"{case.name} {phase}: {len(times)} input frames, {len(tracked)} tracked, {len(keyframes[0])} keyframes, {wall_s:.1f} s")


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
    t_ns: Int64[ndarray, " n"], xyz: Float64[ndarray, "n 3"], gt_t_ns: Int64[ndarray, " g"], gt_xyz: Float64[ndarray, "g 3"]
) -> tuple[float, float, int]:
    """Sim3-aligned (Umeyama) RMSE of the estimated camera centres against the nearest GT centres (<= 5 ms apart).

    Returns (rmse in m, Sim3 scale, number of associated poses); NaN when fewer than three poses associate.
    """
    right: Int64[ndarray, " n"] = np.clip(np.searchsorted(gt_t_ns, t_ns), 1, len(gt_t_ns) - 1)
    nearest: Int64[ndarray, " n"] = np.where(np.abs(gt_t_ns[right - 1] - t_ns) <= np.abs(gt_t_ns[right] - t_ns), right - 1, right)
    keep: np.ndarray = np.abs(gt_t_ns[nearest] - t_ns) <= MAX_ASSOCIATION_NS
    if keep.sum() < 3:
        return float("nan"), float("nan"), int(keep.sum())
    gt_T_est: SimilarityTransform = umeyama_alignment(xyz[keep], gt_xyz[nearest[keep]], allow_scaling=True)
    error: Float64[ndarray, " m"] = np.linalg.norm(gt_T_est.apply(xyz[keep]) - gt_xyz[nearest[keep]], axis=1)
    return float(np.sqrt(np.mean(error**2))), gt_T_est.scale, int(keep.sum())


def score(case: Case, golden: Path) -> dict[str, float]:
    """Banded: ``ate_rmse_m`` (final keyframes vs GT cam_00, Sim3) and ``tracked_fraction``; the rest is reported."""
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        kf_t, kf_xyz, _ = _poses(dataset.filter_contents([KEYFRAMES]).reader(index=TIMELINE).to_arrow_table(), KEYFRAMES)
        tr_t, tr_xyz, _ = _poses(dataset.filter_contents([TRACKED]).reader(index=TIMELINE).to_arrow_table(), TRACKED)
        run_row: dict = dataset.filter_contents([RUN]).reader(index=None).to_arrow_table().to_pylist()[0]
    gt_t, gt_xyz = _gt_camera_centers(case)
    ate, scale, associated = _ate(kf_t, kf_xyz, gt_t, gt_xyz)
    tracked_ate, _, _ = _ate(tr_t, tr_xyz, gt_t, gt_xyz)
    reported: dict[str, float] = {f"run_{k.split(':')[-1]}": float(v[0]) for k, v in run_row.items() if k.startswith(f"{RUN}:") and v is not None}
    return {
        "ate_rmse_m": ate,
        "tracked_fraction": len(tr_t) / len(_input_times(case, _clip(case))),  # the denominator comes from the catalog, not the run
        "keyframes": float(len(kf_t)),
        "keyframes_associated": float(associated),
        "sim3_scale": scale,
        "tracked_ate_rmse_m": tracked_ate,
    } | reported
