"""Validate that EPFL Smart Kitchen MANO-derived joints match ground-truth keypoints.

For each selected (participant, session) pair:

1. Load the EPFL labels via the shared ``EpflSmartKitchenSequence``.
2. Recompute MANO 21-joint hand positions with ``MANOLayerNP`` using the same
   ManoStack the visualizer would feed to it.
3. Compare per-frame against ``xyzc_stack[:, 91:133, :3]`` (the dataset's own
   COCO-133 hand keypoints) over a deterministic subsample of confident frames.
4. Optionally save an RRD via the standard ``visualize_exo_ego`` flow (one exo
   camera, ego off, env mesh off) so the result is browsable in the viewer.

Pass criterion: per-session mean L2 < ``--tol-mean-mm`` and max L2 <
``--tol-max-mm``. Prints one ``[epfl-mano] session=...`` line per session and
one ``[epfl-mano] OVERALL`` summary line. These are the exact patterns the
``/goal`` evaluator reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from numpy import ndarray

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.data.exo.base_exo import ManoStack
from simplecv.data.exoego.epfl_smart_kitchen import (
    EPFL_EXO_CAMERA_NAMES,
    EpflExoCameraName,
    EpflSmartKitchenConfig,
    EpflSmartKitchenSequence,
)
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.rerun_log_utils import RerunTyroConfig

DEFAULT_EPFL_ROOT: Path = Path("/mnt/8tb/data/epfl-smart-kitchen-av1")


@dataclass(slots=True)
class SessionResult:
    participant: str
    session: str
    mean_mm: float
    max_mm: float
    n_frames: int
    rrd_path: Path
    passed: bool


def pick_first_train_sessions(root: Path, n: int) -> list[tuple[str, str]]:
    pose_root: Path = root / "Public_release_pose" / "train"
    sessions: list[tuple[str, str]] = []
    for participant_dir in sorted(pose_root.iterdir()):
        if not participant_dir.is_dir():
            continue
        for session_dir in sorted(participant_dir.iterdir()):
            pose_csv: Path = session_dir / "pose_3d" / "pose3d_mano.csv"
            if pose_csv.exists():
                sessions.append((participant_dir.name, session_dir.name))
                if len(sessions) >= n:
                    return sessions
    return sessions


def compute_endpoint_error(
    mano_stack: ManoStack,
    xyzc_stack: Float32[ndarray, "n_frames 133 4"],
    n_frames: int,
) -> tuple[Float32[ndarray, "n"], Float32[ndarray, "n"], Int[ndarray, "n"]]:
    """Per-frame (mean_mm, max_mm) over a subsample of confident frames.

    Returns mean per joint and max per joint across the sampled frames as well
    as the frame indices used, so callers can report which frames were checked.
    """
    hand_conf: Float32[ndarray, "n_frames 42"] = xyzc_stack[:, 91:133, 3]
    confident_mask: ndarray = np.asarray((hand_conf > 0).all(axis=1))
    confident_idx: Int[ndarray, "k"] = np.where(confident_mask)[0]
    if confident_idx.size == 0:
        raise RuntimeError("No frames have both hands fully confident in xyzc_stack.")

    take: int = int(min(n_frames, confident_idx.size))
    sample_positions: Int[ndarray, "n"] = np.linspace(0, confident_idx.size - 1, num=take, dtype=np.int64)
    sampled: Int[ndarray, "n"] = confident_idx[sample_positions]

    right_layer = MANOLayerNP(side="right", betas=mano_stack.betas_for(0), use_pca=mano_stack.use_pca)
    left_layer = MANOLayerNP(side="left", betas=mano_stack.betas_for(1), use_pca=mano_stack.use_pca)

    poses_r: Float32[ndarray, "n 48"] = mano_stack.so3[sampled, 0]
    poses_l: Float32[ndarray, "n 48"] = mano_stack.so3[sampled, 1]
    trans_r: Float32[ndarray, "n 3"] = mano_stack.trans[sampled, 0]
    trans_l: Float32[ndarray, "n 3"] = mano_stack.trans[sampled, 1]

    _, joints_r = right_layer(poses_r, trans_r)
    _, joints_l = left_layer(poses_l, trans_l)

    gt_left: Float32[ndarray, "n 21 3"] = xyzc_stack[sampled, 91:112, :3]
    gt_right: Float32[ndarray, "n 21 3"] = xyzc_stack[sampled, 112:133, :3]

    err_r_mm: Float32[ndarray, "n 21"] = np.linalg.norm(joints_r - gt_right, axis=-1) * 1000.0
    err_l_mm: Float32[ndarray, "n 21"] = np.linalg.norm(joints_l - gt_left, axis=-1) * 1000.0
    all_err: Float32[ndarray, "n 42"] = np.concatenate([err_l_mm, err_r_mm], axis=1)

    per_frame_mean: Float32[ndarray, "n"] = all_err.mean(axis=1).astype(np.float32)
    per_frame_max: Float32[ndarray, "n"] = all_err.max(axis=1).astype(np.float32)
    return per_frame_mean, per_frame_max, sampled.astype(np.int64)


def build_rrd(
    *,
    cfg: EpflSmartKitchenConfig,
    rrd_path: Path,
) -> None:
    """Generate a slim RRD (one exo camera, ego/env off) for browsing."""
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    if rrd_path.exists():
        rrd_path.unlink()

    rr_cfg = RerunTyroConfig(
        application_id="exoego-forge",
        recording_id=f"epfl-smart-kitchen__{cfg.split}__{cfg.participant_id}__{cfg.session_name}__validate",
        save=rrd_path,
        headless=True,
    )
    viz_cfg = VisualizeConfig(
        rr_config=rr_cfg,
        dataset=cfg,
        log_exo=True,
        log_ego=False,
        log_mano=True,
        log_labels=True,
        log_env_mesh=False,
        log_depths=False,
    )
    sequence = EpflSmartKitchenSequence(cfg)
    visualize_exo_ego(sequence, viz_cfg)
    rec: rr.RecordingStream = rr_cfg.rec_stream
    rec.flush(timeout_sec=120.0)
    # Detach the recording so subsequent sessions can re-init cleanly.
    rec.disconnect()


def process_session(
    *,
    participant: str,
    session: str,
    root: Path,
    tol_mean_mm: float,
    tol_max_mm: float,
    n_frames: int,
    out_dir: Path,
    skip_rrd: bool,
    exo_camera: str,
) -> SessionResult:
    selected_camera: EpflExoCameraName = cast(EpflExoCameraName, exo_camera) if exo_camera in EPFL_EXO_CAMERA_NAMES else EPFL_EXO_CAMERA_NAMES[0]
    cfg = EpflSmartKitchenConfig(
        root_directory=root,
        split="train",
        participant_id=participant,
        session_name=session,
        exo_camera_names=(selected_camera,),
        load_labels=True,
    )
    sequence = EpflSmartKitchenSequence(cfg)
    labels = sequence.exoego_labels
    if labels is None or labels.mano_stack is None:
        raise RuntimeError(f"Labels missing for {participant}/{session}")

    per_frame_mean, per_frame_max, sampled = compute_endpoint_error(labels.mano_stack, labels.xyzc_stack, n_frames=n_frames)
    mean_mm: float = float(per_frame_mean.mean())
    max_mm: float = float(per_frame_max.max())

    session_dir: Path = out_dir / f"{participant}__{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = session_dir / f"{session}.rrd"
    if not skip_rrd:
        build_rrd(cfg=cfg, rrd_path=rrd_path)

    summary: dict[str, object] = {
        "participant": participant,
        "session": session,
        "n_sampled_frames": int(sampled.size),
        "mean_mm": mean_mm,
        "max_mm": max_mm,
        "per_frame_mean_mm": per_frame_mean.tolist(),
        "per_frame_max_mm": per_frame_max.tolist(),
        "sampled_frame_indices": sampled.tolist(),
        "tol_mean_mm": tol_mean_mm,
        "tol_max_mm": tol_max_mm,
        "rrd_path": str(rrd_path),
    }
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    passed: bool = (mean_mm < tol_mean_mm) and (max_mm < tol_max_mm)
    return SessionResult(
        participant=participant,
        session=session,
        mean_mm=mean_mm,
        max_mm=max_mm,
        n_frames=int(sampled.size),
        rrd_path=rrd_path,
        passed=passed,
    )


def _format_line(result: SessionResult) -> str:
    status: str = "PASS" if result.passed else "FAIL"
    return (
        f"[epfl-mano] session={result.participant}/{result.session} "
        f"mean_mm={result.mean_mm:.2f} max_mm={result.max_mm:.2f} "
        f"n_frames={result.n_frames} {status}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_EPFL_ROOT)
    parser.add_argument("--pick-first-train", type=int, default=0)
    parser.add_argument("--participant", action="append", default=[])
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--tol-mean-mm", type=float, default=20.0)
    parser.add_argument("--tol-max-mm", type=float, default=50.0)
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/epfl-mano-debug"))
    parser.add_argument("--skip-rrd", action="store_true")
    parser.add_argument("--exo-camera", default="output0")
    args = parser.parse_args()

    sessions: list[tuple[str, str]] = []
    if args.pick_first_train > 0:
        sessions = pick_first_train_sessions(args.root, args.pick_first_train)
    if args.participant and args.session and len(args.participant) == len(args.session):
        sessions.extend(zip(args.participant, args.session, strict=True))
    if not sessions:
        print("No sessions selected. Pass --pick-first-train N or repeated --participant/--session pairs.", file=sys.stderr)
        return 2

    print(f"[epfl-mano] Running against {len(sessions)} session(s): {sessions}", flush=True)

    results: list[SessionResult] = []
    for participant, session in sessions:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                result: SessionResult = process_session(
                    participant=participant,
                    session=session,
                    root=args.root,
                    tol_mean_mm=args.tol_mean_mm,
                    tol_max_mm=args.tol_max_mm,
                    n_frames=args.n_frames,
                    out_dir=args.out_dir,
                    skip_rrd=args.skip_rrd,
                    exo_camera=args.exo_camera,
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"[epfl-mano] session={participant}/{session} mean_mm=nan max_mm=nan n_frames=0 FAIL ({type(e).__name__}: {e})",
                flush=True,
            )
            results.append(SessionResult(participant, session, float("nan"), float("nan"), 0, Path(), False))
            continue
        print(_format_line(result), flush=True)
        results.append(result)

    passed: int = sum(1 for r in results if r.passed)
    if results:
        finite_mean = [r.mean_mm for r in results if np.isfinite(r.mean_mm)]
        finite_max = [r.max_mm for r in results if np.isfinite(r.max_mm)]
        mean_max: float = float(max(finite_mean)) if finite_mean else float("nan")
        max_max: float = float(max(finite_max)) if finite_max else float("nan")
    else:
        mean_max = float("nan")
        max_max = float("nan")
    overall_status: str = "PASS" if passed == len(results) and passed > 0 else "FAIL"
    print(
        f"[epfl-mano] OVERALL sessions={len(results)} passed={passed} mean_mm_max={mean_max:.2f} max_mm_max={max_max:.2f} {overall_status}",
        flush=True,
    )
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
