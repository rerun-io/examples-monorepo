"""Re-log the ORIGINAL pipeline's saved artifacts as an enriched Rerun recording.

The original DAG takes ~2 hours to run but its outputs are all on disk
(calibration NPZs, ma_2d landmark NPZs, ma_3d verts/params NPZs). This tool
replays them through the same StreamLogger as the streaming port — video,
mesh, bones, SMPL-X transform tree, contact scalars — in minutes, producing a
comparison artifact with identical entity layout to our live recordings.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.fitting.smplx_wrapper import build_smplx_neutral, smplx_forward_per_parts
from mamma.fitting.window_fitter import FitResult
from mamma.landmarks.estimator import CameraLandmarks, LandmarkResult
from mamma.viz.stream_logger import StreamLogger


@dataclass
class RelogConfig:
    rr_config: RerunTyroConfig
    """Rerun behavior (pass --rr-config.save <path> to write the RRD)."""
    data_dir: Path = Path("data/inputs/outdoors/running_jumping")
    """Capture inputs (meta NPZs + videos) the original run consumed."""
    ma3d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_3d/baseline-rj2/outdoors/running_jumping"
    )
    """Original ma_3d output dir (verts_joints + smplx_params NPZs)."""
    ma2d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_2d/baseline-rj2/outdoors/running_jumping"
    )
    """Original ma_2d output dir (per-camera landmark NPZs)."""
    body_id: int = 0
    """Person id in the original outputs."""
    smplx_model_dir: Path = Path("data/body_models")
    """For SMPL-X faces + rest joints (betas-dependent)."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Logged video/overlay resolution (matches our live recordings)."""
    device: str = "cuda"
    """Decode device."""


def main(config: RelogConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    vj = np.load(config.ma3d_dir / f"verts_joints_body_id-{config.body_id:02d}.npz", allow_pickle=True)
    params = np.load(config.ma3d_dir / f"smplx_params_body_id-{config.body_id:02d}.npz", allow_pickle=True)
    vertices: Float32[ndarray, "f v 3"] = vj["pred_vertices"]
    joints: Float32[ndarray, "f j 3"] = vj["pred_joints"]
    pose: Float32[ndarray, "f 165"] = params["smplx_pose"].reshape(vertices.shape[0], -1)
    betas: Float32[ndarray, "nb"] = params["smplx_betas"].reshape(-1)
    trans: Float32[ndarray, "f 3"] = params["smplx_translation"].reshape(vertices.shape[0], 3)
    tri_cloud: Float32[ndarray, "f n 3"] = params["triangulated_3d_pts"]
    floor_contact: Float32[ndarray, "f n"] = params["smplx_floor_contact"]
    n_frames: int = vertices.shape[0]
    print(f"{sequence.name}: {n_frames} frames of original ma_3d output")

    # 2D landmarks per camera (source-resolution px -> engine px).
    scale: float = config.resize_hw[1] / sequence.cameras[0].width
    lm_by_cam: dict[str, ndarray] = {}
    vis_by_cam: dict[str, ndarray] = {}
    for cam in sequence.camera_names:
        d = np.load(config.ma2d_dir / f"{cam}.npz", allow_pickle=True)
        lm_by_cam[cam] = d["landmarks"][:, config.body_id]  # (f, 512, 3)
        vis_by_cam[cam] = d["visibilities"][:, config.body_id]  # (f, 512)

    # Rest joints for the transform-tree decomposition (betas-dependent).
    model = build_smplx_neutral(config.smplx_model_dir, device=config.device)
    nb: int = int(model.num_betas)
    betas_t: Float32[torch.Tensor, "1 nb"] = torch.from_numpy(betas[:nb]).float().to(config.device).unsqueeze(0)
    zero3: Float32[torch.Tensor, "1 3"] = torch.zeros(1, 3, device=config.device)
    with torch.no_grad():
        rest = smplx_forward_per_parts(
            model,
            zero3,
            torch.zeros(1, 63, device=config.device),
            torch.zeros(1, 45, device=config.device),
            torch.zeros(1, 45, device=config.device),
            zero3,
            betas_t,
            torch.zeros(1, 3, device=config.device),
        )
    rest_joints: Float32[ndarray, "55 3"] = rest.joints[0, :55].cpu().numpy()
    faces: ndarray = model.faces

    logger = StreamLogger(sequence, resize_hw=config.resize_hw)
    logger.setup()
    reader = TorchCodecMultiVideoReader(list(sequence.video_paths), device=config.device, resize_hw=config.resize_hw)

    chunk: int = 32
    for start in range(0, n_frames, chunk):
        stop: int = min(start + chunk, n_frames)
        videos: list[UInt8[torch.Tensor, "b 3 h w"]] = [r.get_frames_in_range(start, stop) for r in reader.video_readers]
        for local in range(stop - start):
            f: int = start + local
            frames: list[UInt8[torch.Tensor, "3 h w"]] = [v[local] for v in videos]
            logger.log_tick_video(f, frames)
            landmarks: list[CameraLandmarks] = []
            for cam in sequence.camera_names:
                lm: Float32[torch.Tensor, "n 3"] = torch.from_numpy(lm_by_cam[cam][f].copy()).float()
                lm[:, :2] *= scale
                landmarks.append(
                    {
                        config.body_id: LandmarkResult(
                            obj_id=config.body_id,
                            joints2d=lm,
                            visibility=torch.from_numpy(vis_by_cam[cam][f].copy()).float(),
                            contact=torch.zeros(lm.shape[0]),
                            floor_contact=torch.from_numpy(floor_contact[f].copy()).float(),
                        )
                    }
                )
            if f % 2 == 0:
                logger.log_tick_landmarks(f, landmarks)
            fit = FitResult(
                frame_idx=f,
                vertices=vertices[f],
                joints=joints[f],
                pose=pose[f],
                betas=betas,
                trans=trans[f],
                rest_joints=rest_joints,
            )
            cloud: Float32[torch.Tensor, "n 3"] = torch.from_numpy(tri_cloud[f].copy()).float()
            valid: torch.Tensor = cloud.abs().sum(dim=-1) > 1e-6
            logger.log_tick_fit(f, {config.body_id: fit}, {config.body_id: (cloud, valid)}, faces)
            logger.log_tick_metrics(f, {}, {"floor_contacts": float((floor_contact[f] > 0.5).sum())})
        print(f"\r{stop}/{n_frames}", end="", flush=True)
    logger.flush()
    print(f"\nrelogged {n_frames} frames from {config.ma3d_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(RelogConfig)))
