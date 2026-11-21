"""Freeze the dynamic gate's reference: the dataset's MAMMA prediction of running_jumping, forwarded through SMPL-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mamma.eval.dynamic_reference import write_reference


@dataclass
class Config:
    data_dir: Path = Path("data/inputs/outdoors/running_jumping")
    """Dynamic capture (MAMMA NPZ layout) whose ``pred/params_00.npz`` is frozen."""
    model_dir: Path = Path("data/body_models")
    """Body-model root containing ``smplx/SMPLX_NEUTRAL.npz`` (v1.1, 16 betas)."""
    out: Path = Path("data/golden/ma_3d_dynamic/outdoors/running_jumping/reference.rrd")
    """Reference file written (publish it under ``golden/`` of the mamma-streaming-data dataset)."""
    fps: float = 30.0
    """Frame rate of the clip."""
    device: str = "cuda"
    """Device the SMPL-X forward runs on (the gate's former on-the-fly path ran on CUDA)."""


def main(config: Config) -> int:
    from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch

    pred_npz: Path = config.data_dir / "pred" / "params_00.npz"
    with np.load(pred_npz, allow_pickle=False) as pred:
        betas, poses, trans = pred["betas"], pred["poses"], pred["trans"]
    layer = SmplxLayerTorch(betas=betas.reshape(-1).astype(np.float32), model_root_dir=config.model_dir, flat_hand_mean=False)
    vertices = np.asarray(layer.to(config.device).forward_batched(poses.astype(np.float32), trans.astype(np.float32)).vertices, dtype=np.float32)
    frames = np.arange(len(vertices), dtype=np.int64)
    write_reference(config.out, frames, vertices, config.fps, f"dataset MAMMA prediction {pred_npz} via SMPL-X v1.1 neutral (16 betas), frozen 2026-11-21")
    print(f"wrote {config.out}: {len(frames)} frames x {vertices.shape[1]} vertices")
    return 0
