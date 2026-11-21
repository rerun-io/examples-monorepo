"""Freeze the dynamic gate's reference: the dataset's MAMMA prediction of running_jumping, forwarded through SMPL-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray
from serde import SerdeError, from_dict, serde

from mamma.eval.dynamic_reference import ENTITY, FRAME_TIMELINE


def write_reference(path: Path, frames: Int64[ndarray, "f"], vertices: Float32[ndarray, "f v 3"], fps: float, source: str) -> None:
    """Write ``vertices`` (one row per frame in ``frames``) to ``path``."""
    import rerun as rr

    if len(frames) != len(vertices):
        raise ValueError(f"{len(frames)} frames for {len(vertices)} vertex rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream("mamma-dynamic-reference")
    recording.save(path)
    recording.send_columns(
        ENTITY,
        indexes=[rr.TimeColumn(FRAME_TIMELINE, sequence=frames), rr.TimeColumn("time", duration=frames / fps)],
        columns=rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3)).partition([vertices.shape[1]] * len(frames)),
    )
    recording.send_property("reference", rr.AnyValues(source=source, fps=fps))
    recording.flush()


@serde
@dataclass(frozen=True, slots=True)
class MammaPredParams:
    """One subject's ``pred/params_NN.npz`` as the MAMMA release ships it (third-party format; other fields allowed)."""

    betas: Float32[ndarray, "b"]
    """SMPL-X v1.1 neutral shape coefficients."""
    poses: Float32[ndarray, "f 165"]
    """Per-frame axis-angle pose (root, body, jaw, eyes, hands)."""
    trans: Float32[ndarray, "f 3"]
    """Per-frame root translation, metres."""

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.trans):
            raise ValueError(f"{len(self.poses)} pose rows for {len(self.trans)} translation rows")


def load_pred_params(path: Path) -> MammaPredParams:
    """Typed ``pred/params_NN.npz``; a malformed file raises ``ValueError`` naming ``path``."""
    with np.load(path, allow_pickle=False) as npz:
        fields: dict[str, ndarray] = {name: npz[name] for name in ("betas", "poses", "trans") if name in npz.files}
    try:
        return from_dict(MammaPredParams, fields)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: not a MAMMA prediction record ({error})") from error


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
    pred: MammaPredParams = load_pred_params(pred_npz)
    layer = SmplxLayerTorch(betas=pred.betas, model_root_dir=config.model_dir, flat_hand_mean=False)
    vertices: Float32[ndarray, "f v 3"] = np.asarray(layer.to(config.device).forward_batched(pred.poses, pred.trans).vertices, dtype=np.float32)
    frames = np.arange(len(vertices), dtype=np.int64)
    write_reference(
        config.out, frames, vertices, config.fps, f"dataset MAMMA prediction {pred_npz} via SMPL-X v1.1 neutral (16 betas), frozen 2026-11-21"
    )
    print(f"wrote {config.out}: {len(frames)} frames x {vertices.shape[1]} vertices")
    return 0
