"""Per-frame Gaussian splats from a TRiGS checkpoint, as stock ``rr.GaussianSplats3D`` on the ``frame_index`` timeline.

The TRiGS motion model is evaluated at every clip frame (``TrigsTrainer._deform``: centres and wxyz quaternions at time
t, temporal opacity gamma). Opacity = sigmoid(logit) * gamma; rows below 1/255 are culled. Colours are the SH DC term
(``0.5 + SH_C0 * sh0``) with alpha = opacity; the SH rest (degrees 1-3) is optional. The world frame is the training
world, which is the run recording's world (``open4dgs.trigs_adapter.camera_config``), so the run's cameras line up.

The custom viewer (packages/gsplat-rust-renderer) and the stock viewer both draw these rows; nothing here is
viewer-specific.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float16, Float32, UInt8
from numpy import ndarray
from torch import Tensor
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

from open4dgs.catalog_clip import FRAME_TIMELINE

SH_C0: float = 0.28209479177387814
"""Degree-0 spherical-harmonics constant."""
MIN_OPACITY: float = 1.0 / 255.0
"""Rows at or below this opacity are dropped (they round to alpha 0)."""
SPLAT_ENTITY: str = "/world/person/splats"


@dataclass(frozen=True, slots=True)
class FrameSplats:
    """The visible Gaussians of one frame, in Rerun's conventions."""

    centers: Float32[ndarray, "n 3"]
    scales: Float32[ndarray, "n 3"]
    """Linear standard deviations (exp of the stored log scales)."""
    quaternions: Float32[ndarray, "n 4"]
    """xyzw, unit length."""
    colors: UInt8[ndarray, "n 4"]
    """RGB from the SH DC term, A = opacity at this frame."""
    sh_rest: Float16[ndarray, "n 15 3"] | None
    """SH degrees 1-3 (``shN``), when asked for."""


def wxyz_to_xyzw(quats: Float32[Tensor, "n 4"]) -> Float32[Tensor, "n 4"]:
    """gsplat / TRiGS order (w, x, y, z) -> Rerun order (x, y, z, w)."""
    return torch.roll(quats, shifts=-1, dims=-1)


def dc_rgba(sh0: Float32[Tensor, "n 3"], opacity: Float32[Tensor, " n"]) -> UInt8[Tensor, "n 4"]:
    """RGBA8: RGB = clip(0.5 + SH_C0 * sh0), A = opacity."""
    rgba: Tensor = torch.cat([0.5 + SH_C0 * sh0, opacity[:, None]], dim=-1).clamp(0.0, 1.0)
    return (rgba * 255.0 + 0.5).to(torch.uint8)


def splats_at(params: Mapping[str, Tensor], t: float, trainer: TrigsTrainer, config: TrigsTrainingConfig, with_sh: bool = False) -> FrameSplats:
    """Evaluate the motion model at time ``t`` (seconds) and keep the rows with opacity above 1/255."""
    with torch.no_grad():
        means_t, quats_t, gamma = trainer._deform(params, t, config)  # pyrefly: ignore[bad-argument-type]
        opacity: Tensor = torch.sigmoid(params["opacities"].reshape(-1)) * gamma.reshape(-1)
        keep: Tensor = opacity > MIN_OPACITY
        quats: Tensor = torch.nn.functional.normalize(quats_t[keep], dim=-1)
        sh_rest: ndarray | None = params["shN"][keep].to(torch.float16).cpu().numpy() if with_sh else None
        return FrameSplats(
            centers=means_t[keep].float().cpu().numpy(),
            scales=torch.exp(params["scales"][keep]).float().cpu().numpy(),
            quaternions=wxyz_to_xyzw(quats).float().cpu().numpy(),
            colors=dc_rgba(params["sh0"].reshape(-1, 3)[keep], opacity[keep]).cpu().numpy(),
            sh_rest=sh_rest,
        )


def checkpoint_params(checkpoint: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    """The parameter tensors of an ``open4dgs.trigs_adapter.TrainedClip.save`` checkpoint, on ``device``."""
    params = checkpoint["params"]
    if not isinstance(params, Mapping):
        raise ValueError("checkpoint has no 'params' mapping")
    return {str(name): value.to(device) for name, value in params.items()}


def log_splat_frames(
    recording: rr.RecordingStream, frames: Iterator[tuple[int, FrameSplats]], entity: str = SPLAT_ENTITY, batch_frames: int = 8
) -> list[int]:
    """Send ``(frame_index, splats)`` rows as ``GaussianSplats3D`` columns (``batch_frames`` frames per call); returns the counts."""
    counts: list[int] = []
    batch: list[tuple[int, FrameSplats]] = []

    def flush() -> None:
        if not batch:
            return
        lengths: list[int] = [len(s.centers) for _, s in batch]
        has_sh: bool = batch[0][1].sh_rest is not None
        columns = rr.GaussianSplats3D.columns(
            centers=np.concatenate([s.centers for _, s in batch]),
            scales=np.concatenate([s.scales for _, s in batch]),
            quaternions=np.concatenate([s.quaternions for _, s in batch]),
            colors=np.concatenate([s.colors for _, s in batch]),
            **(
                {
                    "sh_coefficients": np.concatenate([s.sh_rest for _, s in batch if s.sh_rest is not None]),
                    "spherical_harmonics_degree": [3] * sum(lengths),
                }
                if has_sh
                else {}
            ),
        ).partition(lengths)
        recording.send_columns(entity, indexes=[rr.TimeColumn(FRAME_TIMELINE, sequence=[f for f, _ in batch])], columns=columns)
        batch.clear()

    for frame_index, splats in frames:
        counts.append(len(splats.centers))
        batch.append((frame_index, splats))
        if len(batch) >= batch_frames:
            flush()
    flush()
    return counts


def up_view_coordinates(world_T_cam: Float32[ndarray, "c 4 4"]) -> rr.components.ViewCoordinates:
    """Right-handed world coordinates whose up axis is the dominant axis of the mean camera up (OpenCV: up = -y)."""
    up: ndarray = -world_T_cam[:, :3, 1].mean(axis=0)
    axis: int = int(np.argmax(np.abs(up)))
    positive: bool = bool(up[axis] > 0)
    table = {
        (0, True): rr.ViewCoordinates.RIGHT_HAND_X_UP,
        (0, False): rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
        (1, True): rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        (1, False): rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        (2, True): rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        (2, False): rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
    }
    return table[(axis, positive)]
