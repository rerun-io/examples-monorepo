"""Per-frame Gaussian splats from a TRiGS checkpoint, as stock ``rr.GaussianSplats3D`` on the ``frame_index`` timeline.

The TRiGS motion model is evaluated at every clip frame (``TrainedClip.deform``: centres and wxyz quaternions at time
t, temporal opacity gamma). Opacity = sigmoid(logit) * gamma; rows below 1/255 are culled. Colours are the SH DC term
(``0.5 + SH_C0 * sh0``) with alpha = opacity; the SH rest (degrees 1-3) is optional. The world frame is the training
world, which is the run recording's world (``open4dgs.trigs_adapter.camera_config``), so the run's cameras line up.

The custom viewer (packages/gsplat-rust-renderer) and the stock viewer both draw these rows; nothing here is
viewer-specific.

Also what every export of a run shares (``apis.export``, ``apis.eval_asog``): the run's ``run`` property, a
recording with the run's cameras and meshes (``run_recording``) and the checkpoint's parameter tensors.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gsplat_rust_renderer import asog_decode
from gsplat_rust_renderer.sog3d import log_asog
from jaxtyping import Float16, Float32, Float64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE
from open4dgs.properties import rrd_cells
from open4dgs.run_log import RunProperties
from open4dgs.trigs_adapter import TrainedClip

SH_C0: float = 0.28209479177387814
"""Degree-0 spherical-harmonics constant."""
MIN_OPACITY: float = 1.0 / 255.0
"""Rows at or below this opacity are dropped (they round to alpha 0)."""
SPLAT_ENTITY: str = "/world/person/splats"
ASOG_CUT_FORMAT: float = 0.05
"""The ``.asog`` v1 temporal cut (no manifest field; what every other player applies)."""
STOCK_VIEWER_NOTE: str = "a stock Rerun viewer shows one still pose; use --format per-frame for playback in stock viewers"
"""Printed by ``open4dgs-export --format asog``."""


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


def splats_at(model: TrainedClip, t: float, with_sh: bool = False) -> FrameSplats:
    """Evaluate the motion model at time ``t`` (seconds) and keep the rows with opacity above 1/255."""
    params = model.params
    with torch.no_grad():
        means_t, quats_t, gamma = model.deform(t)
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


def _train_camera_poses(reader: RrdReader) -> np.ndarray:
    """(c, 4, 4) world_T_cam of the training cameras, read back from the run's static Transform3D rows."""
    poses: list[np.ndarray] = []
    for chunk in reader.stream().filter(content="/world/cameras/train/**", is_static=True).to_chunks():
        batch = chunk.to_record_batch()
        if "Transform3D:translation" not in batch.schema.names:
            continue
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(batch.column("Transform3D:mat3x3").to_pylist()[0][0], dtype=np.float64).reshape(3, 3).T  # column-major
        pose[:3, 3] = np.asarray(batch.column("Transform3D:translation").to_pylist()[0][0], dtype=np.float64)
        poses.append(pose)
    if not poses:
        raise ValueError(f"{reader.path} has no training camera transforms")
    return np.stack(poses)


def run_recording(
    reader: RrdReader, run_name: str, out: Path, target: Float64[ndarray, " 3"]
) -> tuple[rr.RecordingStream, rr.components.ViewCoordinates, rrb.Spatial3DView]:
    """A recording saved to ``out`` (application ``open4dgs-runs``, id = the run name) holding the run's cameras and SMPL-X
    meshes, ``ViewCoordinates`` on ``/world`` (up = the mean camera up) and a 3D view from training camera 0 at ``target``."""
    poses: np.ndarray = _train_camera_poses(reader)
    view_coordinates: rr.components.ViewCoordinates = up_view_coordinates(poses.astype(np.float32))
    up: np.ndarray = -poses[:, :3, 1].mean(axis=0)
    eye: np.ndarray = poses[0, :3, 3]
    recording: rr.RecordingStream = rr.RecordingStream("open4dgs-runs", recording_id=run_name)
    recording.save(out)
    recording.send_chunks(reader.stream().filter(content=["+ /world/cameras/**", "+ /world/people/**"]))
    recording.log("/world", view_coordinates, static=True)
    view = rrb.Spatial3DView(
        origin="/world",
        name="splats",
        eye_controls=rrb.EyeControls3D(position=eye.tolist(), look_target=target.tolist(), eye_up=(up / np.linalg.norm(up)).tolist()),
    )
    recording.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
    return recording, view_coordinates, view


def write_asog_rrd(run_dir: Path, asog: bytes, out: Path, cut: float, preview_rows: int) -> dict[str, object]:
    """``out``: the run's cameras and meshes (``run_recording``) and the ``asog`` bytes verbatim as the gsplat viewer's
    ``AnimatedSogSplats3D`` payload (``gsplat_rust_renderer.sog3d.log_asog``, ``frame_origin`` = the clip's first frame,
    temporal cut ``cut``) with a stock preview of the clip-middle pose (at most ``preview_rows`` rows); mode 644.

    The ASOG writer of ``open4dgs-export`` (which registers).
    """
    reader: RrdReader = RrdReader(run_dir / "run.rrd")
    frame_start: int = RunProperties.from_cells(rrd_cells(reader, "run")).frame_start
    parsed = asog_decode.parse_asog(asog, f"{run_dir.name} asog")
    target: Float64[ndarray, " 3"] = np.median(asog_decode.decode(parsed).means, axis=0).astype(np.float64)
    recording, view_coordinates, view = run_recording(reader, run_dir.name, out, target)
    t0: float = time.perf_counter()
    log_asog(SPLAT_ENTITY, parsed, float(frame_start), cut, preview_rows, recording=recording)
    log_s: float = time.perf_counter() - t0
    recording.flush()
    recording.disconnect()
    out.chmod(0o644)
    return {
        "asog_rrd": str(out),
        "bytes_asog_rrd": out.stat().st_size,
        "frame_origin": frame_start,
        "active_threshold": cut,
        "preview_rows_max": preview_rows,
        "log_s": log_s,
        "view_coordinates": str(view_coordinates),
        "blueprint_view_id": str(view.id),
    }
