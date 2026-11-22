"""Export a trained run as Gaussian splats: ``<run>/splats.rrd`` -> layer ``splats`` of the run's segment.

The recording holds the run's cameras and SMPL-X meshes (copied from ``run.rrd``), ``ViewCoordinates`` on ``/world``
(up = the mean camera up), a default 3D view from a training camera, and the splats at ``/world/person/splats``:

- ``--format per-frame`` (default): one ``GaussianSplats3D`` row per frame on ``frame_index``; the stock viewer plays it.
- ``--format trigs-motion``: the checkpoint once, static: canonical ``GaussianSplats3D`` + ``TrigsMotion3D``
  (``gsplat_rust_renderer.trigs_motion3d``), which the custom viewer (packages/gsplat-rust-renderer) evaluates at the
  time cursor; stock viewers draw the canonical rows only. Default output ``<run>/trigs_motion.rrd``, layer ``trigs_motion``.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gsplat_rust_renderer.trigs_motion3d import TrigsParams, log_trigs_motion
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from simplecv.catalog_layers import register_layer
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

from open4dgs.asog import decode_asog
from open4dgs.run_log import RUNS_DATASET
from open4dgs.splat_export import SPLAT_ENTITY, checkpoint_params, log_splat_frames, splats_at, up_view_coordinates

LOCAL_CATALOG: str = "rerun+http://pablo-dl-server:51235"


@dataclass(frozen=True, slots=True)
class Config:
    """One export."""

    run: Path
    """Run directory (``run.rrd`` + ``trigs_ckpt.pt``); its name is the segment id."""
    frame_step: int = 1
    """Export every n-th clip frame."""
    with_sh: bool = False
    """Also log SH degrees 1-3 (+90 B per Gaussian per frame); DC colour only by default."""
    device: str = "cpu"
    """Device for the motion-model evaluation (cpu keeps the GPU free; about 1 s per frame for 328k rows)."""
    register: bool = True
    runs_catalog_url: str = LOCAL_CATALOG
    asog: Path | None = None
    """Export this ``.asog`` export of the run (decoded, open4dgs.asog) instead of the checkpoint; never registered."""
    output: Path | None = None
    """Output path (default ``<run>/splats.rrd``, ``<run>/trigs_motion.rrd`` for ``trigs-motion``)."""
    format: Literal["per-frame", "trigs-motion"] = "per-frame"
    """``per-frame``: one ``GaussianSplats3D`` row per frame; ``trigs-motion``: the checkpoint once as ``TrigsMotion3D``."""


def run_property(reader: RrdReader, name: str) -> int:
    for chunk in reader.stream().filter(content="/__properties/run").to_chunks():
        batch = chunk.to_record_batch()
        if name in batch.schema.names:
            return int(batch.column(name).to_pylist()[0][0])
    raise ValueError(f"{reader.path} has no run property {name!r}")


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


def trigs_params(params: dict[str, torch.Tensor]) -> TrigsParams:
    """The checkpoint's raw TRiGS parameters as the viewer contract's f32 arrays (per-row scalars flattened)."""

    def f32(name: str, *shape: int) -> np.ndarray:
        return params[name].detach().float().cpu().numpy().reshape(-1, *shape)  # no shape: one value per row

    zetas: dict[str, np.ndarray] = {f"zeta_{part}_{kind}": f32(f"zeta_{part}_{kind}", 3) for part in ("base", "c0", "c1", "c2") for kind in ("t", "r")}
    return TrigsParams(
        means=f32("means", 3),
        quats=f32("quats", 4),
        scales=f32("scales", 3),
        opacities=f32("opacities"),
        sh0=f32("sh0", 3),
        times=f32("times"),
        log_durations=f32("log_durations"),
        anchors=f32("anchors", 3),
        is_static=f32("is_static") if "is_static" in params else None,
        **zetas,
    )


def main(config: Config) -> dict[str, object]:
    """Write ``<run>/splats.rrd`` (or ``trigs_motion.rrd``), register it as layer ``splats`` (``trigs_motion``); returns sizes and counts."""
    run_name: str = config.run.name
    motion: bool = config.format == "trigs-motion"
    if motion and config.asog is not None:
        raise ValueError("--format trigs-motion exports the checkpoint; --asog exports per frame only")
    out: Path = config.output or config.run / ("trigs_motion.rrd" if motion else "splats.rrd")
    layer: str = "trigs_motion" if motion else "splats"
    reader: RrdReader = RrdReader(config.run / "run.rrd")
    frame_start: int = run_property(reader, "frame_start")
    checkpoint = torch.load(config.run / "trigs_ckpt.pt", map_location="cpu", weights_only=True)
    fps: float = float(checkpoint["fps"])
    num_frames: int = int(checkpoint["num_frames"])
    if motion:
        compact: TrigsParams = trigs_params(checkpoint_params(checkpoint, torch.device("cpu")))
        recording, view_coordinates, view = run_recording(reader, run_name, out, np.median(compact.means, axis=0).astype(np.float64))
        t0 = time.perf_counter()
        cut: float = TrigsTrainingConfig().temporal_active_threshold
        log_trigs_motion(SPLAT_ENTITY, compact, fps, float(frame_start), cut, recording=recording)
        recording.flush()
        recording.disconnect()
        out.chmod(0o644)
        summary: dict[str, object] = {
            "trigs_motion_rrd": str(out),
            "bytes": out.stat().st_size,
            "rows": len(compact.means),
            "fps": fps,
            "frame_origin": frame_start,
            "active_threshold": cut,
            "export_s": time.perf_counter() - t0,
            "view_coordinates": str(view_coordinates),
            "blueprint_view_id": str(view.id),
        }
        if config.register:
            register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out], layer)
            summary["registered"] = f"{RUNS_DATASET}/{run_name}/{layer}"
        print(summary)
        return summary
    device: torch.device = torch.device(config.device)
    params = decode_asog(config.asog, device)[0] if config.asog is not None else checkpoint_params(checkpoint, device)
    trainer: TrigsTrainer = TrigsTrainer(device=str(device))
    trigs_config: TrigsTrainingConfig = TrigsTrainingConfig(fps=fps)  # _deform reads only the TRiGS defaults (gauge, smooth cull)

    first = splats_at(params, 0.0, trainer, trigs_config)
    recording, view_coordinates, view = run_recording(reader, run_name, out, np.median(first.centers, axis=0).astype(np.float64))

    t0: float = time.perf_counter()
    frames = range(0, num_frames, config.frame_step)
    counts: list[int] = log_splat_frames(
        recording, ((frame_start + f, splats_at(params, f / fps, trainer, trigs_config, config.with_sh)) for f in frames), SPLAT_ENTITY
    )
    recording.flush()
    recording.disconnect()
    out.chmod(0o644)
    summary = {
        "splats_rrd": str(out),
        "bytes": out.stat().st_size,
        "frames": len(counts),
        "gaussians_min": min(counts),
        "gaussians_median": int(np.median(counts)),
        "gaussians_max": max(counts),
        "rows": params["means"].shape[0],
        "export_s": time.perf_counter() - t0,
        "view_coordinates": str(view_coordinates),
        "blueprint_view_id": str(view.id),
    }
    if config.register and config.asog is None:
        register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out], layer)
        summary["registered"] = f"{RUNS_DATASET}/{run_name}/{layer}"
    print(summary)
    return summary
