"""Export a trained run as Gaussian splats: a recording -> a layer of the run's segment on open4dgs-runs.

The recording holds the run's cameras and SMPL-X meshes (copied from ``run.rrd``), ``ViewCoordinates`` on ``/world``
(up = the mean camera up), a default 3D view from a training camera, and the splats at ``/world/person/splats``:

- ``--format asog`` (default): the checkpoint encoded as ASOG (``open4dgs.asog.encode_asog``, band-safe; or ``--asog``) stored once as the gsplat
  viewer's ``AnimatedSogSplats3D`` payload plus a one-pose stock preview, exactly what ``open4dgs-export-asog --rrd``
  writes (one writer: ``splat_export.write_asog_rrd``). Default output ``<run>/asog.rrd``, layer ``asog``. A stock
  Rerun viewer shows one still pose.
- ``--format per-frame``: one ``GaussianSplats3D`` row per frame on ``frame_index``; the stock viewer plays it.
  Default output ``<run>/splats.rrd``, layer ``splats``.
- ``--format trigs-motion``: the checkpoint once, static: canonical ``GaussianSplats3D`` + ``TrigsMotion3D``
  (``gsplat_rust_renderer.trigs_motion3d``), which the custom viewer (packages/gsplat-rust-renderer) evaluates at the
  time cursor; stock viewers draw the canonical rows only. Default output ``<run>/trigs_motion.rrd``, layer ``trigs_motion``.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS
from gsplat_rust_renderer.trigs_motion3d import TrigsParams, log_trigs_motion
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from simplecv.catalog_layers import register_layer
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

from open4dgs.asog import AsogSettings, decode_asog, encode_asog
from open4dgs.catalog_clip import LOCAL_CATALOG
from open4dgs.run_log import RUNS_DATASET
from open4dgs.splat_export import (
    ASOG_CUT_FORMAT,
    SPLAT_ENTITY,
    STOCK_VIEWER_NOTE,
    checkpoint_params,
    log_splat_frames,
    run_property,
    run_recording,
    splats_at,
    trigs_motion_params,
    write_asog_rrd,
)


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
    """An ``.asog`` file of the run (e.g. from another encoder): ``asog`` stores it instead of encoding the checkpoint;
    ``per-frame`` exports it decoded instead of the checkpoint and never registers."""
    output: Path | None = None
    """Output path (default ``<run>/asog.rrd``, ``<run>/splats.rrd`` for ``per-frame``, ``<run>/trigs_motion.rrd`` for ``trigs-motion``)."""
    format: Literal["asog", "per-frame", "trigs-motion"] = "asog"
    """``asog`` (default): the ``.asog`` once as ``AnimatedSogSplats3D`` + a one-pose stock preview (stock viewers show one
    still pose); ``per-frame``: one ``GaussianSplats3D`` row per frame (plays in stock viewers); ``trigs-motion``: the
    checkpoint once as ``TrigsMotion3D`` (the gsplat viewer only)."""


def main(config: Config) -> dict[str, object]:
    """Write ``<run>/asog.rrd`` (or ``splats.rrd``, ``trigs_motion.rrd``), register it as layer ``asog`` (``splats``,
    ``trigs_motion``); returns sizes and counts."""
    run_name: str = config.run.name
    if config.format == "asog":
        if config.asog is not None:
            asog: bytes = config.asog.read_bytes()
        else:
            checkpoint = torch.load(config.run / "trigs_ckpt.pt", map_location="cpu", weights_only=True)
            asog = encode_asog(
                checkpoint_params(checkpoint, torch.device("cpu")), float(checkpoint["fps"]), int(checkpoint["num_frames"]), AsogSettings()
            )[0]
        out_asog: Path = config.output or config.run / "asog.rrd"
        summary_asog: dict[str, object] = write_asog_rrd(config.run, asog, out_asog, ASOG_CUT_FORMAT, PREVIEW_ROWS)
        if config.register:
            register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out_asog], "asog")
            summary_asog["registered"] = f"{RUNS_DATASET}/{run_name}/asog"
        print(summary_asog)
        print(f"[open4dgs-export-splats] {STOCK_VIEWER_NOTE}")
        return summary_asog
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
        compact: TrigsParams = trigs_motion_params(checkpoint_params(checkpoint, torch.device("cpu")))
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
    params = decode_asog(config.asog.read_bytes(), device)[0] if config.asog is not None else checkpoint_params(checkpoint, device)
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
