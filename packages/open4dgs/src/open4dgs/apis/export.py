"""Export a trained run as a recording, registered as a layer of the run's segment on open4dgs-runs: ``open4dgs-export``.

The recording holds the run's cameras and SMPL-X meshes (copied from ``run.rrd``), ``ViewCoordinates`` on ``/world``
(up = the mean camera up), a default 3D view from a training camera, and the splats at ``/world/person/splats``:

- ``--format asog`` (default): the checkpoint encoded as ASOG (``open4dgs.asog.encode_asog``, ``--settings``, band-safe by
  default; or ``--asog <file>``) stored once as the gsplat viewer's ``AnimatedSogSplats3D`` payload plus a one-pose
  stock preview (``splat_export.write_asog_rrd``). Default output ``<run>/asog.rrd``, layer ``asog``; ``--asog-file``
  also writes the ``.asog`` for other players. A stock Rerun viewer shows one still pose.
- ``--format per-frame``: one ``GaussianSplats3D`` row per frame on ``frame_index``; the stock viewer plays it.
  Default output ``<run>/splats.rrd``, layer ``splats`` (not registered when it comes from ``--asog``).

What the compression costs is ``open4dgs-eval-asog``.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from simplecv.catalog_layers import register_layer
from simplecv.run_record import RunStart, start_run, write_run_record
from trigs.core.trainer import TrigsTrainingConfig

from open4dgs.asog import AsogSettings, decode_asog, encode_asog, tiny_omega_fraction
from open4dgs.catalog_clip import LOCAL_CATALOG
from open4dgs.properties import rrd_cells
from open4dgs.run_log import RUNS_DATASET, RunProperties
from open4dgs.splat_export import (
    ASOG_CUT_FORMAT,
    ASOG_LAYER,
    SPLAT_ENTITY,
    STOCK_VIEWER_NOTE,
    log_splat_frames,
    run_recording,
    splats_at,
    write_asog_rrd,
)
from open4dgs.trigs_adapter import TrainedClip

ExportFormat = Literal["asog", "per-frame"]
OUTPUTS: dict[str, tuple[str, str]] = {"asog": ("asog.rrd", ASOG_LAYER), "per-frame": ("splats.rrd", "splats")}
"""Per format: the default file in the run directory and the layer name."""


@dataclass(frozen=True, slots=True)
class Config:
    """One export."""

    run: Path
    """Run directory (``run.rrd`` + ``trigs_ckpt.pt``); its name is the segment id."""
    format: ExportFormat = "asog"
    """``asog`` (default): the ``.asog`` once as ``AnimatedSogSplats3D`` + a one-pose stock preview (stock viewers show one
    still pose); ``per-frame``: one ``GaussianSplats3D`` row per frame (plays in stock viewers)."""
    asog: Path | None = None
    """An ``.asog`` file of the run (e.g. from another encoder): ``asog`` stores it instead of encoding the checkpoint;
    ``per-frame`` exports it decoded instead of the checkpoint and never registers."""
    settings: AsogSettings = field(default_factory=AsogSettings)
    """``asog``: encoder preset (``band-safe`` default; ``--settings.preset small``: smaller file for web streaming) and overrides."""
    asog_file: Path | None = None
    """``asog``: also write the encoded ASOG as an ``.asog`` file here (interchange with other players)."""
    asog_cut: Literal["format", "trained"] = "format"
    """``asog``: temporal cut recorded for the viewer: ``format`` = the ``.asog`` v1 constant 0.05 (what other players show),
    ``trained`` = the trainer's 0.01 (the trained look; the same file then plays differently elsewhere)."""
    preview_rows: int = PREVIEW_ROWS
    """``asog``: stock-viewer preview rows of the clip-middle pose (0 = none; the gsplat viewer then draws nothing)."""
    frame_step: int = 1
    """``per-frame``: export every n-th clip frame."""
    with_sh: bool = False
    """``per-frame``: also log SH degrees 1-3 (+90 B per Gaussian per frame); DC colour only by default."""
    device: str = "cpu"
    """``per-frame``: device for the motion-model evaluation (cpu keeps the GPU free; about 1 s per frame for 328k rows)."""
    output: Path | None = None
    """Output path (default ``<run>/asog.rrd``, ``<run>/splats.rrd`` for ``per-frame``)."""
    register: bool = True
    runs_catalog_url: str = LOCAL_CATALOG


def write_asog(config: Config, trained: TrainedClip, out: Path, run: RunStart) -> dict[str, object]:
    """``out``: the ``--asog`` file or the encoded checkpoint as ``AnimatedSogSplats3D`` (``write_asog_rrd``); the summary."""
    summary: dict[str, object] = {"rows_checkpoint": int(trained.params["means"].shape[0])}
    if config.asog is not None:
        asog: bytes = config.asog.read_bytes()
        summary["asog_input"] = str(config.asog)
    else:
        params = dict(trained.params)
        t0: float = time.perf_counter()
        asog, stats = encode_asog(params, trained.fps, trained.num_frames, config.settings)
        summary.update(
            {
                "preset": config.settings.preset,
                "settings": asdict(config.settings.options()),
                "rows_asog": stats.count,
                "bytes_per_splat": len(asog) / stats.count,
                "stream_bytes": stats.stream_bytes,
                "encode_s": time.perf_counter() - t0,
                "tiny_omega_fraction": tiny_omega_fraction(params),
            }
        )
    if config.asog_file is not None:
        config.asog_file.parent.mkdir(parents=True, exist_ok=True)
        config.asog_file.write_bytes(asog)
        config.asog_file.chmod(0o644)
        summary["asog_file"] = str(config.asog_file)
    summary["bytes_checkpoint"] = (config.run / "trigs_ckpt.pt").stat().st_size
    summary["bytes_asog"] = len(asog)
    cut: float = ASOG_CUT_FORMAT if config.asog_cut == "format" else TrigsTrainingConfig().temporal_active_threshold
    summary.update(write_asog_rrd(config.run, asog, out, cut, config.preview_rows, run))
    return summary


def write_per_frame(config: Config, trained: TrainedClip, out: Path, run: RunStart) -> dict[str, object]:
    """``out``: one ``GaussianSplats3D`` row per (``frame_step``-th) clip frame of the checkpoint or the decoded ``--asog``."""
    device: torch.device = torch.device(config.device)
    if config.asog is not None:
        trained = trained.with_params(decode_asog(config.asog.read_bytes(), device)[0])
    reader: RrdReader = RrdReader(config.run / "run.rrd")
    frame_start: int = RunProperties.from_cells(rrd_cells(reader, "run")).frame_start
    first = splats_at(trained, 0.0)
    recording, view_coordinates, view = run_recording(reader, config.run.name, out, np.median(first.centers, axis=0).astype(np.float64))
    t0: float = time.perf_counter()
    frames = range(0, trained.num_frames, config.frame_step)
    counts: list[int] = log_splat_frames(
        recording, ((frame_start + f, splats_at(trained, f / trained.fps, config.with_sh)) for f in frames), SPLAT_ENTITY
    )
    write_run_record(recording, run.finish(OUTPUTS["per-frame"][1], config.run.name))
    recording.flush()
    recording.disconnect()
    return {
        "splats_rrd": str(out),
        "bytes": out.stat().st_size,
        "frames": len(counts),
        "gaussians_min": min(counts),
        "gaussians_median": int(np.median(counts)),
        "gaussians_max": max(counts),
        "rows": trained.params["means"].shape[0],
        "export_s": time.perf_counter() - t0,
        "view_coordinates": str(view_coordinates),
        "blueprint_view_id": str(view.id),
    }


def main(config: Config) -> dict[str, object]:
    """Write the run's ``--format`` recording with this export's run record (``simplecv.run_record``), register it as that
    format's layer; prints and returns the summary."""
    run: RunStart = start_run(config, source=Path(__file__))
    device: torch.device = torch.device(config.device if config.format == "per-frame" else "cpu")
    trained: TrainedClip = TrainedClip.load(config.run / "trigs_ckpt.pt", device)
    file, layer = OUTPUTS[config.format]
    out: Path = config.output or config.run / file
    summary: dict[str, object] = (write_asog if config.format == "asog" else write_per_frame)(config, trained, out, run)
    out.chmod(0o644)
    # A per-frame export of another encoder's .asog is a comparison, not the run's layer.
    if config.register and not (config.format == "per-frame" and config.asog is not None):
        register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out], layer)
        summary["registered"] = f"{RUNS_DATASET}/{config.run.name}/{layer}"
    print(json.dumps(summary, indent=2, default=float))
    if config.format == "asog":
        print(f"[open4dgs-export] {STOCK_VIEWER_NOTE}")
    return summary
