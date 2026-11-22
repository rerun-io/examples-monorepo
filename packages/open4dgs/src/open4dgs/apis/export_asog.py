"""Compress a trained run to Animated SOG (``.asog``) and, optionally, measure what the compression costs.

``<run>/trigs_ckpt.pt`` -> ``<out_dir>/<run>.asog`` (an explicit third-party export, mode 644; the run's data stays
``.rrd``) -> decoded back to TRiGS parameters. With ``--evaluate`` the held-out cameras of the run (its ``run`` property:
dataset, segment, frames, cameras, size, ROI, mask layer) are re-read from the catalog and rendered with the trainer's
renderer from the uncompressed and the decoded parameters, scored like ``open4dgs-train`` (person crop), and compared
with each other (PSNR of decoded vs uncompressed renders). ``decoded_cut05`` renders the decoded clip with the ``.asog``
v1 temporal cut (0.05) instead of the trainer's (0.01): what a stock ``.asog`` player shows.

The ``.asog`` also goes into a Rerun recording, ``<run>/asog.rrd`` (``--rrd``): the file verbatim as the custom gsplat
viewer's ``AnimatedSogSplats3D`` payload (``gsplat_rust_renderer.sog3d.log_asog``) with ``frame_origin`` = the clip's first frame, a stock
preview of the clip-middle pose on the same entity, and the run's cameras, SMPL-X meshes and ``ViewCoordinates`` as
``open4dgs-export-splats`` logs them. It is registered as layer ``asog`` of the run on open4dgs-runs (``--no-register``).
"""

import dataclasses
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from asog import Asog
from asog import read as read_asog
from gsplat_rust_renderer.asog_decode import parse_asog
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, log_asog
from PIL import Image
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from simplecv.catalog_layers import register_layer
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

from open4dgs.asog import AsogSettings, decode_asog, encode_params, tiny_omega_fraction
from open4dgs.catalog_clip import LOCAL_CATALOG, MultiViewClip, read_catalog_clip
from open4dgs.evaluate import FrameMetrics, Metrics, mask_box, mean_metrics, psnr
from open4dgs.masks import read_clip_masks
from open4dgs.run_log import RUNS_DATASET
from open4dgs.splat_export import SPLAT_ENTITY, checkpoint_params, run_properties, run_property, run_recording
from open4dgs.trigs_adapter import TrainConfig, TrainedClip, camera_config

ASOG_CUT_FORMAT: float = 0.05
"""The ``.asog`` v1 temporal cut (no manifest field; what every other player applies)."""


@dataclass(frozen=True, slots=True)
class Config:
    """One export (and optional evaluation)."""

    run: Path
    """Run directory (``run.rrd`` + ``trigs_ckpt.pt``)."""
    out_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/asog")
    """Where ``<run>.asog`` goes."""
    name: str | None = None
    """File stem (default: the run name)."""
    settings: AsogSettings = field(default_factory=AsogSettings)
    """Encoder preset (``band-safe`` default; ``--settings.preset small``: smaller file for web streaming) and per-option overrides."""
    evaluate: bool = False
    """Render the held-out cameras from both models and score them (needs the GPU and the catalog)."""
    mask_catalog_url: str = LOCAL_CATALOG
    """Catalog of the run's masks layer (local for every run so far)."""
    png_dir: Path | None = None
    """With ``evaluate``: ground truth | uncompressed | decoded PNGs at two frames per held-out camera."""
    png_cameras: int = 2
    """Held-out cameras that get PNGs."""
    rrd: Path | None = None
    """Recording with the ``.asog`` as the gsplat viewer's ``AnimatedSogSplats3D`` payload (default ``<run>/asog.rrd``)."""
    asog_cut: Literal["format", "trained"] = "format"
    """Temporal cut recorded for the viewer: ``format`` = the ``.asog`` v1 constant 0.05 (what other players show),
    ``trained`` = the trainer's 0.01 (the trained look; the same file then plays differently elsewhere)."""
    preview_rows: int = PREVIEW_ROWS
    """Stock-viewer preview rows of the clip-middle pose (0 = none; the gsplat viewer then draws nothing)."""
    register: bool = True
    """Register the recording as layer ``asog`` of the run on open4dgs-runs."""
    runs_catalog_url: str = LOCAL_CATALOG


def _model(params: dict[str, torch.Tensor], clip: MultiViewClip, fps: float, num_frames: int, holdout: tuple[int, ...], cut: float) -> TrainedClip:
    trainer: TrigsTrainer = TrigsTrainer(device="cuda")
    trainer._sequence_t_max = (num_frames - 1) / fps
    trigs_config = dataclasses.replace(TrainConfig().to_trigs(fps, 3, holdout), temporal_active_threshold=cut)
    return TrainedClip(
        params=params,  # pyrefly: ignore[bad-argument-type]
        trainer=trainer,
        trigs_config=trigs_config,
        cameras=camera_config(clip.world_T_cam, clip.K, clip.sizes, torch.device("cuda")),
        sh_degree=3,
        fps=fps,
        num_frames=num_frames,
        scalars={},
        train_seconds=0.0,
        peak_gpu_bytes=0,
    )


def _evaluate(config: Config, params: dict[str, torch.Tensor], decoded: dict[str, torch.Tensor], fps: float, num_frames: int) -> dict[str, object]:
    props = run_properties(RrdReader(config.run / "run.rrd"))
    roi: str = str(props.get("roi", ["full"])[0])  # runs before the ROI cameras have no roi property
    clip: MultiViewClip = read_catalog_clip(
        str(props["catalog_url"][0]),
        str(props["dataset"][0]),
        str(props["segment_id"][0]),
        int(props["frame_start"][0]),  # pyrefly: ignore[bad-argument-type]
        int(props["num_frames"][0]),  # pyrefly: ignore[bad-argument-type]
        cameras=[str(c) for c in props["cameras"]],
        max_side=int(props["max_side"][0]) or None,  # pyrefly: ignore[bad-argument-type]
        roi="people" if roi.startswith("people") else None,
        roi_pad=float(roi.split(":")[1]) if roi.startswith("people") else 0.15,
    )
    mask_dataset: str = str(props["mask_source"][0]).removeprefix("layer:").rsplit("/", 1)[0]  # ".../masks" (older runs) or ".../people_matte"
    masks = read_clip_masks(clip, config.mask_catalog_url, mask_dataset, str(props["segment_id"][0]))
    holdout_names: tuple[str, ...] = tuple(str(c) for c in props["holdout_cameras"])
    holdout: tuple[int, ...] = tuple(clip.camera_names.index(n) for n in holdout_names)
    models: dict[str, TrainedClip] = {
        "uncompressed": _model({k: v.cuda() for k, v in params.items()}, clip, fps, num_frames, holdout, 0.01),
        "decoded": _model({k: v.cuda() for k, v in decoded.items()}, clip, fps, num_frames, holdout, 0.01),
        "decoded_cut05": _model({k: v.cuda() for k, v in decoded.items()}, clip, fps, num_frames, holdout, 0.05),
    }
    metrics: Metrics = Metrics(torch.device("cuda"))
    rows: dict[str, list[FrameMetrics]] = {name: [] for name in models}
    vs_full: list[float] = []
    vs_crop: list[float] = []
    png_frames: tuple[int, ...] = (num_frames // 4, (3 * num_frames) // 4)
    for ci, (name, cam) in enumerate(zip(holdout_names, holdout, strict=True)):
        for f in range(num_frames):
            gt = clip.frames[cam][f].cuda().float() / 255.0
            mask = masks[cam][f].cuda().float()
            outputs = {key: model.render(model.cameras, cam, f) for key, model in models.items()}
            renders = {key: rgb for key, (rgb, _) in outputs.items()}
            for key, (rgb, alpha) in outputs.items():
                rows[key].append(metrics.frame(name, int(clip.frame_indices[f]), rgb, alpha[..., 0], gt, mask))
            vs_full.append(psnr(renders["decoded"], renders["uncompressed"]))
            box = mask_box(mask > 0.5)
            if box is not None:
                x0, y0, x1, y1 = box
                vs_crop.append(psnr(renders["decoded"][y0:y1, x0:x1], renders["uncompressed"][y0:y1, x0:x1]))
            if config.png_dir is not None and ci < config.png_cameras and f in png_frames:
                config.png_dir.mkdir(parents=True, exist_ok=True)
                tiles = [gt, renders["uncompressed"], renders["decoded"]]
                strip = np.concatenate([(t.clamp(0, 1) * 255).round().byte().cpu().numpy() for t in tiles], axis=1)
                stem = config.name or config.run.name
                Image.fromarray(strip).save(config.png_dir / f"{stem}-{name.replace('/', '_')}-f{int(clip.frame_indices[f])}-gt_uncompressed_decoded.png")
    finite = [v for v in vs_full if math.isfinite(v)]
    return {
        "holdout_cameras": list(holdout_names),
        **{f"holdout_{key}": mean_metrics(value) for key, value in rows.items()},
        "psnr_decoded_vs_uncompressed_full": float(np.mean(finite)) if finite else math.inf,
        "psnr_decoded_vs_uncompressed_crop": float(np.mean([v for v in vs_crop if math.isfinite(v)])),
    }


def write_rrd(config: Config, asog_path: Path, out: Path) -> dict[str, object]:
    """Log ``asog_path`` with the run's cameras and meshes into ``out``; registers it unless ``--no-register``."""
    run_name: str = config.run.name
    reader: RrdReader = RrdReader(config.run / "run.rrd")
    frame_start: int = run_property(reader, "frame_start")
    cut: float = ASOG_CUT_FORMAT if config.asog_cut == "format" else TrigsTrainingConfig().temporal_active_threshold
    clip: Asog = read_asog(asog_path)
    target = np.median(clip.means, axis=0).astype(np.float64)
    recording, view_coordinates, view = run_recording(reader, run_name, out, target)
    t0: float = time.perf_counter()
    log_asog(SPLAT_ENTITY, parse_asog(asog_path.read_bytes(), str(asog_path)), float(frame_start), cut, config.preview_rows, recording=recording)
    log_s: float = time.perf_counter() - t0
    recording.flush()
    recording.disconnect()
    out.chmod(0o644)
    summary: dict[str, object] = {
        "asog_rrd": str(out),
        "bytes_asog_rrd": out.stat().st_size,
        "frame_origin": frame_start,
        "active_threshold": cut,
        "preview_rows_max": config.preview_rows,
        "log_s": log_s,
        "view_coordinates": str(view_coordinates),
        "blueprint_view_id": str(view.id),
    }
    if config.register:
        register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out], "asog")
        summary["registered"] = f"{RUNS_DATASET}/{run_name}/asog"
    return summary


def main(config: Config) -> dict[str, object]:
    """Write ``<out_dir>/<name>.asog`` and ``<run>/asog.rrd``, decode it back, optionally evaluate; prints and returns the summary."""
    checkpoint = torch.load(config.run / "trigs_ckpt.pt", map_location="cpu", weights_only=True)
    fps: float = float(checkpoint["fps"])
    num_frames: int = int(checkpoint["num_frames"])
    params: dict[str, torch.Tensor] = checkpoint_params(checkpoint, torch.device("cpu"))
    out: Path = config.out_dir / f"{config.name or config.run.name}.asog"
    t0: float = time.perf_counter()
    stats = encode_params(params, fps, num_frames, out, config.settings)
    encode_s: float = time.perf_counter() - t0
    decoded, decode_s = decode_asog(out, torch.device("cpu"))
    splats_rrd: Path = config.run / "splats.rrd"
    summary: dict[str, object] = {
        "asog": str(out),
        "preset": config.settings.preset,
        "settings": asdict(config.settings.options()),
        "rows_checkpoint": int(params["means"].shape[0]),
        "rows_asog": stats.count,
        "bytes_checkpoint": (config.run / "trigs_ckpt.pt").stat().st_size,
        "bytes_asog": out.stat().st_size,
        "bytes_splats_rrd": splats_rrd.stat().st_size if splats_rrd.exists() else None,
        "bytes_per_splat": out.stat().st_size / stats.count,
        "stream_bytes": stats.stream_bytes,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "tiny_omega_fraction": tiny_omega_fraction(params),
    }
    summary.update(write_rrd(config, out, config.rrd or config.run / "asog.rrd"))
    if config.evaluate:
        summary.update(_evaluate(config, params, decoded, fps, num_frames))
    print(json.dumps(summary, indent=2, default=float))
    return summary
