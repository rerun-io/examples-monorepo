"""Compress a trained run to Animated SOG (ASOG) in a Rerun recording and, optionally, measure what the compression costs.

``<run>/trigs_ckpt.pt`` -> ASOG bytes -> ``<run>/asog.rrd`` (the one output; ``--asog-file`` also writes the ``.asog`` for
other players) -> decoded back to TRiGS parameters. With ``--evaluate`` the held-out cameras of the run (its ``run`` property:
dataset, segment, frames, cameras, size, ROI, mask layer) are re-read from the catalog and rendered with the trainer's
renderer from the uncompressed and the decoded parameters, scored like ``open4dgs-train`` (person crop), and compared
with each other (PSNR of decoded vs uncompressed renders). ``decoded_cut05`` renders the decoded clip with the ``.asog``
v1 temporal cut (0.05) instead of the trainer's (0.01): what a stock ``.asog`` player shows.

The recording, ``<run>/asog.rrd`` (``--rrd``), holds the ASOG bytes verbatim as the custom gsplat viewer's
``AnimatedSogSplats3D`` payload (``gsplat_rust_renderer.sog3d.log_asog``) with ``frame_origin`` = the clip's first frame, a stock
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
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS
from PIL import Image
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from simplecv.catalog_layers import register_layer
from trigs.core.trainer import TrigsTrainingConfig

from open4dgs.asog import AsogSettings, decode_asog, encode_asog, tiny_omega_fraction
from open4dgs.catalog_clip import LOCAL_CATALOG, MultiViewClip, read_catalog_clip
from open4dgs.evaluate import FrameMetrics, Metrics, mask_box, mean_metrics, psnr
from open4dgs.masks import read_clip_masks
from open4dgs.properties import rrd_cells
from open4dgs.run_log import RUNS_DATASET, RunProperties
from open4dgs.splat_export import ASOG_CUT_FORMAT, write_asog_rrd
from open4dgs.trigs_adapter import TrainedClip


@dataclass(frozen=True, slots=True)
class Config:
    """One export (and optional evaluation)."""

    run: Path
    """Run directory (``run.rrd`` + ``trigs_ckpt.pt``)."""
    asog_file: Path | None = None
    """Also write the ASOG as an ``.asog`` file here (interchange with other players); by default the recording is the only output."""
    settings: AsogSettings = field(default_factory=AsogSettings)
    """Encoder preset (``band-safe`` default; ``--settings.preset small``: smaller file for web streaming) and per-option overrides."""
    evaluate: bool = False
    """Render the held-out cameras from both models and score them (needs the GPU and the catalog)."""
    mask_catalog_url: str = LOCAL_CATALOG
    """Catalog of the run's ``people_matte`` layer (local for every run so far)."""
    png_dir: Path | None = None
    """With ``evaluate``: ground truth | uncompressed | decoded PNGs at two frames per held-out camera."""
    png_cameras: int = 2
    """Held-out cameras that get PNGs."""
    rrd: Path | None = None
    """Recording with the ASOG as the gsplat viewer's ``AnimatedSogSplats3D`` payload (default ``<run>/asog.rrd``;
    always written; ``open4dgs-export-splats --format asog`` writes the same recording)."""
    asog_cut: Literal["format", "trained"] = "format"
    """Temporal cut recorded for the viewer: ``format`` = the ``.asog`` v1 constant 0.05 (what other players show),
    ``trained`` = the trainer's 0.01 (the trained look; the same file then plays differently elsewhere)."""
    preview_rows: int = PREVIEW_ROWS
    """Stock-viewer preview rows of the clip-middle pose (0 = none; the gsplat viewer then draws nothing)."""
    register: bool = True
    """Register the recording as layer ``asog`` of the run on open4dgs-runs."""
    runs_catalog_url: str = LOCAL_CATALOG


def evaluation_models(path: Path, decoded: dict[str, torch.Tensor], clip: MultiViewClip | None, holdout: tuple[int, ...], device: torch.device) -> dict[str, TrainedClip]:
    """The checkpoint at ``path`` as it was trained (``TrainedClip.load``: its own settings, cameras and capture times),
    the same model with the ``decoded`` ASOG parameters, and that one at the ``.asog`` temporal cut (0.05).

    ``clip`` and ``holdout`` only fill in a checkpoint written before ``CheckpointMeta``."""
    uncompressed: TrainedClip = TrainedClip.load(path, device, clip, holdout)
    decoded_model: TrainedClip = uncompressed.with_params(decoded)
    return {
        "uncompressed": uncompressed,
        "decoded": decoded_model,
        "decoded_cut05": dataclasses.replace(
            decoded_model, trigs_config=dataclasses.replace(decoded_model.trigs_config, temporal_active_threshold=ASOG_CUT_FORMAT)
        ),
    }


def _evaluate(config: Config, decoded: dict[str, torch.Tensor]) -> dict[str, object]:
    run: RunProperties = RunProperties.from_cells(rrd_cells(RrdReader(config.run / "run.rrd"), "run"))
    clip: MultiViewClip = read_catalog_clip(
        run.catalog_url,
        run.dataset,
        run.segment_id,
        run.frame_start,
        run.num_frames,
        cameras=list(run.cameras),
        max_side=run.max_side,
        roi="people" if run.roi_pad is not None else None,
        roi_pad=run.roi_pad if run.roi_pad is not None else 0.15,
    )
    if run.mask_dataset is None:
        raise ValueError(f"{config.run}: trained on mesh silhouettes; --evaluate reads the people_matte layer")
    masks = read_clip_masks(clip, config.mask_catalog_url, run.mask_dataset, run.segment_id)
    holdout_names: tuple[str, ...] = run.holdout_cameras
    holdout: tuple[int, ...] = tuple(clip.camera_names.index(n) for n in holdout_names)
    models: dict[str, TrainedClip] = evaluation_models(config.run / "trigs_ckpt.pt", decoded, clip, holdout, torch.device("cuda"))
    num_frames: int = models["uncompressed"].num_frames
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
                Image.fromarray(strip).save(config.png_dir / f"{config.run.name}-{name.replace('/', '_')}-f{int(clip.frame_indices[f])}-gt_uncompressed_decoded.png")
    finite = [v for v in vs_full if math.isfinite(v)]
    return {
        "holdout_cameras": list(holdout_names),
        **{f"holdout_{key}": mean_metrics(value) for key, value in rows.items()},
        "psnr_decoded_vs_uncompressed_full": float(np.mean(finite)) if finite else math.inf,
        "psnr_decoded_vs_uncompressed_crop": float(np.mean([v for v in vs_crop if math.isfinite(v)])),
    }


def write_rrd(config: Config, asog: bytes, out: Path) -> dict[str, object]:
    """Log the ``asog`` bytes with the run's cameras and meshes into ``out`` (``splat_export.write_asog_rrd``, the writer
    ``open4dgs-export-splats --format asog`` uses too); registers it as layer ``asog`` unless ``--no-register``."""
    cut: float = ASOG_CUT_FORMAT if config.asog_cut == "format" else TrigsTrainingConfig().temporal_active_threshold
    summary: dict[str, object] = write_asog_rrd(config.run, asog, out, cut, config.preview_rows)
    if config.register:
        register_layer(CatalogClient(config.runs_catalog_url).get_dataset(RUNS_DATASET), [out], "asog")
        summary["registered"] = f"{RUNS_DATASET}/{config.run.name}/asog"
    return summary


def main(config: Config) -> dict[str, object]:
    """Write ``<run>/asog.rrd`` (and ``--asog-file``), decode the ASOG back, optionally evaluate; prints and returns the summary."""
    trained: TrainedClip = TrainedClip.load(config.run / "trigs_ckpt.pt", torch.device("cpu"))
    params: dict[str, torch.Tensor] = dict(trained.params)
    t0: float = time.perf_counter()
    asog, stats = encode_asog(params, trained.fps, trained.num_frames, config.settings)
    encode_s: float = time.perf_counter() - t0
    if config.asog_file is not None:
        config.asog_file.parent.mkdir(parents=True, exist_ok=True)
        config.asog_file.write_bytes(asog)
        config.asog_file.chmod(0o644)
    decoded, decode_s = decode_asog(asog, torch.device("cpu"))
    splats_rrd: Path = config.run / "splats.rrd"
    summary: dict[str, object] = {
        "asog_file": str(config.asog_file) if config.asog_file is not None else None,
        "preset": config.settings.preset,
        "settings": asdict(config.settings.options()),
        "rows_checkpoint": int(params["means"].shape[0]),
        "rows_asog": stats.count,
        "bytes_checkpoint": (config.run / "trigs_ckpt.pt").stat().st_size,
        "bytes_asog": len(asog),
        "bytes_splats_rrd": splats_rrd.stat().st_size if splats_rrd.exists() else None,
        "bytes_per_splat": len(asog) / stats.count,
        "stream_bytes": stats.stream_bytes,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "tiny_omega_fraction": tiny_omega_fraction(params),
    }
    summary.update(write_rrd(config, asog, config.rrd or config.run / "asog.rrd"))
    if config.evaluate:
        summary.update(_evaluate(config, decoded))
    print(json.dumps(summary, indent=2, default=float))
    return summary
