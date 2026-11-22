"""Compress a trained run to Animated SOG (``.asog``) and, optionally, measure what the compression costs.

``<run>/trigs_ckpt.pt`` -> ``<out_dir>/<run>.asog`` (an explicit third-party export, mode 644; the run's data stays
``.rrd``) -> decoded back to TRiGS parameters. With ``--evaluate`` the held-out cameras of the run (its ``run`` property:
dataset, segment, frames, cameras, size, ROI, mask layer) are re-read from the catalog and rendered with the trainer's
renderer from the uncompressed and the decoded parameters, scored like ``open4dgs-train`` (person crop), and compared
with each other (PSNR of decoded vs uncompressed renders). ``decoded_cut05`` renders the decoded clip with the ``.asog``
v1 temporal cut (0.05) instead of the trainer's (0.01): what a stock ``.asog`` player shows.
"""

import dataclasses
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rerun.chunk import RrdReader
from trigs.core.trainer import TrigsTrainer

from open4dgs.asog import AsogSettings, decode_asog, encode_params, tiny_omega_fraction
from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.evaluate import FrameMetrics, Metrics, mask_box, mean_metrics, psnr
from open4dgs.masks import read_clip_masks
from open4dgs.splat_export import checkpoint_params
from open4dgs.trigs_adapter import TrainConfig, TrainedClip, camera_config

LOCAL_CATALOG: str = "rerun+http://pablo-dl-server:51235"


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


def _run_properties(reader: RrdReader) -> dict[str, list[object]]:
    out: dict[str, list[object]] = {}
    for chunk in reader.stream().filter(content="/__properties/run").to_chunks():
        batch = chunk.to_record_batch()
        for name in batch.schema.names:
            values = batch.column(name).to_pylist()[0]
            if isinstance(values, list):
                out[name.split(":")[-1]] = values
    return out


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
    props = _run_properties(RrdReader(config.run / "run.rrd"))
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
    mask_dataset: str = str(props["mask_source"][0]).removeprefix("layer:").removesuffix("/masks")
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


def main(config: Config) -> dict[str, object]:
    """Write ``<out_dir>/<name>.asog``, decode it back, optionally evaluate; prints and returns the summary."""
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
    if config.evaluate:
        summary.update(_evaluate(config, params, decoded, fps, num_frames))
    print(json.dumps(summary, indent=2, default=float))
    return summary
