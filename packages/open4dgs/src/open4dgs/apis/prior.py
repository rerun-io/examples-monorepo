"""Compute the MoGe-2 prior for a catalog clip once and register it as layer ``moge2_prior``.

The clip is read with ``read_catalog_clip`` exactly as the trainer reads it (same window, cameras and ``max_side``),
so the maps sit on the trainer's pixel grid; ``read_clip_priors`` refuses a clip of another window or size.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.prior import MODEL_NAME, ClipPriors, compute_clip_priors, register_prior_layer, write_prior_layer

LOCAL_CATALOG_URL: str = "rerun+http://pablo-dl-server:51235"


@dataclass
class Config:
    catalog_url: str = LOCAL_CATALOG_URL
    """Catalog that holds the segment's video and cameras."""
    dataset: str = "fdanyone-views"
    segment_id: str = "pexels-2785536__v06-seed42"
    frame_start: int = 0
    num_frames: int = 30
    cameras: list[str] | None = None
    """Camera names such as ``rig_01/cam_00``; default every camera (the trainer reads only the training ones)."""
    max_side: int | None = 640
    """Must equal the training run's ``max_side``."""
    register_url: str = LOCAL_CATALOG_URL
    register_dataset: str | None = None
    """Default: the source dataset when it is on ``register_url``, else ``open4dgs-mamma`` (Spark is read-only)."""
    register: bool = True
    output_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/prior")
    """The layer recording goes to ``<output_dir>/<register_dataset>/<segment_id>.moge2_prior.rrd``."""
    batch_size: int = 8
    resolution_level: int = 9
    """MoGe-2 detail level (0-9)."""


def main(config: Config) -> None:
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor

    target: str = config.register_dataset or (config.dataset if config.catalog_url == config.register_url else "open4dgs-mamma")
    t0: float = time.perf_counter()
    clip: MultiViewClip = read_catalog_clip(
        config.catalog_url, config.dataset, config.segment_id, config.frame_start, config.num_frames, cameras=config.cameras, max_side=config.max_side
    )
    t_read: float = time.perf_counter() - t0
    predictor = MoGeV2TorchPredictor(encoder="vitl", heads="geometry", resolution_level=config.resolution_level)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    priors: ClipPriors = compute_clip_priors(clip, clip.camera_names, predictor, batch_size=config.batch_size)
    torch.cuda.synchronize()
    t_moge: float = time.perf_counter() - t0
    n_images: int = len(clip.camera_names) * len(clip.frame_indices)
    path: Path = config.output_dir / target / f"{config.segment_id}.moge2_prior.rrd"
    properties: dict[str, str | int | float] = {
        "model": MODEL_NAME,
        "predictor": "MoGeV2TorchPredictor fp16 autocast",
        "resolution_level": config.resolution_level,
        "fov_source": "calibrated fx (fov_x = 2 atan(w / 2 fx))",
        "source": f"{config.catalog_url}/{config.dataset}",
        "frame_start": int(clip.frame_indices[0]),
        "frame_end": int(clip.frame_indices[-1]),
        "num_frames": len(clip.frame_indices),
        "max_side": config.max_side or 0,
        "cameras": ",".join(clip.camera_names),
        "image_sizes": ",".join(f"{w}x{h}" for w, h in clip.sizes),
        "loader": "open4dgs.catalog_clip.read_catalog_clip (undistorted)",
    }
    t0 = time.perf_counter()
    write_prior_layer(path, config.segment_id, priors, properties)
    t_write: float = time.perf_counter() - t0
    print(
        f"[open4dgs-prior] {n_images} images ({len(clip.camera_names)} cams x {len(clip.frame_indices)} frames, sizes {sorted(set(clip.sizes))}): "
        f"read {t_read:.1f} s, MoGe-2 {t_moge:.1f} s ({n_images / t_moge:.1f} img/s), write {t_write:.1f} s, {path.stat().st_size / 1e6:.0f} MB -> {path}"
    )
    if config.register:
        register_prior_layer(config.register_url, target, path)
        print(f"registered layer moge2_prior of {target}/{config.segment_id} on {config.register_url}")
