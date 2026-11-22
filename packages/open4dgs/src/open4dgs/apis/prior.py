"""Compute the MoGe-2 prior for a catalog clip once and register it as layer ``moge2_prior``.

The clip is read with ``read_catalog_clip`` exactly as the trainer reads it (same window, cameras and ``max_side``),
so the maps sit on the trainer's pixel grid; ``read_clip_priors`` refuses a clip of another window or size.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from open4dgs.catalog_clip import LOCAL_CATALOG, MultiViewClip, read_catalog_clip
from open4dgs.prior import compute_prior_layer


@dataclass
class Config:
    catalog_url: str = LOCAL_CATALOG
    """Catalog that holds the segment's video and cameras."""
    dataset: str = "fdanyone-views"
    segment_id: str = "pexels-2785536__v06-seed42"
    frame_start: int = 0
    num_frames: int = 30
    cameras: list[str] | None = None
    """Camera names such as ``rig_01/cam_00``; default every camera (the trainer reads only the training ones)."""
    max_side: int | None = 640
    """Must equal the training run's ``max_side``."""
    roi: Literal["people"] | None = None
    """Must equal the training run's ``roi`` (person-centred virtual cameras, ``open4dgs.roi_cameras``)."""
    roi_pad: float = 0.15
    """Must equal the training run's ``roi_pad``."""
    register_url: str = LOCAL_CATALOG
    register_dataset: str | None = None
    """Default: the source dataset when it is on ``register_url``, else ``open4dgs-mamma`` (Spark is read-only)."""
    register: bool = True
    output_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/prior")
    """The layer recording goes to ``<output_dir>/<register_dataset>/<segment_id>.moge2_prior-<grid>.rrd`` (``prior.compute_prior_layer``)."""
    batch_size: int = 8
    resolution_level: int = 9
    """MoGe-2 detail level (0-9)."""


def main(config: Config) -> None:
    target: str = config.register_dataset or (config.dataset if config.catalog_url == config.register_url else "open4dgs-mamma")
    t0: float = time.perf_counter()
    clip: MultiViewClip = read_catalog_clip(
        config.catalog_url,
        config.dataset,
        config.segment_id,
        config.frame_start,
        config.num_frames,
        cameras=config.cameras,
        max_side=config.max_side,
        roi=config.roi,
        roi_pad=config.roi_pad,
    )
    t_read: float = time.perf_counter() - t0
    _, path, timings = compute_prior_layer(
        clip,
        config.segment_id,
        source=f"{config.catalog_url}/{config.dataset}",
        catalog_url=config.register_url,
        dataset=target,
        output_dir=config.output_dir,
        max_side=config.max_side,
        register=config.register,
        batch_size=config.batch_size,
        resolution_level=config.resolution_level,
    )
    print(
        f"[open4dgs-prior] {timings['images']:.0f} images ({len(clip.camera_names)} cams x {len(clip.frame_indices)} frames, sizes {sorted(set(clip.sizes))}): "
        f"read {t_read:.1f} s, MoGe-2 {timings['moge_s']:.1f} s ({timings['images'] / timings['moge_s']:.1f} img/s), write {timings['write_s']:.1f} s, "
        f"{path.stat().st_size / 1e6:.0f} MB -> {path}"
    )
    if config.register:
        print(f"registered layer moge2_prior of {target}/{config.segment_id} on {config.register_url} ({timings['register_s']:.1f} s)")
