"""Log a Streamed SOG (PlayCanvas LOD: lod-meta.json + SOG files) as `SogLodSplats3D` with a stock-viewer preview."""

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, lod_preview_level, log_sog_lod
from gsplat_rust_renderer.sog_lod import SogLodSet, read_sog_lod


@dataclass
class LogSogLodConfig:
    """Log a Streamed SOG to the custom Rust viewer (stock viewers draw the preview only)."""

    rr_config: RerunTyroConfig
    """Rerun connection/output configuration. Use --rr-config.save <file.rrd> to write a recording."""
    path: Path
    """lod-meta.json or the directory holding it."""
    entity: str = "/world/splats"
    """Entity of the payload and its preview."""
    preview_rows: int = PREVIEW_ROWS
    """Stock-viewer preview rows (every k-th splat of the coarsest level with at least this many); must be > 0."""


def main(config: LogSogLodConfig) -> None:
    lod: SogLodSet = read_sog_lod(config.path)
    # SOG keeps the 3DGS PLY frame (x right, y down, z forward).
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_sog_lod(config.entity, lod, config.preview_rows)
    stored: int = sum(len(b) for sog in lod.sogs for b in sog.planes.values())
    print(
        f"logged {config.path} (Streamed SOG, {lod.levels} levels, {len(lod.files)} files, {sum(lod.file_counts)} rows, "
        f"{stored} plane bytes, preview from level {lod_preview_level(lod, config.preview_rows)}) as static {config.entity}"
    )
