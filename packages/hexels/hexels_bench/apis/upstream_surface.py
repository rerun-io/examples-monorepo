"""Stream upstream CUDA surface renders of a published surface checkpoint to stdout (runs in the `hexels` env).

`hexels-surface-run --methods hexels-v3-surf-30k-pub-cuda` (hexels-surface env) starts this and fuses the
frames with the same code as the WGSL frames, so depth is the only difference between the two meshes.
Everything upstream prints goes to stderr; stdout carries only frames (`surface_frames.write_frame`).
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from hexels_bench.checkpoint import find_checkpoint, load_run_config
from hexels_bench.layout import Dataset, Roots
from hexels_bench.surface_frames import write_frame
from hexels_bench.upstream import surface_frames


@dataclass
class Config:
    """Render every view of a published surface checkpoint with upstream CUDA and stream the frames."""

    dataset: Dataset
    """dtu or tnt."""
    scene: str
    """Upstream scene name."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def main(config: Config) -> None:
    frames_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # upstream logs to stdout; keep the frame stream clean
    model_dir: Path = config.roots.published_surface_model(config.dataset, config.scene)
    scene_dir: Path = config.roots.data / Path(load_run_config(model_dir / "config.json").source_path).relative_to("data")
    with frames_out:
        for frame in surface_frames(model_dir, find_checkpoint(model_dir), scene_dir):
            write_frame(frames_out, frame)
    print(f"streamed {config.dataset}/{config.scene} upstream surface frames", file=sys.stderr)
