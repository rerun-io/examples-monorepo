"""Export an upstream Hexels run (config.json + checkpoints/ckpt_*.pt) to a hexel PLY."""

from dataclasses import dataclass
from pathlib import Path

from hexels_bench.checkpoint import UpstreamCheckpoint, find_checkpoint, load_checkpoint, load_run_config
from hexels_bench.hexel_ply import HexelPly, write_hexel_ply


@dataclass
class Config:
    """Export one upstream run directory to a hexel PLY."""

    model_dir: Path
    """Upstream run directory holding config.json and checkpoints/."""
    out: Path
    """Destination hexel PLY."""
    checkpoint: Path | None = None
    """Checkpoint to export; defaults to the highest-step checkpoints/ckpt_*.pt."""


def export_run(model_dir: Path, out: Path, checkpoint: Path | None = None) -> HexelPly:
    checkpoint_path: Path = checkpoint if checkpoint is not None else find_checkpoint(model_dir)
    loaded: UpstreamCheckpoint = load_checkpoint(checkpoint_path, load_run_config(model_dir / "config.json"))
    ply = HexelPly(cloud=loaded.cloud, source=str(checkpoint_path), step=loaded.step)
    write_hexel_ply(out, ply)
    return ply


def main(config: Config) -> None:
    ply: HexelPly = export_run(config.model_dir, config.out, config.checkpoint)
    print(f"wrote {config.out}: {ply.cloud.count} hexels, variant {ply.cloud.variant}, step {ply.step}")
