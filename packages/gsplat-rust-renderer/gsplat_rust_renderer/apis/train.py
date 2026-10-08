"""Prepare one dataset and run the in-process trainer with a recording sink."""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from gsplat_rust_renderer.apis.prepare_nerf_init import Config as InitConfig
from gsplat_rust_renderer.apis.prepare_nerf_init import main as initialize_nerf
from gsplat_rust_renderer.nerfbaselines import DATA_ROOT, BlenderScene, download_tandt


@dataclass(frozen=True, slots=True)
class Config:
    scene: Literal[BlenderScene, "truck", "train"] = "lego"
    """NeRF-synthetic or Tanks and Temples scene."""
    iterations: int = 30000
    """Training steps; checkpoints are exported every 7000 steps."""
    mode: Literal["record", "live", "video"] = "record"
    """Recording only, connection to a viewer, or a flat spinning video layout."""
    binary: Path = Path("target/release/gsplat-train")
    """Built trainer executable."""
    data_root: Path = DATA_ROOT.parent
    """Dataset root for downloaded COLMAP scenes."""
    output_root: Path = DATA_ROOT.parent / "brush-runs"
    """Recordings and checkpoint exports, grouped by scene, steps and mode."""


def main(config: Config, extra_flags: list[str]) -> None:
    """Resolve the dataset once and pass typed settings to the Rust CLI."""
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.scene in ("truck", "train"):
        source: Path = download_tandt(config.data_root / "tandt") / config.scene
    else:
        initialization = InitConfig(scene=cast(BlenderScene, config.scene))
        initialize_nerf(initialization)
        source = initialization.output_root / config.scene
    if not source.is_dir():
        raise FileNotFoundError(f"Training dataset is missing: {source}")
    output: Path = (config.output_root / config.scene / str(config.iterations) / config.mode).resolve()
    command: list[str] = [
        str(config.binary.resolve()), str(source.resolve()),
        "--total-train-iters", str(config.iterations), "--export-every", "7000",
        "--save", str(output / "training.rrd"), "--export-path", str(output),
    ]
    if config.mode == "live":
        command += ["--connect", "rerun+http://127.0.0.1:9876/proxy"]
    elif config.mode == "video":
        command += ["--video"]
    forwarded: list[str] = extra_flags[1:] if extra_flags[:1] == ["--"] else extra_flags
    subprocess.run(command + forwarded, check=True)
