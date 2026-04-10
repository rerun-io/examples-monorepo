from dataclasses import dataclass
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mini_dpvo.api.inference import inference_dpvo
from mini_dpvo.config import cfg as base_cfg


@dataclass
class DPVODemoConfig:
    """DPVO visual odometry demo."""

    rr_config: RerunTyroConfig
    """Rerun recording configuration."""
    imagedir: str = "data/movies/IMG_0493.MOV"
    """Path to image directory or video file."""
    network_path: str = "checkpoints/dpvo.pth"
    """Path to DPVO network weights."""
    calib: str | None = None
    """Path to calibration file. If None, uses DUSt3R for estimation."""
    stride: int = 2
    """Frame stride for sampling."""
    skip: int = 0
    """Number of frames to skip at the start."""
    buffer: int = 2048
    """Maximum number of keyframes."""
    config: Path = Path("config/fast.yaml")
    """DPVO config YAML file."""


if __name__ == "__main__":
    demo_cfg: DPVODemoConfig = tyro.cli(DPVODemoConfig)

    base_cfg.merge_from_file(str(demo_cfg.config))
    base_cfg.BUFFER_SIZE = demo_cfg.buffer

    print("Running with config...")
    print(base_cfg)

    inference_dpvo(
        cfg=base_cfg,
        network_path=demo_cfg.network_path,
        imagedir=demo_cfg.imagedir,
        calib=demo_cfg.calib,
        stride=demo_cfg.stride,
        skip=demo_cfg.skip,
    )
