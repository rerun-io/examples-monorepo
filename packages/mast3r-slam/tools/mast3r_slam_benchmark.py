from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference


@dataclass
class BenchmarkConfig:
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    dataset: str = "data/normal-apt-tour.mp4"
    config: str = "config/base.yaml"
    img_size: int = 512
    max_frames: int | None = None
    benchmark_dir: Path = Path("benchmark")
    no_viz: bool = False


if __name__ == "__main__":
    cfg = tyro.cli(BenchmarkConfig)
    mast3r_slam_inference(
        InferenceConfig(
            rr_config=cfg.rr_config,
            dataset=cfg.dataset,
            config=cfg.config,
            no_viz=cfg.no_viz,
            img_size=cfg.img_size,
            max_frames=cfg.max_frames,
            benchmark_dir=cfg.benchmark_dir,
        )
    )
