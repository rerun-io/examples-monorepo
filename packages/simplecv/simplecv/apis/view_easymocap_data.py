from dataclasses import dataclass
from pathlib import Path

import rerun as rr

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.easymocap import load_cameras
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video


@dataclass
class ViewEasyMocapConfig:
    data_dir: Path
    rr_config: RerunTyroConfig


def view_easymocap(config: ViewEasyMocapConfig) -> None:
    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.ULB, static=True)
    cameras: list[PinholeParameters] = load_cameras(data_path=config.data_dir)
    video_paths: list[Path] = sorted((config.data_dir / "videos").glob("*.mp4"))
    for cam, video_path in zip(cameras, video_paths, strict=True):
        camera_log_path: Path = parent_log_path / cam.name
        image_log_path: Path = camera_log_path / "pinhole" / "image"
        log_video(video_path, image_log_path)
        log_pinhole(camera=cam, cam_log_path=camera_log_path, static=True)
