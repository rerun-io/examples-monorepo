"""Prepare DataForge's RoboCap geometry and blueprint for the live Rust recorder."""

from dataclasses import dataclass
from pathlib import Path

from simplecv.camera_parameters import Fisheye62Parameters
from simplecv.data.ego.robocap_ego import CAMERA_DISPLAY_ORDER

from dataforge.datasets.robocap import RobocapConfig, RobocapDataset, build_blueprint
from dataforge.writing import atomic_recording


@dataclass
class Config:
    """Export reusable static display data, without camera or IMU recordings."""

    output: Path
    """Destination recording path; must not already exist."""
    source_device: str = "f408193e6447b3b0"
    """Device identifier whose factory calibration supplies the display geometry."""
    root: Path = Path("/mnt/nas/datasets/robocap")
    """RoboCap dataset root containing factory calibration and mesh assets."""


def main(config: Config) -> None:
    """Use the existing calibration importer and scene logger to prepare the live display."""
    if config.output.exists():
        raise FileExistsError(config.output)
    dataset: RobocapDataset = RobocapDataset(RobocapConfig(root=config.root))
    cameras: dict[str, Fisheye62Parameters] = dataset.calibration(config.source_device)
    missing_cameras: list[str] = [name for name in CAMERA_DISPLAY_ORDER if name not in cameras]
    if missing_cameras:
        raise ValueError(f"Calibration for {config.source_device} is missing cameras: {', '.join(missing_cameras)}")
    with atomic_recording(
        config.output,
        application_id="robocap",
        # Contract with the Rust display loader: "{calibration_device}-display".
        recording_id=f"{config.source_device}-display",
        default_blueprint=build_blueprint(list(CAMERA_DISPLAY_ORDER), pose_source="slam_rs"),
    ) as recording:
        dataset.log_scene(recording, cameras)
