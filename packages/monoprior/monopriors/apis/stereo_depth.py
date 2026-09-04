"""Stereo depth on one rectified pair with Middlebury-style calibration, visualized as an exoego:v2 rig in Rerun."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.stereo_depth import STEREO_PREDICTORS, BaseStereoPredictor, StereoDepthPrediction, get_stereo_predictor
from monopriors.rr_logging_utils import create_stereo_depth_blueprint, log_stereo_pred


@dataclass(slots=True)
class MiddleburyCalibration:
    """The subset of a Middlebury v3 ``calib.txt`` a rectified pair needs (ETH3D two-view uses the same format)."""

    K_33: Float32[np.ndarray, "3 3"]
    """``cam0`` intrinsics."""
    baseline_m: float
    """``baseline`` (stored in millimetres in the file) converted to metres."""


def read_middlebury_calib(path: Path) -> MiddleburyCalibration:
    """Parse ``cam0=[fx 0 cx; 0 fy cy; 0 0 1]`` and ``baseline=<mm>`` from a Middlebury v3 ``calib.txt``."""
    text: str = path.read_text()
    cam0_match: re.Match[str] | None = re.search(r"cam0=\[(.*?)\]", text)
    baseline_match: re.Match[str] | None = re.search(r"baseline=([\d.]+)", text)
    if cam0_match is None or baseline_match is None:
        raise ValueError(f"{path} lacks cam0/baseline entries")
    K_33: Float32[np.ndarray, "3 3"] = np.array([[float(v) for v in row.split()] for row in cam0_match.group(1).split(";")], dtype=np.float32)
    return MiddleburyCalibration(K_33=K_33, baseline_m=float(baseline_match.group(1)) / 1000.0)


@dataclass
class StereoDepthCLIConfig:
    """CLI configuration for stereo depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    scene_dir: Path = Path("data/examples/stereo/eth3d/two_view_training/playground_1l")
    """Directory with ``im0.png`` (left), ``im1.png`` (right) and a Middlebury-style ``calib.txt``."""
    predictor_name: STEREO_PREDICTORS = "LiteAnyStereoPredictor"
    """Which stereo predictor to use."""
    model_size: Literal["s", "m", "l", "h"] = "m"
    """LiteAnyStereo V2 variant."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend."""
    max_depth_m: float = 20.0
    """Depth beyond this is dropped from the logged depth image / point cloud."""


def read_rgb(path: Path) -> UInt8[np.ndarray, "h w 3"]:
    bgr_hw3: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        raise FileNotFoundError(f"Failed to read image {path}")
    return cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)


def main(config: StereoDepthCLIConfig) -> None:
    left_rgb: UInt8[np.ndarray, "h w 3"] = read_rgb(config.scene_dir / "im0.png")
    right_rgb: UInt8[np.ndarray, "h w 3"] = read_rgb(config.scene_dir / "im1.png")
    calibration: MiddleburyCalibration = read_middlebury_calib(config.scene_dir / "calib.txt")

    predictor: BaseStereoPredictor = get_stereo_predictor(config.predictor_name)(device=config.device, model_size=config.model_size)
    stereo_pred: StereoDepthPrediction = predictor(left_rgb, right_rgb, K_33=calibration.K_33, baseline_m=calibration.baseline_m)

    parent_log_path: Path = Path("world")
    rr.send_blueprint(create_stereo_depth_blueprint(parent_log_path))
    log_stereo_pred(parent_log_path, stereo_pred, left_rgb, right_rgb, max_depth_m=config.max_depth_m)
