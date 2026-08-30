"""Stereo depth on one rectified pair with Middlebury-style calibration, visualized as an exoego:v2 rig in Rerun."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, UInt8
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.stereo_depth import AnnotatedStereoPredictorUnion, BaseStereoPredictor, LiteAnyStereoConfig, StereoDepthPrediction
from monopriors.rr_logging_utils import create_stereo_depth_blueprint, log_stereo_pred

ETH3D_MAX_DISP: float = 192.0
"""Ground-truth disparity cutoff used for comparable ETH3D metrics across predictors."""


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
    predictor: AnnotatedStereoPredictorUnion = field(default_factory=LiteAnyStereoConfig)
    """Stereo model to run; pick with the subcommand, e.g. ``liteanystereo --predictor.model-size h`` or ``fast-foundationstereo``."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend."""
    max_depth_m: float = 20.0
    """Depth beyond this is dropped from the logged depth image / point cloud."""
    remove_flying_pixels: bool = True
    """Zero depth on depth edges so the point cloud has no streaks between surfaces."""
    depth_edge_threshold: float = 0.5
    """Depth-gradient magnitude (metres per pixel) that counts as an edge."""


def read_rgb(path: Path) -> UInt8[np.ndarray, "h w 3"]:
    bgr_hw3: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        raise FileNotFoundError(f"Failed to read image {path}")
    return cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)


def read_pfm(path: Path) -> Float32[np.ndarray, "h w"]:
    """Read a grayscale PFM disparity image.

    Args:
        path: PFM file whose first line is ``Pf``.

    Returns:
        Bottom-to-top corrected disparity, ``Float32[ndarray, "h w"]``.
    """
    with path.open("rb") as file:
        if file.readline().strip() != b"Pf":
            raise ValueError(f"{path} is not a grayscale PFM")
        width, height = (int(value) for value in file.readline().split())
        scale: float = float(file.readline())
        data: Float32[np.ndarray, "pixels"] = np.fromfile(file, dtype="<f4" if scale < 0.0 else ">f4", count=width * height)
    return np.flipud(data.reshape(height, width)).astype(np.float32)


def stereo_metrics(
    disparity_hw: Float32[np.ndarray, "h w"],
    ground_truth_hw: Float32[np.ndarray, "h w"],
    nonoccluded_hw: UInt8[np.ndarray, "h w"],
    max_disp: float,
) -> tuple[float, float]:
    """Compute ETH3D EPE and bad1 on finite, non-occluded disparities below an evaluation cutoff.

    Args:
        disparity_hw: Predicted left disparity, ``Float32[ndarray, "h w"]``.
        ground_truth_hw: Ground-truth left disparity, ``Float32[ndarray, "h w"]``.
        nonoccluded_hw: ETH3D non-occlusion mask, ``UInt8[ndarray, "h w"]``; 255 marks valid pixels.
        max_disp: Evaluation cutoff; exclude ground-truth disparities at or above this value.

    Returns:
        Mean endpoint error in pixels and bad1 percentage.
    """
    valid_hw: Bool[np.ndarray, "h w"] = np.isfinite(ground_truth_hw) & (ground_truth_hw < max_disp) & (nonoccluded_hw == 255)
    error_hw: Float32[np.ndarray, "h w"] = np.abs(disparity_hw - ground_truth_hw)
    return float(error_hw[valid_hw].mean()), 100.0 * float((error_hw[valid_hw] > 1.0).mean())


def main(config: StereoDepthCLIConfig) -> None:
    left_rgb: UInt8[np.ndarray, "h w 3"] = read_rgb(config.scene_dir / "im0.png")
    right_rgb: UInt8[np.ndarray, "h w 3"] = read_rgb(config.scene_dir / "im1.png")
    calibration: MiddleburyCalibration = read_middlebury_calib(config.scene_dir / "calib.txt")

    predictor: BaseStereoPredictor = config.predictor.setup(device=config.device)
    stereo_pred: StereoDepthPrediction = predictor(left_rgb, right_rgb, K_33=calibration.K_33, baseline_m=calibration.baseline_m)

    gt_dir: Path = config.scene_dir.parent.with_name(f"{config.scene_dir.parent.name}_gt") / config.scene_dir.name
    ground_truth_path: Path = gt_dir / "disp0GT.pfm"
    nonoccluded_path: Path = gt_dir / "mask0nocc.png"
    if ground_truth_path.is_file() and nonoccluded_path.is_file():
        ground_truth_hw: Float32[np.ndarray, "h w"] = read_pfm(ground_truth_path)
        nonoccluded_hw: UInt8[np.ndarray, "h w"] | None = cv2.imread(str(nonoccluded_path), cv2.IMREAD_GRAYSCALE)
        if nonoccluded_hw is None:
            raise FileNotFoundError(f"Failed to read image {nonoccluded_path}")
        metrics: tuple[float, float] = stereo_metrics(stereo_pred.disparity, ground_truth_hw, nonoccluded_hw, max_disp=ETH3D_MAX_DISP)
        print(f"{type(predictor).__name__} ETH3D: EPE {metrics[0]:.3f} px, bad1 {metrics[1]:.2f}%")

    parent_log_path: Path = Path("world")
    rr.send_blueprint(create_stereo_depth_blueprint(parent_log_path))
    log_stereo_pred(
        parent_log_path,
        stereo_pred,
        left_rgb,
        right_rgb,
        max_depth_m=config.max_depth_m,
        remove_flying_pixels=config.remove_flying_pixels,
        depth_edge_threshold=config.depth_edge_threshold,
    )
