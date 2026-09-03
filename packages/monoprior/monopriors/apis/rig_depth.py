"""X-Lens metric rig depth on the ETH3D ``playground_1l`` pinhole pair."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import Rig, RigCalibration, stereo_rig_calibration

from monopriors.apis.stereo_depth import ETH3D_MAX_DISP, MiddleburyCalibration, read_middlebury_calib, read_pfm, read_rgb, stereo_metrics
from monopriors.models.rig_depth import (
    AnnotatedRigDepthPredictorUnion,
    BaseRigDepthPredictor,
    RigDepthPrediction,
    XLensConfig,
    camera_type,
    unit_rays,
)


@dataclass(slots=True)
class ETH3DRigPair:
    """Cropped ETH3D inputs, calibration, rig geometry, and ground truth."""

    images: UInt8[ndarray, "2 h w 3"]
    rays: Float32[ndarray, "2 h w 3"]
    cam_types: Int64[ndarray, "2"]
    cam_T_ref: Float64[ndarray, "2 4 4"]
    calibration: MiddleburyCalibration
    rig_calibration: RigCalibration
    ground_truth_disparity: Float32[ndarray, "h w"]
    nonoccluded: UInt8[ndarray, "h w"]


@dataclass
class Config:
    """CLI configuration for X-Lens on an ETH3D calibrated pinhole pair."""

    rr_config: RerunTyroConfig
    """Rerun viewer, save, connect, or headless configuration."""
    scene_dir: Path = Path("data/examples/stereo/eth3d/two_view_training/playground_1l")
    """Directory containing ``im0.png``, ``im1.png``, and ``calib.txt``."""
    predictor: AnnotatedRigDepthPredictorUnion = field(default_factory=XLensConfig)
    """Calibrated rig-depth predictor."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution device."""
    max_depth_m: float = 20.0
    """Maximum displayed depth in metres."""


def metric_depth_to_disparity(
    depth_m_hw: Float32[ndarray, "h w"],
    calibration: MiddleburyCalibration,
) -> Float32[ndarray, "h w"]:
    """Convert metric z-depth to Middlebury disparity, including ``doffs``."""
    disparity_hw: Float32[ndarray, "h w"] = np.full(depth_m_hw.shape, np.nan, dtype=np.float32)
    valid_hw: Bool[ndarray, "h w"] = np.isfinite(depth_m_hw) & (depth_m_hw > 0.0)
    disparity_hw[valid_hw] = (
        float(calibration.K_33[0, 0]) * calibration.baseline_m / depth_m_hw[valid_hw] + calibration.doffs_px
    )
    return disparity_hw


def eth3d_rig_metrics(
    depth_m_hw: Float32[ndarray, "h w"],
    ground_truth_disparity_hw: Float32[ndarray, "h w"],
    nonoccluded_hw: UInt8[ndarray, "h w"],
    calibration: MiddleburyCalibration,
) -> tuple[float, float, float]:
    """Compute ETH3D EPE, bad1, and metric-depth absolute-relative error."""
    predicted_disparity_hw: Float32[ndarray, "h w"] = metric_depth_to_disparity(depth_m_hw, calibration)
    epe_px, bad1_percent = stereo_metrics(
        predicted_disparity_hw,
        ground_truth_disparity_hw,
        nonoccluded_hw,
        max_disp=ETH3D_MAX_DISP,
    )
    valid_hw: Bool[ndarray, "h w"] = (
        np.isfinite(ground_truth_disparity_hw)
        & np.isfinite(predicted_disparity_hw)
        & (ground_truth_disparity_hw < ETH3D_MAX_DISP)
        & (ground_truth_disparity_hw > calibration.doffs_px)
        & (nonoccluded_hw == 255)
    )
    gt_depth_hw: Float32[ndarray, "h w"] = np.zeros_like(ground_truth_disparity_hw)
    gt_depth_hw[valid_hw] = (
        float(calibration.K_33[0, 0]) * calibration.baseline_m / (ground_truth_disparity_hw[valid_hw] - calibration.doffs_px)
    )
    abs_rel: float = float((np.abs(depth_m_hw[valid_hw] - gt_depth_hw[valid_hw]) / gt_depth_hw[valid_hw]).mean())
    return epe_px, bad1_percent, abs_rel


def load_eth3d_rig_pair(scene_dir: Path) -> ETH3DRigPair:
    """Load and top-left crop ETH3D to a valid X-Lens patch grid."""
    left_rgb: UInt8[ndarray, "h w 3"] = read_rgb(scene_dir / "im0.png")
    right_rgb: UInt8[ndarray, "h w 3"] = read_rgb(scene_dir / "im1.png")
    height: int = min(left_rgb.shape[0], right_rgb.shape[0]) // 14 * 14
    width: int = min(left_rgb.shape[1], right_rgb.shape[1]) // 14 * 14
    if height < 28 or width < 28:
        raise ValueError(f"ETH3D crop must be at least 28x28, got {height}x{width}")
    images: UInt8[ndarray, "2 h w 3"] = np.stack((left_rgb[:height, :width], right_rgb[:height, :width]))

    calibration: MiddleburyCalibration = read_middlebury_calib(scene_dir / "calib.txt")
    rig_calibration: RigCalibration = stereo_rig_calibration(calibration.K_33, calibration.baseline_m, width, height)
    rays: Float32[ndarray, "2 h w 3"] = np.stack([unit_rays(sensor.pinhole) for sensor in rig_calibration.cameras])
    cam_types: Int64[ndarray, "2"] = np.asarray([camera_type(sensor.pinhole) for sensor in rig_calibration.cameras], dtype=np.int64)
    cam_T_ref: Float64[ndarray, "2 4 4"] = np.stack(
        [np.asarray(sensor.pinhole.extrinsics.world_T_cam, dtype=np.float64) for sensor in rig_calibration.cameras]
    )

    gt_dir: Path = scene_dir.parent.with_name(f"{scene_dir.parent.name}_gt") / scene_dir.name
    ground_truth_disparity: Float32[ndarray, "h w"] = read_pfm(gt_dir / "disp0GT.pfm")[:height, :width]
    nonoccluded: UInt8[ndarray, "h w"] | None = cv2.imread(str(gt_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    if nonoccluded is None:
        raise FileNotFoundError(f"Failed to read image {gt_dir / 'mask0nocc.png'}")
    return ETH3DRigPair(
        images=images,
        rays=rays,
        cam_types=cam_types,
        cam_T_ref=cam_T_ref,
        calibration=calibration,
        rig_calibration=rig_calibration,
        ground_truth_disparity=ground_truth_disparity,
        nonoccluded=nonoccluded[:height, :width],
    )


def create_rig_depth_blueprint() -> rrb.Blueprint:
    """Show the rig in 3D beside both RGB/depth views and left-view errors."""
    rig_path = "world/rig_00"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", contents=["$origin/**", f"- {rig_path}/cam_00/rig_depth/**"]),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_00/pinhole/image", name="left image"),
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_01/pinhole/image", name="right image"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_00/pinhole/depth", name="left depth"),
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_01/pinhole/depth", name="right depth"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_00/rig_depth/gt_disparity", name="GT disparity"),
                    rrb.Spatial2DView(origin=f"{rig_path}/cam_00/rig_depth/disparity_error", name="disparity error"),
                ),
            ),
            column_shares=(2, 3),
        ),
        collapse_panels=True,
    )


def log_eth3d_prediction(pair: ETH3DRigPair, prediction: RigDepthPrediction, max_depth_m: float) -> None:
    """Log the ETH3D stereo rig, X-Lens depths, ground truth, and error."""
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    log_rig_static(Rig(index=0, calibration=pair.rig_calibration, image_plane_distance=0.25))
    for index in range(2):
        pinhole_path: str = f"world/rig_00/cam_{index:02d}/pinhole"
        rr.log(f"{pinhole_path}/image", rr.Image(pair.images[index]).compress(jpeg_quality=90))
        depth_hw: Float32[ndarray, "h w"] = prediction.depth_m[index].detach().cpu().numpy()
        display_depth_hw: Float32[ndarray, "h w"] = np.where(
            np.isfinite(depth_hw) & (depth_hw > 0.0) & (depth_hw <= max_depth_m), depth_hw, 0.0
        ).astype(np.float32)
        rr.log(f"{pinhole_path}/depth", rr.DepthImage(display_depth_hw, meter=1.0, depth_range=(0.0, max_depth_m)))

    predicted_disparity_hw: Float32[ndarray, "h w"] = metric_depth_to_disparity(
        prediction.depth_m[0].detach().cpu().numpy(), pair.calibration
    )
    disparity_error_hw: Float32[ndarray, "h w"] = np.abs(predicted_disparity_hw - pair.ground_truth_disparity).astype(np.float32)
    rr.log("world/rig_00/cam_00/rig_depth/gt_disparity", rr.DepthImage(pair.ground_truth_disparity, meter=1.0))
    rr.log("world/rig_00/cam_00/rig_depth/disparity_error", rr.DepthImage(disparity_error_hw, meter=1.0))


def benchmark_predictor(predictor: BaseRigDepthPredictor, pair: ETH3DRigPair) -> float:
    """Return warm X-Lens latency in milliseconds (10 warm-ups, 50 samples)."""
    for _ in range(10):
        predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started: float = time.perf_counter()
    for _ in range(50):
        predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / 50.0


def main(config: Config) -> None:
    """Run, evaluate, benchmark, and log X-Lens on ETH3D."""
    pair: ETH3DRigPair = load_eth3d_rig_pair(config.scene_dir)
    predictor: BaseRigDepthPredictor = config.predictor.setup(device=config.device)
    prediction: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    epe_px, bad1_percent, abs_rel = eth3d_rig_metrics(
        prediction.depth_m[0].detach().cpu().numpy(),
        pair.ground_truth_disparity,
        pair.nonoccluded,
        pair.calibration,
    )
    latency_ms: float = benchmark_predictor(predictor, pair)
    print(
        f"{type(predictor).__name__} ETH3D: EPE {epe_px:.3f} px, bad1 {bad1_percent:.2f}%, "
        f"abs-rel {abs_rel:.4f}, warm {latency_ms:.2f} ms/frameset"
    )
    rr.send_blueprint(create_rig_depth_blueprint())
    log_eth3d_prediction(pair, prediction, config.max_depth_m)
