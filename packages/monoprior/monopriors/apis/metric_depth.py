"""Metric depth estimation node.

Provides a self-contained API for running any ``BaseMetricPredictor`` on a
single RGB image to produce metric-scale depth, confidence, and camera
intrinsics. Uses the existing factory pattern from ``models/metric_depth/``.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jaxtyping import Float, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.metric_depth import (
    METRIC_PREDICTORS,
    BaseMetricPredictor,
    MetricDepthPrediction,
    get_metric_predictor,
)


@dataclass
class MetricDepthNodeConfig:
    """Configuration for metric depth estimation — works with any metric predictor."""

    predictor_name: METRIC_PREDICTORS = "MoGeV2MetricPredictor"
    """Which metric depth predictor to use (MoGeV2MetricPredictor, UniDepthMetricPredictor, etc.)."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend."""


@dataclass
class MetricDepthCLIConfig:
    """CLI configuration for metric depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/multiview/car_landscape_12/IMG_2933.jpg")
    """Path to input image."""
    metric_config: MetricDepthNodeConfig = field(default_factory=MetricDepthNodeConfig)
    """Metric depth prediction configuration."""


def create_metric_predictor(config: MetricDepthNodeConfig) -> BaseMetricPredictor:
    """Instantiate a metric depth predictor from config.

    Args:
        config: Metric depth node configuration.

    Returns:
        An initialised ``BaseMetricPredictor`` instance.
    """
    return get_metric_predictor(config.predictor_name)(device=config.device)


def run_metric_depth(
    *,
    rgb: UInt8[ndarray, "H W 3"],
    predictor: BaseMetricPredictor,
    K_33: Float[ndarray, "3 3"] | None = None,
) -> MetricDepthPrediction:
    """Run metric depth estimation on a single image.

    Args:
        rgb: Input RGB image.
        predictor: Pre-initialised metric depth predictor.
        K_33: Optional camera intrinsics. If None, the predictor estimates its own.

    Returns:
        MetricDepthPrediction with depth_meters, confidence, and K_33.
    """
    return predictor(rgb=rgb, K_33=K_33)


def main(config: MetricDepthCLIConfig) -> None:
    """CLI entry point for metric depth estimation with Rerun visualization."""
    import cv2
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
    from simplecv.rerun_log_utils import log_pinhole

    parent_log_path: Path = Path("world")

    # Load image
    bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(config.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image {config.image_path}")
    rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Init predictor and run
    predictor: BaseMetricPredictor = create_metric_predictor(config.metric_config)
    result: MetricDepthPrediction = run_metric_depth(rgb=rgb, predictor=predictor)

    # Setup Rerun
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"{parent_log_path}",
                contents=["+ $origin/**", f"- {parent_log_path}/camera/pinhole/depth"],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/confidence"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # Log pinhole
    h: int = rgb.shape[0]
    w: int = rgb.shape[1]
    intrinsics: Intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(result.K_33[0, 0]),
        fl_y=float(result.K_33[1, 1]),
        cx=float(result.K_33[0, 2]),
        cy=float(result.K_33[1, 2]),
        width=w,
        height=h,
    )
    extrinsics: Extrinsics = Extrinsics(
        world_R_cam=np.eye(3, dtype=np.float32),
        world_t_cam=np.zeros(3, dtype=np.float32),
    )
    pinhole: PinholeParameters = PinholeParameters(name="camera_0", extrinsics=extrinsics, intrinsics=intrinsics)
    cam_log_path: Path = parent_log_path / "camera"
    log_pinhole(pinhole, cam_log_path=cam_log_path, image_plane_distance=0.05, static=True)

    rr.log(f"{cam_log_path}/pinhole/image", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(), static=True)
    rr.log(f"{cam_log_path}/pinhole/depth", rr.DepthImage(result.depth_meters, meter=1), static=True)
    rr.log(f"{cam_log_path}/pinhole/confidence", rr.Image(result.confidence, color_model=rr.ColorModel.L), static=True)
