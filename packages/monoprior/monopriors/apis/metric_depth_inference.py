from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.metric_depth import (
    METRIC_PREDICTORS,
    MetricDepthPrediction,
    get_metric_predictor,
)
from monopriors.models.metric_depth.base_metric_depth import BaseMetricPredictor
from monopriors.rr_logging_utils import log_metric_pred


@dataclass
class MetricDepthConfig:
    """Configuration for single-image metric depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/single-image/room.jpg")
    """Path to the input image."""
    predictor_name: METRIC_PREDICTORS = "MoGeV2MetricPredictor"
    """Which metric depth predictor to use."""
    depth_edge_threshold: float = 0.1
    """Threshold for removing flying pixels at depth edges."""


def metric_depth_from_img(config: MetricDepthConfig) -> None:
    parent_log_path: Path = Path("world")
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint=blueprint)

    bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.imread(str(config.image_path))
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    predictor: BaseMetricPredictor = get_metric_predictor(config.predictor_name)(device="cuda")
    metric_pred: MetricDepthPrediction = predictor(rgb=rgb_hw3, K_33=None)

    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_metric_pred(parent_log_path, metric_pred, rgb_hw3, depth_edge_threshold=config.depth_edge_threshold)
