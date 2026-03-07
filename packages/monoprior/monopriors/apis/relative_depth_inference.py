from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.relative_depth import (
    RELATIVE_PREDICTORS,
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor
from monopriors.rr_logging_utils import log_relative_pred


@dataclass
class PredictorConfig:
    rr_config: RerunTyroConfig
    image_path: Path = Path("data/examples/single-image/room.jpg")
    predictor_name: RELATIVE_PREDICTORS = "MogeV1Predictor"
    depth_edge_threshold: float = 0.1


def resize_image(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    current_dim = max(image.shape[0], image.shape[1])
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_h: int = int(image.shape[0] * scale_factor)
        new_w: int = int(image.shape[1] * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def relative_depth_from_img(config: PredictorConfig) -> None:
    parent_log_path = Path("world")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/confidence"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint=blueprint)
    bgr_hw3 = cv2.imread(str(config.image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    max_dim = 1024 // 2
    rgb_hw3 = resize_image(rgb_hw3, max_dim)

    predictor: BaseRelativePredictor = get_relative_predictor(config.predictor_name)(device="cuda")
    relative_pred: RelativeDepthPrediction = predictor.__call__(rgb=rgb_hw3, K_33=None)
    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_relative_pred(parent_log_path, relative_pred, rgb_hw3, depth_edge_threshold=config.depth_edge_threshold)
