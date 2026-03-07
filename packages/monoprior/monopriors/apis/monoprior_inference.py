from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.monoprior import (
    DsineAndUnidepth,
    MoGeV2MonoPrior,
    MonoPriorModel,
    MonoPriorPrediction,
)
from monopriors.rr_logging_utils import log_metric_pred, log_normal_pred

MONOPRIOR_MODELS = Literal["MoGeV2MonoPrior", "DsineAndUnidepth"]
"""Available composite monoprior models."""


def get_monoprior_model(model_name: MONOPRIOR_MODELS) -> MonoPriorModel:
    """Instantiate a composite monoprior model by name."""
    match model_name:
        case "MoGeV2MonoPrior":
            return MoGeV2MonoPrior()
        case "DsineAndUnidepth":
            return DsineAndUnidepth()
        case _:
            raise ValueError(f"Unknown monoprior model: {model_name}")


@dataclass
class MonoPriorConfig:
    """Configuration for single-image monoprior (metric depth + surface normals)."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/single-image/room.jpg")
    """Path to the input image."""
    model_name: MONOPRIOR_MODELS = "MoGeV2MonoPrior"
    """Which composite monoprior model to use."""
    depth_edge_threshold: float = 0.1
    """Threshold for removing flying pixels at depth edges."""


def monoprior_from_img(config: MonoPriorConfig) -> None:
    parent_log_path: Path = Path("world")
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/normals"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/confidence"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint=blueprint)

    bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.imread(str(config.image_path))
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    model: MonoPriorModel = get_monoprior_model(config.model_name)
    pred: MonoPriorPrediction = model(rgb=rgb_hw3, K_33=None)

    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_metric_pred(parent_log_path, pred.metric_pred, rgb_hw3, depth_edge_threshold=config.depth_edge_threshold)
    log_normal_pred(parent_log_path, pred.normal_pred, rgb_hw3, K_33=pred.metric_pred.K_33)
