from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.surface_normal import (
    AnnotatedNormalPredictorUnion,
    DSineNormalConfig,
    SurfaceNormalPrediction,
)
from monopriors.models.surface_normal.base_normal_model import BaseNormalPredictor
from monopriors.rr_logging_utils import log_normal_pred


@dataclass
class NormalPredictorConfig:
    """Configuration for single-image surface normal estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_path: Path = Path("data/examples/single-image/room.jpg")
    """Path to the input image."""
    predictor: AnnotatedNormalPredictorUnion = field(default_factory=DSineNormalConfig)
    """Surface-normal model to run; select it with a predictor subcommand."""


def surface_normal_from_img(config: NormalPredictorConfig) -> None:
    parent_log_path: Path = Path("world")
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/normals"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/confidence"),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint=blueprint)

    bgr_hw3: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(str(config.image_path))
    if bgr_hw3 is None:
        raise FileNotFoundError(f"Failed to read image {config.image_path}")
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    predictor: BaseNormalPredictor = config.predictor.setup(device="cuda")
    normal_pred: SurfaceNormalPrediction = predictor(rgb=rgb_hw3, K_33=None)

    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_normal_pred(parent_log_path, normal_pred, rgb_hw3)
