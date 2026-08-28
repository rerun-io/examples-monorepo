"""ZipDepth on one image, visualized in Rerun as RGB + relative depth + backprojected cloud.

Same layout as ``monoprior-relative-depth``; this entry point additionally accepts a local
checkpoint so trained ``final_model.pth`` files can be inspected.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import UInt8
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor
from monopriors.rr_logging_utils import log_relative_pred
from simplecv.rerun_log_utils import RerunTyroConfig


@dataclass
class InferRerunConfig:
    rr_config: RerunTyroConfig
    image: Path = Path("assets/examples/im0.jpg")
    checkpoint: Path | None = None
    """Local .pth; defaults to the released weights from the Hub."""
    input_size: int = 384
    device: Literal["auto", "cuda", "cpu"] = "auto"
    depth_edge_threshold: float = 0.1


def infer_rerun(config: InferRerunConfig) -> None:
    parent_log_path = Path("world")
    rr.send_blueprint(
        rrb.Blueprint(
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
    )
    bgr = cv2.imread(str(config.image))
    if bgr is None:
        raise FileNotFoundError(config.image)
    rgb: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    device: Literal["cuda", "cpu"] = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    predictor = ZipDepthPredictor(device=device, checkpoint=config.checkpoint, input_size=config.input_size)
    pred = predictor(rgb, None)
    print(f"  Disparity range: [{pred.disparity.min():.3f}, {pred.disparity.max():.3f}]")

    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_relative_pred(parent_log_path, pred, rgb, depth_edge_threshold=config.depth_edge_threshold)
