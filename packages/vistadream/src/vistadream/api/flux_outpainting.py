from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Bool, Float
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.models.relative_depth import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.resize_utils import add_border_and_mask, process_image


@dataclass
class FluxOutpaintingConfig:
    """
    Configuration for Flux Outpainting.
    """

    rr_config: RerunTyroConfig
    image_path: Path
    offload: bool = True
    num_steps: int = 25
    guidance: float = 30.0
    expansion_percent: float = 0.2


def main(config: FluxOutpaintingConfig) -> None:
    """
    Main function to run the Flux Outpainting process.
    """
    parent_log_path: Path = Path("/world")
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                contents=[
                    "+ $origin/**",
                    f"- {pinhole_path}/mask",
                ]
            ),
            rrb.Grid(
                rrb.Spatial2DView(
                    origin=f"{pinhole_path}/mask",
                ),
                rrb.Spatial2DView(
                    origin=f"{pinhole_path}/input_image",
                ),
                rrb.Spatial2DView(
                    origin=f"{pinhole_path}/image",
                ),
                rrb.Spatial2DView(
                    origin=f"{pinhole_path}/depth",
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint=blueprint)
    rr.set_time("time", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig())

    input_image: Image.Image = Image.open(config.image_path).convert("RGB")
    # ensures image is correctly sized and processed
    input_image: Image.Image = process_image(input_image)

    # Auto-generate outpainting setup: user-controlled border expansion
    border_percent: float = config.expansion_percent / 2.0  # Convert to fraction per side (divide by 2 for each side)
    input_image, mask = add_border_and_mask(
        input_image,
        zoom_all=1.0,
        zoom_left=border_percent,
        zoom_right=border_percent,
        zoom_up=border_percent,
        zoom_down=border_percent,
        overlap=0,
    )
    rr.log(f"{pinhole_path}/input_image", rr.Image(input_image, color_model=rr.ColorModel.RGB))
    rr.log(f"{pinhole_path}/mask", rr.Image(mask, color_model=rr.ColorModel.L))

    width, height = input_image.size

    outpainted_image: Image.Image = flux_inpainter(rgb_hw3=np.array(input_image), mask=np.array(mask))
    rr.log(f"{pinhole_path}/image", rr.Image(outpainted_image, color_model=rr.ColorModel.RGB))
    rgb_hw3 = np.array(outpainted_image.convert("RGB"))
    predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")
    relative_pred: RelativeDepthPrediction = predictor.__call__(rgb=rgb_hw3, K_33=None)

    intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=relative_pred.K_33[0, 0].item(),
        fl_y=relative_pred.K_33[1, 1].item(),
        cx=relative_pred.K_33[0, 2].item(),
        cy=relative_pred.K_33[1, 2].item(),
        width=rgb_hw3.shape[1],
        height=rgb_hw3.shape[0],
    )

    extri = Extrinsics(
        world_R_cam=np.eye(3, dtype=np.float32),
        world_t_cam=np.zeros(3, dtype=np.float32),
    )
    pinhole_params = PinholeParameters(name="pinhole", intrinsics=intri, extrinsics=extri)
    log_pinhole(camera=pinhole_params, cam_log_path=cam_log_path)

    depth_hw: Float[np.ndarray, "h w"] = relative_pred.depth
    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.01)
    depth_hw: Float[np.ndarray, "h w"] = depth_hw * ~edges_mask

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw, meter=1.0))

    depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )
