"""Multi-view geometry prediction node.

Provides a self-contained API for running a multi-view geometry predictor
(VGGT or gravity-aligned G3T) to produce oriented camera
poses, depths, confidences, and intrinsics from a list of RGB images.
This is the first node in the multi-view calibration pipeline.

The network genuinely takes a list — it processes all views jointly to
produce consistent multi-view geometry. This is the only node where list
input matches the network's actual contract.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Final

import numpy as np
from beartype.vale import Is
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.multiview.multiview_model import MultiviewPred, robust_filter_confidences
from monopriors.models.multiview.multiview_pointcloud import mv_pred_to_filtered_pointcloud
from monopriors.models.multiview.multiview_predictor import (
    CenterMethod,
    ImagePreprocessingMode,
    MultiviewPredictor,
    MultiviewPredictorConfig,
)

KeepTopPercent = Annotated[int | float, Is[lambda percent: 1 <= percent <= 100]]
MAX_POINT_CLOUD_POINTS: Final[int] = 150_000


@dataclass(frozen=True, slots=True)
class MultiviewGeometryConfig:
    """Configuration for multi-view geometry prediction."""

    keep_top_percent: KeepTopPercent = 30.0
    """Percentage of high-confidence pixels retained after filtering.
    Value in [1, 100]; e.g. 30 keeps the top 30% and uses up to 30% of the
    bounded point-cloud budget."""
    preprocessing_mode: ImagePreprocessingMode = "pad"
    """Image preprocessing strategy for this inference run."""
    center_method: CenterMethod = "none"
    """How to center canonicalized camera poses after backend inference."""
    verbose: bool = False
    """Emit per-camera detail logging when True."""

    @property
    def point_cloud_budget(self) -> int:
        """Maximum point count after confidence filtering."""
        return max(1, round(MAX_POINT_CLOUD_POINTS * float(self.keep_top_percent) / 100.0))


@dataclass
class MultiviewGeometryResult:
    """Output of multi-view geometry prediction."""

    mv_pred_list: list[MultiviewPred]
    """Oriented multi-view predictions (poses, depths, confidences, intrinsics)."""
    depth_confidences: list[UInt8[ndarray, "H W"]]
    """Binary confidence masks after robust filtering (0 or 255)."""


def run_multiview_geometry(
    *,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    multiview_predictor: MultiviewPredictor,
    config: MultiviewGeometryConfig,
) -> MultiviewGeometryResult:
    """Run multi-view geometry prediction.

    Runs the selected backend on all views jointly, canonicalizes the resulting poses,
    and computes robust confidence masks.

    Args:
        rgb_list: Ordered RGB frames across cameras.
        multiview_predictor: Pre-initialised backend-neutral predictor.
        config: Geometry prediction configuration.

    Returns:
        MultiviewGeometryResult with oriented predictions and confidence masks.
    """
    mv_pred_list: list[MultiviewPred] = multiview_predictor(
        rgb_list,
        preprocessing_mode=config.preprocessing_mode,
        center_method=config.center_method,
    )

    depth_confidences: list[UInt8[ndarray, "H W"]] = [
        robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=config.keep_top_percent)
        for mv_pred in mv_pred_list
    ]

    return MultiviewGeometryResult(
        mv_pred_list=mv_pred_list,
        depth_confidences=depth_confidences,
    )


@dataclass
class MultiviewGeometryCLIConfig:
    """CLI configuration for multi-view geometry prediction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_dir: Path = Path("data/examples/multiview/car_landscape_12")
    """Directory containing input images."""
    predictor_config: MultiviewPredictorConfig = field(default_factory=MultiviewPredictorConfig)
    """Model construction configuration."""
    geometry_config: MultiviewGeometryConfig = field(default_factory=lambda: MultiviewGeometryConfig(verbose=True))
    """Multi-view geometry prediction configuration."""


def main(config: MultiviewGeometryCLIConfig) -> None:
    """CLI entry point for multi-view geometry prediction with Rerun visualization."""
    import rerun as rr
    import rerun.blueprint as rrb
    from simplecv.rerun_log_utils import log_pinhole

    from monopriors.apis.multiview_calibration import (
        PARENT_LOG_PATH,
        SUPPORTED_IMAGE_EXTENSIONS,
        load_rgb_images,
    )

    # Load images
    image_paths: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_paths.extend(config.image_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)
    assert len(image_paths) > 0, f"No images found in {config.image_dir}"
    rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(image_paths)

    # Init predictor
    multiview_predictor: MultiviewPredictor = MultiviewPredictor(config.predictor_config)

    # Run geometry
    result: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=rgb_list,
        multiview_predictor=multiview_predictor,
        config=config.geometry_config,
    )

    # Setup Rerun blueprint
    from monopriors.gradio_ui.multiview_geometry_ui import create_multiview_blueprint

    final_view: rrb.ContainerLike = create_multiview_blueprint(parent_log_path=PARENT_LOG_PATH, num_images=len(rgb_list))
    blueprint: rrb.Blueprint = rrb.Blueprint(final_view, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{PARENT_LOG_PATH}", rr.ViewCoordinates.RFU, static=True)

    # Log per-camera results
    for mv_pred, depth_conf in zip(result.mv_pred_list, result.depth_confidences, strict=True):
        cam_log_path: Path = PARENT_LOG_PATH / mv_pred.cam_name
        pinhole_log_path: Path = cam_log_path / "pinhole"
        log_pinhole(mv_pred.pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.05, static=True)
        rr.log(
            f"{pinhole_log_path}/image",
            rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
            static=True,
        )
        rr.log(
            f"{pinhole_log_path}/confidence", rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(), static=True
        )
        filtered_depth: Float32[ndarray, "H W"] = np.where(depth_conf > 0, mv_pred.depth_map, 0)
        rr.log(f"{pinhole_log_path}/filtered_depth", rr.DepthImage(filtered_depth, meter=1), static=True)
        rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(mv_pred.depth_map, meter=1), static=True)

    # Log point cloud
    pointcloud, point_colors = mv_pred_to_filtered_pointcloud(
        result.mv_pred_list,
        result.depth_confidences,
        target_points=config.geometry_config.point_cloud_budget,
    )
    rr.log(
        f"{PARENT_LOG_PATH}/point_cloud",
        rr.Points3D(pointcloud, colors=point_colors),
        static=True,
    )
