"""Minimal standalone demo wiring for SAM 3D Body with Rerun visualization."""

import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Literal

import cv2
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8
from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, get_metric_predictor
from numpy import ndarray
from sam3_rerun.api.predictor import SAM3Config, SAM3Predictor, SAM3Results
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm
from yacs.config import CfgNode

from sam3d_body.api.visualization import create_view, set_annotation_context, visualize_sample
from sam3d_body.build_models import load_sam_3d_body_hf
from sam3d_body.models.meta_arch import SAM3DBody
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction, SAM3DBodyEstimator


@dataclass
class SAM3DBodyE2EConfig:
    """Bundle of sub-configurations required for the end-to-end demo."""

    sam3_config: SAM3Config
    """Settings for the underlying SAM3 detector."""
    intrinsics_estimator: Literal["MoGeV2MetricPredictor"] = "MoGeV2MetricPredictor"
    """Identifier of the predictor used to estimate camera intrinsics."""
    mhr_path: Path = Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    """Path to the MHR mesh/pose asset file required by the head network."""
    checkpoint_path: Path = Path("checkpoints/sam-3d-body-dinov3/model.ckpt")
    """Core SAM 3D Body model checkpoint (.ckpt)."""


class SAM3DBodyE2E:
    """Convenience facade that chains detection, intrinsics estimation, and 3D reconstruction."""

    def __init__(self, config: SAM3DBodyE2EConfig):
        self.sam3_predictor = SAM3Predictor(config.sam3_config)
        self.intrinsics_predictor: BaseMetricPredictor = get_metric_predictor(config.intrinsics_estimator)(device="cuda")
        load_output: tuple[SAM3DBody, CfgNode] = load_sam_3d_body_hf(repo_id="facebook/sam-3d-body-dinov3")
        model: SAM3DBody = load_output[0]
        self.sam3d_body_estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
        )

    def predict_single_image(self, rgb_hw3: UInt8[ndarray, "h w 3"]) -> tuple[list[FinalPosePrediction], MetricDepthPrediction]:
        """Estimate 3D poses for a single frame.

        Pipeline:
        1. Use the configured intrinsics predictor to derive camera intrinsics ``K_33``.
        2. Run SAM3 to obtain person masks and boxes.
        3. Feed detections and intrinsics into ``SAM3DBodyEstimator`` for per-person 3D bodies.

        Args:
            rgb_hw3: RGB image with shape ``[H, W, 3]`` and dtype ``uint8``.

        Returns:
            A list of ``FinalPosePrediction`` entries—one per detected person.
        """
        # estimate the camera intrinsics and metric depth
        metric_pred: MetricDepthPrediction = self.intrinsics_predictor(rgb=rgb_hw3, K_33=None)
        K_33: Float32[ndarray, "3 3"] = metric_pred.K_33

        sam3_results: SAM3Results = self.sam3_predictor.predict_single_image(rgb_hw3)

        outputs: list[FinalPosePrediction] = self.sam3d_body_estimator.process_one_image(
            rgb_hw3,
            xyxy=sam3_results.boxes,
            masks=sam3_results.masks,
            masks_score=sam3_results.scores,
            K_33=K_33,
        )
        return outputs, metric_pred


@dataclass(slots=True)
class Sam3DBodyDemoConfig:
    """Configuration for the standalone demo runner."""

    rr_config: RerunTyroConfig
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""

    sam3_e2e_config: SAM3DBodyE2EConfig
    """Configuration for the end-to-end SAM 3D Body model."""

    image_folder: Path | None = None
    """Directory containing input images to process."""

    image_path: Path | None = None
    """Path to a single input image to process."""

    max_frames: int | None = None
    """Optional limit on the number of images to process; ``None`` processes all images."""


def main(cfg: Sam3DBodyDemoConfig):
    """Run the Rerun-enabled demo on a folder or single image.

    Args:
        cfg: Aggregated configuration containing Rerun settings, SAM3 model options,
            and input image selection.
    """
    # Setup Rerun
    parent_log_path = Path("/world")
    set_annotation_context()
    view: rrb.ContainerLike = create_view(log_depth=True)
    blueprint = rrb.Blueprint(view, collapse_panels=True)
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    if cfg.image_path is not None:
        images_list = [str(cfg.image_path)]
    elif cfg.image_folder is not None:
        image_extensions: list[str] = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]
        images_list: list[str] = sorted([image for ext in image_extensions for image in glob(os.path.join(cfg.image_folder, ext))])
    else:
        raise ValueError("Either image_path or image_folder must be specified.")

    # load end to end model
    sam3D_body_e2e = SAM3DBodyE2E(cfg.sam3_e2e_config)

    for idx, image_path in enumerate(tqdm(images_list)):
        rr.set_time(timeline="image_sequence", sequence=idx)
        # load image and convert to RGB
        bgr_hw3: UInt8[ndarray, "h w 3"] = cv2.imread(image_path)
        rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

        outputs: tuple[list[FinalPosePrediction], MetricDepthPrediction] = sam3D_body_e2e.predict_single_image(rgb_hw3)
        pred_list: list[FinalPosePrediction] = outputs[0]
        metric_pred: MetricDepthPrediction = outputs[1]

        if len(pred_list) == 0:
            # Detector/FOV failed on this frame; avoid crashing the visualization step.
            print(f"[warn] No detections for {image_path}; skipping.")
            continue

        visualize_sample(
            pred_list=pred_list,
            rgb_hw3=rgb_hw3,
            parent_log_path=parent_log_path,
            faces=sam3D_body_e2e.sam3d_body_estimator.faces,
            metric_depth_pred=metric_pred,
        )
