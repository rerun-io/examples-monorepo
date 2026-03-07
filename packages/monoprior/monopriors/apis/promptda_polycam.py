from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from jaxtyping import UInt8, UInt16
from numpy import ndarray
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from monopriors.models.depth_completion.base_completion_depth import (
    CompletionDepthPrediction,
)
from monopriors.models.depth_completion.prompt_da import PromptDAPredictor


@dataclass
class PDAPolycamConfig:
    polycam_zip_path: Path
    rr_config: RerunTyroConfig
    max_size: int = 1008


def log_polycam_data(
    parent_path: Path,
    polycam_data: PolycamData,
    depth_pred: UInt16[ndarray, "h w"],
    rescale_factor: int = 1,
) -> None:
    cam_path: Path = parent_path / "cam"
    pinhole_path: Path = cam_path / "pinhole"

    rgb: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
    depth: UInt16[np.ndarray, "h w"] = polycam_data.depth_hw
    confidence: UInt8[np.ndarray, "h w"] = polycam_data.confidence_hw

    # resize images to be half the size
    target_height: int = rgb.shape[0] // rescale_factor
    target_width: int = rgb.shape[1] // rescale_factor
    rgb_resized = cv2.resize(rgb, (target_width, target_height))
    depth_resized = cv2.resize(depth, (target_width, target_height))
    confidence_resized = cv2.resize(confidence, (target_width, target_height))
    depth_pred_resized = cv2.resize(depth_pred, (target_width, target_height))

    # rescale intrinsics to match the image size
    rescaled_intrinsics: Intrinsics = rescale_intri(
        camera_intrinsics=polycam_data.pinhole_params.intrinsics,
        target_height=target_height,
        target_width=target_width,
    )

    polycam_data.pinhole_params.intrinsics = rescaled_intrinsics

    log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=cam_path)
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_resized).compress(jpeg_quality=75))
    rr.log(f"{pinhole_path}/confidence", rr.SegmentationImage(confidence_resized))
    rr.log(f"{pinhole_path}/arkit_depth", rr.DepthImage(depth_resized, meter=1000))
    rr.log(f"{pinhole_path}/pred_depth", rr.DepthImage(depth_pred_resized, meter=1000))


def filter_depth(
    depth_mm: UInt16[np.ndarray, "h w"],
    confidence: UInt8[np.ndarray, "h w"],
    confidence_threshold: DepthConfidenceLevel,
    max_depth_meter: float,
) -> UInt16[np.ndarray, "h w"]:
    filtered_depth_mm: UInt16[np.ndarray, "h w"] = depth_mm.copy()
    filtered_depth_mm[confidence < confidence_threshold] = 0
    filtered_depth_mm[depth_mm > max_depth_meter * 1000] = 0

    return filtered_depth_mm


def pda_polycam_inference(
    config: PDAPolycamConfig,
) -> None:
    parent_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)

    max_depth_meter: float = 4.0
    pred_fuser = Open3DFuser(fusion_resolution=0.04, max_fusion_depth=max_depth_meter)

    model = PromptDAPredictor(device="cuda", model_type="large", max_size=config.max_size)
    pbar = tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))
    polycam_data: PolycamData
    for frame_idx, polycam_data in enumerate(pbar):
        rr.set_time("frame_idx", sequence=frame_idx)
        # convert image data to tensor
        depth_pred: CompletionDepthPrediction = model(
            rgb=polycam_data.rgb_hw3, prompt_depth=polycam_data.original_depth_hw
        )

        # filter depthmaps based on confidence, only keep with max confidence
        pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
            depth_mm=depth_pred.depth_mm,
            confidence=polycam_data.confidence_hw,
            confidence_threshold=DepthConfidenceLevel.MEDIUM,
            max_depth_meter=max_depth_meter,
        )

        # fuse the predicted depth and the ground truth depth
        pred_fuser.fuse_frames(
            depth_hw=pred_filtered_depth_mm,
            K_33=polycam_data.pinhole_params.intrinsics.k_matrix,
            cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
            rgb_hw3=polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_path,
            polycam_data=polycam_data,
            depth_pred=depth_pred.depth_mm,
            rescale_factor=1,
        )

        pred_mesh = pred_fuser.get_mesh()
        pred_mesh.compute_vertex_normals()

        rr.log(
            f"{parent_path}/pred_mesh",
            rr.Mesh3D(
                vertex_positions=pred_mesh.vertices,
                triangle_indices=pred_mesh.triangles,
                vertex_normals=pred_mesh.vertex_normals,
                vertex_colors=pred_mesh.vertex_colors,
            ),
        )

    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()

    rr.log(
        f"{parent_path}/pred_mesh",
        rr.Mesh3D(
            vertex_positions=pred_mesh.vertices,
            triangle_indices=pred_mesh.triangles,
            vertex_normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.vertex_colors,
        ),
    )
