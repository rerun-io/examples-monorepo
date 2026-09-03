from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import torch
from jaxtyping import Float, Float32, UInt8, UInt16
from numpy import ndarray
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.depth import quantize_depth_m_to_mm
from simplecv.ops.tsdf_depth_fuser import Open3DFuser, log_fused_mesh
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torch import Tensor
from tqdm import tqdm
from trtkit import TorchBackendConfig

from monopriors.models.depth_completion.prompt_da import (
    PromptDAConfig,
    PromptDAPredictor,
    network_image_hw,
)


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
    rgb_resized: UInt8[np.ndarray, "target_height target_width 3"] = np.asarray(
        cv2.resize(rgb, (target_width, target_height)), dtype=np.uint8
    )
    depth_resized: UInt16[np.ndarray, "target_height target_width"] = np.asarray(
        cv2.resize(depth, (target_width, target_height)), dtype=np.uint16
    )
    confidence_resized: UInt8[np.ndarray, "target_height target_width"] = np.asarray(
        cv2.resize(confidence, (target_width, target_height)), dtype=np.uint8
    )
    depth_pred_resized: UInt16[np.ndarray, "target_height target_width"] = np.asarray(
        cv2.resize(depth_pred, (target_width, target_height)), dtype=np.uint16
    )

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

    first_frame: PolycamData = next(iter(polycam_dataset))
    image_hw: tuple[int, int] = network_image_hw(first_frame.rgb_hw3.shape[:2], config.max_size)
    model: PromptDAPredictor = PromptDAConfig(
        backend=TorchBackendConfig(),
        image_height=image_hw[0],
        image_width=image_hw[1],
    ).setup()
    pbar = tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))
    polycam_data: PolycamData
    for frame_idx, polycam_data in enumerate(pbar):
        rr.set_time("frame_idx", sequence=frame_idx)
        rgb_bhwc: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(polycam_data.rgb_hw3).unsqueeze(0).cuda()
        prompt_bhw: Float32[Tensor, "1 192 256"] = (
            torch.from_numpy(polycam_data.original_depth_hw).unsqueeze(0).to(device="cuda", dtype=torch.float32) / 1000.0
        )
        depth_bhw: Float32[Tensor, "1 h w"] = model(rgb_bhwc, prompt_bhw)
        depth_pred_mm: UInt16[np.ndarray, "h w"] = quantize_depth_m_to_mm(depth_bhw[0]).cpu().numpy()

        # filter depthmaps based on confidence, only keep with max confidence
        pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
            depth_mm=depth_pred_mm,
            confidence=polycam_data.confidence_hw,
            confidence_threshold=DepthConfidenceLevel.MEDIUM,
            max_depth_meter=max_depth_meter,
        )

        # fuse the predicted depth and the ground truth depth
        # simplecv's Intrinsics stores k_matrix as float64; Open3DFuser.fuse_frames accepts Float
        K_33: Float[np.ndarray, "3 3"] | None = polycam_data.pinhole_params.intrinsics.k_matrix
        if K_33 is None:
            raise ValueError("Polycam frame is missing camera intrinsics.")
        pred_fuser.fuse_frames(
            depth_hw=pred_filtered_depth_mm,
            K_33=K_33,
            cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
            rgb_hw3=polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_path,
            polycam_data=polycam_data,
            depth_pred=depth_pred_mm,
            rescale_factor=1,
        )

        pred_mesh = pred_fuser.get_mesh()
        pred_mesh.compute_vertex_normals()

        log_fused_mesh(
            f"{parent_path}/pred_mesh",
            pred_mesh,
            static=False,
            face_rendering=rr.components.MeshFaceRendering.DoubleSided,
        )

    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()

    log_fused_mesh(
        f"{parent_path}/pred_mesh",
        pred_mesh,
        static=False,
        face_rendering=rr.components.MeshFaceRendering.DoubleSided,
    )
