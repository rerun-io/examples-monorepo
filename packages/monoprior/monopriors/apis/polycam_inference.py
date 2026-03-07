import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8, UInt16
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.polycam import PolycamData, PolycamDataset, load_polycam_data
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm

from monopriors.monoprior_models import DsineAndUnidepth, MonoPriorPrediction


@dataclass
class PolycamConfig:
    polycam_zip_path: Path
    rr_config: RerunTyroConfig


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    assert zip_path.suffix == ".zip", "Not a zip file"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def validate_zip(zip_path: Path) -> bool:
    return zip_path.is_file() and zip_path.exists() and zip_path.suffix == ".zip"


def find_keyframes_dir(extract_dir: Path) -> Path | None:
    for path in extract_dir.rglob("*"):
        if path.is_dir() and path.name == "keyframes":
            return path
    return None


def log_mono_pred(
    parent_path: Path,
    rgb: UInt8[np.ndarray, "h w 3"],
    mono_pred: MonoPriorPrediction,
    pinhole_params: PinholeParameters,
    gt_depth: UInt16[np.ndarray, "h w"] | None = None,
):
    cam_path = parent_path / "cam"
    pinhole_path = cam_path / "pinhole"

    rr.log(
        f"{cam_path}",
        rr.Transform3D(
            translation=pinhole_params.extrinsics.cam_t_world,
            mat3x3=pinhole_params.extrinsics.cam_R_world,
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=pinhole_params.intrinsics.k_matrix,
            width=pinhole_params.intrinsics.width,
            height=pinhole_params.intrinsics.height,
            camera_xyz=getattr(
                rr.ViewCoordinates,
                pinhole_params.intrinsics.camera_conventions,
            ),
        ),
    )

    rr.log(f"{pinhole_path}/image", rr.Image(rgb))
    rr.log(
        f"{pinhole_path}/depth",
        rr.DepthImage(mono_pred.metric_pred.depth_meters, meter=1.0),
    )
    rr.log(f"{pinhole_path}/normal", rr.Image(mono_pred.normal_pred.normal_hw3))
    # rr.log(
    #     f"{pinhole_path}/normal_conf",
    #     rr.DepthImage(mono_pred.normal_pred.confidence_hw1),
    # )
    if gt_depth is not None:
        rr.log(f"{pinhole_path}/gt_depth", rr.DepthImage(gt_depth, meter=1000))
        # diff_depth_l1 = np.abs((depth_np.squeeze() - gt_depth))
        # normalize to 0-255
        # diff_depth_l1 = (diff_depth_l1 / diff_depth_l1.max() * 255).astype(np.uint8)
        # rr.log(f"{pinhole_path}/depth_error_l1", rr.Image(diff_depth_l1))


def polycam_inference(config: PolycamConfig) -> None:
    polycam_dataset: PolycamDataset = load_polycam_data(config.polycam_zip_path)

    model = DsineAndUnidepth()
    gt_fuser = Open3DFuser()
    pred_fuser = Open3DFuser()

    parent_path: Path = Path("world")
    rr.log(f"{parent_path}", rr.ViewCoordinates.RUB, static=True)

    pbar = tqdm(polycam_dataset, total=len(polycam_dataset))
    polycam_data: PolycamData
    for idx, polycam_data in enumerate(pbar):
        rr.set_time("timestep", sequence=idx)
        rgb_hw3: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
        pinhole_params: PinholeParameters = polycam_data.pinhole_params
        K_33: Float32[np.ndarray, "3 3"] = pinhole_params.intrinsics.k_matrix.astype(np.float32)

        pred: MonoPriorPrediction = model.__call__(rgb_hw3, K_33)
        # convert to mm and Uint16
        pred_depth_hw: UInt16[np.ndarray, "h w"] = (pred.metric_pred.depth_meters * 1000).astype(np.uint16)

        gt_fuser.fuse_frames(
            polycam_data.depth_hw,
            K_33,
            pinhole_params.extrinsics.cam_T_world.astype(np.float32),
            rgb_hw3,
        )
        pred_fuser.fuse_frames(
            pred_depth_hw,
            K_33,
            pinhole_params.extrinsics.cam_T_world.astype(np.float32),
            rgb_hw3,
        )

        log_mono_pred(
            parent_path=parent_path,
            rgb=rgb_hw3,
            mono_pred=pred,
            pinhole_params=pinhole_params,
            gt_depth=polycam_data.depth_hw,
        )
    gt_mesh = gt_fuser.get_mesh()
    gt_mesh.compute_vertex_normals()

    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()

    rr.log(
        f"{parent_path}/gt_mesh",
        rr.Mesh3D(
            vertex_positions=gt_mesh.vertices,
            triangle_indices=gt_mesh.triangles,
            vertex_normals=gt_mesh.vertex_normals,
            vertex_colors=gt_mesh.vertex_colors,
        ),
    )

    rr.log(
        f"{parent_path}/pred_mesh",
        rr.Mesh3D(
            vertex_positions=pred_mesh.vertices,
            triangle_indices=pred_mesh.triangles,
            vertex_normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.vertex_colors,
        ),
    )
