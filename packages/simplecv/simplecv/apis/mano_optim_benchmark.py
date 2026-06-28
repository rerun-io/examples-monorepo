from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.apis.view_exoego import (
    compute_vertex_normals_batch,
    create_blueprint,
    filter_out_of_bounds_keypoints,
    log_exoego_batch,
)
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.mano.mano_np import ManoSimpleLayerNP
from simplecv.ops.mano.optim_jax_single import OptimInput, OptimResult, PoseOptimConfig, SingleHandOptim
from simplecv.ops.mano.optim_jax_single_shape import (
    OptimShapeInput,
    OptimShapeResult,
    PoseShapeOptimConfig,
    SingleHandShapeOptim,
)
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class ManoOptimBenchConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedExoEgoDatasetUnion
    max_frames: int | None = None
    use_single_hand: bool = False
    single_hand_side: Literal["left", "right"] = "left"


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Coco Wholebody - Optimized", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


def main(cfg: ManoOptimBenchConfig):
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = cfg.dataset.setup()
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    exo_timestamps: list[Int[ndarray, "n_frames"]] = []

    exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    # log stationary exo cameras and video assets
    for exo_cam in exo_sequence.exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        log_pinhole(
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=exo_sequence.image_plane_distance,
            static=True,
        )

    for video_file, exo_video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        exo_timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_file, exo_video_log_path, timeline=timeline)
        exo_timestamps.append(exo_timestamps_ns)

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
    )
    rr.send_blueprint(blueprint)

    shortest_timestamp: Int[ndarray, "n_frames"] = min(exo_timestamps, key=len)
    # limit only the iteration (keep full-length for batch logging consistency)
    frames_to_iter: Int[ndarray, "n_frames"] = (
        shortest_timestamp[: cfg.max_frames] if cfg.max_frames is not None else shortest_timestamp
    )

    # This benchmark logs exo cameras in its own flat layout (parent_log_path / name),
    # so the 2D projections must target the same cam nodes (not the rig layout).
    log_exoego_batch(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        shortest_timestamp=shortest_timestamp,
        exo_cam_paths={exo_cam.name: parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list},
        ego_cam_paths={},
        log_ego=False,
        log_exo=True,
        log_mano=False,
    )
    print(f"Total time taken: {timer() - start_time:.2f} seconds")

    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    exo_cam_param_list: list[PinholeParameters] = exo_sequence.exo_cam_list
    # chosen_idx: tuple = (0, 5)
    gt_so3_seq: Float[ndarray, "n_frames 48"] | None = None
    if exoego_labels is not None:
        xyzc_stack: Float[ndarray, "n_frames n_kpts=133 4"] = exoego_labels.xyzc_stack
        xyz_stack: Float[ndarray, "n_frames n_kpts=133 3"] = xyzc_stack[:, :, :3]
        xyz_hom_stack: Float[ndarray, "n_frames n_kpts=133 4"] = np.concatenate(
            [xyz_stack, np.ones_like(xyz_stack[..., :1])], axis=-1
        )
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in exo_cam_param_list]
        )
        # Pall_exo = Pall_exo[list(chosen_idx), :, :]
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = proj_3d_vectorized(
            xyz_hom=xyz_hom_stack, P=Pall_exo
        )
        # any keypoints that are out of frame are turned into nan when margin percentage is 0.0
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = filter_out_of_bounds_keypoints(
            uv_exo_stack, exo_cam_param_list[0], margin_percentage=0.0
        )
        print(uv_exo_stack.shape)
        # beta values for mano
        # gt_beta: Float[ndarray, "10"] | None = (
        #     exoego_labels.mano_stack.betas if exoego_labels.mano_stack is not None else None
        # )
        # Also capture GT pose for the side we optimize against (0=right,1=left)
        if exoego_labels.mano_stack is not None:
            side_idx_pose: int = 1 if cfg.single_hand_side == "left" else 0
            gt_so3_seq: Float32[ndarray, "n_frames 48"] = exoego_labels.mano_stack.so3[:, side_idx_pose, :]

    ###################################
    # Step 1. Optimize for Mano Shape #
    ###################################
    hand_side: Literal["left", "right"] = "left"
    n_frames_optim: int = 10
    mano_layer: ManoSimpleLayerNP = ManoSimpleLayerNP(side=hand_side, mano_root=Path("data/"))

    optim_shape_cfg = PoseShapeOptimConfig(
        Pall=Pall_exo, hand_side=hand_side, n_frames_optim=n_frames_optim, n_optim_iters=30
    )
    optimizer_shape = SingleHandShapeOptim(config=optim_shape_cfg)
    # take the first 30 frames and feed them into the optimizer
    optim_shape_input: OptimShapeInput = OptimShapeInput(
        uv_pred=uv_exo_stack[:n_frames_optim],
        beta_init=np.zeros((10,), dtype=np.float32),
        so3_init=np.zeros((n_frames_optim, 48), dtype=np.float32),
        trans_init=np.array([[0.0, 0.0, 0.6]]).repeat(n_frames_optim, axis=0).astype(np.float32),
    )
    optim_shape_tuple: tuple[OptimShapeResult, LevenbergMarquardtState] = optimizer_shape(optim_shape_input)
    optim_shape_result: OptimShapeResult = optim_shape_tuple[0]
    beta_optim: Float[ndarray, "10"] = optim_shape_result.beta_optim
    # beta_optim: Float[ndarray, "10"] = np.random.randn(10) * 5.0

    # from icecream import ic

    # ic(optim_shape_result.beta_optim - gt_beta)
    # ic(optim_shape_result.beta_optim, gt_beta)
    ######################################################################
    # Step 2. With Found Mano Shape, Optimize for pose across all frames #
    ######################################################################
    # Optimizers over MANO pose (so3) + translation
    optim_cfg = PoseOptimConfig(
        beta=beta_optim,
        Pall=Pall_exo,
        hand_side=hand_side,
    )

    optimizer = SingleHandOptim(config=optim_cfg)
    # so3_init: Float[ndarray, "1 48"] = gt_so3_seq[0:1, ...] if gt_so3_seq is not None else np.zeros((1, 48))
    so3_init: Float[ndarray, "1 48"] = np.zeros((1, 48))
    # start with a reasonable depth
    trans_init: Float[ndarray, "1 3"] = np.array([[0.0, 0.0, 0.6]])
    # Track pose errors if GT is available
    pose_mse_list: list[float] = []
    for ts_idx, timestamp in enumerate(tqdm(frames_to_iter, desc="Optimizing Frames", unit="frame")):
        rr.set_time(timeline="video_time", duration=1e-9 * timestamp)
        _bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[ts_idx]

        # 2D keypoints per-view for COCO-133 → slice hands (already in Mediapipe order)
        uv_frame: Float[ndarray, "n_frames=1 n_views 133 2"] = uv_exo_stack[ts_idx : ts_idx + 1, :, :, :]
        optim_input: OptimInput = OptimInput(uv_pred=uv_frame, so3_init=so3_init, trans_init=trans_init)
        optim_tuple: tuple[OptimResult, LevenbergMarquardtState] = optimizer(optim_input)
        optim_result: OptimResult = optim_tuple[0]
        mano_out: tuple[Float32[ndarray, "b n_verts=778 3"], Float32[ndarray, "b joints_and_tips=21 3"]] = mano_layer(
            th_pose_coeffs=optim_result.so3_optim,
            th_betas=beta_optim[np.newaxis, :],
            th_trans=optim_result.trans_optim,
        )
        # set new init params
        so3_init = optim_result.so3_optim
        trans_init = optim_result.trans_optim
        xyz_optim: Float[ndarray, "1 n_kpts=133 3"] = np.full((1, 133, 3), np.nan)
        match optim_cfg.hand_side:
            case "left":
                xyz_optim[..., LEFT_HAND_IDX, :] = mano_out[1] / 1000
            case "right":
                xyz_optim[..., RIGHT_HAND_IDX, :] = mano_out[1] / 1000
        rr.log(
            f"{parent_log_path}/optim_xyz",
            rr.Points3D(
                xyz_optim[0],
                colors=(0, 255, 0),
                class_ids=1,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
        )

        # Log MANO mesh: static faces from the MANO layer, dynamic per-frame vertices
        faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.th_faces.astype(np.int32)
        mesh_color_rgba: tuple[int, int, int, int] = (255, 0, 0, 255)
        mesh_entity_path: Path = parent_log_path / f"mano_{mano_layer.side}_mesh"

        # Stream vertex positions and normals over time under the same entity using send_columns
        verts_np: Float32[ndarray, "n_frames n_verts=778 3"] = mano_out[0] / 1000
        n_frames_mesh: int = min(len(verts_np), len(shortest_timestamp))
        vertex_normals: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_np[0:n_frames_mesh], faces_np
        )

        rr.log(
            f"{mesh_entity_path}",
            rr.Mesh3D(
                vertex_positions=verts_np[0],
                vertex_normals=vertex_normals[0],
                triangle_indices=faces_np,
                albedo_factor=mesh_color_rgba,
            ),
        )

        # ----------------------------------------------
        # Pose GT comparison (exclude global rotation 0:3)
        # ----------------------------------------------
        if gt_so3_seq is not None:
            gt_pose_frame: Float[ndarray, "48"] = gt_so3_seq[ts_idx]
            diff: Float[ndarray, "45"] = optim_result.so3_optim[0, 3:48] - gt_pose_frame[3:48]
            pose_mse = float(np.mean(diff**2))
            pose_mse_list.append(pose_mse)

    if pose_mse_list:
        print(f"Average pose MSE over {len(pose_mse_list)} frames: {np.mean(pose_mse_list):.6f}")
