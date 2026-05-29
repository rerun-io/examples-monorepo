from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.apis.view_exoego import create_blueprint, filter_out_of_bounds_keypoints
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.mano.optim_jax_multi import LossWeights, ManoOptimization, OptimizationResults
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
    max_frames: int | None = 10


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
                rr.AnnotationInfo(id=1, label="Left Hand"),
                rr.AnnotationInfo(id=2, label="Right Hand"),
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

    print(f"Total time taken: {timer() - start_time:.2f} seconds")

    # log_exoego_batch(
    #     exoego_sequence,
    #     parent_log_path=parent_log_path,
    #     timeline=timeline,
    #     shortest_timestamp=shortest_timestamp,
    #     log_ego=False,
    #     log_exo=True,
    # )

    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    exo_cam_param_list: list[PinholeParameters] = exo_sequence.exo_cam_list
    if exoego_labels is not None:
        xyzc_stack: Float[ndarray, "n_frames n_kpts=133 4"] = exoego_labels.xyzc_stack
        xyz_stack: Float[ndarray, "n_frames n_kpts=133 3"] = xyzc_stack[:, :, :3]
        xyz_hom_stack: Float[ndarray, "n_frames n_kpts=133 4"] = np.concatenate(
            [xyz_stack, np.ones_like(xyz_stack[..., :1])], axis=-1
        )
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in exo_cam_param_list]
        )
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = proj_3d_vectorized(
            xyz_hom=xyz_hom_stack, P=Pall_exo
        )
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = filter_out_of_bounds_keypoints(
            uv_exo_stack, exo_cam_param_list[0]
        )
        # beta values for mano
        gt_beta: Float[ndarray, "10"] | None = (
            exoego_labels.mano_stack.betas if exoego_labels.mano_stack is not None else None
        )
        # gt_so3: Float32[ndarray, "n_frames n_hands=2 48"] | None = (
        #     exoego_labels.mano_stack.so3 if exoego_labels.mano_stack is not None else None
        # )
        # gt_trans: Float32[ndarray, "n_frames n_hands=2 3"] | None = (
        #     exoego_labels.mano_stack.trans if exoego_labels.mano_stack is not None else None
        # )

    # Optimizer over MANO pose (so3) + translation using 2D keypoints across views
    optimizer_fn = ManoOptimization(
        beta=gt_beta, Pall=Pall_exo, loss_weights=LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0)
    )
    for ts_idx, timestamp in enumerate(tqdm(frames_to_iter, desc="Logging frames", unit="frame")):
        rr.set_time(timeline="video_time", duration=1e-9 * timestamp)
        _bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[ts_idx]

        # 2D keypoints per-view for COCO-133 → slice hands (already in Mediapipe order)
        uv_frame: Float[ndarray, "n_views 133 2"] = uv_exo_stack[ts_idx]
        uv_left_mediapipe: Float[ndarray, "n_views 21 2"] = uv_frame[:, LEFT_HAND_IDX, :]
        uv_right_mediapipe: Float[ndarray, "n_views 21 2"] = uv_frame[:, RIGHT_HAND_IDX, :]

        # Optimize MANO parameters for both hands from 2D
        optimization_output = optimizer_fn(
            uv_left_pred_batch=uv_left_mediapipe,
            uv_right_pred_batch=uv_right_mediapipe,
        )
        optimization_results: OptimizationResults = optimization_output[0]

        # Extract results
        xyz_mano_opt: Float[ndarray, "2 21 3"] = optimization_results.xyz_mano
        # so3_opt: Float[ndarray, "2 48"] = optimization_results.so3
        # trans_opt: Float[ndarray, "2 3"] = optimization_results.trans

        # Log 3D keypoints for left/right
        rr.log(f"{parent_log_path}/optimized_left_kpts", rr.Points3D(xyz_mano_opt[0]))
        rr.log(f"{parent_log_path}/optimized_right_kpts", rr.Points3D(xyz_mano_opt[1]))

        # Also overlay 2D reprojections per exo cam for visual validation
        for exo_cam_idx, exo_cam in enumerate(exo_cam_param_list):
            exo_cam_path: Path = parent_log_path / exo_cam.name
            exo_pinhole_path: Path = exo_cam_path / "pinhole"

            for side_idx, side_name in enumerate(["left", "right"]):
                xyz_hom: Float[ndarray, "1 21 4"] = np.concatenate(
                    [xyz_mano_opt[side_idx][None, ...], np.ones((1, 21, 1), dtype=xyz_mano_opt.dtype)], axis=-1
                )
                uv_proj: Float[ndarray, "1 n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_hom, P=Pall_exo)
                uv_proj_cam: Float[ndarray, "21 2"] = uv_proj[0, exo_cam_idx]
                rr.log(f"{exo_pinhole_path}/optimized_{side_name}", rr.Points2D(uv_proj_cam))
