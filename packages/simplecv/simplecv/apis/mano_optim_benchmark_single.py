from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

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
from simplecv.ops.mano.optim_jax_multi import LossWeights, ManoOptimization
from simplecv.ops.mano.optim_jax_single import SingleHandOptimization
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

    # Optimizers over MANO pose (so3) + translation
    single_opt_cache: dict[tuple[int, ...], SingleHandOptimization] = {}
    multi_opt_cache: dict[tuple[int, ...], ManoOptimization] = {}
    for ts_idx, timestamp in enumerate(tqdm(frames_to_iter, desc="Logging frames", unit="frame")):
        rr.set_time(timeline="video_time", duration=1e-9 * timestamp)
        _bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[ts_idx]

        # 2D keypoints per-view for COCO-133 → slice hands (already in Mediapipe order)
        uv_frame: Float[ndarray, "n_views 133 2"] = uv_exo_stack[ts_idx]
        uv_left_mediapipe: Float[ndarray, "n_views 21 2"] = uv_frame[:, LEFT_HAND_IDX, :]
        uv_right_mediapipe: Float[ndarray, "n_views 21 2"] = uv_frame[:, RIGHT_HAND_IDX, :]

        if cfg.use_single_hand:
            # Select side
            side = cfg.single_hand_side.lower()
            uv_sel = uv_left_mediapipe if side == "left" else uv_right_mediapipe
            # Keep only camera views where all 2D keypoints are finite; align P accordingly
            views_all_finite: ndarray = np.isfinite(uv_sel).all(axis=(1, 2))
            valid_view_indices: ndarray = np.nonzero(views_all_finite)[0]
            if valid_view_indices.size == 0:
                print(f"Skipping frame {ts_idx} for {side}: no views with all-finite 2D keypoints")
                continue

            uv_use: Float[ndarray, "n_views_valid 21 2"] = uv_sel[valid_view_indices]
            P_use: Float[ndarray, "n_views_valid 3 4"] = Pall_exo[valid_view_indices]

            view_key: tuple[int, ...] = tuple(int(i) for i in valid_view_indices.tolist())
            if view_key not in single_opt_cache:
                single_opt_cache[view_key] = SingleHandOptimization(
                    beta=gt_beta,
                    Pall=P_use,
                    hand_side="left" if side == "left" else "right",
                    loss_weights=LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0),
                )
            single_results, state = single_opt_cache[view_key](uv_use)
            xyz_mano_opt_lr: Float[ndarray, "21 3"] = single_results.xyz_mano
            from simplecv.print_utils import debug_numpy as lo

            print(f"{side.title()} 3D Keypoints", lo(xyz_mano_opt_lr))
            # Simple GT metric (ignore NaNs in GT)
            gt_xyz_lr: Float[ndarray, "21 3"] = xyz_stack[ts_idx, LEFT_HAND_IDX if side == "left" else RIGHT_HAND_IDX]
            valid = ~np.isnan(gt_xyz_lr).any(axis=-1)
            if np.any(valid):
                mae_m = float(np.mean(np.linalg.norm(gt_xyz_lr[valid] - xyz_mano_opt_lr[valid], axis=-1)))
                print(f"Mean |Δ| to GT ({side}) [m]: {mae_m:.4f}")
            # Assert no NaNs in optimized result
            assert np.isfinite(xyz_mano_opt_lr).all(), "NaNs/Infs in optimized 3D keypoints"
            # Log 3D keypoints for the selected side
            rr.log(f"{parent_log_path}/optimized_{side}_kpts", rr.Points3D(xyz_mano_opt_lr))
            # Overlay re-projections
            xyz_hom_lr: Float[ndarray, "1 21 4"] = np.concatenate(
                [xyz_mano_opt_lr[None, ...], np.ones((1, 21, 1), dtype=xyz_mano_opt_lr.dtype)], axis=-1
            )
            uv_proj_lr: Float[ndarray, "1 n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_lr, P=Pall_exo)
            for exo_cam_idx, exo_cam in enumerate(exo_cam_param_list):
                exo_cam_path: Path = parent_log_path / exo_cam.name
                exo_pinhole_path: Path = exo_cam_path / "pinhole"
                rr.log(f"{exo_pinhole_path}/optimized_{side}", rr.Points2D(uv_proj_lr[0, exo_cam_idx]))
            continue  # skip multi-hand branch

        # Multi-hand branch with view alignment, GT checking, and NaN guard
        views_left_ok: ndarray = np.isfinite(uv_left_mediapipe).all(axis=(1, 2))
        views_right_ok: ndarray = np.isfinite(uv_right_mediapipe).all(axis=(1, 2))
        views_ok: ndarray = views_left_ok & views_right_ok
        valid_idx: ndarray = np.nonzero(views_ok)[0]
        if valid_idx.size == 0:
            print("Skipping frame: no views with all-finite 2D keypoints for both hands")
            continue

        uv_left_use: Float[ndarray, "n_views_valid 21 2"] = uv_left_mediapipe[valid_idx]
        uv_right_use: Float[ndarray, "n_views_valid 21 2"] = uv_right_mediapipe[valid_idx]
        P_use: Float[ndarray, "n_views_valid 3 4"] = Pall_exo[valid_idx]

        view_key_mh: tuple[int, ...] = tuple(int(i) for i in valid_idx.tolist())
        if view_key_mh not in multi_opt_cache:
            multi_opt_cache[view_key_mh] = ManoOptimization(
                beta=gt_beta, Pall=P_use, loss_weights=LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0)
            )
        optimization_results, state = multi_opt_cache[view_key_mh](
            uv_left_pred_batch=uv_left_use,
            uv_right_pred_batch=uv_right_use,
        )
        xyz_mano_opt: Float[ndarray, "2 21 3"] = optimization_results.xyz_mano
        from simplecv.print_utils import debug_numpy as lo
        print("Left 3D Keypoints", lo(xyz_mano_opt[0]))
        print("Right 3D Keypoints", lo(xyz_mano_opt[1]))

        # GT metrics and NaN checks
        gt_left: Float[ndarray, "21 3"] = xyz_stack[ts_idx, LEFT_HAND_IDX]
        gt_right: Float[ndarray, "21 3"] = xyz_stack[ts_idx, RIGHT_HAND_IDX]
        valid_l = ~np.isnan(gt_left).any(axis=-1)
        valid_r = ~np.isnan(gt_right).any(axis=-1)
        if np.any(valid_l):
            mae_left = float(np.mean(np.linalg.norm(gt_left[valid_l] - xyz_mano_opt[0][valid_l], axis=-1)))
            print(f"Mean |Δ| to GT (left) [m]: {mae_left:.4f}")
        if np.any(valid_r):
            mae_right = float(np.mean(np.linalg.norm(gt_right[valid_r] - xyz_mano_opt[1][valid_r], axis=-1)))
            print(f"Mean |Δ| to GT (right) [m]: {mae_right:.4f}")
        assert np.isfinite(xyz_mano_opt).all(), "NaNs/Infs in optimized 3D keypoints (both hands)"

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
