"""Shared Rerun visualization helpers for cuVSLAM tracking."""

from pathlib import Path

import cuvslam
import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.ops.conventions import CC, convert_pose
from simplecv.rerun_log_utils import log_pinhole, log_video

from robocap_slam.data.base import BaseTrackDataset

# Robocap rig mesh alignment: rotation (Euler XYZ degrees) and translation offset
ROBOCAP_MESH_EULER_XYZ_DEG = [-90, 0, -80]
ROBOCAP_MESH_TRANSLATION = [0.0, -0.15, 0.025]


def color_from_id(identifier: int) -> list[int]:
    """Generate pseudo-random colour from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


def compute_orient_transform(
    world_from_rig_matrices: list[Float64[ndarray, "4 4"]],
    rig_from_cam: Float64[ndarray, "4 4"],
    center: bool = True,
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"]] | None:
    """Compute gravity-alignment transform from collected poses.

    Args:
        world_from_rig_matrices: List of 4x4 world_from_rig matrices collected during tracking.
        rig_from_cam: 4x4 rig_from_camera extrinsics for the first camera.
        center: Whether to center poses around mean position.

    Returns:
        (R, t) rotation and translation for the orient transform, or None if < 10 poses.
    """
    if len(world_from_rig_matrices) < 10:
        return None

    # world_T_cam = world_from_rig @ rig_from_cam (CV convention)
    poses_cv = np.stack(world_from_rig_matrices) @ rig_from_cam

    # Convert CV -> GL convention
    poses_gl = convert_pose(poses_cv, CC.CV, CC.GL)

    center_method = "poses" if center else "none"
    result = auto_orient_and_center_poses(poses_gl, method="up", center_method=center_method)
    transform_34 = result.transform  # (3, 4)

    R = transform_34[:3, :3]
    t = transform_34[:3, 3]
    return R, t


def log_orient_transform(R: Float64[ndarray, "3 3"], t: Float64[ndarray, "3"]) -> None:
    """Log the gravity-alignment transform as a static Transform3D on 'world'.

    Also updates root ViewCoordinates to RFU (Z-up) to match auto_orient's output,
    which aligns camera-up with +Z.
    """
    rr.log("world", rr.Transform3D(mat3x3=R, translation=t), static=True)
    rr.log("/", rr.ViewCoordinates.RFU, static=True)


def log_static_cameras_and_videos(
    dataset: BaseTrackDataset,
    timeline: str = "video_time",
) -> None:
    """Log pinholes, extrinsics, and video assets (all static, called once).

    Args:
        dataset: Track dataset providing camera params and video paths.
        timeline: Timeline name for video frame timestamps.
    """
    for i, cam_name in enumerate(dataset.cam_names):
        cam_log_path: Path = Path(f"world/rig/cam{i}")

        # Log pinhole intrinsics + extrinsics (static)
        log_pinhole(
            dataset.cam_params[cam_name],
            cam_log_path,
            image_plane_distance=dataset.image_plane_distance,
            static=True,
        )

        # Log video asset + frame references (static)
        if cam_name in dataset.video_paths:
            log_video(
                dataset.video_paths[cam_name],
                cam_log_path / "pinhole" / "video",
                timeline=timeline,
            )


def log_frame_visuals(
    n_cameras: int,
    observations: list,
    landmarks: list,
    odom_pose: cuvslam.Pose,
) -> None:
    """Log per-frame rig pose, landmarks, and 2D observations.

    Args:
        n_cameras: Number of cameras in the rig.
        observations: Per-camera observation lists from tracker.
        landmarks: Landmark list from tracker.
        odom_pose: Current odometry pose estimate.
    """
    rr.log(
        "world/rig",
        rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation),
    )

    if landmarks:
        lm_xyz = [lm.coords for lm in landmarks]
        lm_colors = [color_from_id(lm.id) for lm in landmarks]
        rr.log("world/rig/landmarks", rr.Points3D(lm_xyz, radii=0.02, colors=lm_colors))

    for i in range(n_cameras):
        obs_uv = [[o.u, o.v] for o in observations[i]]
        obs_colors = [color_from_id(o.id) for o in observations[i]]
        rr.log(
            f"world/rig/cam{i}/pinhole/observations",
            rr.Points2D(obs_uv, radii=5, colors=obs_colors),
        )


def log_rig_mesh(mesh_path: Path | None) -> None:
    """Log a GLB mesh asset under the rig entity so it follows the rig pose.

    Args:
        mesh_path: Path to the GLB mesh file, or None to skip.
    """
    if mesh_path is None:
        return

    # Static transform to align mesh with rig coordinate frame
    R = Rotation.from_euler("xyz", ROBOCAP_MESH_EULER_XYZ_DEG, degrees=True).as_matrix()
    rr.log("world/rig/mesh", rr.Transform3D(mat3x3=R, translation=ROBOCAP_MESH_TRANSLATION), static=True)
    rr.log("world/rig/mesh", rr.Asset3D(path=mesh_path), static=True)


def log_final_landmarks(tracker: cuvslam.Tracker) -> None:
    """Log final accumulated landmarks after tracking completes."""
    final_landmarks = tracker.get_final_landmarks()
    if final_landmarks:
        rr.log(
            "world/final_landmarks",
            rr.Points3D(list(final_landmarks.values()), radii=0.01),
        )
