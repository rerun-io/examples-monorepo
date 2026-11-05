"""The exocentric camera rig of an HO-Cap/ExoEgo sequence, as the multiview body demo reads it.

Cameras are static, so the rig is read once from the first sample: the cameras with intrinsics, their
camera-to-world transforms and the world-to-image projection matrices the multiview optimizer fits against.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample


@dataclass(frozen=True, slots=True)
class ExoRig:
    """The exo cameras usable for multiview fitting, in sequence camera order."""

    cam_indices: list[int]
    """Index of each usable camera in the sample's ``exo_cam_params_list``."""
    cam_names: list[str]
    """Name of each usable camera."""
    world_T_cams: list[Float32[ndarray, "4 4"]]
    """Camera-to-world transform of each usable camera."""
    projection_matrices: Float32[ndarray, "n_views 3 4"]
    """World-to-image projection ``K @ [R|t]`` of each usable camera."""

    @property
    def num_views(self) -> int:
        """Number of usable cameras."""
        return len(self.cam_indices)


def exo_camera_names(sequence: BaseExoEgoSequence) -> list[str]:
    """Names of the sequence's exo cameras; empty when it has no exo sequence."""
    return [cam.name for cam in sequence.exo_sequence.exo_cam_list if cam is not None] if sequence.exo_sequence else []


def exo_video_sources(sequence: BaseExoEgoSequence) -> list[tuple[str, bytes | Path]]:
    """Each exo stream's name with its video: the in-memory blob of an RRD sequence, or the file path (HoCap)."""
    if sequence.exo_sequence is None:
        return []
    exo_video_names: list[str] = sequence.exo_sequence.exo_video_names
    exo_video_paths: list[Path] = sequence.exo_sequence.exo_video_paths
    # Get video blobs if available (RRD sequences), otherwise use file paths (HoCap)
    exo_video_blobs: dict[str, bytes] | None = getattr(sequence.exo_sequence, "_video_blobs", None)
    return [
        (stream_name, exo_video_blobs[stream_name] if exo_video_blobs and stream_name in exo_video_blobs else video_file)
        for stream_name, video_file in zip(exo_video_names, exo_video_paths, strict=True)
    ]


def build_projection_matrix(
    K_33: Float32[ndarray, "3 3"],
    world_T_cam: Float32[ndarray, "4 4"],
) -> Float32[ndarray, "3 4"]:
    """Build projection matrix P = K @ [R|t] from intrinsics and extrinsics.

    Args:
        K_33: Camera intrinsics (3x3).
        world_T_cam: Transformation from camera to world coordinates.

    Returns:
        3x4 projection matrix for projecting world points to image coordinates.
    """
    # cam_T_world is the inverse of world_T_cam
    cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_cam).astype(np.float32)
    # Extract [R|t] (3x4)
    Rt: Float32[ndarray, "3 4"] = cam_T_world[:3, :]
    # P = K @ [R|t]
    P: Float32[ndarray, "3 4"] = K_33 @ Rt
    return P


def load_exo_rig(first_sample: ExoEgoSample) -> ExoRig:
    """The cameras of the first sample that have intrinsics, with their projection matrices."""
    projection_matrices_list: list[Float32[ndarray, "3 4"]] = []
    valid_cam_indices: list[int] = []
    valid_cam_names: list[str] = []
    world_T_cam_list: list[Float32[ndarray, "4 4"]] = []

    if first_sample.exo_cam_params_list is not None:
        for cam_idx, cam_params in enumerate(first_sample.exo_cam_params_list):
            if cam_params is None:
                continue
            k_raw = cam_params.intrinsics.k_matrix
            if k_raw is None:
                continue
            K_33: Float32[ndarray, "3 3"] = k_raw.astype(np.float32)
            world_T_cam: Float32[ndarray, "4 4"] = cam_params.extrinsics.world_T_cam.astype(np.float32)
            P: Float32[ndarray, "3 4"] = build_projection_matrix(K_33, world_T_cam)
            projection_matrices_list.append(P)
            valid_cam_indices.append(cam_idx)
            valid_cam_names.append(cam_params.name)
            world_T_cam_list.append(world_T_cam)

    projection_matrices: Float32[ndarray, "n_views 3 4"] = (
        np.stack(projection_matrices_list, axis=0) if projection_matrices_list else np.zeros((0, 3, 4), dtype=np.float32)
    )
    return ExoRig(valid_cam_indices, valid_cam_names, world_T_cam_list, projection_matrices)
