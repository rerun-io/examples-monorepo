from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, CamNameType, EgoData
from simplecv.data.exoego.epfl_smart_kitchen import (
    EPFL_EGO_CAMERA_NAME,
    HoloLensPoses,
    _camera_size,
    _distortion_from_entry,
    _matrix3,
    load_camera_matrix,
    load_hololens_world_to_camera_poses,
    video_path_for_camera,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig
else:
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as EpflSmartKitchenConfig


class EpflSmartKitchenEgoSequence(BaseEgoSequence[EpflSmartKitchenConfig]):
    """HoloLens ego stream for EPFL-Smart-Kitchen."""

    def load_video_paths(self) -> list[Path]:
        video_path: Path = video_path_for_camera(self.config, EPFL_EGO_CAMERA_NAME)
        if not video_path.exists():
            raise FileNotFoundError(f"EPFL HoloLens RGB video not found: {video_path}")
        return [video_path]

    def load_ego_cams(self) -> dict[str, list[CameraParam]]:
        camera_matrix: dict[str, dict] = load_camera_matrix(self.config)
        if EPFL_EGO_CAMERA_NAME not in camera_matrix:
            raise KeyError(f"EPFL camera matrix is missing {EPFL_EGO_CAMERA_NAME!r}")
        entry: dict = camera_matrix[EPFL_EGO_CAMERA_NAME]
        video_path: Path = video_path_for_camera(self.config, EPFL_EGO_CAMERA_NAME)
        image_size: tuple[int, int] = _camera_size(entry, video_path)
        width: int = image_size[0]
        height: int = image_size[1]
        k_matrix: Float32[ndarray, "3 3"] = _matrix3(entry["K"], field_name=f"{EPFL_EGO_CAMERA_NAME}.K")
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=k_matrix,
            width=width,
            height=height,
        )
        distortion: BrownConradyDistortion | None = _distortion_from_entry(EPFL_EGO_CAMERA_NAME, entry)
        holo_poses: HoloLensPoses = load_hololens_world_to_camera_poses(self.config)
        cam_T_world_list: list[Float32[ndarray, "4 4"]] = holo_poses.cam_T_world_list
        pose_valid_mask: Bool[ndarray, "n_frames"] = holo_poses.valid_mask
        # Where the HoloLens lost tracking (blank world2holo) the loader carried the last
        # good pose forward as a placeholder. Rendering that stale pose is exactly the bug:
        # it freezes the ego frustum and paints 2D keypoints through a dead camera. Hand
        # those frames NaN extrinsics instead -- Rerun draws no frustum for a NaN transform,
        # and the keypoint projection comes out NaN (so no 2D dots), while the black
        # "tracking died here" video is left untouched. Genuine poses are unchanged, so
        # clean sessions (and every non-EPFL dataset) render byte-for-byte as before.
        nan_cam_R_world: Float32[ndarray, "3 3"] = np.full((3, 3), np.nan, dtype=np.float32)
        nan_cam_t_world: Float32[ndarray, "3"] = np.full(3, np.nan, dtype=np.float32)
        camera_list: list[CameraParam] = []
        for frame_idx, cam_T_world in enumerate(cam_T_world_list):
            extrinsics: Extrinsics
            if bool(pose_valid_mask[frame_idx]):
                extrinsics = Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                )
            else:
                extrinsics = Extrinsics(
                    cam_R_world=nan_cam_R_world.copy(),
                    cam_t_world=nan_cam_t_world.copy(),
                )
            camera_params: PinholeParameters = PinholeParameters(
                name=EPFL_EGO_CAMERA_NAME,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                distortion=distortion,
            )
            camera_list.append(camera_params)
        return {EPFL_EGO_CAMERA_NAME: camera_list}

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[CamNameType, list[CameraParam]],
    ) -> tuple[dict[CamNameType, list[CameraParam]], dict[CamNameType, Path]]:
        if len(video_path_list) != 1:
            raise ValueError(f"Expected one EPFL HoloLens video, got {len(video_path_list)}")
        if set(ego_cam_dict) != {EPFL_EGO_CAMERA_NAME}:
            raise ValueError(f"Expected one EPFL HoloLens camera, got {set(ego_cam_dict)}")
        camera_name: CamNameType = next(iter(ego_cam_dict))
        return ego_cam_dict, {camera_name: video_path_list[0]}

    def __getitem__(self, idx: int) -> EgoData:
        bgr_list = self.ego_video_readers[idx]
        camera_list: list[CameraParam] = self.ego_cam_dict[EPFL_EGO_CAMERA_NAME]
        camera_idx: int = min(idx, len(camera_list) - 1)
        if idx >= len(camera_list) and not getattr(self, "_warned_pose_clamp", False):
            warnings.warn(
                f"EPFL HoloLens camera poses have {len(camera_list)} entries but frame {idx} was requested; "
                "reusing the final pose for trailing video frames.",
                stacklevel=2,
            )
            self._warned_pose_clamp = True
        return EgoData(cam_params_list=[camera_list[camera_idx]], bgr_list=bgr_list)

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return ViewCoordinates.RIGHT_HAND_Z_UP
