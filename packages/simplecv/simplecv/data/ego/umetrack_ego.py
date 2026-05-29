from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde.json import from_json

from simplecv.apis.view_umetrack_data import CAMERA_FILE_NAMES, UmeTrackAnnotation, UmeTrackCameras
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData

if TYPE_CHECKING:
    from simplecv.data.exoego.umetrack import UmeTrackConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as UmeTrackConfig


def create_fisheye_camparams(umtrack_camera_list: list[UmeTrackCameras]) -> list[Fisheye62Parameters]:
    """Instantiate fisheye cameras with camera_parameters intrinsics/extrinsics."""

    cameras: list[Fisheye62Parameters] = []
    for camera_name, umetrack_camera in enumerate(umtrack_camera_list):
        intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(umetrack_camera.fx),
            fl_y=float(umetrack_camera.fy),
            cx=float(umetrack_camera.cx),
            cy=float(umetrack_camera.cy),
            height=int(umetrack_camera.image_size_y),
            width=int(umetrack_camera.image_size_x),
        )
        extrinsics = Extrinsics(
            cam_R_world=np.eye(3, dtype=np.float32),
            cam_t_world=np.zeros(3, dtype=np.float32),
        )
        distortion = KannalaBrandtDistortion(
            k1=float(umetrack_camera.k1),
            k2=float(umetrack_camera.k2),
            k3=float(umetrack_camera.k3),
            k4=float(umetrack_camera.k4),
            k5=float(umetrack_camera.k5),
            k6=float(umetrack_camera.k6),
            p1=float(umetrack_camera.p1),
            p2=float(umetrack_camera.p2),
        )
        camera_params = Fisheye62Parameters(
            name=f"camera_{camera_name}",
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=distortion,
        )
        cameras.append(camera_params)

    return cameras


class UmeTrackEgoSequence(BaseEgoSequence[UmeTrackConfig]):
    """Egocentric view loader for UmeTrack with extrinsics in meters."""

    def __len__(self) -> int:
        assert len(self.ego_video_readers) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        raise NotImplementedError("UmeTrack ego data loading not implemented yet.")

    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        recording_dir: Path = (
            self.config.root_directory
            / self.config.data_type
            / self.config.hand_interaction
            / self.config.split
            / f"user_{self.config.user:02d}"
            / f"recording_{self.config.recording_id:02d}"
        )
        assert recording_dir.exists() and recording_dir.is_dir(), (
            f"Recording directory {recording_dir} does not exist or is not a directory."
        )

        ordered_video_paths: list[Path] = []
        for stem in CAMERA_FILE_NAMES.values():
            video_path: Path = recording_dir / f"{stem}.mp4"
            assert video_path.exists(), f"Expected ego video {video_path} is missing."
            ordered_video_paths.append(video_path)

        return ordered_video_paths

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        ############################################
        # Get Intrinsic Parameters for Ego Cameras #
        ############################################
        annotation_path: Path = (
            self.config.root_directory
            / self.config.data_type
            / self.config.hand_interaction
            / self.config.split
            / f"user_{self.config.user:02d}"
            / f"recording_{self.config.recording_id:02d}"
            / f"recording_{self.config.recording_id:02d}.json"
        )
        assert annotation_path.exists(), f"Annotation file {annotation_path} does not exist."
        annotation: UmeTrackAnnotation = from_json(UmeTrackAnnotation, annotation_path.read_text())
        # these only contain intrinsics and distortion parameters, extrinsics are zeroed out
        fishey_params: list[Fisheye62Parameters] = create_fisheye_camparams(umtrack_camera_list=annotation.cameras)
        # these contain the extrinsics in meters, need to assign them to the fisheye cameras
        world_T_cam_all_mm: Float32[ndarray, "n_frames n_cams 4 4"] = annotation.camera_to_world_transforms
        scale_to_meters: float = 1e-3
        self.world_T_cam_all: Float32[ndarray, "n_frames n_cams 4 4"] = world_T_cam_all_mm.copy()
        self.world_T_cam_all[..., :3, 3] *= np.float32(scale_to_meters)
        final_fisheye_params: dict[str, list[Fisheye62Parameters]] = {k: [] for k in CAMERA_FILE_NAMES}
        prev_cam_T_world: Float32[ndarray, "n_cams 4 4"] = np.tile(
            np.eye(4, dtype=np.float32),
            (len(CAMERA_FILE_NAMES), 1, 1),
        )
        for cam_idx, cam_name in enumerate(CAMERA_FILE_NAMES):
            # intrinsics and distortion are already set and the same for all frames per camera
            cam_params: Fisheye62Parameters = fishey_params[cam_idx]
            updated_fisheye_list: list[Fisheye62Parameters] = []
            for frame_idx in range(self.world_T_cam_all.shape[0]):
                world_T_cam: Float32[ndarray, "4 4"] = self.world_T_cam_all[frame_idx, cam_idx]
                # deal with potential singular matrices by reusing the last valid extrinsics
                try:
                    cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_cam)
                except np.linalg.LinAlgError:
                    cam_T_world = prev_cam_T_world[cam_idx]
                    world_T_cam = np.linalg.inv(cam_T_world)
                    self.world_T_cam_all[frame_idx, cam_idx] = world_T_cam
                else:
                    if np.all(np.isfinite(cam_T_world)):
                        prev_cam_T_world[cam_idx] = cam_T_world
                    else:
                        cam_T_world = prev_cam_T_world[cam_idx]
                        world_T_cam = np.linalg.inv(cam_T_world)
                        self.world_T_cam_all[frame_idx, cam_idx] = world_T_cam

                cam_R_world: Float32[ndarray, "3 3"] = cam_T_world[:3, :3]
                cam_t_world: Float32[ndarray, "3"] = cam_T_world[:3, 3]
                extrinsics: Extrinsics = Extrinsics(
                    cam_R_world=cam_R_world,
                    cam_t_world=cam_t_world,
                )
                new_fisheye_params: Fisheye62Parameters = Fisheye62Parameters(
                    name=cam_params.name,
                    intrinsics=cam_params.intrinsics,
                    distortion=cam_params.distortion,
                    extrinsics=extrinsics,
                )
                updated_fisheye_list.append(new_fisheye_params)
            final_fisheye_params[cam_name] = updated_fisheye_list

        cam_dict: dict[str, list[Fisheye62Parameters]] = final_fisheye_params

        return cam_dict

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[Fisheye62Parameters]]
    ) -> tuple[dict[str, list[Fisheye62Parameters]], dict[str, Path]]:
        """Align cameras and videos based on the sequence."""

        assert video_path_list, "No ego videos provided for alignment."
        assert ego_cam_dict, "No ego cameras available for alignment."
        assert len(video_path_list) == len(CAMERA_FILE_NAMES), (
            f"Expected {len(CAMERA_FILE_NAMES)} ego videos, received {len(video_path_list)}."
        )

        aligned_cam_dict: dict[str, list[Fisheye62Parameters]] = {}
        aligned_video_map: dict[str, Path] = {}

        for idx, (cam_name, expected_stem) in enumerate(CAMERA_FILE_NAMES.items()):
            assert cam_name in ego_cam_dict, f"Camera {cam_name} missing from ego camera dictionary."
            video_path: Path = video_path_list[idx]
            assert video_path.stem == expected_stem, (
                f"Video at index {idx} stem '{video_path.stem}' does not match expected '{expected_stem}'."
            )

            aligned_cam_dict[cam_name] = ego_cam_dict[cam_name]
            aligned_video_map[cam_name] = video_path

        return aligned_cam_dict, aligned_video_map

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.025
