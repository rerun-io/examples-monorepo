import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from serde import field as serde_field
from serde import from_dict, serde
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.video_io import VideoReader
from simplecv.video_utils import RESOLUTION_MAP, reencode_video_optimal

if TYPE_CHECKING:
    from simplecv.data.exoego.assembly101 import Assembly101Config
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as Assembly101Config


@serde
class ExoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C10404: Float32[ndarray, "4 4"] = serde_field(rename="C10404:rgb")
    C10118: Float32[ndarray, "4 4"] = serde_field(rename="C10118:rgb")
    C10119: Float32[ndarray, "4 4"] = serde_field(rename="C10119:rgb")
    C10095: Float32[ndarray, "4 4"] = serde_field(rename="C10095:rgb")
    C10379: Float32[ndarray, "4 4"] = serde_field(rename="C10379:rgb")
    C10395: Float32[ndarray, "4 4"] = serde_field(rename="C10395:rgb")
    C10115: Float32[ndarray, "4 4"] = serde_field(rename="C10115:rgb")
    C10390: Float32[ndarray, "4 4"] = serde_field(rename="C10390:rgb")


@serde
class EgoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C21176875: Float32[ndarray, "4 4"] = serde_field(rename="21176875:mono10bit")
    C21179183: Float32[ndarray, "4 4"] = serde_field(rename="21179183:mono10bit")
    C21110305: Float32[ndarray, "4 4"] = serde_field(rename="21110305:mono10bit")
    C21176623: Float32[ndarray, "4 4"] = serde_field(rename="21176623:mono10bit")


@serde
class Hand3DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 3"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = serde_field(rename="1")


@serde
class Hand2DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 2"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 2"] = serde_field(rename="1")


@serde
class Exo2DKeypoints:
    C10395: Hand2DKeypoints = serde_field(rename="C10395:rgb")
    C10379: Hand2DKeypoints = serde_field(rename="C10379:rgb")
    C10404: Hand2DKeypoints = serde_field(rename="C10404:rgb")
    C10119: Hand2DKeypoints = serde_field(rename="C10119:rgb")
    C10115: Hand2DKeypoints = serde_field(rename="C10115:rgb")
    C10390: Hand2DKeypoints = serde_field(rename="C10390:rgb")
    C10118: Hand2DKeypoints = serde_field(rename="C10118:rgb")
    C10095: Hand2DKeypoints = serde_field(rename="C10095:rgb")


@serde
class Ego2DKeypoints:
    C21176875: Hand2DKeypoints = serde_field(rename="21176875:mono10bit")
    C21179183: Hand2DKeypoints = serde_field(rename="21179183:mono10bit")
    C21110305: Hand2DKeypoints = serde_field(rename="21110305:mono10bit")
    C21176623: Hand2DKeypoints = serde_field(rename="21176623:mono10bit")


class Assembly101ExoSequence(BaseExoSequence[Assembly101Config]):
    """Assembly101 exocentric view loader with meters-based extrinsics and labels."""

    def __len__(self) -> int:
        assert len(self._video_path_list) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.exo_video_readers)

    def __getitem__(self, idx: int) -> None:
        # bgr_list: list[UInt8[ndarray, "H W 3"]] = self.exo_video_readers[idx]
        # if self.config.load_labels:
        #     xyz: Float32[ndarray, "2 21 3"] = self.exo_batch_data.xyz_stack[idx]
        #     uv_dict: dict[str, Float32[ndarray, "2 21 2"]] = {
        #         cam_name: uv_stack[idx] for cam_name, uv_stack in self.exo_batch_data.uv_stack_dict.items()
        #     }
        # else:
        #     xyz = None
        #     uv_dict = None
        # return ExoData(
        #     cam_params_list=self.exo_cam_list,
        #     bgr_list=bgr_list,
        #     xyz=xyz,
        #     uv_dict=uv_dict,
        # )
        return None

    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        video_dir: Path = self.config.root_directory / "videos" / "av1-720-new" / self.config.sequence_name
        assert video_dir.exists(), f"Directory {video_dir} does not exist"
        exo_video_files: list[Path] = sorted(
            [file for file in video_dir.iterdir() if file.is_file() and not file.name.startswith("HMC")]
        )
        if self.config.resize is not None:
            exo_video_files: list[Path] = [
                reencode_video_optimal(p, resize=self.config.resize)
                for p in tqdm(
                    exo_video_files,
                    desc="Re-encoding videos",
                    disable=not self.config.verbose,
                    leave=False,
                    position=1,
                )
            ]

        return exo_video_files

    def load_exo_cams(self) -> list[PinholeParameters]:
        extrinsics_exo_path: Path = (
            self.config.root_directory
            / "assembly101_camera_and_hand_poses"
            / "camera_extrinsics_fixed"
            / f"{self.config.sequence_name}.json"
        )
        assert extrinsics_exo_path.exists(), f"File {extrinsics_exo_path} does not exist"
        assembly_hands_annotation_path: Path = self.config.root_directory / "assembly-hands"
        train_assembly_hands_json: Path = (
            assembly_hands_annotation_path / "annotations" / "train" / "assemblyhands_train_exo_calib_v1-1.json"
        )
        assert train_assembly_hands_json.exists(), f"File {train_assembly_hands_json} does not exist"
        with open(extrinsics_exo_path) as f:
            extrinsics_fixed = json.load(f)

        # sort extrinsics_fixed by camera name
        extrinsics_fixed = dict(sorted(extrinsics_fixed.items()))
        exo_raw_extri: ExoExtriCameras = from_dict(ExoExtriCameras, extrinsics_fixed)
        # assembly101 does not have camera intrinsics, so need to get them from assemblyhands
        with open(train_assembly_hands_json) as f:
            train_assembly_hands_dict: dict = json.load(f)

        all_calib_dict: dict = train_assembly_hands_dict["calibration"]
        # assume that all cameras have the same intrinsics for each capture, so only get a single one
        instrinsics_dict: dict[str, list[list[float]]] = next(iter(all_calib_dict.values()))["intrinsics"]
        # Original resolution of the videos in assembly101
        height: int = 1080
        width: int = 1920

        # get the current resolution from video files
        video_reader = VideoReader(filename=self._video_path_list[0])
        current_w, current_h = video_reader.resolution

        pinhole_list: list[PinholeParameters] = []

        cam_name: str
        exo_camera: Float32[ndarray, "4 4"]
        for cam_name, exo_camera in asdict(exo_raw_extri).items():
            intri_np: Float32[ndarray, "3 3"] = np.array(instrinsics_dict[f"{cam_name}_rgb"], dtype=np.float32)
            intri: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=float(intri_np[0, 0]),
                fl_y=float(intri_np[1, 1]),
                cx=float(intri_np[0, 2]),
                cy=float(intri_np[1, 2]),
                height=height,
                width=width,
            )
            # resize intri if resizing is enabled
            if self.config.resize is not None:
                new_w, new_h = RESOLUTION_MAP[self.config.resize]
                intri = Intrinsics(
                    camera_conventions=intri.camera_conventions,
                    fl_x=float(intri.fl_x * new_w / width),
                    fl_y=float(intri.fl_y * new_h / height),
                    cx=float(intri.cx * new_w / width),
                    cy=float(intri.cy * new_h / height),
                    height=new_h,
                    width=new_w,
                )
            if current_w != width or current_h != height:
                # adjust intrinsics to the current resolution of the video
                intri = Intrinsics(
                    camera_conventions=intri.camera_conventions,
                    fl_x=float(intri.fl_x * current_w / width),
                    fl_y=float(intri.fl_y * current_h / height),
                    cx=float(intri.cx * current_w / width),
                    cy=float(intri.cy * current_h / height),
                    height=current_h,
                    width=current_w,
                )

            extri = Extrinsics(
                world_R_cam=exo_camera[:3, :3],
                world_t_cam=exo_camera[:3, 3] * np.float32(1e-3),
            )
            pinhole_param = PinholeParameters(
                name=cam_name,
                intrinsics=intri,
                extrinsics=extri,
            )
            pinhole_list.append(pinhole_param)

        pinhole_list = list(sorted(pinhole_list, key=lambda x: x.name))
        return pinhole_list

    @property
    def depth_paths(self) -> None:
        """Get mapping from joint ID to joint name."""
        return None

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.1
