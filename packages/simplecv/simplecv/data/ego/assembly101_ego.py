import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde import InternalTagging, serde
from serde import field as serde_field
from serde.json import from_json
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, EgoData

if TYPE_CHECKING:
    from simplecv.data.exoego.assembly101 import Assembly101Config
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as Assembly101Config
CameraNames = Literal["e1", "e2", "e3", "e4"]
SerialNo = Literal[
    "21176875",  # e1
    "21176623",  # e2
    "21110305",  # e3
    "21179183",  # e4
    "84346135",  # e1
    "84347414",  # e2
    "84355350",  # e3
    "84358933",  # e4
]

SERIAL_TO_ALIAS: Mapping[SerialNo, CameraNames] = {
    # --- e1 -----------------------------------------------------
    "21176875": "e1",
    "84346135": "e1",
    # --- e2 -----------------------------------------------------
    "21176623": "e2",
    "84347414": "e2",
    # --- e3 -----------------------------------------------------
    "21110305": "e3",
    "84355350": "e3",
    # --- e4 -----------------------------------------------------
    "21179183": "e4",
    "84358933": "e4",
}


@serde
class EgoExtri211:
    e1: Float32[ndarray, "4 4"] = serde_field(rename="21176875:mono10bit")
    e2: Float32[ndarray, "4 4"] = serde_field(rename="21176623:mono10bit")
    e3: Float32[ndarray, "4 4"] = serde_field(rename="21110305:mono10bit")
    e4: Float32[ndarray, "4 4"] = serde_field(rename="21179183:mono10bit")


# ----- cameras that come in as HMC_843… -------------------------------
@serde
class EgoExtri843:
    e1: Float32[ndarray, "4 4"] = serde_field(rename="84346135:mono10bit")
    e2: Float32[ndarray, "4 4"] = serde_field(rename="84347414:mono10bit")
    e3: Float32[ndarray, "4 4"] = serde_field(rename="84355350:mono10bit")
    e4: Float32[ndarray, "4 4"] = serde_field(rename="84358933:mono10bit")


@serde
class OpenCV:
    ImageSizeX: int
    ImageSizeY: int
    ModelViewMatrix: Float32[ndarray, "4 4"]
    SerialNo: str
    cx: float
    cy: float
    fx: float
    fy: float
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float
    p3: float
    p4: float


@serde
class OVFishEye62:
    ImageSizeX: int
    ImageSizeY: int
    ModelViewMatrix: Float32[ndarray, "4 4"]
    SerialNo: str
    cx: float
    cy: float
    fx: float
    fy: float
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float
    p3: float
    p4: float


@serde(tagging=InternalTagging("DistortionModel"))
class CameraRecord:
    Camera: OpenCV | OVFishEye62
    MountingLocation: str
    Type: str
    ParentName: str | None = None


def pick_schema(extrinsics_ego: dict[str, Any]) -> type[EgoExtri843] | type[EgoExtri211]:
    first_ego_extri: dict[str, Any] = next(iter(extrinsics_ego.values()))
    first_key: str = next(iter(first_ego_extri.keys()))
    schema: type[EgoExtri843] | type[EgoExtri211] = EgoExtri843 if first_key.startswith("8") else EgoExtri211
    return schema


class Assembly101EgoSequence(BaseEgoSequence[Assembly101Config]):
    """Egocentric view loader for Assembly101 with extrinsics in meters."""

    def __len__(self) -> int:
        assert len(self.ego_video_readers) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        # bgr_list: list[UInt8[ndarray, "H W 3"]] = self.ego_video_readers[idx]
        return EgoData(
            # cam_params_list=self._ego_cam_dict,
            # bgr_list=bgr_list,
        )

    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        video_dir: Path = self.config.root_directory / "videos" / "av1-720-new" / self.config.sequence_name
        assert video_dir.exists(), f"Directory {video_dir} does not exist"
        ego_video_files: list[Path] = sorted(
            [file for file in video_dir.iterdir() if file.is_file() and file.name.startswith("HMC")]
        )

        return ego_video_files

    def load_ego_cams(self) -> dict[str, list[CameraParam]]:
        ############################################
        # Get Intrinsic Parameters for Ego Cameras #
        ############################################
        intrinsics_ego_dir: Path = self.config.root_directory / "assemblyhands-toolkit" / "calib" / "nimble_json_calib"
        intri_jsons: list[Path] = sorted(intrinsics_ego_dir.glob("*.json"))

        # the original assembly101 dataset does not provide camera intrinsics in the fisheye 62 format, so we load them from
        # assemblyhands-toolkit and only use the last record in each file assuming distorion parameters are the same for all cameras in a sequence
        intri_json_path: Path = intri_jsons[0]
        assert intri_json_path.exists(), f"File {intri_json_path} does not exist"

        intri_text: str = intri_json_path.read_text()
        records: list[CameraRecord] = from_json(list[CameraRecord], intri_text)
        assert records, f"No camera records found in {intri_json_path}"
        record: CameraRecord = records[-1]
        camera_record: OpenCV | OVFishEye62 = record.Camera
        if not isinstance(camera_record, OVFishEye62):
            raise ValueError(f"Expected OVFishEye62 camera record, but received {type(camera_record).__name__}")
        fisheye62: OVFishEye62 = camera_record

        distortion: KannalaBrandtDistortion = KannalaBrandtDistortion(
            k1=fisheye62.k1,
            k2=fisheye62.k2,
            k3=fisheye62.k3,
            k4=fisheye62.k4,
            k5=fisheye62.k5,
            k6=fisheye62.k6,
            p1=fisheye62.p1,
            p2=fisheye62.p2,
        )
        intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(fisheye62.fx),
            fl_y=float(fisheye62.fy),
            cx=float(fisheye62.cx),
            cy=float(fisheye62.cy),
            height=fisheye62.ImageSizeY,
            width=fisheye62.ImageSizeX,
        )

        ############################################
        # Get Extrinsic Parameters for Ego Cameras #
        ############################################
        extrinsics_ego_path: Path = (
            self.config.root_directory
            / "assembly101_camera_and_hand_poses"
            / "camera_extrinsics_ego"
            / f"{self.config.sequence_name}.json"
        )
        assert extrinsics_ego_path.exists(), f"File {extrinsics_ego_path} does not exist"
        with open(extrinsics_ego_path) as f:
            extrinsics_ego: dict[str, Any] = json.load(f)

        schema: type[EgoExtri843] | type[EgoExtri211] = pick_schema(extrinsics_ego)
        extrinsics_text: str = extrinsics_ego_path.read_text()
        if schema is EgoExtri843:
            raw_ego_cameras: dict[str, EgoExtri843] = from_json(dict[str, EgoExtri843], extrinsics_text)
            ego_extri_cameras: dict[str, EgoExtri843 | EgoExtri211] = {
                key: raw_ego_cameras[key] for key in sorted(raw_ego_cameras, key=int)
            }
        else:
            raw_ego_cameras_211: dict[str, EgoExtri211] = from_json(dict[str, EgoExtri211], extrinsics_text)
            ego_extri_cameras = {key: raw_ego_cameras_211[key] for key in sorted(raw_ego_cameras_211, key=int)}

        # create list of Fisheye62Parameters for ego cameras
        ego_fisheye_dict: dict[str, list[Fisheye62Parameters]] = {
            "e1": [],
            "e2": [],
            "e3": [],
            "e4": [],
        }
        ego_cam: EgoExtri211 | EgoExtri843
        for _, ego_cam in tqdm(
            ego_extri_cameras.items(),
            desc="Processing ego cameras",
            disable=not self.config.verbose,
            leave=False,
            position=1,
        ):
            for key in ego_fisheye_dict:
                cam_T_world = getattr(ego_cam, key)
                extri: Extrinsics = Extrinsics(
                    world_R_cam=cam_T_world[:3, :3],
                    world_t_cam=cam_T_world[:3, 3] * np.float32(1e-3),
                )
                ego_fisheye_dict[key].append(
                    Fisheye62Parameters(
                        name=f"{key}",
                        intrinsics=intrinsics,
                        extrinsics=extri,
                        distortion=distortion,
                    )
                )

        return cast(dict[str, list[CameraParam]], ego_fisheye_dict)

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[CameraNames, list[Fisheye62Parameters]]
    ) -> tuple[dict[CameraNames, list[Fisheye62Parameters]], dict[CameraNames, Path]]:
        """Align cameras and videos based on the sequence."""
        assert len(video_path_list) == len(ego_cam_dict), (
            f"Number of videos {len(video_path_list)} does not match number of cameras {len(ego_cam_dict)}"
        )

        video_to_cam_map: dict[CameraNames, Path] = {}
        for video_path in video_path_list:
            serial_num: str = video_path.stem.split("_")[1]  # Extract the serial number from the video file name
            # make sure the serial number is in the expected format
            assert serial_num in SERIAL_TO_ALIAS, f"Serial number {serial_num} not found in SERIAL_TO_ALIAS"
            cam_name: CameraNames = SERIAL_TO_ALIAS[serial_num]
            video_to_cam_map[cam_name] = video_path
        # make sure that the video_to_cam_map has the same keys as ego_cam_dict
        assert set(video_to_cam_map.keys()) == set(ego_cam_dict.keys()), (
            f"Keys of video_to_cam_map {set(video_to_cam_map.keys())} do not match keys of ego_cam_dict {set(ego_cam_dict.keys())}"
        )
        # Sort the video_to_cam_map by camera name and do the same for ego_cam_dict
        ego_cam_dict = dict(sorted(ego_cam_dict.items()))
        video_to_cam_map = dict(sorted(video_to_cam_map.items()))

        return ego_cam_dict, video_to_cam_map

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
