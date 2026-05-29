from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import rerun as rr
from jaxtyping import Float, Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from scipy.spatial.transform import Rotation as R
from serde import serde
from serde.yaml import from_yaml

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.video_utils import create_temp_video_from_img_dir

if TYPE_CHECKING:
    from simplecv.data.exoego.hocap import HocapConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as HocapConfig

# External (exo) cameras are identified by numerical IDs
ExoCameraIDs = Literal[
    "037522251142",
    "043422252387",
    "046122250168",
    "105322251225",
    "105322251564",
    "108222250342",
    "115422250549",
    "117222250549",
]

# Egocentric cameras (HoloLens)
EgoCameraIDs = Literal["hololens_kv5h72"]

# Other cameras (tags, etc.)
OtherCameraIDs = Literal["tag_0", "tag_1"]
# Combined camera IDs using type union
CameraIDs = ExoCameraIDs | EgoCameraIDs | OtherCameraIDs
SubjectIDs = Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]


@serde
class HOCapIntrinsics:
    """Common parameters for both color and depth cameras"""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: Float[ndarray, "..."] | None = None
    k_33: Float[ndarray, "3 3"] = field(default_factory=lambda: np.eye(3, dtype=np.float32))

    def __post_init__(self) -> None:
        # fmt: off
        self.k_33 = np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ])
        # fmt: on


@serde
class HOCapExtrinsicsData:
    """Camera extrinsics calibration data"""

    extrinsics: dict[CameraIDs, Float32[ndarray, "..."]]
    rs_master: str
    world_T_cam_dict: dict[CameraIDs, Float32[ndarray, "4 4"]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # convert extrinsics to 3x4 matrices
        self.extrinsics: dict[CameraIDs, Float32[ndarray, "3 4"]] = {
            cam_id: extri.reshape(3, 4) for cam_id, extri in self.extrinsics.items()
        }
        # convert extrinsics to 4x4 matrices
        self.extrinsics: dict[CameraIDs, Float32[ndarray, "4 4"]] = {
            cam_id: np.vstack((extri, np.array([0, 0, 0, 1], dtype=np.float32)))
            for cam_id, extri in self.extrinsics.items()
        }
        # qrtag extrinsics
        _tag_1: Float32[ndarray, "4 4"] = self.extrinsics["tag_1"]
        _tag_1_inv = np.linalg.inv(_tag_1)

        # calculate world_T_cam from master realsense device -> world coordinates
        self.world_T_cam_dict: dict[CameraIDs, Float32[ndarray, "4 4"]] = {}
        for cam_id, _extr2master in self.extrinsics.items():
            if cam_id not in ["tag_0", "tag_1"]:  # skip tags
                self.world_T_cam_dict[cam_id] = _tag_1_inv @ _extr2master


@serde
class HOCapIntrinsicsData:
    """Camera intrinsics calibration data"""

    serial: CameraIDs
    color: HOCapIntrinsics
    depth: HOCapIntrinsics | None = None
    depth2color: Float[ndarray, "..."] | None = None


def quat_to_mat(quat: Float[ndarray, "batch 7"]) -> Float[ndarray, "batch 4 4"]:
    """Convert quaternion to rotation matrix."""
    # Placeholder for quaternion to rotation matrix conversion
    # Extract quaternion (q) and translation (t)
    q: Float[ndarray, "batch 4"] = quat[..., :4]  # Quaternion (4 elements)
    t: Float[ndarray, "batch 3"] = quat[..., 4:]  # Translation (3 elements)

    # Convert quaternion to rotation matrix and fill in the pose matrix
    r = R.from_quat(q)

    N = quat.shape[0]
    p = np.tile(np.eye(4), (N, 1, 1))  # Create N identity matrices

    p[..., :3, :3] = r.as_matrix()  # Fill rotation part
    p[..., :3, 3] = t  # Fill translation part
    return p


class HocapEgoSequence(BaseEgoSequence[HocapConfig]):

    def load_video_paths(self) -> list[Path]:
        sequence_path: Path = (
            self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        )
        assert sequence_path.exists(), f"Path {sequence_path} does not exist."

        # Load video path s for the ego camera (HoloLens)
        img_dir: Path = sequence_path / get_args(EgoCameraIDs)[0]  # Assuming the first camera is the HoloLens
        assert img_dir.exists(), f"Path {img_dir} does not exist."
        video_path: Path = img_dir / "output.mp4"
        if not video_path.exists():
            video_path: Path = create_temp_video_from_img_dir(
                img_dir,
                fps=30,
                quality="low",
                image_extension="jpg",
                save_file=False,
            )

        return [video_path]

    def load_ego_cams(self) -> dict[str, list[PinholeParameters]]:
        calibration_path: Path = self.config.root_directory / "calibration"
        intrinsics_path: Path = calibration_path / "intrinsics"

        sequence_path: Path = (
            self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        )

        assert intrinsics_path.exists(), f"Path {intrinsics_path} does not exist."
        # TODO don't use self here after getting final iterable
        assert sequence_path.exists(), f"Path {sequence_path} does not exist."

        extrinsics_npy_path: Path = sequence_path / "poses_pv.npy"
        # first 4 elements are the rotation (quaternion) and translation are the last 3 elements
        extri_quat: Float[ndarray, "num_frames 7"] = np.load(extrinsics_npy_path)
        world_T_cam_batch: Float[ndarray, "num_frames 4 4"] = quat_to_mat(extri_quat)

        holo_intri_yaml_paths = list(intrinsics_path.glob("holo*.yaml"))
        if not holo_intri_yaml_paths:
            raise FileNotFoundError(f"No HoloLens intrinsics YAML found in {intrinsics_path}")
        holo_intri_yaml_path: Path = holo_intri_yaml_paths[0]
        holo_intri: HOCapIntrinsicsData = from_yaml(HOCapIntrinsicsData, holo_intri_yaml_path.read_text())

        ego_cam_list: list[PinholeParameters] = []

        world_T_cam: Float[ndarray, "4 4"]
        for world_T_cam in world_T_cam_batch:
            # Check if the serial contains 'hololens' using match statement
            intri = Intrinsics(
                camera_conventions="RDF",
                fl_x=holo_intri.color.fx,
                fl_y=holo_intri.color.fy,
                cx=holo_intri.color.ppx,
                cy=holo_intri.color.ppy,
                width=holo_intri.color.width,
                height=holo_intri.color.height,
            )
            # For other cameras, print their serial and extrinsics if available
            # need to do some stuff to make this
            extri = Extrinsics(world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3])
            ego_cam_list.append(PinholeParameters(name=holo_intri.serial, intrinsics=intri, extrinsics=extri))

        ego_cam_dict: dict[str, list[PinholeParameters]] = {get_args(EgoCameraIDs)[0]: ego_cam_list}
        return ego_cam_dict

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[PinholeParameters]]
    ) -> tuple[dict[str, list[PinholeParameters]], dict[str, Path]]:
        """Align cameras and videos based on the sequence."""
        assert len(video_path_list) == 1, f"Expected single video, got {len(video_path_list)}"
        assert len(ego_cam_dict) == 1, f"Expected single camera, got {len(ego_cam_dict)}"

        # Get the single camera name and video path
        cam_name = list(ego_cam_dict.keys())[0]
        video_path = video_path_list[0]

        video_to_cam_map = {cam_name: video_path}

        return ego_cam_dict, video_to_cam_map

    def __getitem__(self, idx) -> EgoData:
        return EgoData()

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
