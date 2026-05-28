from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation as R
from serde import serde
from serde.yaml import from_yaml
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence, ExoBatchData, ManoStack
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


class ImageLabel(TypedDict):
    """Structure of the label data from npz files."""

    cam_K: Float32[ndarray, "3 3"]  # Camera intrinsic matrix
    obj_poses: Float32[ndarray, "num_objs 4 4"]  # Object poses as 4x4 transformation matrices
    hand_joints_3d: Float32[ndarray, "2 21 3"]  # 3D hand joint coordinates (left/right hands)
    hand_joints_2d: Int[ndarray, "2 21 2"]  # 2D hand joint coordinates as int64 (left/right hands)
    seg_mask: UInt8[ndarray, "480 640"]  # Segmentation mask
    obj_class_inds: Int[ndarray, "num_objs"]  # Object class indices as int64
    obj_class_names: ndarray  # Object class names as strings


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
class HOCapIntrinsics:
    """Common parameters for both color and depth cameras"""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: Float32[ndarray, "..."] | None = None
    k_33: Float32[ndarray, "3 3"] = field(default_factory=lambda: np.eye(3, dtype=np.float32))

    def __post_init__(self) -> None:
        # fmt: off
        self.k_33 = np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ])
        # fmt: on


@serde
class HOCapIntrinsicsData:
    """Camera intrinsics calibration data"""

    serial: CameraIDs
    color: HOCapIntrinsics
    depth: HOCapIntrinsics | None = None
    depth2color: Float32[ndarray, "..."] | None = None


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


class HocapExoSequence(BaseExoSequence[HocapConfig]):
    def __init__(self, cfg: HocapConfig) -> None:
        super().__init__(cfg)
        self._depth_paths: list[dict[ExoCameraIDs, Path]] = self.load_depth_paths(
            data_path=self.config.root_directory,
            sequence_name=self.config.sequence_name,
            subject_id=self.config.subject_id,
        )

    def __len__(self) -> int:
        assert len(self._video_path_list) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.exo_video_readers)

    def __getitem__(self, idx: int) -> None:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for sequence of length {len(self)}")

        # bgr_list: list[UInt8[ndarray, "480 640 3"]] = self.exo_video_readers[idx]
        # if self.config.load_labels:
        #     xyz: Float32[ndarray, "2 21 3"] = self.exo_batch_data.xyz_stack[idx]
        #     uv_dict: dict[str, Float32[ndarray, "2 21 2"]] = {
        #         cam_name: uv_stack[idx] for cam_name, uv_stack in self.exo_batch_data.uv_stack_dict.items()
        #     }
        # else:
        #     xyz = None
        #     uv_dict = None
        # return ExoData(cam_params_list=self.exo_cam_list, bgr_list=bgr_list, xyz=xyz, uv_dict=uv_dict)
        return None

    def load_exo_batch_data(self, data_path: Path, sequence_name: str, subject_id: SubjectIDs) -> ExoBatchData:
        label_path: Path = data_path / f"subject_{subject_id}" / sequence_name
        assert label_path.exists(), f"Path {label_path} does not exist."

        xyz_list: list[Float32[ndarray, "2 21 3"]] = []
        uv_stack_dict: dict[str, Float32[ndarray, "num_frames 2 21 2"]] = {}
        for cam_idx, exo_cam in enumerate(
            tqdm(
                self.exo_cam_list,
                desc="Loading 3D labels",
                disable=not self.config.verbose,
                leave=False,
                position=1,
            )
        ):
            camera_label_path: Path = label_path / exo_cam.name
            assert camera_label_path.exists(), f"Path {camera_label_path} does not exist."
            npz_paths: list[Path] = sorted(camera_label_path.glob("*.npz"))
            uv_list: list[Float32[ndarray, "2 21 2"]] = []
            # for now only load the 2d + 3d hand joints
            for npz_path in npz_paths:
                npz_data = np.load(npz_path)
                # label_dict: ImageLabel = {key: npz_data[key] for key in npz_data}
                uv: Int[ndarray, "2 21 2"] = npz_data["hand_joints_2d"]
                uv_list.append(uv.astype(np.float32))
                if cam_idx == 0:
                    xyz_cam: Float32[ndarray, "2 21 3"] = npz_data["hand_joints_3d"]
                    # convert to world coordinates
                    world_T_cam: Float32[ndarray, "4 4"] = exo_cam.extrinsics.world_T_cam.astype(
                        np.float32
                    )  # means camera to world, or world from cam

                    # Assuming hand_joints_3d_cam.shape == (2, 21, 3) and world_T_cam.shape == (4, 4)
                    # Create homogeneous coordinates by concatenating a ones column along the last axis
                    ones: Float32[ndarray, "2 21 1"] = np.ones(
                        (*xyz_cam.shape[:-1], 1),
                        dtype=xyz_cam.dtype,
                    )
                    xyz_cam_homogeneous: Float32[ndarray, "2 21 4"] = np.concatenate([xyz_cam, ones], axis=-1)

                    # filger out -1 (not detected) values
                    xyz_cam_homogeneous = np.where(xyz_cam_homogeneous == -1, np.nan, xyz_cam_homogeneous)

                    # Transform all joints at once using matrix multiplication.
                    # The multiplication is broadcast over the first two dimensions.
                    xyz_world_homogeneous: Float32[ndarray, "2 21 4"] = xyz_cam_homogeneous @ world_T_cam.T

                    # Extract the 3D world coordinates (ignore the homogeneous component)
                    xyz_world: Float32[ndarray, "2 21 3"] = xyz_world_homogeneous[..., :3]
                    xyz_list.append(xyz_world)

            uv_stack: Float32[ndarray, "num_frames 2 21 2"] = np.stack(uv_list)
            uv_stack_dict[exo_cam.name] = uv_stack

        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = np.stack(xyz_list)
        mano_stack: ManoStack = self.load_mano_poses(data_path, sequence_name, subject_id)
        return ExoBatchData(uv_stack_dict=uv_stack_dict, xyz_stack=xyz_stack, mano_stack=mano_stack)

    def load_video_paths(self) -> list[Path]:
        sequence_path: Path = (
            self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        )
        assert sequence_path.exists(), f"Path {sequence_path} does not exist."

        video_path_list: list[Path] = []
        # Load video paths for each exo camera, in
        for exo_cam in tqdm(
            self.load_exo_cams(),
            desc="Loading videos",
            disable=not self.config.verbose,
            leave=False,
            position=1,
        ):
            img_dir: Path = sequence_path / exo_cam.name
            assert img_dir.exists(), f"Path {img_dir} does not exist."
            video_path: Path = img_dir / "output.mp4"
            if not video_path.exists():
                video_path: Path = create_temp_video_from_img_dir(
                    img_dir,
                    fps=30,
                    quality="low",
                    image_extension="jpg",
                    save_file=True,
                )

            video_path_list.append(video_path)
        return video_path_list

    def load_depth_paths(
        self, data_path: Path, sequence_name: str, subject_id: SubjectIDs
    ) -> list[dict[ExoCameraIDs, Path]]:
        """Load depth image paths organized by timestamp."""
        sequence_path: Path = data_path / f"subject_{subject_id}" / sequence_name
        assert sequence_path.exists(), f"Path {sequence_path} does not exist."
        # First, collect all depth paths per camera
        camera_depth_paths: dict[ExoCameraIDs, list[Path]] = {}
        for exo_cam in tqdm(
            self.exo_cam_list,
            desc="Indexing depth images",
            disable=not self.config.verbose,
            leave=False,
            position=1,
        ):
            depth_dir: Path = sequence_path / exo_cam.name
            assert depth_dir.exists(), f"Path {depth_dir} does not exist."
            depth_paths: list[Path] = sorted(depth_dir.glob("*.png"))
            camera_depth_paths[exo_cam.name] = depth_paths

        # Then organize by timestamp
        # Get minimum number of frames across all cameras
        min_frames: int = min(len(paths) for paths in camera_depth_paths.values())

        # Create list of dictionaries - each dict represents one timestamp
        depth_paths_list: list[dict[ExoCameraIDs, Path]] = []
        for frame_idx in range(min_frames):
            frame_depth_dict: dict[ExoCameraIDs, Path] = {}
            for camera_name, paths in camera_depth_paths.items():
                frame_depth_dict[camera_name] = paths[frame_idx]
            depth_paths_list.append(frame_depth_dict)

        return depth_paths_list

    def load_exo_cams(self) -> list[PinholeParameters]:
        calibration_path: Path = self.config.root_directory / "calibration"
        extrinsics_directory: Path = calibration_path / "extrinsics"
        intrinsics_path: Path = calibration_path / "intrinsics"

        assert extrinsics_directory.exists(), f"Path {extrinsics_directory} does not exist."
        assert intrinsics_path.exists(), f"Path {intrinsics_path} does not exist."

        sequence_path: Path = (
            self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        )
        assert sequence_path.exists(), f"Path {sequence_path} does not exist."

        extrinsic_yaml: Path = extrinsics_directory / "extrinsics_20231014.yaml"
        assert extrinsic_yaml.exists(), f"Path {extrinsic_yaml} does not exist."

        extri_hocap: HOCapExtrinsicsData = from_yaml(HOCapExtrinsicsData, extrinsic_yaml.read_text())

        hocap_intri_list: list[HOCapIntrinsicsData] = []
        for intrinsics_yaml in intrinsics_path.glob("*.yaml"):
            # Skip any file with "hololens" in the name
            with open(intrinsics_yaml) as file:
                intri_str: str = file.read()
            hocap_intri_list.append(from_yaml(HOCapIntrinsicsData, intri_str))

        exo_cam_list: list[PinholeParameters] = []
        for hocap_intri in hocap_intri_list:
            # Check if the serial contains 'hololens' using match statement
            match hocap_intri.serial:
                # ego perspective
                case serial if "hololens" in serial:
                    if self.config.verbose:
                        tqdm.write(f"Found HoloLens camera: {hocap_intri.serial}")
                # exo perspective
                case _:
                    intri = Intrinsics(
                        camera_conventions="RDF",
                        fl_x=hocap_intri.color.fx,
                        fl_y=hocap_intri.color.fy,
                        cx=hocap_intri.color.ppx,
                        cy=hocap_intri.color.ppy,
                        width=hocap_intri.color.width,
                        height=hocap_intri.color.height,
                    )
                    # For other cameras, print their serial and extrinsics if available
                    world_T_cam: Float32[ndarray, "4 4"] = extri_hocap.world_T_cam_dict.get(hocap_intri.serial)
                    # need to do some stuff to make this
                    extri = Extrinsics(world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3])
                    exo_cam_list.append(PinholeParameters(name=hocap_intri.serial, intrinsics=intri, extrinsics=extri))

        return exo_cam_list

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1

    @property
    def depth_paths(self) -> list[dict[ExoCameraIDs, Path]]:
        return self._depth_paths

    # @property
    # def depth_paths(self) -> list[dict[ExoCameraIDs, Path]]:
    #     return self._depth_paths
