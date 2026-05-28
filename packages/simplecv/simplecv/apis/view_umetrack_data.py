import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import field as serde_field
from serde import serde
from serde.json import from_json

from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.umetrack_temp.cameras import Camera
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    LANDMARK,
    UME_HAND_CONNECTIONS,
    HandModelNumpy,
    HandPoseLabels,
    SingleHandPose,
    landmarks_from_hand_pose,
)
from simplecv.umetrack_temp.perspective_cropping import (
    gen_crop_parameters_from_points,
    get_crop_points_from_hand_pose,
    rank_hand_visibility_in_cameras,
    warp_image_between_cameras,
)
from simplecv.umetrack_temp.projection import project_points
from simplecv.video_io import MultiVideoReader

HAND_TYPE = ("left", "right")
KEYPOINT_IDS = np.array([landmark.value for landmark in LANDMARK], dtype=np.uint16)
CAMERA_PANEL_ORDER: list[tuple[str, int]] = [
    ("TL", 0),
    ("BL", 1),
    ("BR", 2),
    ("TR", 3),
]
CAMERA_FILE_NAMES: dict[str, str] = {
    "TL": "top_left",
    "BL": "bottom_left",
    "BR": "bottom_right",
    "TR": "top_right",
}


@serde
class UmeTrackCameras:
    """Intrinsic parameters for a single UmeTrack fisheye camera.

    Notes:
        Synthetic sequences encode the fifth and sixth radial coefficients under
        the legacy keys `p3` and `p4`; see https://github.com/facebookresearch/UmeTrack_data/issues/4.
    """

    image_size_x: int = serde_field(rename="ImageSizeX")
    """Horizontal image resolution in pixels."""

    image_size_y: int = serde_field(rename="ImageSizeY")
    """Vertical image resolution in pixels."""

    fx: float
    """Focal length along the X axis in pixels."""

    fy: float
    """Focal length along the Y axis in pixels."""

    cx: float
    """Principal point X coordinate in pixels."""

    cy: float
    """Principal point Y coordinate in pixels."""

    distortion_model: Literal["FishEye62"] = serde_field(rename="DistortionModel")
    """Distortion model identifier exported by UmeTrack."""

    k1: float
    """First radial distortion coefficient."""

    k2: float
    """Second radial distortion coefficient."""

    k3: float
    """Third radial distortion coefficient."""

    k4: float
    """Fourth radial distortion coefficient."""

    p1: float
    """First tangential distortion coefficient."""

    p2: float
    """Second tangential distortion coefficient."""

    k5: float = serde_field(default=0.0)
    """Fifth radial distortion coefficient (defaults to 0.0 if not specified)."""

    k6: float = serde_field(default=0.0)
    """Sixth radial distortion coefficient (defaults to 0.0 if not specified)."""


@serde
class UmeTrackAnnotation:
    cameras: list[UmeTrackCameras]
    """Intrinsic definitions for each UmeTrack fisheye camera."""
    camera_angles: list[float]
    hand_model: HandModelNumpy
    joint_angles: Float32[ndarray, "n_frames n_hands=2 n_joints=22"]
    hand_confidences: Float32[np.ndarray, "n_frames n_hands=2"]
    wrist_transforms: Float32[np.ndarray, "n_frames n_hands=2 4 4"]
    camera_to_world_transforms: Float32[ndarray, "n_frames n_cams 4 4"]


@dataclass
class UmeTrackData:
    multi_view_images: UInt8[ndarray, "n_cams frame_h single_cam_w 3"]
    multi_world_T_cam: Float32[ndarray, "n_cams 4 4"]
    gt_tracking: dict[int, SingleHandPose]


class DataStream:
    def __init__(
        self,
        annotation_path: Path,
        video_paths: list[Path],
    ) -> None:
        """
        Datastream for a single sequence. This provides the following:
        - camera intrinsics
        - camera extrinsics (these change per frame)
        - multiview monocular images
        - hand pose labels
        """
        self._n_cams: int = len(CAMERA_PANEL_ORDER)
        self.multi_video_stream: MultiVideoReader = MultiVideoReader(video_paths)

        annotation: UmeTrackAnnotation = from_json(UmeTrackAnnotation, annotation_path.read_text())
        # camera intrinsics
        self.fisheye_cameras = create_cameras(annotation.cameras)
        # camera extrinsics (slam pose of each camera)
        self.world_T_cam_all: Float32[ndarray, "n_frames n_cams 4 4"] = annotation.camera_to_world_transforms
        self.hand_model: HandModelNumpy = annotation.hand_model

        self.hand_pose_labels = HandPoseLabels(
            camera_angles=[float(angle) for angle in annotation.camera_angles],
            camera_to_world_transforms=self.world_T_cam_all,
            hand_model=self.hand_model,
            joint_angles=annotation.joint_angles,
            wrist_transforms=annotation.wrist_transforms,
            hand_confidences=annotation.hand_confidences,
        )
        self._warned_singular_pose = False
        self._warned_short_sequence = False

    def __len__(self):
        return len(self.hand_pose_labels)

    def __iter__(self) -> Iterator[UmeTrackData]:
        """
        Returns:
            multi_view_images: (n_cams, frame_h, single_cam_w, 3)
            cam_to_world: (n_cams, 4, 4)
            gt_tracking: dict[int, SingleHandPose]
        """
        for frame_idx in range(self.__len__()):
            gt_tracking: dict[int, SingleHandPose] = {}
            camera_frame_list = self.multi_video_stream[frame_idx]
            if len(camera_frame_list) != self._n_cams:
                raise ValueError(
                    f"Expected {self._n_cams} camera frames, received {len(camera_frame_list)} "
                    f"for frame index {frame_idx}."
                )
            camera_frames: list[UInt8[ndarray, "frame_h single_cam_w 3"]] = [
                np.asarray(frame, dtype=np.uint8) for frame in camera_frame_list
            ]
            multi_view_images: UInt8[ndarray, "n_cams frame_h single_cam_w 3"] = np.stack(camera_frames, axis=0)
            multi_world_T_cam: Float32[ndarray, "n_cams 4 4"] = self.world_T_cam_all[frame_idx]

            # Skip frames where any camera pose is singular or invalid.
            if not np.all(np.isfinite(multi_world_T_cam)):
                if not self._warned_singular_pose:
                    warnings.warn(
                        "Encountered non-finite camera pose; skipping affected frames.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_singular_pose = True
                continue

            if any(abs(np.linalg.det(world_T_cam[:3, :3])) < 1e-8 for world_T_cam in multi_world_T_cam):
                if not self._warned_singular_pose:
                    warnings.warn(
                        "Encountered singular camera pose; skipping affected frames.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_singular_pose = True
                continue
            for hand_idx in range(0, 2):
                if self.hand_pose_labels.hand_confidences[frame_idx, hand_idx] > 0:
                    gt_tracking[hand_idx] = SingleHandPose(
                        joint_angles=self.hand_pose_labels.joint_angles[frame_idx, hand_idx].astype(
                            np.float32, copy=False
                        ),
                        wrist_xform=self.hand_pose_labels.wrist_transforms[frame_idx, hand_idx].astype(
                            np.float32, copy=False
                        ),
                        hand_confidence=float(self.hand_pose_labels.hand_confidences[frame_idx, hand_idx]),
                    )

            # set camera extrinsics as they change per frame (slam tracking)
            for cam_idx, world_T_cam in enumerate(multi_world_T_cam):
                cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_cam)
                self.fisheye_cameras[cam_idx].set_extrinsic(cam_T_world)

            data: UmeTrackData = UmeTrackData(
                multi_view_images=multi_view_images,
                multi_world_T_cam=multi_world_T_cam,
                gt_tracking=gt_tracking,
            )
            yield data


def create_cameras(umtrack_camera_list: list[UmeTrackCameras]) -> list[Camera]:
    """Instantiate fisheye cameras with camera_parameters intrinsics/extrinsics."""

    cameras: list[Camera] = []
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
        cameras.append(Camera(camera_params))

    return cameras


def setup_logging(parent_log_path: Path) -> None:
    """
    setup logging for rerun along with annotations context for each hand
    """
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RUB, static=True)
    class_descriptions = []
    for hand_idx, hand_type in enumerate(HAND_TYPE):
        class_descriptions.append(
            ClassDescription(
                info=AnnotationInfo(label=f"{hand_type} hand", id=hand_idx),
                keypoint_annotations=[AnnotationInfo(id=lm.value) for lm in LANDMARK],
                keypoint_connections=list(UME_HAND_CONNECTIONS),
            ),
        )
    rr.log(f"{parent_log_path}", rr.AnnotationContext(class_descriptions), static=True)


def create_umetrack_view(
    log_path: Path,
    camera_ids: Sequence[str],
    camera_filter: list[Literal["TL", "TR", "BL", "BR"]] | None = None,
) -> rrb.ContainerLike:
    """Send a Rerun blueprint tailored for the UmeTrack visualization layout."""

    if camera_filter is None:
        camera_filter = ["BL", "BR"]
    spatial_view = rrb.Spatial3DView(origin=f"{log_path}", name="3D View")

    camera_rows: list[rrb.ContainerLike] = []
    for display_name, camera_idx in CAMERA_PANEL_ORDER:
        if display_name not in camera_filter:
            continue
        if camera_idx >= len(camera_ids):
            continue
        camera_id: str = camera_ids[camera_idx]
        crop_views = rrb.Vertical(
            contents=[
                rrb.Spatial2DView(
                    origin=f"{log_path}/ego/{camera_id}_left_crop/pinhole/image",
                    contents=[
                        "+ $origin/**",
                    ],
                    name=f"{display_name} Left Crop",
                ),
                rrb.Spatial2DView(
                    origin=f"{log_path}/ego/{camera_id}_right_crop/pinhole/image",
                    contents=[
                        "+ $origin/**",
                    ],
                    name=f"{display_name} Right Crop",
                ),
            ],
            row_shares=[1, 1],
            name=f"{display_name} Crops",
        )

        camera_view = rrb.Spatial2DView(
            origin=f"{log_path}/ego/{camera_id}/pinhole/video",
            contents=[
                "+ $origin/**",
            ],
            name=f"{display_name} Camera",
        )

        camera_rows.append(
            rrb.Horizontal(
                contents=[crop_views, camera_view],
                column_shares=[1, 2],
                name=f"{display_name} Panel",
            )
        )

    right_column = rrb.Vertical(contents=camera_rows, row_shares=[1] * len(camera_rows), name="Camera Panels")

    final_container: rrb.Horizontal = rrb.Horizontal(
        contents=[spatial_view, right_column],
        column_shares=[3, 2],
        name="UmeTrack Layout",
    )

    return final_container


@dataclass
class UmeTrackVisualizeConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    data_dir: Path
    """Parent directory containing one or more `recording_*` folders."""
    sequence_id: int = 0
    """Recording index to visualize (0-indexed within the parent directory)."""
    min_required_vis_landmarks: int = 19
    """Minimum number of visible landmarks required to consider a hand visible in a camera."""
    num_crop_points: Literal[21, 42, 63] = 63
    """Number of points to use when generating the crop region around the hand."""


def main(config: UmeTrackVisualizeConfig) -> None:
    if not config.data_dir.exists():
        raise FileNotFoundError(config.data_dir)
    if not config.data_dir.is_dir():
        raise NotADirectoryError(f"{config.data_dir} is not a directory")
    recording_dirs: list[Path] = sorted(path for path in config.data_dir.glob("recording_*") if path.is_dir())
    if not recording_dirs:
        raise FileNotFoundError(
            f"Expected at least one 'recording_*' directory under {config.data_dir}. "
            "Pass the parent directory containing the recordings."
        )
    if not 0 <= config.sequence_id < len(recording_dirs):
        raise ValueError(
            f"sequence_id {config.sequence_id} is out of range for {len(recording_dirs)} recordings in {config.data_dir}."
        )
    selected_dir: Path = recording_dirs[config.sequence_id]
    video_paths: list[Path] = list(selected_dir.glob("*.mp4"))
    # sort video paths based on CAMERA_FILE_NAMES order
    video_paths: list[Path] = sorted(video_paths, key=lambda p: list(CAMERA_FILE_NAMES.values()).index(p.stem))
    # there should always be 4 videos for ume track
    if len(video_paths) != 4:
        raise FileNotFoundError(f"Expected 4 camera videos in {selected_dir}, found {len(video_paths)}")

    annotation_matches: list[Path] = sorted(selected_dir.glob("*.json"))
    if not annotation_matches:
        raise FileNotFoundError(f"Missing annotation JSON in {selected_dir}")

    annotation_path: Path = annotation_matches[0]
    datastream = DataStream(annotation_path=annotation_path, video_paths=video_paths)
    camera_angles: list[float] = datastream.hand_pose_labels.camera_angles

    parent_log_path: Path = Path("world")
    camera_ids: list[str] = [video_path.stem for video_path in video_paths]
    camera_entity_paths: list[Path] = [parent_log_path / "ego" / camera_id for camera_id in camera_ids]
    timeline: str = "video_time"
    setup_logging(parent_log_path)

    video_log_paths: list[Path] = [camera_path / "pinhole" / "video" for camera_path in camera_entity_paths]
    # log the video paths
    timestamps_ns_list: list[Int[ndarray, "num_frames"]] = []
    for video_path, video_log_path in zip(video_paths, video_log_paths, strict=True):
        timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_source=video_path, video_log_path=video_log_path, timeline=timeline
        )
        timestamps_ns_list.append(timestamps_ns)

    shortest_timestamp: Int[ndarray, "n_frames"] = min(timestamps_ns_list, key=len)

    view_container: rrb.ContainerLike = create_umetrack_view(parent_log_path, camera_ids)
    blueprint = rrb.Blueprint(view_container, collapse_panels=True)
    rr.send_blueprint(blueprint)
    frame_data: UmeTrackData
    active_crop_entities: set[str] = set()
    for frame_idx, frame_data in enumerate(datastream):
        frame_timestamp_ns: np.int64 | None = (
            shortest_timestamp[frame_idx] if frame_idx < len(shortest_timestamp) else None
        )
        if frame_timestamp_ns is None:
            raise IndexError(f"No video timestamp available for frame index {frame_idx}.")
        rr.set_time(timeline, timestamp=1e-9 * frame_timestamp_ns)
        hand_model: HandModelNumpy = datastream.hand_model
        frame_active_crop_entities: set[str] = set()
        landmarks_dict: dict[str, Float32[ndarray, "n_kpts=21 3"] | None] = {
            "left": None,
            "right": None,
        }

        # log hand pose data
        for hand_idx, hand_type in enumerate(HAND_TYPE):
            if hand_idx in frame_data.gt_tracking:
                hand_pose: SingleHandPose = frame_data.gt_tracking[hand_idx]
                landmark: Float32[ndarray, "n_kpts=21 3"] = landmarks_from_hand_pose(hand_model, hand_pose, hand_idx)
                rr.log(
                    f"{parent_log_path}/{hand_type}/landmark",
                    rr.Points3D(landmark, keypoint_ids=KEYPOINT_IDS, class_ids=hand_idx, show_labels=False),
                )

                ## Generating Crop based on 3d keypoints
                # first are gt, second are neutral, third are open
                crop_points: Float32[np.ndarray, "n_crop_points 3"] = get_crop_points_from_hand_pose(
                    hand_model, hand_pose, hand_idx, num_crop_points=config.num_crop_points
                )
                cam_indices: list[int] = rank_hand_visibility_in_cameras(
                    cameras=datastream.fisheye_cameras,
                    hand_model=hand_model,
                    hand_pose=hand_pose,
                    hand_idx=hand_idx,
                    min_required_vis_landmarks=config.min_required_vis_landmarks,
                )
                # creating new perspective cameras
                for cam_idx in cam_indices:
                    current_cam: Camera = datastream.fisheye_cameras[cam_idx]
                    perspective_cam_params: PinholeParameters = gen_crop_parameters_from_points(
                        current_cam,
                        crop_points,
                        new_image_size=(96, 96),
                        mirror_img_x=False,  # True if hand_type == "right" else False,
                        camera_angle=camera_angles[cam_idx],
                        focal_multiplier=0.95,
                    )
                    perspective_cam = Camera(perspective_cam_params)

                    # perform image warping from src camera to dst camera
                    current_image: UInt8[ndarray, "img_h img_w 3"] = frame_data.multi_view_images[cam_idx]
                    crop: UInt8[ndarray, "crop_h=96 crop_w=96 3"] = warp_image_between_cameras(
                        current_cam, perspective_cam, current_image
                    )

                    crop_camera_id: str = f"{camera_ids[cam_idx]}_{hand_type}_crop"
                    crop_camera_entity_path: Path = parent_log_path / "ego" / crop_camera_id
                    crop_pinhole_log_path: str = f"{crop_camera_entity_path}/pinhole"
                    try:
                        log_pinhole(
                            perspective_cam_params,
                            crop_camera_entity_path,
                            image_plane_distance=50.0,
                        )
                        rr.log(f"{crop_pinhole_log_path}/image", rr.Image(crop).compress(jpeg_quality=75))
                        uv_cropped: Float[ndarray, "n_kpts=21 2"] = project_points(landmark, perspective_cam)
                        rr.log(
                            f"{crop_pinhole_log_path}/image/{hand_type}_landmark",
                            rr.Points2D(uv_cropped, keypoint_ids=KEYPOINT_IDS, class_ids=hand_idx, show_labels=False),
                        )
                    except Exception:
                        rr.log(f"{crop_camera_entity_path}", rr.Clear(recursive=True))
                        continue

                    frame_active_crop_entities.add(crop_camera_entity_path.as_posix())

                landmarks_dict[hand_type] = landmark

        stale_entities: set[str] = active_crop_entities - frame_active_crop_entities
        for stale_entity in stale_entities:
            rr.log(stale_entity, rr.Clear(recursive=True))
        active_crop_entities = frame_active_crop_entities

        # log original camera camera data and projected landmarks
        for camera_idx, camera in enumerate(datastream.fisheye_cameras):
            camera_entity_path: Path = camera_entity_paths[camera_idx]
            pinhole_log_path: Path = camera_entity_path / "pinhole"
            log_pinhole(
                camera.camera_parameters,
                camera_entity_path,
                image_plane_distance=50.0,
            )
            for hand_idx, hand_type in enumerate(HAND_TYPE):
                landmarks: Float32[ndarray, "n_kpts=21 3"] | None = landmarks_dict[hand_type]
                if landmarks is not None:
                    hand_landmarks: Float32[ndarray, "n_kpts=21 3"] = landmarks
                    uv: Float[np.ndarray, "num_points 2"] = project_points(hand_landmarks, camera)
                    rr.log(
                        f"{pinhole_log_path}/video/{hand_type}_landmark",
                        rr.Points2D(
                            uv,
                            keypoint_ids=KEYPOINT_IDS,
                            class_ids=hand_idx,
                            show_labels=False,
                        ),
                    )
