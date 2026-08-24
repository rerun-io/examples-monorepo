from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.configs.dataset_paths import EGO_DEX_ROOT
from simplecv.data.skeleton.avp_fullbody import AVP_ID2NAME, AVP_IDS, AVP_LINKS
from simplecv.ops.triangulate import projectN3
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import log_pinhole, log_video
from simplecv.video_utils import reencode_video_optimal


@dataclass
class ConvertEgoConfig:
    dataset: Literal["ego-dex"] = "ego-dex"
    root_directory: Path = EGO_DEX_ROOT / "test"
    save_directory: Path = EGO_DEX_ROOT / "test-rrd"
    sequence_name: str | Literal["all"] = "wipe_screen"
    send_as_batch: bool = True


@dataclass
class EgoDataSequence:
    video_path: Path
    pinhole_list: list[PinholeParameters]
    llm_description: str
    xyz_stack: Float32[ndarray, "n_frames 68 3"]
    conf_stack: Float32[ndarray, "n_frames 68 1"]


def parse_hdf5_file(hdf5_path: Path, video_path: Path) -> EgoDataSequence:
    h5py_file = h5py.File(f"{hdf5_path}", "r")
    confidences = h5py_file["confidences"]
    transforms = h5py_file["transforms"]

    world_T_camera: Float32[ndarray, "n_frames 4 4"] = transforms.get("camera")[:]
    joints_list: list[Float32[ndarray, "n_frames 3"]] = []
    for joint_name in AVP_ID2NAME.values():
        joint_transform: Float32[ndarray, "n_frames 4 4"] = transforms.get(joint_name)[:]
        joint_xyz: Float32[ndarray, "n_frames 3"] = joint_transform[:, :3, 3]
        joints_list.append(joint_xyz)

    joints_xyz: Float32[ndarray, "n_frames 68 3"] = np.stack(joints_list, axis=1)

    conf_list: list[Float32[ndarray, "n_frames 3"]] = []
    for joint_name in AVP_ID2NAME.values():
        conf: Float32[ndarray, "n_frames"] = confidences.get(joint_name)[:]
        conf_list.append(conf)

    conf_stack: Float32[ndarray, "n_frames 68"] = np.stack(conf_list, axis=1)
    conf_stack = rearrange(conf_stack, "n_frames n_joints -> n_frames n_joints 1")

    intrinsics: Float32[ndarray, "3 3"] = np.array(
        [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    fl_x: float = float(intrinsics[0, 0])
    fl_y: float = float(intrinsics[1, 1])
    cx: float = float(intrinsics[0, 2])
    cy: float = float(intrinsics[1, 2])

    pinhole_list: list[PinholeParameters] = []
    for i in range(world_T_camera.shape[0]):
        pinhole = PinholeParameters(
            name="AVP Camera",
            intrinsics=Intrinsics(
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
                camera_conventions="RDF",
                height=int(cy * 2),
                width=int(cx * 2),
            ),
            extrinsics=Extrinsics(world_R_cam=world_T_camera[i][:3, :3], world_t_cam=world_T_camera[i][:3, 3]),
        )
        pinhole_list.append(pinhole)

    ego_sequence = EgoDataSequence(
        video_path=video_path,
        pinhole_list=pinhole_list,
        llm_description=h5py_file.attrs["llm_description"],
        xyz_stack=joints_xyz,
        conf_stack=conf_stack,
    )
    return ego_sequence


def set_pose_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Apple Vision Pro Keypoints", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in AVP_ID2NAME.items()],
                    keypoint_connections=AVP_LINKS,
                ),
            ]
        ),
        static=True,
    )


def convert_ego(config: ConvertEgoConfig) -> None:
    print("Starting ego data viewer...")
    # get every directory under the root directory
    if config.sequence_name == "all":
        sequence_paths: list[Path] = sorted([p for p in config.root_directory.glob("*") if p.is_dir()])
    else:
        sequence_paths: list[Path] = [config.root_directory / config.sequence_name]

    for sequence_path in tqdm(sequence_paths, desc="Processing sequences"):
        assert sequence_path.exists(), f"Sequence path {sequence_path} does not exist."
        video_paths: list[Path] = sorted(sequence_path.glob("*.mp4"))
        hdf5_paths: list[Path] = sorted(sequence_path.glob("*.hdf5"))
        # check that theres at least one video file
        if len(video_paths) == 0:
            print(f"No video files found in {sequence_path}. Skipping.")
            continue

        for video_path, hdf5_path in tqdm(
            zip(video_paths, hdf5_paths, strict=True),
            desc=f"Processing videos in {sequence_path.name}",
            total=len(video_paths),
            leave=False,
        ):
            assert video_path.stem == hdf5_path.stem
            ego_sequence: EgoDataSequence = parse_hdf5_file(hdf5_path, video_path)

            parent_log_path = Path("world")
            timeline = "video_time"

            # make sure the save directory exists
            save_path: Path = config.save_directory / sequence_path.name / f"{int(video_path.stem):05d}.rrd"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)

            cam_log_path: Path = parent_log_path / "camera"
            pinhole_log_path: Path = cam_log_path / "pinhole"
            video_log_path: Path = pinhole_log_path / "video"

            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="llm_description"),
                        rrb.Spatial2DView(origin=str(video_log_path)),
                        row_shares=[3, 10],
                    ),
                    column_shares=[2, 1],
                ),
                collapse_panels=True,
            )
            rr.init(application_id=f"{config.sequence_name}_{video_path.stem}", spawn=False)
            rr.save(path=save_path, default_blueprint=blueprint)
            set_pose_annotation_context()
            rr.log("/", rr.ViewCoordinates.RUB, static=True)

            new_video_path: Path = reencode_video_optimal(input_video_path=ego_sequence.video_path)
            frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                new_video_path, video_log_path=video_log_path, timeline=timeline
            )
            rr.log(
                "llm_description",
                rr.TextDocument(text=ego_sequence.llm_description, media_type="text/markdown"),
                static=True,
            )
            if config.send_as_batch:
                log_batched(
                    parent_log_path=parent_log_path,
                    timeline=timeline,
                    frame_timestamps_ns=frame_timestamps_ns,
                    ego_sequence=ego_sequence,
                )
            else:
                log_incremental(
                    parent_log_path=parent_log_path,
                    timeline=timeline,
                    frame_timestamps_ns=frame_timestamps_ns,
                    ego_sequence=ego_sequence,
                )


def log_batched(
    parent_log_path: Path,
    timeline: str,
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    ego_sequence: EgoDataSequence,
) -> None:
    """Logs batched ego-centric data, including 3D keypoints and pinhole cameras, to Rerun.

    This function processes a sequence of ego-centric data, converting keypoint
    confidence scores into RGB colors for visualization and logging them as 3D points
    in Rerun. It also logs pinhole camera information for each frame in the sequence.
    The keypoints are logged as a single time-varying `rr.Points3D` entity,
    while pinhole cameras are logged per frame.

    Args:
        parent_log_path (Path): The base Rerun entity path under which all data
            will be logged (e.g., "ego_data").
        timeline (str): The name of the Rerun timeline to use for logging
            time-series data (e.g., "log_time").
        frame_timestamps_ns (Int[ndarray, "num_frames"]): A NumPy array of
            timestamps in nanoseconds for each frame in the sequence.
        ego_sequence (EgoDataSequence): An object containing the ego-centric data.
            It is expected to provide the following attributes:
            - `conf_stack` (Float32[ndarray, "n_frames num_keypoints 1"]):
              Confidence scores for each keypoint in each frame. `num_keypoints`
              is typically 68 in the context of AVP keypoints.
            - `xyz_stack` (Float32[ndarray, "n_frames num_keypoints 3"]):
              3D coordinates (e.g., in world or camera space) of each keypoint
              in each frame.
            - `pinhole_list` (list): A list of pinhole camera model objects,
              one corresponding to each frame in `frame_timestamps_ns`.
              Each object in this list will be passed to `log_pinhole`.
    """
    conf_stack: Float32[ndarray, "n_frames 68 1"] = ego_sequence.conf_stack
    n_kpts: int = conf_stack.shape[1]

    all_colors_stack: UInt8[ndarray, "n_frames 68 3"] = confidence_scores_to_rgb(confidence_scores=conf_stack)
    rearranged_colors: UInt8[ndarray, "_ 3"] = rearrange(all_colors_stack, "n_frames kpts rgb -> (n_frames kpts) rgb")
    joints_log_path: Path = parent_log_path / "avp_keypoints"

    rr.log(
        f"{joints_log_path}",
        rr.Points3D.from_fields(
            radii=0.005,
            class_ids=0,
            keypoint_ids=AVP_IDS,
            show_labels=False,
        ),
        static=True,
    )
    rr.send_columns(
        f"{joints_log_path}",
        indexes=[rr.TimeColumn(timeline, duration=1e-9 * frame_timestamps_ns)],
        columns=[
            *rr.Points3D.columns(
                positions=rearrange(
                    ego_sequence.xyz_stack,
                    "num_frames kpts dim -> (num_frames kpts) dim",
                ),
                colors=rearranged_colors,  # Added dynamic confidence based colors
            ).partition(lengths=[n_kpts] * len(frame_timestamps_ns)),
        ],
    )

    for ts, pinhole in tqdm(
        zip(frame_timestamps_ns, ego_sequence.pinhole_list, strict=True),
        desc="Logging pinhole cameras",
        total=len(frame_timestamps_ns),
    ):
        rr.set_time(timeline=timeline, duration=np.timedelta64(int(ts), "ns"))
        cam_log_path: Path = parent_log_path / "camera"
        log_pinhole(
            pinhole,
            cam_log_path=cam_log_path,
            image_plane_distance=0.1,
            static=False,
        )


def log_incremental(
    parent_log_path: Path,
    timeline: str,
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    ego_sequence: EgoDataSequence,
):
    joints_log_path: Path = parent_log_path / "avp_keypoints"

    xyz_stack: Float32[ndarray, "n_frames 68 3"] = ego_sequence.xyz_stack
    conf_stack: Float32[ndarray, "n_frames 68 1"] = ego_sequence.conf_stack

    rr.log(
        "llm_description", rr.TextDocument(text=ego_sequence.llm_description, media_type="text/markdown"), static=True
    )
    for ts_idx, (ts, pinhole) in enumerate(
        tqdm(
            zip(frame_timestamps_ns, ego_sequence.pinhole_list, strict=True),
            desc="Logging pinhole cameras",
            total=len(frame_timestamps_ns),
        )
    ):
        rr.set_time(timeline=timeline, duration=np.timedelta64(int(ts), "ns"))
        cam_log_path = parent_log_path / "camera"
        log_pinhole(
            pinhole,
            cam_log_path=cam_log_path,
            image_plane_distance=0.1,
            static=False,
        )
        current_xyz: Float32[ndarray, "68 3"] = xyz_stack[ts_idx, ...]
        current_conf: Float32[ndarray, "68 1"] = conf_stack[ts_idx, ...]
        rr.log(
            f"{joints_log_path}",
            Points3DWithConfidence(
                positions=current_xyz,
                confidences=current_conf.squeeze(),
                class_ids=0,
                keypoint_ids=AVP_IDS,
                show_labels=False,
                radii=0.005,
            ),
        )
        pinhole_log_path = cam_log_path / "pinhole"
        video_log_path: Path = pinhole_log_path / "video"
        xyz_hom: Float32[ndarray, "68 4"] = np.hstack(
            (current_xyz, np.ones((current_xyz.shape[0], 1), dtype=np.float32))
        )
        Pall: Float32[ndarray, "3 4"] = pinhole.projection_matrix.astype(np.float32)
        uv: Float32[ndarray, "1 68 3"] = projectN3(kpts3d=xyz_hom, Pall=Pall[np.newaxis, ...]).astype(np.float32)

        # Mark out-of-bounds uv coordinates as NaN
        uv[..., 0] = np.where((uv[..., 0] < 0) | (uv[..., 0] > pinhole.intrinsics.width), np.nan, uv[..., 0])
        uv[..., 1] = np.where((uv[..., 1] < 0) | (uv[..., 1] > pinhole.intrinsics.height), np.nan, uv[..., 1])

        # Log the 2D keypoints. uv[..., -1] is the homogeneous camera-space
        # depth, NOT a confidence -- the tracked confidence lives in conf_stack.
        rr.log(
            f"{video_log_path}/keypoints",
            Points2DWithConfidence(
                positions=uv[0, :, 0:2],  # Remove the view dimension
                confidences=current_conf.squeeze(),
                class_ids=0,
                keypoint_ids=AVP_IDS,
                show_labels=False,
            ),
        )
