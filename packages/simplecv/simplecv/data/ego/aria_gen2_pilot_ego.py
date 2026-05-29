"""Egocentric view loader for Aria Gen2 Pilot dataset.

Aria Gen2 has 5 cameras (RGB + 4 SLAM) with different labels from Gen1.
Reuses HOT3D calibration and trajectory parsers since the MPS format is identical.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.data.hot3d_utils import (
    AriaSequenceCalibration,
    AriaStreamCalibration,
    load_calibration,
    lookup_nearest_poses,
    parse_mps_closed_loop_trajectory,
    parse_online_calibration_first,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.aria_gen2_pilot import AriaGen2PilotConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as AriaGen2PilotConfig

# Aria Gen2 ego camera streams: label → MP4 filename stem.
# Gen2 has 4 SLAM cameras (front-left/right, side-left/right) instead of Gen1's 2.
ARIA_GEN2_EGO_STREAMS: dict[str, str] = {
    "camera-rgb": "rgb",
    "slam-front-left": "slam_front_left",
    "slam-front-right": "slam_front_right",
    "slam-side-left": "slam_side_left",
    "slam-side-right": "slam_side_right",
}

SIMPLECV_DIR: str = "_simplecv"


class AriaGen2PilotEgoSequence(BaseEgoSequence[AriaGen2PilotConfig]):
    """Egocentric view loader for Aria Gen2 Pilot."""

    def __init__(self, cfg: AriaGen2PilotConfig) -> None:
        self._ego_streams: dict[str, str] = ARIA_GEN2_EGO_STREAMS
        super().__init__(cfg)

    def __len__(self) -> int:
        assert len(self.ego_video_readers) > 0, "No videos found."
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        raise NotImplementedError("Aria Gen2 Pilot ego data loading not implemented yet.")

    def _sequence_dir(self) -> Path:
        return Path(self.config.base_directory) / self.config.sequence_name

    def load_video_paths(self) -> list[Path]:
        """Load paths to preprocessed AV1 MP4 videos for all ego cameras."""
        seq_dir: Path = self._sequence_dir()
        simplecv_dir: Path = seq_dir / SIMPLECV_DIR

        video_paths: list[Path] = []
        for label, stem in self._ego_streams.items():
            video_path: Path = simplecv_dir / f"{stem}.mp4"
            assert video_path.exists(), (
                f"Preprocessed video for '{label}' not found at {video_path}. "
                f"Run: pixi run preprocess-aria-gen2-pilot --root {self.config.base_directory} "
                f"--sequence {self.config.sequence_name} --no-skip-existing"
            )
            video_paths.append(video_path)

        return video_paths

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        """Load per-frame fisheye camera parameters for all ego cameras.

        Calibration: online_calibration.jsonl (MPS-refined).
        Trajectory: MPS closed_loop_trajectory.csv (device-time domain, 1kHz).
        """
        seq_dir: Path = self._sequence_dir()

        # Load calibration
        cal: AriaSequenceCalibration = self._load_calibration(seq_dir)
        cal_by_label: dict[str, AriaStreamCalibration] = {s.stream_label: s for s in cal.streams}

        # Load trajectory
        traj_result: tuple[Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"]] = self._load_trajectory(seq_dir)
        traj_ts_ns: Int64[ndarray, "n_poses"] = traj_result[0]
        world_T_device_all: Float32[ndarray, "n_poses 4 4"] = traj_result[1]

        # Load VRS device-time timestamps per stream
        vrs_ts_path: Path = seq_dir / SIMPLECV_DIR / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())

        # Use RGB timestamps as the canonical reference for ALL cameras.
        # Native rates: RGB 10fps, SLAM cameras 30fps, hand tracking 30fps,
        # SLAM trajectory 1kHz.  The SLAM *videos* play at their native
        # 30fps, but camera *poses* are sampled at the RGB 10fps rate
        # because the viewer assumes all ego cameras share the same frame
        # count.  By looking up trajectory poses at RGB timestamps for
        # every camera, all cameras get identical frame counts and
        # temporal alignment.
        # TODO: support per-stream native frame rates so SLAM camera poses,
        # SLAM video frames, and hand labels all run at their full 30fps
        # instead of being downsampled to the RGB 10fps timeline.
        rgb_label: str = "camera-rgb"
        assert rgb_label in vrs_ts_data, "No RGB timestamps in timestamps_ns.json"
        canonical_device_ts: Int64[ndarray, "n_frames"] = np.array(vrs_ts_data[rgb_label], dtype=np.int64)
        n_frames: int = len(canonical_device_ts)

        world_T_device_frames: Float32[ndarray, "n_frames 4 4"] = lookup_nearest_poses(
            query_ts_ns=canonical_device_ts,
            trajectory_ts_ns=traj_ts_ns,
            world_T_device=world_T_device_all,
        )

        # Build per-frame camera params for each stream
        all_cam_dict: dict[str, list[Fisheye62Parameters]] = {}

        for label in self._ego_streams:
            stream_cal: AriaStreamCalibration | None = cal_by_label.get(label)
            assert stream_cal is not None, f"No calibration found for stream '{label}'"

            device_T_camera: Float32[ndarray, "4 4"] = np.array(stream_cal.device_T_camera, dtype=np.float32)

            intrinsics: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=stream_cal.fl_x,
                fl_y=stream_cal.fl_y,
                cx=stream_cal.cx,
                cy=stream_cal.cy,
                height=stream_cal.height,
                width=stream_cal.width,
            )
            distortion: KannalaBrandtDistortion = KannalaBrandtDistortion(
                k1=stream_cal.k1, k2=stream_cal.k2, k3=stream_cal.k3,
                k4=stream_cal.k4, k5=stream_cal.k5, k6=stream_cal.k6,
                p1=stream_cal.p1, p2=stream_cal.p2,
            )

            cam_list: list[Fisheye62Parameters] = []
            prev_cam_T_world: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)

            for frame_idx in range(n_frames):
                world_T_device: Float32[ndarray, "4 4"] = world_T_device_frames[frame_idx]
                world_T_camera: Float32[ndarray, "4 4"] = world_T_device @ device_T_camera

                try:
                    cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_camera)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        f"Singular world_T_camera for '{label}' at frame {frame_idx}, "
                        f"reusing previous pose.",
                        stacklevel=2,
                    )
                    cam_T_world = prev_cam_T_world
                else:
                    if np.all(np.isfinite(cam_T_world)):
                        prev_cam_T_world = cam_T_world
                    else:
                        warnings.warn(
                            f"Non-finite cam_T_world for '{label}' at frame {frame_idx}, "
                            f"reusing previous pose.",
                            stacklevel=2,
                        )
                        cam_T_world = prev_cam_T_world

                extrinsics: Extrinsics = Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                )
                cam_list.append(Fisheye62Parameters(
                    name=label, intrinsics=intrinsics, distortion=distortion, extrinsics=extrinsics,
                ))

            all_cam_dict[label] = cam_list

        return all_cam_dict

    def _load_calibration(self, seq_dir: Path) -> AriaSequenceCalibration:
        """Load calibration, preferring preprocessed cache with correct dimensions."""
        preprocessed: Path = seq_dir / SIMPLECV_DIR / "calibration.json"
        if preprocessed.exists():
            return load_calibration(preprocessed)

        online_cal: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
        assert online_cal.exists(), f"No calibration source found in {seq_dir}"
        return parse_online_calibration_first(online_cal)

    def _load_trajectory(self, seq_dir: Path) -> tuple[Int64[ndarray, "n"], Float32[ndarray, "n 4 4"]]:
        """Load MPS SLAM closed-loop trajectory (device-time domain, 1kHz)."""
        mps_traj: Path = seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv"
        assert mps_traj.exists(), (
            f"MPS trajectory not found at {mps_traj}. "
            f"Ensure mps_slam_trajectories was downloaded."
        )
        mps_result: tuple[
            Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"], Float32[ndarray, "n_poses"]
        ] = parse_mps_closed_loop_trajectory(mps_traj)
        ts_ns: Int64[ndarray, "n_poses"] = mps_result[0]
        world_T_device: Float32[ndarray, "n_poses 4 4"] = mps_result[1]
        return ts_ns, world_T_device

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[Fisheye62Parameters]]
    ) -> tuple[dict[str, list[Fisheye62Parameters]], dict[str, Path]]:
        """Align cameras and videos by stream label order."""
        stream_labels: list[str] = list(self._ego_streams.keys())
        assert len(video_path_list) == len(stream_labels), (
            f"Expected {len(stream_labels)} ego videos, got {len(video_path_list)}"
        )

        aligned_cam_dict: dict[str, list[Fisheye62Parameters]] = {}
        aligned_video_map: dict[str, Path] = {}

        for idx, label in enumerate(stream_labels):
            assert label in ego_cam_dict, f"Camera '{label}' missing from ego camera dictionary"
            aligned_cam_dict[label] = ego_cam_dict[label]
            aligned_video_map[label] = video_path_list[idx]

        return aligned_cam_dict, aligned_video_map

    @property
    def image_plane_distance(self) -> int | float:
        """Image plane distance for camera visualization in meters."""
        return 0.03
