"""Egocentric view loader for HOT3D (Aria and Quest 3).

Supports Aria (3 cameras: RGB + 2 SLAM) and Quest 3 (2 cameras: SLAM only).
Calibration and trajectory sources are headset-aware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.data.hot3d_utils import (
    Hot3dSequenceCalibration,
    Hot3dStreamCalibration,
    detect_headset,
    load_calibration,
    lookup_nearest_poses,
    parse_camera_models_json,
    parse_headset_trajectory,
    parse_mps_closed_loop_trajectory,
    parse_online_calibration_first,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.hot3d import Hot3dConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as Hot3dConfig

# Ego camera streams per headset: label → MP4 filename stem.
ARIA_EGO_STREAMS: dict[str, str] = {
    "camera-rgb": "rgb",
    "camera-slam-left": "slam_left",
    "camera-slam-right": "slam_right",
}
QUEST_EGO_STREAMS: dict[str, str] = {
    "camera-slam-left": "slam_left",
    "camera-slam-right": "slam_right",
}

SIMPLECV_DIR: str = "_simplecv"


def _ego_streams_for_headset(headset: str) -> dict[str, str]:
    """Return the ego stream mapping for the given headset type."""
    if headset == "Quest3":
        return QUEST_EGO_STREAMS
    return ARIA_EGO_STREAMS


class Hot3dEgoSequence(BaseEgoSequence[Hot3dConfig]):
    """Egocentric view loader for HOT3D (Aria and Quest 3)."""

    def __init__(self, cfg: Hot3dConfig) -> None:
        seq_dir: Path = Path(cfg.root_directory) / cfg.sequence_name
        self._headset: str = detect_headset(seq_dir)
        self._ego_streams: dict[str, str] = _ego_streams_for_headset(self._headset)
        super().__init__(cfg)

    def __len__(self) -> int:
        assert len(self.ego_video_readers) > 0, "No videos found."
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        raise NotImplementedError("Hot3D ego data loading not implemented yet.")

    def _sequence_dir(self) -> Path:
        return Path(self.config.root_directory) / self.config.sequence_name

    def load_video_paths(self) -> list[Path]:
        """Load paths to preprocessed AV1 MP4 videos for all ego cameras."""
        seq_dir: Path = self._sequence_dir()
        simplecv_dir: Path = seq_dir / SIMPLECV_DIR

        video_paths: list[Path] = []
        for label, stem in self._ego_streams.items():
            video_path: Path = simplecv_dir / f"{stem}.mp4"
            assert video_path.exists(), (
                f"Preprocessed video for '{label}' not found at {video_path}. "
                f"Run: pixi run preprocess-hot3d --root {self.config.root_directory} "
                f"--sequence {self.config.sequence_name} --no-skip-existing"
            )
            video_paths.append(video_path)

        return video_paths

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        """Load per-frame fisheye camera parameters for all ego cameras.

        Calibration source hierarchy:
        - **Aria**: Prefer ``online_calibration.jsonl`` (MPS-refined, per-timestamp).
          Meta invested in online calibration refinement for Aria sequences, so this
          is the most accurate source. Falls back to ``camera_models.json`` if missing.
        - **Quest 3**: Use ``camera_models.json`` (no MPS available).

        Trajectory source:
        - **Aria**: ``mps/slam/closed_loop_trajectory.csv`` (device-time domain).
        - **Quest 3**: ``headset_trajectory.csv`` (native VRS time domain).
        """
        seq_dir: Path = self._sequence_dir()

        # ── Load calibration ─────────────────────────────────────────────
        cal: Hot3dSequenceCalibration = self._load_calibration(seq_dir)
        cal_by_label: dict[str, Hot3dStreamCalibration] = {s.stream_label: s for s in cal.streams}

        # ── Load trajectory ──────────────────────────────────────────────
        traj_ts_ns: Int64[ndarray, "n_poses"]
        world_T_device_all: Float32[ndarray, "n_poses 4 4"]
        traj_ts_ns, world_T_device_all = self._load_trajectory(seq_dir)

        # ── Load VRS device-time timestamps per stream ───────────────────
        vrs_ts_path: Path = seq_dir / SIMPLECV_DIR / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())

        # ── Build per-frame camera params for each stream ────────────────
        all_cam_dict: dict[str, list[Fisheye62Parameters]] = {}

        for label in self._ego_streams:
            stream_cal: Hot3dStreamCalibration | None = cal_by_label.get(label)
            assert stream_cal is not None, f"No calibration found for stream '{label}'"
            assert label in vrs_ts_data, (
                f"No timestamps for stream '{label}' in timestamps_ns.json. "
                f"Re-run preprocessing: pixi run preprocess-hot3d --root {self.config.root_directory} "
                f"--sequence {self.config.sequence_name} --no-skip-existing"
            )

            device_T_camera: Float32[ndarray, "4 4"] = np.array(stream_cal.device_T_camera, dtype=np.float32)
            stream_device_ts: Int64[ndarray, "n_frames"] = np.array(vrs_ts_data[label], dtype=np.int64)
            n_frames: int = len(stream_device_ts)

            world_T_device_frames: Float32[ndarray, "n_frames 4 4"] = lookup_nearest_poses(
                query_ts_ns=stream_device_ts,
                trajectory_ts_ns=traj_ts_ns,
                world_T_device=world_T_device_all,
            )

            intrinsics: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=stream_cal.fl_x,
                fl_y=stream_cal.fl_y,
                cx=stream_cal.cx,
                cy=stream_cal.cy,
                height=stream_cal.height,
                width=stream_cal.width,
            )
            # FISHEYE624 → Fisheye62: use k1-k6 radial + p1-p2 tangential.
            # Thin-prism terms (s1-s4) from the full FISHEYE624 model are
            # dropped — validated <1px error on HOT3D data.
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
                    cam_T_world = prev_cam_T_world
                else:
                    if np.all(np.isfinite(cam_T_world)):
                        prev_cam_T_world = cam_T_world
                    else:
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

    def _load_calibration(self, seq_dir: Path) -> Hot3dSequenceCalibration:
        """Load calibration with headset-aware source selection.

        Aria: prefer online_calibration.jsonl (MPS-refined per-timestamp intrinsics)
        over camera_models.json (factory). Meta's online calibration pipeline
        refines intrinsics during SLAM — while current measurements show <0.01px
        difference, the refined version is the intended source for Aria.

        Quest 3: use camera_models.json (no MPS available).
        """
        # Check for preprocessed calibration first (from earlier preprocessing run)
        preprocessed: Path = seq_dir / SIMPLECV_DIR / "calibration.json"
        if preprocessed.exists():
            return load_calibration(preprocessed)

        # Aria: prefer online_calibration.jsonl (MPS-refined)
        online_cal: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
        if online_cal.exists():
            return parse_online_calibration_first(online_cal)

        # Fallback (Quest 3 or Aria without MPS): camera_models.json
        cam_models: Path = seq_dir / "camera_models.json"
        assert cam_models.exists(), f"No calibration source found in {seq_dir}"
        return parse_camera_models_json(cam_models)

    def _load_trajectory(self, seq_dir: Path) -> tuple[Int64[ndarray, "n"], Float32[ndarray, "n 4 4"]]:
        """Load device trajectory with headset-aware source and time domain.

        Aria: MPS closed_loop_trajectory.csv (device-time domain, 1kHz).
        Quest 3: headset_trajectory.csv (native VRS time domain, ~30Hz).
        """
        # Aria: prefer MPS SLAM trajectory (higher rate, device-time domain)
        mps_traj: Path = seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv"
        if mps_traj.exists():
            mps_result: tuple[
                Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"], Float32[ndarray, "n_poses"]
            ] = parse_mps_closed_loop_trajectory(mps_traj)
            ts_ns: Int64[ndarray, "n_poses"] = mps_result[0]
            world_T_device: Float32[ndarray, "n_poses 4 4"] = mps_result[1]
            return ts_ns, world_T_device

        # Quest 3 (or Aria without MPS): headset_trajectory.csv
        headset_traj: Path = seq_dir / "headset_trajectory.csv"
        assert headset_traj.exists(), f"No trajectory source found in {seq_dir}"
        return parse_headset_trajectory(headset_traj)

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
