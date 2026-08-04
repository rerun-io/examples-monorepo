"""Aria Gen2 Pilot dataset adapter for the ExoEgo visualization pipeline.

Ego-only (Aria device, no exo cameras). Uses preprocessed MP4 videos
(from H.265 VRS), MPS SLAM trajectories for camera poses, and MPS hand
tracking providing full 21-landmark 3D hand keypoints in device frame.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int, Int64
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from tqdm import tqdm

from simplecv.configs.dataset_paths import ARIA_GEN2_PILOT_ROOT
from simplecv.data.ego.aria_gen2_pilot_ego import AriaGen2PilotEgoSequence
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.hot3d_utils import lookup_nearest_poses, parse_mps_closed_loop_trajectory
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133


@dataclass
class AriaGen2PilotConfig(BaseExoEgoDatasetConfig):
    """Configuration for Aria Gen2 Pilot sequences."""

    _target: type = field(default_factory=lambda: AriaGen2PilotSequence)
    base_directory: Path = ARIA_GEN2_PILOT_ROOT
    """Base directory containing sequence subdirectories."""
    sequence_name: str = "cook_0"
    """Sequence folder name (e.g. 'walk_1', 'cook_0', 'eat_0')."""


class AriaGen2PilotSequence(BaseExoEgoSequence[AriaGen2PilotConfig]):
    """Aria Gen2 Pilot dataset adapter (ego-only, Aria device)."""

    def __init__(self, cfg: AriaGen2PilotConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    def _sequence_dir(self) -> Path:
        return Path(self.config.base_directory) / self.config.sequence_name

    @classmethod
    def sequence_identity_for_config(cls, cfg: AriaGen2PilotConfig) -> SequenceIdentity:
        return SequenceIdentity(dataset="aria-gen2", parts=(cfg.sequence_name,))

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=None,
            exo_bgr_list=None,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[AriaGen2PilotConfig] | None:
        return AriaGen2PilotEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[AriaGen2PilotConfig] | None:
        return None  # ego-only

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego videos (and labels if available)."""
        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()

        if self.ego_sequence is not None:
            for name, video_path in zip(
                self.ego_sequence.ego_video_names,
                self.ego_sequence.ego_video_paths,
                strict=True,
            ):
                stream_name: str = f"ego/{name}"
                timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """Load COCO-133 hand keypoints from MPS hand tracking results.

        MPS hand tracking provides 21 landmarks per hand in **device frame**
        (meters), using the same ordering as Assembly-Hands / UmeTrack.
        We transform them to world frame using the MPS SLAM trajectory and
        then map to COCO-133 via ``assembly21_to_coco133``.

        Timestamp alignment:
        - MPS hand tracking timestamps are in device-time ``tracking_timestamp_us``
          (same domain as the MPS SLAM trajectory).
        - We filter to one label per video frame via nearest-neighbor matching
          against the VRS device-time timestamps saved during preprocessing.
        - Label timestamps are then normalized to the 0-based video timeline
          (subtract VRS recording start time).
        """
        seq_dir: Path = self._sequence_dir()
        hand_csv: Path = seq_dir / "mps" / "hand_tracking" / "hand_tracking_results.csv"
        if not hand_csv.exists():
            return None

        import pandas as pd

        df: pd.DataFrame = pd.read_csv(hand_csv)
        if df.empty:
            return None

        # ── Timestamps: μs → ns (device-time domain) ─────────────────────
        hand_ts_ns: Int64[ndarray, "n_hand"] = (df["tracking_timestamp_us"].values * np.int64(1000)).astype(np.int64)

        # ── Load trajectory for device→world transform ────────────────────
        traj_csv: Path = seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv"
        assert traj_csv.exists(), f"Trajectory not found at {traj_csv}"
        traj_result: tuple[
            Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"], Float32[ndarray, "n_poses"]
        ] = parse_mps_closed_loop_trajectory(traj_csv)
        traj_ts_ns: Int64[ndarray, "n_poses"] = traj_result[0]
        world_T_device_all: Float32[ndarray, "n_poses 4 4"] = traj_result[1]

        # Look up world_T_device for every hand tracking frame
        world_T_device_hand: Float32[ndarray, "n_hand 4 4"] = lookup_nearest_poses(
            query_ts_ns=hand_ts_ns,
            trajectory_ts_ns=traj_ts_ns,
            world_T_device=world_T_device_all,
        )

        # ── Extract 21 landmarks per hand (device frame, meters) ──────────
        n_hand: int = len(df)
        left_conf: Float32[ndarray, "n_hand"] = df["left_tracking_confidence"].values.astype(np.float32)
        right_conf: Float32[ndarray, "n_hand"] = df["right_tracking_confidence"].values.astype(np.float32)

        # Build (n_hand, 21, 3) arrays for left and right in device frame
        left_device: Float32[ndarray, "n_hand 21 3"] = np.zeros((n_hand, 21, 3), dtype=np.float32)
        right_device: Float32[ndarray, "n_hand 21 3"] = np.zeros((n_hand, 21, 3), dtype=np.float32)
        for lm_idx in range(21):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                left_col: str = f"t{axis}_left_landmark_{lm_idx}_device"
                right_col: str = f"t{axis}_right_landmark_{lm_idx}_device"
                left_device[:, lm_idx, axis_idx] = df[left_col].values.astype(np.float32)
                right_device[:, lm_idx, axis_idx] = df[right_col].values.astype(np.float32)

        # ── Transform device→world ────────────────────────────────────────
        # world_point = R @ device_point + t
        R_all: Float32[ndarray, "n_hand 3 3"] = world_T_device_hand[:, :3, :3]
        t_all: Float32[ndarray, "n_hand 3"] = world_T_device_hand[:, :3, 3]

        # Vectorized: (n_hand, 21, 3) = einsum('nij,nkj->nki', R, pts) + t[:,None,:]
        left_world: Float32[ndarray, "n_hand 21 3"] = np.einsum("nij,nkj->nki", R_all, left_device) + t_all[:, np.newaxis, :]
        right_world: Float32[ndarray, "n_hand 21 3"] = np.einsum("nij,nkj->nki", R_all, right_device) + t_all[:, np.newaxis, :]

        # Zero out invalid detections (confidence <= 0)
        left_invalid: ndarray = left_conf <= 0
        right_invalid: ndarray = right_conf <= 0
        left_world[left_invalid] = np.nan
        right_world[right_invalid] = np.nan

        # ── Filter to video-frame-aligned entries ─────────────────────────
        # Hand tracking and SLAM cameras both run at 30fps, but we
        # downsample to the RGB 10fps timeline (one label per RGB frame)
        # because the viewer shares a single frame count across all ego
        # cameras and labels.
        # TODO: emit labels at native 30fps once the viewer supports
        # per-stream frame rates.
        vrs_ts_path: Path = seq_dir / "_simplecv" / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())
        rgb_stream_name: str = "camera-rgb"
        if rgb_stream_name not in vrs_ts_data:
            available_streams: list[str] = list(vrs_ts_data.keys())
            raise KeyError(
                f"RGB timestamps missing from {vrs_ts_path}: expected key "
                f"{rgb_stream_name!r}, found {available_streams!r}"
            )
        vrs_ref_ts: Int64[ndarray, "n_video"] = np.array(vrs_ts_data[rgb_stream_name], dtype=np.int64)

        # Nearest-neighbor: for each video frame, find closest hand tracking frame
        insertion: Int64[ndarray, "n_video"] = np.searchsorted(hand_ts_ns, vrs_ref_ts, side="left").astype(np.int64)
        left_idx: Int64[ndarray, "n_video"] = np.clip(insertion - 1, 0, n_hand - 1).astype(np.int64)
        right_idx: Int64[ndarray, "n_video"] = np.clip(insertion, 0, n_hand - 1).astype(np.int64)
        left_delta: Int64[ndarray, "n_video"] = np.abs(vrs_ref_ts - hand_ts_ns[left_idx])
        right_delta: Int64[ndarray, "n_video"] = np.abs(hand_ts_ns[right_idx] - vrs_ref_ts)
        aligned_idx: Int64[ndarray, "n_video"] = np.where(left_delta <= right_delta, left_idx, right_idx).astype(np.int64)

        left_aligned: Float32[ndarray, "n_video 21 3"] = left_world[aligned_idx]
        right_aligned: Float32[ndarray, "n_video 21 3"] = right_world[aligned_idx]
        left_conf_aligned: Float32[ndarray, "n_video"] = left_conf[aligned_idx]
        right_conf_aligned: Float32[ndarray, "n_video"] = right_conf[aligned_idx]

        # ── Map to COCO-133 ───────────────────────────────────────────────
        num_frames: int = len(vrs_ref_ts)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[:, :, 3] = np.float32(0.0)

        n_gaps_left: int = 0
        n_gaps_right: int = 0

        for frame_idx in range(num_frames):
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)

            # Left hand — NaN if not detected (confidence <= 0)
            l_conf: float = float(left_conf_aligned[frame_idx])
            if l_conf > 0:
                landmarks_lr[0] = left_aligned[frame_idx]
            else:
                n_gaps_left += 1

            # Right hand — NaN if not detected (confidence <= 0)
            r_conf: float = float(right_conf_aligned[frame_idx])
            if r_conf > 0:
                landmarks_lr[1] = right_aligned[frame_idx]
            else:
                n_gaps_right += 1

            xyzc_stack[frame_idx] = assembly21_to_coco133(landmarks_lr)

            # Set confidence for hand keypoints (matching HOT3D pattern)
            adjustments: tuple[tuple[int, int], ...] = ((0, 91), (1, 112))
            wrist_indices: tuple[int, int] = (9, 10)
            thumb_base_indices: tuple[int, int] = (92, 113)
            for hand_idx, coco_offset in adjustments:
                conf: float = float(left_conf_aligned[frame_idx]) if hand_idx == 0 else float(right_conf_aligned[frame_idx])
                conf = max(conf, 0.0)
                if conf <= 0.0:
                    xyzc_stack[frame_idx, coco_offset: coco_offset + 21, 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(0.0)
                else:
                    xyzc_stack[frame_idx, coco_offset: coco_offset + 21, 3] = np.float32(conf)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(conf)
                    if not np.isnan(xyzc_stack[frame_idx, thumb_base_indices[hand_idx], :3]).all():
                        xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(conf)

        if self.config.verbose and n_gaps_left > 0:
            warnings.warn(
                f"Left hand not detected in {n_gaps_left}/{num_frames} frames "
                f"({n_gaps_left / num_frames * 100:.0f}%). Those frames have NaN keypoints.",
                stacklevel=2,
            )
        if self.config.verbose and n_gaps_right > 0:
            warnings.warn(
                f"Right hand not detected in {n_gaps_right}/{num_frames} frames "
                f"({n_gaps_right / num_frames * 100:.0f}%). Those frames have NaN keypoints.",
                stacklevel=2,
            )

        # Labels are aligned one-per-video-frame, so their normalized
        # timestamps follow the video timeline exactly rather than the
        # nearest hand sample timestamps (which may have small jitter).
        normalized_ts: Int64[ndarray, "num_frames"] = (vrs_ref_ts - vrs_ref_ts[0]).astype(np.int64)

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=normalized_ts,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: AriaGen2PilotConfig) -> Generator["AriaGen2PilotSequence", None, None]:
        """Iterate over all sequences in the dataset directory.

        Yields one ``AriaGen2PilotSequence`` per folder with preprocessed MP4s.
        """
        for seq_dir in cls._iter_sequence_dirs(cfg):
            episode_cfg: AriaGen2PilotConfig = replace(
                cfg,
                sequence_name=seq_dir.name,
            )
            try:
                yield cls(episode_cfg)
            except Exception as exc:  # pragma: no cover
                tqdm.write(f"[skip] {seq_dir.name}: {exc}")

    @classmethod
    def num_sequences_for_config(cls, cfg: AriaGen2PilotConfig) -> int:
        return len(cls._iter_sequence_dirs(cfg))

    @staticmethod
    def _iter_sequence_dirs(cfg: AriaGen2PilotConfig) -> list[Path]:
        root: Path = cfg.base_directory
        assert root.exists(), f"Aria Gen2 Pilot root directory {root} does not exist."

        def _has_preprocessed_streams(d: Path) -> bool:
            simplecv_dir: Path = d / "_simplecv"
            return simplecv_dir.is_dir() and any(simplecv_dir.glob("*.mp4"))

        return natsorted([d for d in root.iterdir() if d.is_dir() and _has_preprocessed_streams(d)])

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Aria: gravity = [0,0,-9.81] → +Z is up."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP
