"""HOT3D dataset adapter for the ExoEgo visualization pipeline (Aria + Quest 3).

Ego-only (no exo cameras), following the UmeTrack pattern. Uses preprocessed
AV1 MP4 videos (from VRS), device trajectories for camera poses, and
UmeTrack-format hand annotations. Supports both Aria and Quest 3 headsets.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int, Int64
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from tqdm import tqdm

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.hot3d_ego import Hot3dEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.hot3d_utils import (
    build_4x4,
    detect_headset,
    load_timecode_to_devicetime_mapping,
    quat_wxyz_to_matrix,
)
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, SingleHandPose, landmarks_from_hand_pose


@dataclass
class Hot3dConfig(BaseExoEgoDatasetConfig):
    """Configuration for HOT3D sequences (Aria and Quest 3)."""

    _target: type = field(default_factory=lambda: Hot3dSequence)
    base_directory: Path = Path("/mnt/8tb/data/hot3d")
    """Base directory containing ``aria/`` and ``quest3/`` subdirectories."""
    headset: Literal["aria", "quest3"] = "aria"
    """Headset type. Determines subdirectory, camera streams, and trajectory source."""
    sequence_name: str = "P0001_10a27bf7"
    """Sequence folder name (e.g. 'P0001_10a27bf7')."""

    @property
    def root_directory(self) -> Path:
        """Computed root: ``base_directory / headset``."""
        return self.base_directory / self.headset


class Hot3dSequence(BaseExoEgoSequence[Hot3dConfig]):
    """HOT3D Aria dataset adapter emitting 3D hand annotations in meters."""

    def __init__(self, cfg: Hot3dConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        seq_dir: Path = Path(cfg.root_directory) / cfg.sequence_name
        self._headset: str = detect_headset(seq_dir)
        super().__init__(cfg)

    def _sequence_dir(self) -> Path:
        return Path(self.config.root_directory) / self.config.sequence_name

    @classmethod
    def sequence_identity_for_config(cls, cfg: Hot3dConfig) -> SequenceIdentity:
        return SequenceIdentity(dataset=f"hot3d-{cfg.headset}", parts=(cfg.sequence_name,))

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

    def _build_ego(self) -> BaseEgoSequence[Hot3dConfig] | None:
        return Hot3dEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Hot3dConfig] | None:
        return None  # ego-only (like UmeTrack)

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

    @staticmethod
    def _has_hand_labels(seq_dir: Path) -> bool:
        """Return whether this sequence has usable hand pose labels."""
        metadata_path: Path = seq_dir / "metadata.json"
        has_hand_gt: bool | None = None
        if metadata_path.exists():
            metadata: dict = json.loads(metadata_path.read_text())
            if "have_hand_object_pose_gt" in metadata:
                has_hand_gt = bool(metadata["have_hand_object_pose_gt"])

        if has_hand_gt is False:
            return False

        required_paths: tuple[Path, ...] = (
            seq_dir / "umetrack_hand_user_profile.json",
            seq_dir / "umetrack_hand_pose_trajectory.jsonl",
        )
        missing_or_empty: list[Path] = [
            path for path in required_paths if not path.exists() or path.stat().st_size == 0
        ]
        if not missing_or_empty:
            return True

        if has_hand_gt is True:
            missing_text: str = ", ".join(str(path) for path in missing_or_empty)
            raise AssertionError(f"Hand GT is marked available but required label files are missing or empty: {missing_text}")
        return False

    def load_labels(self) -> ExoEgoLabels | None:
        """Load COCO-133 hand keypoints in meters from HOT3D UmeTrack annotations.

        HOT3D stores hand annotations in UmeTrack JSONL format:
        - ``umetrack_hand_pose_trajectory.jsonl``: per-frame hand poses
        - ``umetrack_hand_user_profile.json``: hand model definition

        The hand model rest positions are in millimeters (same as UmeTrack).
        The wrist_xform translation in HOT3D is in meters, so we convert it to
        mm before FK, then scale the output back to meters.
        """
        seq_dir: Path = self._sequence_dir()
        if not self._has_hand_labels(seq_dir):
            return None

        # ── Load hand model ──────────────────────────────────────────────
        profile_path: Path = seq_dir / "umetrack_hand_user_profile.json"
        assert profile_path.exists(), f"Hand user profile not found at {profile_path}"

        from serde import from_dict

        # HOT3D wraps the hand model in {"hand_model": {...}} envelope
        profile_data: dict = json.loads(profile_path.read_text())
        hand_model: HandModelNumpy = from_dict(HandModelNumpy, profile_data["hand_model"])

        # ── Parse JSONL hand annotations ─────────────────────────────────
        annotations_path: Path = seq_dir / "umetrack_hand_pose_trajectory.jsonl"
        assert annotations_path.exists(), f"Hand annotations not found at {annotations_path}"

        frame_data: list[dict] = []
        with open(annotations_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frame_data.append(json.loads(line))

        # ── Map JSONL timestamps to device-time domain ────────────────────
        # Aria: JSONL uses "timecode" domain, VRS uses "device time".
        #   The timecode_devicetime_mapping.csv provides the 1:1 translation.
        # Quest 3: JSONL timestamps are already in the same domain as VRS.
        mapping_path: Path = seq_dir / "timecode_devicetime_mapping.csv"
        if mapping_path.exists():
            # Aria: convert timecode → device-time
            devicetime_ns_all: Int64[ndarray, "n_entries"] = load_timecode_to_devicetime_mapping(mapping_path)
            assert len(devicetime_ns_all) == len(frame_data), (
                f"Timecode mapping has {len(devicetime_ns_all)} entries but JSONL has {len(frame_data)}"
            )
        else:
            # Quest 3: JSONL timestamps are already in device-time domain
            devicetime_ns_all = np.array([entry["timestamp_ns"] for entry in frame_data], dtype=np.int64)

        # ── Filter to video-frame-aligned entries only ────────────────────
        # The JSONL may have more entries than video frames (Aria: ~2x, one
        # per SLAM camera; Quest: ~1:1).  Keep nearest entry per video frame.
        vrs_ts_path: Path = seq_dir / "_simplecv" / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())
        # Use the first available stream's timestamps as reference
        first_stream: str = next(iter(vrs_ts_data))
        vrs_ref_ts: Int64[ndarray, "n_video"] = np.array(vrs_ts_data[first_stream], dtype=np.int64)

        # For each video frame, find the nearest label by device-time.
        # searchsorted gives the first index >= query; compare with the left
        # neighbor to pick the truly closest timestamp.
        insertion_indices: Int64[ndarray, "n_video"] = np.searchsorted(
            devicetime_ns_all, vrs_ref_ts, side="left"
        ).astype(np.int64)
        left_indices: Int64[ndarray, "n_video"] = np.clip(insertion_indices - 1, 0, len(devicetime_ns_all) - 1).astype(
            np.int64
        )
        right_indices: Int64[ndarray, "n_video"] = np.clip(insertion_indices, 0, len(devicetime_ns_all) - 1).astype(
            np.int64
        )
        left_deltas: Int64[ndarray, "n_video"] = np.abs(vrs_ref_ts - devicetime_ns_all[left_indices])
        right_deltas: Int64[ndarray, "n_video"] = np.abs(devicetime_ns_all[right_indices] - vrs_ref_ts)
        rgb_label_indices: Int64[ndarray, "n_video"] = np.where(
            left_deltas <= right_deltas, left_indices, right_indices
        ).astype(np.int64)

        frame_data_filtered: list[dict] = [frame_data[int(i)] for i in rgb_label_indices]
        devicetime_ns_filtered: Int64[ndarray, "n_video"] = devicetime_ns_all[rgb_label_indices]

        num_frames: int = len(frame_data_filtered)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[:, :, 3] = np.float32(0.0)

        prev_landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)

        # ── Per-frame UmeTrack hand forward kinematics ───────────────────
        # For each filtered frame, convert UmeTrack wrist_xform + joint_angles
        # into 3D hand landmarks in world frame (meters), then map to COCO-133.
        # The UmeTrack hand model is in mm, so wrist translation is converted
        # to mm before FK, and output is scaled back to meters.
        for frame_idx, entry in enumerate(frame_data_filtered):
            hand_poses: dict = entry.get("hand_poses", {})
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
            hand_confidences: Float32[ndarray, "2"] = np.zeros(2, dtype=np.float32)

            # HOT3D JSONL key "0" = Left hand, "1" = Right hand
            # (verified by comparing wrist positions against HOT3D clips GT)
            # simplecv: LEFT_HAND_INDEX=0, RIGHT_HAND_INDEX=1
            for hand_key, hand_idx in [("0", 0), ("1", 1)]:
                if hand_key not in hand_poses:
                    continue

                pose_data: dict = hand_poses[hand_key]
                confidence: float = float(pose_data.get("hand_confidence", 0.0))
                hand_confidences[hand_idx] = np.float32(confidence)

                if confidence > 0.0:
                    # Step 1: Build wrist 4x4 transform (world_T_wrist) from quaternion + translation.
                    # Translation is in meters in the JSONL but the hand model is in mm.
                    wrist_data: dict = pose_data["wrist_xform"]
                    q_wxyz: list[float] = wrist_data["q_wxyz"]
                    t_xyz: list[float] = wrist_data["t_xyz"]
                    R_wrist: Float32[ndarray, "3 3"] = quat_wxyz_to_matrix(q_wxyz)
                    t_xyz_mm: list[float] = [v * 1000.0 for v in t_xyz]
                    wrist_xform: Float32[ndarray, "4 4"] = build_4x4(R_wrist, t_xyz_mm)

                    # Step 2: Run UmeTrack forward kinematics to get 21 hand landmarks.
                    # landmarks_from_hand_pose applies the hand model skeleton + skinning
                    # with X-flip for right hand (hand_idx=1).
                    joint_angles: Float32[ndarray, "22"] = np.array(pose_data["joint_angles"], dtype=np.float32)
                    hand_pose: SingleHandPose = SingleHandPose(
                        joint_angles=joint_angles,
                        wrist_xform=wrist_xform,
                        hand_confidence=confidence,
                    )
                    landmarks_mm: Float32[ndarray, "21 3"] = landmarks_from_hand_pose(
                        hand_model, hand_pose, hand_idx
                    ).astype(np.float32, copy=False)

                    # Step 3: Convert mm → meters for world-frame output.
                    scale_to_meters: float = 1e-3
                    landmarks_world: Float32[ndarray, "21 3"] = landmarks_mm * scale_to_meters
                    landmarks_lr[hand_idx] = landmarks_world
                    prev_landmarks_lr[hand_idx] = landmarks_world
                else:
                    # No detection: carry forward last valid landmarks (gap filling)
                    landmarks_lr[hand_idx] = prev_landmarks_lr[hand_idx]

            # Step 4: Map 2×21 hand landmarks into the 133-keypoint COCO-WholeBody layout.
            xyzc_stack[frame_idx] = assembly21_to_coco133(landmarks_lr)

            # Set confidence for hand keypoints
            visible_hands: Float32[ndarray, "2"] = np.maximum(hand_confidences, np.float32(0.0))
            adjustments: tuple[tuple[int, int], ...] = ((0, 91), (1, 112))
            wrist_indices: tuple[int, int] = (9, 10)
            thumb_base_indices: tuple[int, int] = (92, 113)
            for hand_idx, coco_offset in adjustments:
                conf: float = float(visible_hands[hand_idx])
                if conf <= 0.0:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(0.0)
                else:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(conf)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(conf)
                    if not np.isnan(xyzc_stack[frame_idx, thumb_base_indices[hand_idx], :3]).all():
                        xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(conf)

        # ── Load MANO parameters (if available) ────────────────────────
        mano_stack: ManoStack | None = self._load_mano_poses(seq_dir, rgb_label_indices)

        # Normalize label timestamps to the video container's 0-based timeline.
        vrs_start_ns: np.int64 = np.int64(vrs_ref_ts[0])
        normalized_label_ts: Int64[ndarray, "num_frames"] = devicetime_ns_filtered - vrs_start_ns

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=normalized_label_ts,
            mano_stack=mano_stack,
        )

    def _load_mano_poses(self, seq_dir: Path, rgb_label_indices: Int64[ndarray, "n_video"]) -> ManoStack | None:
        """Load MANO parameters from ``mano_hand_pose_trajectory.jsonl``.

        HOT3D MANO format per hand:
        - ``pose``: 15 PCA hand coefficients (zero-padded to 45 for MANO layer)
        - ``wrist_xform.q_wxyz``: global rotation as quaternion → axis-angle (3)
        - ``wrist_xform.t_xyz``: translation in meters
        - ``betas``: 10 shape params (per-frame but typically constant)

        ManoStack convention: index 0=right, 1=left.
        HOT3D JSONL keys: "0"=left, "1"=right.
        """
        from scipy.spatial.transform import Rotation

        mano_path: Path = seq_dir / "mano_hand_pose_trajectory.jsonl"
        if not mano_path.exists() or mano_path.stat().st_size == 0:
            return None

        # JSONL: one JSON object per line. pyserde doesn't support JSONL natively,
        # and the schema varies per frame (some frames have empty hand_poses), so
        # we parse line-by-line with json.loads.
        mano_frames: list[dict] = []
        with open(mano_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    mano_frames.append(json.loads(line))
        if not mano_frames:
            return None

        # Filter to same RGB-aligned indices as UmeTrack
        mano_filtered: list[dict] = [mano_frames[int(i)] for i in rgb_label_indices]
        num_frames: int = len(mano_filtered)

        so3: Float32[ndarray, "num_frames 2 48"] = np.zeros((num_frames, 2, 48), dtype=np.float32)
        trans: Float32[ndarray, "num_frames 2 3"] = np.zeros((num_frames, 2, 3), dtype=np.float32)
        betas: Float32[ndarray, "10"] | None = None

        # HOT3D key → ManoStack hand index: "0"=left→1, "1"=right→0
        hand_key_to_mano_idx: dict[str, int] = {"0": 1, "1": 0}

        for frame_idx, entry in enumerate(mano_filtered):
            hand_poses: dict = entry.get("hand_poses", {})

            for hand_key, mano_hand_idx in hand_key_to_mano_idx.items():
                if hand_key not in hand_poses:
                    continue

                pose_data: dict = hand_poses[hand_key]

                # Extract betas from first available frame
                if betas is None and "betas" in pose_data:
                    betas = np.array(pose_data["betas"], dtype=np.float32)

                # Global rotation: quaternion [w,x,y,z] → axis-angle (3)
                wrist_data: dict = pose_data["wrist_xform"]
                q_wxyz: list[float] = wrist_data["q_wxyz"]
                w, x, y, z = q_wxyz
                global_rot: Float32[ndarray, "3"] = Rotation.from_quat([x, y, z, w]).as_rotvec().astype(np.float32)

                # PCA hand pose: HOT3D provides 15 PCA coefficients, but our
                # MANOLayerNP expects 45 (ncomps=45 hardcoded). Zero-padding is
                # mathematically correct since PCA is linear — the 30 zero
                # coefficients contribute nothing to the output.
                pca_coeffs: list[float] = pose_data["pose"]
                hand_pca: Float32[ndarray, "45"] = np.zeros(45, dtype=np.float32)
                hand_pca[: len(pca_coeffs)] = np.array(pca_coeffs, dtype=np.float32)

                # Concatenate: [3 global_rot, 45 PCA hand] = 48
                so3[frame_idx, mano_hand_idx, :3] = global_rot
                so3[frame_idx, mano_hand_idx, 3:] = hand_pca

                # Translation in meters
                trans[frame_idx, mano_hand_idx] = np.array(wrist_data["t_xyz"], dtype=np.float32)

        if betas is None:
            return None

        return ManoStack(betas=betas, so3=so3, trans=trans)

    @classmethod
    def iter_episode_sequences(cls, cfg: Hot3dConfig) -> Generator["Hot3dSequence", None, None]:
        """Iterate over all sequences in the HOT3D headset root directory.

        Yields one ``Hot3dSequence`` per sequence folder that contains
        at least one preprocessed stream MP4 under ``_simplecv/``.
        """
        for seq_dir in cls._iter_sequence_dirs(cfg):
            episode_cfg: Hot3dConfig = replace(
                cfg,
                sequence_name=seq_dir.name,
            )
            try:
                yield cls(episode_cfg)
            except Exception as exc:  # pragma: no cover
                tqdm.write(f"[skip] {seq_dir.name}: {exc}")

    @classmethod
    def num_sequences_for_config(cls, cfg: Hot3dConfig) -> int:
        return len(cls._iter_sequence_dirs(cfg))

    @staticmethod
    def _iter_sequence_dirs(cfg: Hot3dConfig) -> list[Path]:
        root: Path = cfg.root_directory
        assert root.exists(), f"HOT3D root directory {root} does not exist."

        # Check for any expected stream MP4 (Aria has rgb.mp4, Quest has slam_left/right.mp4)
        def _has_preprocessed_streams(d: Path) -> bool:
            simplecv_dir: Path = d / "_simplecv"
            return simplecv_dir.is_dir() and any(simplecv_dir.glob("*.mp4"))

        return natsorted([d for d in root.iterdir() if d.is_dir() and _has_preprocessed_streams(d)])

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Headset-aware world frame.

        Aria: gravity = [0,0,-9.81] → +Z is up.
        Quest 3: Y ≈ 1.0m (head height) → +Y is up (OpenXR convention).
        """
        if self._headset == "Quest3":
            return rr.ViewCoordinates.RIGHT_HAND_Y_UP
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP
