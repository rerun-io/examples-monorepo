from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde.json import from_json
from tqdm import tqdm

from simplecv.apis.view_umetrack_data import UmeTrackAnnotation
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.umetrack_ego import UmeTrackEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, SingleHandPose, landmarks_from_hand_pose


@dataclass
class UmeTrackConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: UmeTrackSequence)
    root_directory: Path = Path("/mnt/8tb/data/umetrack-split")
    data_type: Literal["synthetic", "real"] = "real"
    split: Literal["training", "testing"] = "training"
    hand_interaction: Literal["separate_hand", "hand_hand"] = "separate_hand"
    user: int = 15
    recording_id: int = 0
    sequence_name: str = ""


class UmeTrackSequence(BaseExoEgoSequence[UmeTrackConfig]):
    """UmeTrack dataset adapter emitting 3D annotations in meters."""

    def __init__(self, cfg: UmeTrackConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    @classmethod
    def sequence_identity_for_config(cls, cfg: UmeTrackConfig) -> SequenceIdentity:
        return SequenceIdentity(
            dataset="umetrack",
            parts=(
                cfg.data_type,
                cfg.hand_interaction,
                cfg.split,
                f"user_{cfg.user:02d}",
                f"recording_{cfg.recording_id:02d}",
            ),
        )

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = (None, None)  # no exo cameras
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[UmeTrackConfig] | None:
        return UmeTrackEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[UmeTrackConfig] | None:
        return None

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego videos (labels if available)."""

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

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters for the current sequence."""

        recording_dir: Path = (
            self.config.root_directory
            / self.config.data_type
            / self.config.hand_interaction
            / self.config.split
            / f"user_{self.config.user:02d}"
            / f"recording_{self.config.recording_id:02d}"
        )
        annotation_path: Path = recording_dir / f"recording_{self.config.recording_id:02d}.json"
        assert annotation_path.exists(), f"Annotation file {annotation_path} does not exist."

        annotation: UmeTrackAnnotation = from_json(UmeTrackAnnotation, annotation_path.read_text())
        hand_model_tensor: HandModelNumpy = annotation.hand_model

        num_frames: int = annotation.joint_angles.shape[0]
        # initialize with NaNs and zero confidence
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[:, :, 3] = np.float32(0.0)

        scale_to_meters: float = 1e-3
        prev_landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
        for frame_idx in range(num_frames):
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
            hand_confidences: Float32[ndarray, "2"] = annotation.hand_confidences[frame_idx].astype(
                np.float32, copy=False
            )
            for hand_idx in range(2):
                confidence: float = float(hand_confidences[hand_idx])
                if confidence > 0.0:
                    joint_angles: Float32[ndarray, "22"] = annotation.joint_angles[frame_idx, hand_idx].astype(
                        np.float32, copy=False
                    )
                    wrist_transform: Float32[ndarray, "4 4"] = annotation.wrist_transforms[frame_idx, hand_idx].astype(
                        np.float32, copy=False
                    )
                    hand_pose: SingleHandPose = SingleHandPose(
                        joint_angles=joint_angles,
                        wrist_xform=wrist_transform,
                        hand_confidence=confidence,
                    )
                    landmarks_world: Float32[ndarray, "21 3"] = landmarks_from_hand_pose(
                        hand_model_tensor, hand_pose, hand_idx
                    ).astype(np.float32, copy=False)
                    scaled_landmarks: Float32[ndarray, "21 3"] = landmarks_world * scale_to_meters
                    landmarks_lr[hand_idx] = scaled_landmarks
                    prev_landmarks_lr[hand_idx] = scaled_landmarks
                else:
                    landmarks_lr[hand_idx] = prev_landmarks_lr[hand_idx]

            xyzc_stack[frame_idx] = assembly21_to_coco133(landmarks_lr)
            visible_hands: Float32[ndarray, "2"] = np.maximum(hand_confidences, np.float32(0.0))
            adjustments: tuple[tuple[int, int], ...] = ((0, 91), (1, 112))
            wrist_indices: tuple[int, int] = (9, 10)
            thumb_base_indices: tuple[int, int] = (92, 113)
            for hand_idx, coco_offset in adjustments:
                confidence: float = float(visible_hands[hand_idx])
                if confidence <= 0.0:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(0.0)
                else:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(confidence)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(confidence)
                    if not np.isnan(xyzc_stack[frame_idx, thumb_base_indices[hand_idx], :3]).all():
                        xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(confidence)

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: UmeTrackConfig) -> Generator["UmeTrackSequence", None, None]:
        """
        Iterate over every recording in the UmeTrack dataset rooted at ``cfg.root_directory``.

        The layout is expected to follow ``data_type/hand_interaction/split/user_xx/recording_xx``.
        Nested loops keep the traversal explicit: each directory level is validated and the
        discovered identifiers override the corresponding fields in a copied configuration.

        Args:
            cfg: Configuration object describing the dataset root and default options.

        Yields:
            Instances of :class:`UmeTrackSequence`, one per recording discovered on disk.
        """
        for data_type, hand_interaction, split, user_id, recording_id, recording_dir in cls._iter_episode_specs(cfg):
            episode_cfg: UmeTrackConfig = replace(
                cfg,
                data_type=data_type,
                hand_interaction=hand_interaction,
                split=split,
                user=user_id,
                recording_id=recording_id,
                sequence_name=(
                    f"{data_type}/{hand_interaction}/{split}/"
                    f"user_{user_id:02d}/recording_{recording_id:02d}"
                ),
            )

            try:
                yield cls(episode_cfg)
            except Exception as exc:  # pragma: no cover - defensive skip for corrupted recordings
                tqdm.write(f"[skip] {recording_dir}: {exc}")

    @classmethod
    def num_sequences_for_config(cls, cfg: UmeTrackConfig) -> int:
        return len(cls._iter_episode_specs(cfg))

    @staticmethod
    def _iter_episode_specs(cfg: UmeTrackConfig) -> list[tuple[str, str, str, int, int, Path]]:
        root_dir: Path = cfg.root_directory
        assert root_dir.exists(), f"UmeTrack root directory {root_dir} does not exist."

        episode_specs: list[tuple[str, str, str, int, int, Path]] = []
        data_type_dirs: list[Path] = natsorted([d for d in root_dir.iterdir() if d.is_dir()])
        for data_type_dir in data_type_dirs:
            data_type: str = data_type_dir.name
            if data_type not in {"synthetic", "real"}:
                continue

            hand_dirs: list[Path] = natsorted([d for d in data_type_dir.iterdir() if d.is_dir()])
            for hand_dir in hand_dirs:
                hand_interaction: str = hand_dir.name
                if hand_interaction not in {"separate_hand", "hand_hand"}:
                    continue

                split_dirs: list[Path] = natsorted([d for d in hand_dir.iterdir() if d.is_dir()])
                for split_dir in split_dirs:
                    split: str = split_dir.name
                    if split not in {"training", "testing"}:
                        continue

                    user_dirs: list[Path] = natsorted([d for d in split_dir.glob("user_*") if d.is_dir()])
                    for user_dir in user_dirs:
                        try:
                            user_id: int = int(user_dir.name.split("_")[-1])
                        except ValueError:
                            continue

                        recording_dirs: list[Path] = natsorted([d for d in user_dir.glob("recording_*") if d.is_dir()])
                        for recording_dir in recording_dirs:
                            try:
                                recording_id: int = int(recording_dir.name.split("_")[-1])
                            except ValueError:
                                continue
                            episode_specs.append(
                                (data_type, hand_interaction, split, user_id, recording_id, recording_dir)
                            )
        return episode_specs

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
