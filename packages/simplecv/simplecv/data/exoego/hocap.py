from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import get_args

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Bool, Float32, Int, UInt16
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde import serde
from serde.yaml import from_yaml

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.hocap_ego import ExoCameraIDs, HocapEgoSequence, HOCapExtrinsicsData
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exo.hocap_exo import HocapExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX


@serde
class CalibratedMano:
    betas: Float32[ndarray, "10"]


@dataclass
class HocapConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: HocapSequence)
    root_directory: Path = Path("data/hocap/sample")
    subject_id: str = "8"
    sequence_name: str = "20231024_180733"


class HocapSequence(BaseExoEgoSequence[HocapConfig]):
    """HoloCap dataset adapter emitting 3D annotations in meters."""

    def __init__(self, cfg: HocapConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    @classmethod
    def sequence_identity_for_config(cls, cfg: HocapConfig) -> SequenceIdentity:
        return SequenceIdentity(dataset="hocap", parts=(f"subject_{cfg.subject_id}", cfg.sequence_name))

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        ego_depth_list: list[UInt16[ndarray, "H W"]] | None = self._sample_ego_depths(ts_ns)
        exo_depth_list: list[UInt16[ndarray, "H W"]] | None = self._sample_exo_depths(ts_ns)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None:
            if labels.timestamps_ns is not None:
                label_idx: int = self.timestamp_to_frame_index(ts_ns, labels.timestamps_ns)
            else:
                max_idx: int = int(labels.xyzc_stack.shape[0] - 1)
                label_idx = min(canonical_idx, max_idx)
            xyzc_stack_frame = labels.xyzc_stack[label_idx]
            mano_stack_frame: ManoStack | None = None
            if labels.mano_stack is not None:
                mano_stack_frame = ManoStack(
                    betas=labels.mano_stack.betas,
                    so3=labels.mano_stack.so3[label_idx : label_idx + 1],
                    trans=labels.mano_stack.trans[label_idx : label_idx + 1],
                    use_pca=labels.mano_stack.use_pca,
                )
            timestamps_ns = labels.timestamps_ns
            labels = ExoEgoLabels(
                xyzc_stack=xyzc_stack_frame[np.newaxis, ...],
                timestamps_ns=timestamps_ns,
                mano_stack=mano_stack_frame,
            )

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            ego_depth_list=ego_depth_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            exo_depth_list=exo_depth_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence | None:
        return HocapEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[HocapConfig] | None:
        return HocapExoSequence(cfg=self.config)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego/exo videos (and labels if present)."""

        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()

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

        if self.exo_sequence is not None:
            for name, video_path in zip(
                self.exo_sequence.exo_video_names,
                self.exo_sequence.exo_video_paths,
                strict=True,
            ):
                stream_name = f"exo/{name}"
                timestamps = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._exo_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 joints and MANO parameters in meters for this sequence."""
        # 2D keypoints are not available for HoloLens, so we will load 3D labels from the first camera
        calibration_path: Path = self.config.root_directory / "calibration"
        extrinsics_directory: Path = calibration_path / "extrinsics"

        extrinsic_yaml: Path = extrinsics_directory / "extrinsics_20231014.yaml"
        assert extrinsic_yaml.exists(), f"Path {extrinsic_yaml} does not exist."

        extri_hocap: HOCapExtrinsicsData = from_yaml(HOCapExtrinsicsData, extrinsic_yaml.read_text())

        label_path: Path = self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        assert label_path.exists(), f"Path {label_path} does not exist."
        # hololens does not contain labels, so we need to load the 3d labels from any  other camera
        cam_name: ExoCameraIDs = get_args(ExoCameraIDs)[0]  # Assuming the first camera is the HoloLens
        world_T_cam: Float32[ndarray, "4 4"] | None = extri_hocap.world_T_cam_dict.get(cam_name)
        assert world_T_cam is not None, f"Extrinsics for camera {cam_name} not found in {extrinsic_yaml}."
        cam_dir: Path = label_path / cam_name  # Assuming the first camera is the HoloLens

        npz_paths: list[Path] = sorted(cam_dir.glob("*.npz"))
        assert len(npz_paths) > 0, f"No .npz files found in {cam_dir}. Expected at least one file."

        xyz_list: list[Float32[ndarray, "133 3"]] = []
        for npz_path in npz_paths:
            npz_data = np.load(npz_path)
            xyz_cam: Float32[ndarray, "2 21 3"] = npz_data["hand_joints_3d"]
            # convert to world coordinates
            # Assuming hand_joints_3d_cam.shape == (2, 21, 3) and world_T_cam.shape == (4, 4)
            # Create homogeneous coordinates by concatenating a ones column along the last axis
            ones: Float32[ndarray, "2 21 1"] = np.ones(
                (*xyz_cam.shape[:-1], 1),
                dtype=xyz_cam.dtype,
            )
            xyz_cam_homogeneous: Float32[ndarray, "2 21 4"] = np.concatenate([xyz_cam, ones], axis=-1)

            # filter out -1 (not detected) values
            xyz_cam_homogeneous = np.where(xyz_cam_homogeneous == -1, np.nan, xyz_cam_homogeneous)

            # Transform all joints at once using matrix multiplication.
            # The multiplication is broadcast over the first two dimensions.
            xyz_world_homogeneous: Float32[ndarray, "2 21 4"] = xyz_cam_homogeneous @ world_T_cam.T

            # Extract the 3D world coordinates (ignore the homogeneous component)
            xyz_world: Float32[ndarray, "2 21 3"] = xyz_world_homogeneous[..., :3]
            xyz_list.append(xyz_world)

        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = np.stack(xyz_list)
        right_xyz: Float32[ndarray, "num_frames 21 3"] = xyz_stack[:, 0, :, :]
        left_xyz: Float32[ndarray, "num_frames 21 3"] = xyz_stack[:, 1, :, :]

        coco_xyz_stack: Float32[ndarray, "num_frames 133 3"] = np.full(
            (len(npz_paths), 133, 3), np.nan, dtype=np.float32
        )
        # fill in the right and left hand joints
        coco_xyz_stack[:, RIGHT_HAND_IDX, :] = right_xyz
        coco_xyz_stack[:, LEFT_HAND_IDX, :] = left_xyz
        # propagate NaNs into the confidence channel so downstream averages ignore gaps
        missing_mask: Bool[ndarray, "num_frames 133"] = np.asarray(
            np.isnan(coco_xyz_stack).any(axis=-1),
            dtype=np.bool_,
        )
        conf_mask: Float32[ndarray, "num_frames 133"] = np.where(missing_mask, np.nan, 1.0).astype(np.float32)
        conf_stack: Float32[ndarray, "num_frames 133 1"] = conf_mask[..., np.newaxis]
        # create xyzc stack
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.concatenate([coco_xyz_stack, conf_stack], axis=-1)

        # get mano parameters
        mano_stack: ManoStack = self.load_mano_poses(self.config.root_directory, self.config.sequence_name)
        return ExoEgoLabels(xyzc_stack=xyzc_stack, mano_stack=mano_stack)

    def load_mano_poses(self, data_path: Path, sequence_name: str) -> ManoStack:
        subject_mano_yaml: Path = data_path / "calibration" / "mano" / f"subject_{self.config.subject_id}.yaml"
        assert subject_mano_yaml.exists(), f"Path {subject_mano_yaml} does not exist."

        subject_mano: CalibratedMano = from_yaml(CalibratedMano, subject_mano_yaml.read_text())
        poses_path: Path = data_path / f"subject_{self.config.subject_id}" / sequence_name
        assert poses_path.exists(), f"Path {poses_path} does not exist."
        mano_poses_path: Path = poses_path / "poses_m.npy"
        # 0 for right hand, 1 for left hand
        poses_raw: Float32[ndarray, "num_sides num_frames 51"] = np.load(mano_poses_path)
        # permute to num_frames num_sides 51
        poses_nfhs: Float32[ndarray, "num_frames num_sides 51"] = rearrange(
            poses_raw, "num_sides num_frames pose -> num_frames num_sides pose"
        )
        so3: Float32[ndarray, "num_frames num_sides 48"] = poses_nfhs[..., :48]
        trans: Float32[ndarray, "num_frames num_sides 3"] = poses_nfhs[..., 48:51]

        return ManoStack(betas=subject_mano.betas, so3=so3, trans=trans)

    @classmethod
    def iter_episode_sequences(cls, cfg: HocapConfig) -> Generator["HocapSequence", None, None]:
        """
        Iterates over all episode sequences in the dataset specified by the given configuration.

        This class method yields `HocapSequence` instances for each sequence found in the dataset directory structure.
        It expects the dataset to be organized with subject directories named "subject_*", each containing sequence directories.

        Args:
            cfg (HocapConfig): Configuration object specifying the root directory and other parameters.

        Yields:
            HocapSequence: An instance for each sequence found, with configuration updated for the current subject and sequence.

        Notes:
            - Uses natural sorting for subject and sequence directories.
            - Prints subject ID and sequence name for each iteration using `icecream.ic`.
            - Pauses execution for user input after each sequence (likely for debugging).
        """
        for subject_id, seq_dir in cls._iter_episode_specs(cfg):
            new_cfg = replace(
                cfg,
                subject_id=subject_id,
                sequence_name=seq_dir.name,
            )
            yield cls(new_cfg)

    @classmethod
    def num_sequences_for_config(cls, cfg: HocapConfig) -> int:
        return len(cls._iter_episode_specs(cfg))

    @staticmethod
    def _iter_episode_specs(cfg: HocapConfig) -> list[tuple[str, Path]]:
        root: Path = cfg.root_directory
        subject_dirs: list[Path] = natsorted([d for d in root.glob("subject_*") if d.is_dir()])
        return [
            (subj_dir.name.split("_")[-1], seq_dir)
            for subj_dir in subject_dirs
            for seq_dir in natsorted([d for d in subj_dir.iterdir() if d.is_dir()])
        ]

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
