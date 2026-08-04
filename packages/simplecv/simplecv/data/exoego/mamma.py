"""MAMMA (MPI markerless multi-person mocap) exoego adapter.

Exo-only multi-view rig: per-camera calibration ships as NPZ files
(``cam_int`` 3x3 K, ``cam_ext`` 4x4 world->camera, ``cam_img_w``/``cam_img_h``)
under ``meta/`` (markerless subsets) or ``gt/`` (MammaEval subsets), with one
``<CAM>.mp4`` per camera. Download with ``tools/download_mamma.py`` and build
the AV1 yuv420 mirror with ``tools/preprocess_mamma.py`` (the shipped videos
are yuv444 H.264/H.265, which the NVDEC/rerun hot path cannot decode).

Expected per-sequence layout under ``root_directory``::

    <subset>/<sequence>/
      meta/{<CAM>.npz, global.npz}   # or gt/ for mamma_eval_* subsets
      videos_av1/<CAM>.mp4           # preferred (preprocess_mamma output)
      videos_light/<CAM>.mp4         # iPhone subsets (H.265 CRF24 yuv444)
      videos_crf24/<CAM>.mp4         # IOI subsets (H.265 CRF24 yuv444)
"""

import warnings
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Float, Float32, Int
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.configs.dataset_paths import MAMMA_AV1_1080_ROOT
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, SmplxStack
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity

VIDEO_DIR_PREFERENCE: tuple[str, ...] = ("videos_av1", "videos_light", "videos_crf24", "videos_crf16", "videos")
"""Per-sequence video directories, most preferred first (AV1 mirror, then smallest source)."""

CALIB_DIR_CANDIDATES: tuple[str, ...] = ("meta", "gt")
"""Calibration directory names: ``meta/`` (markerless subsets) or ``gt/`` (MammaEval subsets)."""

DOWNLOAD_HINT: str = "Run `pixi run -e simplecv simplecv-download-mamma` (and `simplecv-preprocess-mamma`) first."
"""Recovery hint appended to missing-data errors."""


@dataclass
class MammaConfig(BaseExoEgoDatasetConfig):
    """MAMMA dataset config (exo-only multi-view rig)."""

    _target: type = field(default_factory=lambda: MammaSequence)
    """Target sequence class to instantiate."""
    root_directory: Path = MAMMA_AV1_1080_ROOT
    """Root directory containing ``<subset>/<sequence>/{meta|gt,videos*}`` trees."""
    sequence_name: str = "mamma_markerless_iphones/indoors/crossing_arms"
    """Sequence directory relative to ``root_directory`` (slash-separated)."""
    camera_names: tuple[str, ...] = ()
    """Cameras to load; empty auto-discovers calibrated cameras that have a video."""


def sequence_dir_for_config(cfg: MammaConfig) -> Path:
    """Return the sequence directory (``root_directory / sequence_name``)."""
    return cfg.root_directory / cfg.sequence_name


def calibration_dir_for_sequence(sequence_dir: Path) -> Path:
    """Return the calibration dir (``meta/`` or ``gt/``) for one sequence."""
    for name in CALIB_DIR_CANDIDATES:
        calib_dir: Path = sequence_dir / name
        if (calib_dir / "global.npz").exists():
            return calib_dir
    raise FileNotFoundError(
        f"No MAMMA calibration dir ({'/'.join(CALIB_DIR_CANDIDATES)} with global.npz) under {sequence_dir}. {DOWNLOAD_HINT}"
    )


def video_dir_for_sequence(sequence_dir: Path) -> Path:
    """Return the first existing non-empty video dir by preference order."""
    for name in VIDEO_DIR_PREFERENCE:
        video_dir: Path = sequence_dir / name
        if video_dir.is_dir() and any(video_dir.glob("*.mp4")):
            return video_dir
    raise FileNotFoundError(f"No MAMMA video dir ({'/'.join(VIDEO_DIR_PREFERENCE)}) with mp4s under {sequence_dir}. {DOWNLOAD_HINT}")


def _cameras_with_video(sequence_dir: Path, calib_dir: Path, video_dir: Path) -> tuple[str, ...]:
    """Cameras with both a calibration NPZ and a video in the given dirs.

    Warns when the video dir covers fewer cameras than the calibration dir —
    e.g. an interrupted ``simplecv-preprocess-mamma`` leaves a partial
    ``videos_av1/`` mirror that would otherwise silently shrink the rig.
    """
    calibrated: list[str] = natsorted(npz_path.stem for npz_path in calib_dir.glob("*.npz") if npz_path.stem != "global")
    names: list[str] = [name for name in calibrated if (video_dir / f"{name}.mp4").exists()]
    if not names:
        raise FileNotFoundError(
            f"No camera with both {calib_dir.name}/<cam>.npz and {video_dir.name}/<cam>.mp4 under {sequence_dir}. {DOWNLOAD_HINT}"
        )
    if len(names) < len(calibrated):
        missing: list[str] = [name for name in calibrated if name not in names]
        warnings.warn(
            f"MAMMA sequence {sequence_dir.name}: {video_dir.name}/ has videos for {len(names)}/{len(calibrated)} calibrated "
            f"cameras (missing {missing}). Loading a reduced rig; re-run simplecv-preprocess-mamma if this mirror is incomplete.",
            stacklevel=2,
        )
    return tuple(names)


def discover_camera_names(sequence_dir: Path) -> tuple[str, ...]:
    """Cameras with both a calibration NPZ and a video, naturally sorted."""
    return _cameras_with_video(sequence_dir, calibration_dir_for_sequence(sequence_dir), video_dir_for_sequence(sequence_dir))


class MammaSequenceLayout(NamedTuple):
    """Resolved on-disk layout of one MAMMA sequence (computed once)."""

    calib_dir: Path
    video_dir: Path
    camera_names: tuple[str, ...]


def resolve_sequence_layout(cfg: MammaConfig) -> MammaSequenceLayout:
    """Resolve the calibration dir, video dir, and camera names for a config once.

    Avoids re-globbing when both ``load_video_paths`` and ``load_exo_cams`` need
    the same immutable layout; honors an explicit ``cfg.camera_names`` override.
    """
    sequence_dir: Path = sequence_dir_for_config(cfg)
    calib_dir: Path = calibration_dir_for_sequence(sequence_dir)
    video_dir: Path = video_dir_for_sequence(sequence_dir)
    camera_names: tuple[str, ...] = cfg.camera_names or _cameras_with_video(sequence_dir, calib_dir, video_dir)
    return MammaSequenceLayout(calib_dir=calib_dir, video_dir=video_dir, camera_names=camera_names)


def discover_sequence_names(root_directory: Path) -> list[str]:
    """Find all sequence dirs (holding ``meta/global.npz`` or ``gt/global.npz``) under the root."""
    names: set[str] = set()
    for global_npz in root_directory.rglob("global.npz"):
        if global_npz.parent.name in CALIB_DIR_CANDIDATES:
            names.add(global_npz.parent.parent.relative_to(root_directory).as_posix())
    return natsorted(names)


def pinhole_from_camera_npz(npz_path: Path, video_path: Path) -> PinholeParameters:
    """Build ``PinholeParameters`` from one MAMMA camera NPZ.

    Mirrors the MAMMA reference loader: ``cam_ext`` is world->camera
    (``x_cam = cam_ext @ x_world``); translations with ``|t| > 200`` are
    millimeters and are normalized to meters. Intrinsics are given at the
    native capture resolution — when the video was re-encoded at a different
    size, K is rescaled to the video resolution.
    """
    data = np.load(npz_path, allow_pickle=True)
    name: str = str(data["cam_name"]) if "cam_name" in data.files else npz_path.stem
    k_matrix: Float[ndarray, "3 3"] = np.asarray(data["cam_int"], dtype=np.float64)
    cam_T_world: Float[ndarray, "4 4"] = np.asarray(data["cam_ext"], dtype=np.float64)
    if np.abs(cam_T_world[:3, 3]).max() > 200:
        cam_T_world = cam_T_world.copy()
        cam_T_world[:3, 3] /= 1000.0
    native_width: int = int(data["cam_img_w"])
    native_height: int = int(data["cam_img_h"])

    capture = cv2.VideoCapture(str(video_path))
    video_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if video_width <= 0 or video_height <= 0:
        raise ValueError(f"Could not probe video resolution for {video_path}")
    if (video_width, video_height) != (native_width, native_height):
        scale: Float[ndarray, "3 3"] = np.diag([video_width / native_width, video_height / native_height, 1.0])
        k_matrix = scale @ k_matrix

    intrinsics: Intrinsics = Intrinsics.from_k_matrix(
        camera_conventions="RDF",
        k_matrix=k_matrix,
        width=video_width,
        height=video_height,
    )
    extrinsics: Extrinsics = Extrinsics(
        cam_R_world=cam_T_world[:3, :3],
        cam_t_world=cam_T_world[:3, 3],
    )
    return PinholeParameters(name=name, intrinsics=intrinsics, extrinsics=extrinsics)


def _coco133_from_smplx_person(smplx_stack: SmplxStack, *, person_idx: int) -> Float32[ndarray, "num_frames 133 4"]:
    """Derive one person's COCO-133 keypoints from the SMPL-X forward's joints.

    Runs the (chunked, GPU when available) SMPL-X forward with the face contour
    enabled so all 133 keypoints — body, feet, 68 face, 42 hands — are mapped.
    Raises RuntimeError/ImportError when the body model cannot be loaded.
    """
    import torch

    from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_xyzc
    from simplecv.ops.smplx.smplx_torch import SmplxForwardResult, SmplxLayerTorch

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smplx_layer: SmplxLayerTorch = (
        SmplxLayerTorch(
            betas=smplx_stack.betas[person_idx],
            model_type=smplx_stack.model_type,
            gender=smplx_stack.gender,
            flat_hand_mean=smplx_stack.flat_hand_mean,
            use_face_contour=True,
            v_template=smplx_stack.v_template[person_idx] if smplx_stack.v_template is not None else None,
        )
        .to(device)
        .eval()
    )
    poses_person: Float32[ndarray, "num_frames n_pose"] = np.ascontiguousarray(smplx_stack.poses[:, person_idx])
    trans_person: Float32[ndarray, "num_frames 3"] = np.ascontiguousarray(smplx_stack.trans[:, person_idx])
    smplx_forward: SmplxForwardResult = smplx_layer.forward_batched(poses_person, trans_person)
    return smplx_joints_to_coco133_xyzc(smplx_forward.joints)


def _warn_on_mixed_person_model_settings(*, genders: list[str], flat_hand_means: list[bool], source: Path) -> None:
    """Warn when people in one sequence were fit with different body-model settings.

    ``SmplxStack`` carries a single ``gender``/``flat_hand_mean`` for all people
    (multi-person label support is a deferred TODO — see ``ExoEgoLabels``), so a
    mixed fit would silently forward later persons through the wrong model. All
    data seen so far is uniform (markerless preds are all-neutral, eval singles
    are one person); this makes the day that changes loud instead of silent.
    """
    if len(set(genders)) > 1:
        warnings.warn(f"{source}: people mix body-model genders {genders}; using person 0's ({genders[0]!r}) for ALL people", stacklevel=3)
    if len(set(flat_hand_means)) > 1:
        warnings.warn(f"{source}: people mix flat_hand_mean {flat_hand_means}; using person 0's ({flat_hand_means[0]}) for ALL people", stacklevel=3)


def _smplx_stack_from_pred_params(params_paths: list[Path]) -> SmplxStack:
    """Build a ``SmplxStack`` from the markerless subsets' ``pred/params_XX.npz`` files.

    Each per-person NPZ ships MAMMA-estimated SMPL-X params: ``poses (n_frames, 165)``
    axis-angle full pose, ``betas (16,)``, and ``trans (n_frames, 3)`` in world meters.
    """
    poses_list: list[Float[ndarray, "n_frames 165"]] = []
    betas_list: list[Float[ndarray, "n_betas"]] = []
    trans_list: list[Float[ndarray, "n_frames 3"]] = []
    flat_hand_means: list[bool] = []
    genders: list[str] = []
    for params_path in params_paths:
        data = np.load(params_path, allow_pickle=True)
        poses_list.append(np.asarray(data["poses"], dtype=np.float32))
        betas_list.append(np.asarray(data["betas"], dtype=np.float32).reshape(-1))
        trans_list.append(np.asarray(data["trans"], dtype=np.float32))
        flat_hand_means.append(bool(data["flat_hand_mean"]) if "flat_hand_mean" in data.files else False)
        genders.append(str(data["gender"]) if "gender" in data.files else "neutral")
    _warn_on_mixed_person_model_settings(genders=genders, flat_hand_means=flat_hand_means, source=params_paths[0].parent)

    num_frames: int = min(person_poses.shape[0] for person_poses in poses_list)
    return SmplxStack(
        betas=np.stack(betas_list, axis=0),
        poses=np.stack([person_poses[:num_frames] for person_poses in poses_list], axis=1),
        trans=np.stack([person_trans[:num_frames] for person_trans in trans_list], axis=1),
        model_type="smplx",
        flat_hand_mean=flat_hand_means[0],
        gender=genders[0],
    )


def _smplx_stack_from_eval_gt(gt_npz_path: Path) -> SmplxStack | None:
    """Build a ``SmplxStack`` from an eval subset's MoSh GT ``gt/global.npz``.

    Eval GT ships ``pose_world (n_frames, n_people, 165)`` axis-angle full pose,
    ``pose_trans_world (n_frames, n_people, 3)`` in world meters (native smplx
    root-joint pivot), ``shape (n_people, 300)`` betas (all zeros — the subject
    geometry lives in ``v_template (n_people, 10475, 3)``, MoSh's
    ``mosh_v_template`` output), a per-person ``gender``, and ``flat_hand_mean``.
    Returns None when the file is missing or holds no SMPL-X params.
    """
    if not gt_npz_path.exists():
        return None
    data = np.load(gt_npz_path, allow_pickle=True)
    if "pose_world" not in data.files:
        return None
    genders: list[str] = [str(person_gender) for person_gender in np.atleast_1d(data["gender"])]
    flat_hand_mean: bool = bool(data["flat_hand_mean"])
    _warn_on_mixed_person_model_settings(genders=genders, flat_hand_means=[flat_hand_mean], source=gt_npz_path)
    return SmplxStack(
        betas=np.asarray(data["shape"], dtype=np.float32),
        poses=np.asarray(data["pose_world"], dtype=np.float32),
        trans=np.asarray(data["pose_trans_world"], dtype=np.float32),
        model_type="smplx",
        flat_hand_mean=flat_hand_mean,
        gender=genders[0],
        v_template=np.asarray(data["v_template"], dtype=np.float32),
    )


class MammaSequence(BaseExoEgoSequence[MammaConfig]):
    """MAMMA adapter (exo-only static rig; COCO-133 keypoints derived from SMPL-X)."""

    def __init__(self, cfg: MammaConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    @classmethod
    def sequence_identity_for_config(cls, cfg: MammaConfig) -> SequenceIdentity:
        return SequenceIdentity(dataset="mamma", parts=tuple(cfg.sequence_name.split("/")))

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=None,
            ego_bgr_list=None,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[MammaConfig] | None:
        return None  # exo-only

    def _build_exo(self) -> BaseExoSequence[MammaConfig] | None:
        # Deferred to break the mamma <-> mamma_exo import cycle (mamma_exo
        # imports the module-level helpers above at its module top).
        from simplecv.data.exo.mamma_exo import MammaExoSequence

        return MammaExoSequence(cfg=self.config)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for the exo videos."""
        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._exo_stream_names.clear()

        if self.exo_sequence is not None:
            for name, video_path in zip(
                self.exo_sequence.exo_video_names,
                self.exo_sequence.exo_video_paths,
                strict=True,
            ):
                stream_name: str = f"exo/{name}"
                timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._exo_stream_names.append(stream_name)

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """Build SMPL-X body labels from the eval GT or ``pred/params_XX.npz``.

        Eval subsets carry MoSh GT (with a subject ``v_template``) in
        ``gt/global.npz``; markerless subsets ship MAMMA-estimated SMPL-X params
        per person in ``pred/params_XX.npz``. Both become a ``SmplxStack`` (see
        the two ``_smplx_stack_from_*`` builders), and the GT wins when both are
        present — the labels are logged as ground truth.

        MAMMA ships no COCO-133 keypoints directly, so they are derived from the
        SMPL-X forward's regressed joints (person 0 — ``ExoEgoLabels.xyzc_stack``
        is single-skeleton; every person still gets a mesh via ``smplx_stack``).
        When the license-gated body model is unavailable the stack degrades to
        the NaN placeholder (robocap-style) and the viewer skips keypoints.
        """
        sequence_dir: Path = sequence_dir_for_config(self.config)
        smplx_stack: SmplxStack | None = _smplx_stack_from_eval_gt(sequence_dir / "gt" / "global.npz")
        if smplx_stack is None:
            params_paths: list[Path] = natsorted((sequence_dir / "pred").glob("params_*.npz"))
            if params_paths:
                smplx_stack = _smplx_stack_from_pred_params(params_paths)
        if smplx_stack is None:
            return None

        num_frames: int = smplx_stack.poses.shape[0]
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[..., 3] = np.float32(0.0)
        try:
            xyzc_stack = _coco133_from_smplx_person(smplx_stack, person_idx=0)
        except (RuntimeError, ImportError) as exc:
            warnings.warn(f"MAMMA COCO-133 keypoints unavailable (SMPL-X body model failed to load): {exc}", stacklevel=2)
        return ExoEgoLabels(xyzc_stack=xyzc_stack, smplx_stack=smplx_stack)

    @classmethod
    def iter_episode_sequences(cls, cfg: MammaConfig) -> Generator["MammaSequence", None, None]:
        """Yield one sequence per discovered sequence dir under the root."""
        for sequence_name in discover_sequence_names(cfg.root_directory):
            yield cls(replace(cfg, sequence_name=sequence_name))

    @classmethod
    def num_sequences_for_config(cls, cfg: MammaConfig) -> int:
        return len(discover_sequence_names(cfg.root_directory))

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return ViewCoordinates.RIGHT_HAND_Z_UP
