"""Meta SAM 3D Body as a posekit role: person boxes (+ masks, camera) -> MHR whole-body meshes with hands.

``Sam3dBodyPredictor`` is a :class:`~posekit.models.base.TopDownBodyPose3d` over the vendored upstream
inference source (``src/sam3d_body``, see VENDORED.md). ``mode="full"`` runs the body pass, one hand pass
per hand crop (the left crop mirrored), a keypoint-prompted body decoder pass, and merges the hands into
the body with MHR forward kinematics: three backbone passes per person. ``mode="body"`` runs the body
pass only. Crops are cut on the CPU, one frame at a time, exactly as upstream does: this is the eager
reference the goldens and the speed baseline measure.

Upstream behaviour kept on purpose (see VENDORED.md): the hand decoder's ``hand_logits`` are computed and
never read; the crop size is fixed at 512; 2D keypoints are projected about the image centre even when a
camera with another principal point is given.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from huggingface_hub import snapshot_download
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.cameras import CalibratedCamera
from posekit.models.base import TopDownBodyPose3d
from posekit.predictions import BodyPoses3d, BoxDetections, Keypoints3d, validate_frames_rgb
from posekit.skeletons import MHR_70, KeypointSkeleton

Sam3dBodyMode: TypeAlias = Literal["full", "body"]
"""``full``: body plus both hand decoders merged into it; ``body``: the body decoder alone."""
Sam3dBodyPass: TypeAlias = Literal["body", "left_hand", "right_hand", "prompt"]
"""Network passes of one frame: body, mirrored left-hand crop, right-hand crop, keypoint-prompted body decoder."""
Sam3dBodyStage: TypeAlias = Literal[
    "backbone_body", "backbone_left_hand", "backbone_right_hand",
    "decoder_body", "decoder_left_hand", "decoder_right_hand", "decoder_prompt",
    "mhr", "other", "total",
]
"""Stage names of ``stage_ms``. Decoder stages include the MHR heads their layers run; ``mhr`` is the forward kinematics
outside the decoders (the hand merge); ``other`` is the rest: CPU crops, batch upload, hand boxes, merge math, copies."""

SAM3D_BODY_REVISION: str = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
"""Pinned commit of facebook/sam-3d-body-dinov3 (gated); model.ckpt sha256 b5a2f9d3...."""
SAM3D_BODY_FILES: tuple[str, ...] = ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt")
"""What the model reads from that repo."""
ROOT_INDICES: tuple[int, int] = (9, 10)
"""MHR-70 left and right hip: their mean is the root of ``z_root_relative``."""


@dataclass(frozen=True, slots=True)
class Sam3dBodyConfig:
    """SAM 3D Body weights and inference mode; the gated checkpoint downloads at a pinned revision unless a local copy is given."""

    mode: Sam3dBodyMode = "full"
    """``full`` (body + hands, three backbone passes per person) or ``body`` (one pass)."""
    checkpoint_dir: Path | None = None
    """Directory holding model.ckpt, model_config.yaml and assets/mhr_model.pt; None downloads them."""
    hf_repo_id: str = "facebook/sam-3d-body-dinov3"
    """Hugging Face repo holding the checkpoint (gated: request access, then ``hf auth login``)."""
    hf_revision: str = SAM3D_BODY_REVISION
    """Pinned commit of that repo."""

    def checkpoint_path(self) -> Path:
        """The checkpoint directory: ``checkpoint_dir``, or the pinned download in the Hugging Face cache."""
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir
        return Path(snapshot_download(repo_id=self.hf_repo_id, revision=self.hf_revision, allow_patterns=list(SAM3D_BODY_FILES)))

    def setup(self) -> "Sam3dBodyPredictor":
        """Load the weights and construct the predictor."""
        return Sam3dBodyPredictor(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sam3dBodyReconstruction(BodyPoses3d):
    """MHR whole-body people: camera-space joints and meshes plus the MHR parameters behind them."""

    translation: Float32[Tensor, "n 3"]
    """Camera-space translation of the mesh, metres; joints, vertices and joint coordinates already include it."""
    focal_length: Float32[Tensor, "n"]
    """Focal length the model projected with, pixels."""
    global_rotation: Float32[Tensor, "n 3"]
    """Root rotation, XYZ Euler angles in radians."""
    body_pose: Float32[Tensor, "n 133"]
    """MHR body pose parameters (continuous)."""
    hand_pose: Float32[Tensor, "n 108"]
    """MHR hand pose PCA coefficients, left 54 then right 54."""
    scale: Float32[Tensor, "n 28"]
    """MHR scale PCA coefficients."""
    shape: Float32[Tensor, "n 45"]
    """MHR identity shape PCA coefficients."""
    expression: Float32[Tensor, "n 72"]
    """MHR face expression PCA coefficients."""
    joint_coordinates: Float32[Tensor, "n j 3"]
    """Camera-space MHR skeleton joint centres, metres."""
    joint_rotations: Float32[Tensor, "n j 3 3"]
    """Global rotation of each MHR skeleton joint."""
    left_hand_boxes: Float32[Tensor, "n 4"] | None
    """Source-image ``xyxy`` crop of the left-hand pass; None in body mode."""
    right_hand_boxes: Float32[Tensor, "n 4"] | None
    """Source-image ``xyxy`` crop of the right-hand pass; None in body mode."""

    def take(self, rows: Int64[Tensor, "m"]) -> "Sam3dBodyReconstruction":
        """The given rows, e.g. the people of one frame."""
        keypoints: Keypoints3d = self.keypoints
        return Sam3dBodyReconstruction(
            keypoints=Keypoints3d(
                keypoints.xy[rows], keypoints.z_root_relative[rows], keypoints.scores[rows], keypoints.frame_indices[rows], keypoints.skeleton,
                keypoints.root_indices, None if keypoints.xyz_camera is None else keypoints.xyz_camera[rows],
            ),
            boxes=self.boxes[rows],
            vertices=self.vertices[rows],
            faces=self.faces,
            translation=self.translation[rows],
            focal_length=self.focal_length[rows],
            global_rotation=self.global_rotation[rows],
            body_pose=self.body_pose[rows],
            hand_pose=self.hand_pose[rows],
            scale=self.scale[rows],
            shape=self.shape[rows],
            expression=self.expression[rows],
            joint_coordinates=self.joint_coordinates[rows],
            joint_rotations=self.joint_rotations[rows],
            left_hand_boxes=None if self.left_hand_boxes is None else self.left_hand_boxes[rows],
            right_hand_boxes=None if self.right_hand_boxes is None else self.right_hand_boxes[rows],
        )


def _person_rows(predictions: list[Any], field: str) -> Float32[ndarray, "n ..."]:
    """One upstream ``FinalPosePrediction`` field stacked over people, as float32."""
    return np.stack([np.asarray(getattr(prediction, field)) for prediction in predictions]).astype(np.float32, copy=False)


class _StageClock:
    """CUDA events around every network pass of one model, recorded only while enabled.

    Upstream's ``forward_step`` tells which pass runs (body, then the left and right hand crops); its keypoint-prompted
    decoder pass runs through ``run_keypoint_prompt``. MHR forward kinematics inside a decoder is part of that decoder.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.enabled: bool = False
        self.events: list[tuple[Sam3dBodyStage, torch.cuda.Event, torch.cuda.Event]] = []
        self.current: Sam3dBodyPass = "body"
        self.hand_passes: int = 0
        self.in_decoder: bool = False
        self._wrap(model, "forward_step", self._pass_from_decoder_type)
        self._wrap(model, "run_keypoint_prompt", lambda *_: "prompt")
        for head in (model.head_pose, model.head_pose_hand):
            self._wrap_timed(head, "mhr_forward", lambda: None if self.in_decoder else "mhr")
        self._hook(model.get_submodule("backbone"), lambda: f"backbone_{self.current}", nested=False)
        for decoder in ("decoder", "decoder_hand"):
            self._hook(model.get_submodule(decoder), lambda: f"decoder_{self.current}", nested=True)

    def start_frame(self) -> None:
        self.current = "body"
        self.hand_passes = 0

    def _pass_from_decoder_type(self, _batch: dict, decoder_type: str = "body") -> Sam3dBodyPass:
        if decoder_type == "body":
            return "body"
        self.hand_passes += 1
        return "left_hand" if self.hand_passes == 1 else "right_hand"

    def _wrap(self, owner: object, name: str, label: Callable[..., Sam3dBodyPass]) -> None:
        method = getattr(owner, name)

        def wrapped(*args: object, **kwargs: object) -> object:
            self.current = label(*args, **kwargs)
            return method(*args, **kwargs)

        setattr(owner, name, wrapped)

    def _wrap_timed(self, owner: object, name: str, stage: Callable[[], Sam3dBodyStage | None]) -> None:
        method = getattr(owner, name)

        def wrapped(*args: object, **kwargs: object) -> object:
            label: Sam3dBodyStage | None = stage() if self.enabled else None
            if label is None:
                return method(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = method(*args, **kwargs)
            end.record()
            self.events.append((label, start, end))
            return result

        setattr(owner, name, wrapped)

    def _hook(self, module: torch.nn.Module, stage: Callable[[], str], *, nested: bool) -> None:
        def before(*_: object) -> None:
            if nested:
                self.in_decoder = True
            if self.enabled:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                self.events.append((stage(), start, torch.cuda.Event(enable_timing=True)))  # type: ignore[arg-type]

        def after(*_: object) -> None:
            if nested:
                self.in_decoder = False
            if self.enabled:
                self.events[-1][2].record()

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    def stage_ms(self, total_ms: float) -> dict[Sam3dBodyStage, float]:
        """GPU time per stage of the recorded passes, plus ``other`` and ``total``; their work must have completed."""
        totals: dict[Sam3dBodyStage, float] = {}
        for stage, start, end in self.events:
            totals[stage] = totals.get(stage, 0.0) + start.elapsed_time(end)
        totals["other"] = total_ms - sum(totals.values())
        totals["total"] = total_ms
        return totals


class Sam3dBodyPredictor(TopDownBodyPose3d):
    """RGB frames + person boxes (+ masks, + camera) -> MHR whole-body people with hands."""

    skeleton: KeypointSkeleton = MHR_70

    def __init__(self, config: Sam3dBodyConfig) -> None:
        from sam3d_body.build_models import load_sam_3d_body
        from sam3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

        self.config: Sam3dBodyConfig = config
        checkpoint: Path = config.checkpoint_path()
        model, _ = load_sam_3d_body(checkpoint / "model.ckpt", device="cuda", mhr_path=checkpoint / "assets" / "mhr_model.pt")
        self.estimator: SAM3DBodyEstimator = SAM3DBodyEstimator(model)
        self.faces: Int64[Tensor, "f 3"] = torch.from_numpy(self.estimator.faces.astype(np.int64)).cuda()
        self.num_vertices: int = int(self.faces.max()) + 1
        self.num_mhr_joints: int = int(model.get_parameter("head_pose.joint_rotation").shape[0])
        self.profile_stages: bool = False
        """Record per-frame stage times in ``stage_ms``; off, inference adds no events or synchronization."""
        self.stage_ms: dict[Sam3dBodyStage, list[float]] = {}
        self.clock: _StageClock = _StageClock(model)

    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections, camera: CalibratedCamera | None = None
    ) -> Sam3dBodyReconstruction:
        """Reconstruct every detected person in the source camera frame.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the CUDA device.
            detections: Person boxes; ``masks`` (with ``scores`` as their weight) prompt the model when present.
            camera: Its intrinsics replace the default field of view (image diagonal as focal length); lens distortion
                is not modelled, so pass undistorted frames.

        Returns:
            One row per detection, in detection order.
        """
        validate_frames_rgb(frames_rgb)
        device: torch.device = frames_rgb.device
        intrinsics: Float32[ndarray, "3 3"] | None = None if camera is None else camera.intrinsics.astype(np.float32)
        frames: UInt8[ndarray, "b h w 3"] = frames_rgb.cpu().numpy()
        boxes: Float32[ndarray, "n 4"] = detections.xyxy_numpy()
        scores: Float32[ndarray, "n"] = detections.scores.detach().cpu().numpy().astype(np.float32, copy=False)
        masks: ndarray | None = None if detections.masks is None else detections.masks.cpu().numpy()
        frame_of_row: Int64[ndarray, "n"] = detections.frame_indices.cpu().numpy()
        predictions: list[Any] = []
        frame_indices: list[int] = []
        rows: list[int] = []
        profile: bool = self.profile_stages
        self.clock.enabled = profile
        for frame_index in range(len(frames)):
            selected: Int64[ndarray, "m"] = np.flatnonzero(frame_of_row == frame_index)
            if len(selected) == 0:
                continue
            self.clock.start_frame()
            self.clock.events.clear()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if profile:
                start.record()
            people: list[Any] = self.estimator.process_one_image(
                frames[frame_index],
                xyxy=boxes[selected],
                masks=None if masks is None else masks[selected].astype(np.float32),
                masks_score=None if masks is None else scores[selected],
                K_33=intrinsics,
                inference_type=self.config.mode,
            )
            if profile:
                end.record()
                torch.cuda.synchronize()
                for stage, milliseconds in self.clock.stage_ms(start.elapsed_time(end)).items():
                    self.stage_ms.setdefault(stage, []).append(milliseconds)
            predictions.extend(people)
            frame_indices.extend([frame_index] * len(people))
            rows.extend(selected.tolist())
        self.clock.enabled = False
        return self._to_reconstruction(predictions, frame_indices, scores[rows], device)

    def _to_reconstruction(self, predictions: list[Any], frame_indices: list[int], scores: Float32[ndarray, "n"], device: torch.device) -> Sam3dBodyReconstruction:
        """Upstream per-person numpy results as one camera-space reconstruction on ``device``."""
        count: int = len(predictions)
        joints: int = MHR_70.num_keypoints
        if count == 0:
            empty = torch.empty((0, 4), dtype=torch.float32, device=device)
            return Sam3dBodyReconstruction(
                keypoints=Keypoints3d.empty(MHR_70, device, root_indices=ROOT_INDICES, with_xyz_camera=True),
                boxes=empty,
                vertices=torch.empty((0, self.num_vertices, 3), device=device),
                faces=self.faces,
                translation=torch.empty((0, 3), device=device),
                focal_length=torch.empty((0,), device=device),
                global_rotation=torch.empty((0, 3), device=device),
                body_pose=torch.empty((0, 133), device=device),
                hand_pose=torch.empty((0, 108), device=device),
                scale=torch.empty((0, 28), device=device),
                shape=torch.empty((0, 45), device=device),
                expression=torch.empty((0, 72), device=device),
                joint_coordinates=torch.empty((0, self.num_mhr_joints, 3), device=device),
                joint_rotations=torch.empty((0, self.num_mhr_joints, 3, 3), device=device),
                left_hand_boxes=empty if self.config.mode == "full" else None,
                right_hand_boxes=empty if self.config.mode == "full" else None,
            )

        def tensor(field: str) -> Float32[Tensor, "n ..."]:
            return torch.from_numpy(_person_rows(predictions, field)).to(device)

        translation: Float32[Tensor, "n 3"] = tensor("pred_cam_t")
        xyz: Float32[Tensor, "n 70 3"] = tensor("pred_keypoints_3d") + translation[:, None]
        root_z: Float32[Tensor, "n 1"] = xyz[:, list(ROOT_INDICES), 2].mean(dim=1, keepdim=True)
        full: bool = self.config.mode == "full"
        return Sam3dBodyReconstruction(
            keypoints=Keypoints3d(
                xy=tensor("pred_keypoints_2d"),
                z_root_relative=xyz[:, :, 2] - root_z,
                scores=torch.from_numpy(scores).to(device)[:, None].expand(-1, joints).contiguous(),
                frame_indices=torch.tensor(frame_indices, dtype=torch.int64, device=device),
                skeleton=MHR_70,
                root_indices=ROOT_INDICES,
                xyz_camera=xyz,
            ),
            boxes=tensor("bbox"),
            vertices=tensor("pred_vertices") + translation[:, None],
            faces=self.faces,
            translation=translation,
            focal_length=tensor("focal_length"),
            global_rotation=tensor("global_rot"),
            body_pose=tensor("body_pose_params"),
            hand_pose=tensor("hand_pose_params"),
            scale=tensor("scale_params"),
            shape=tensor("shape_params"),
            expression=tensor("expr_params"),
            joint_coordinates=tensor("pred_joint_coords") + translation[:, None],
            joint_rotations=tensor("pred_global_rots"),
            left_hand_boxes=tensor("lhand_bbox") if full else None,
            right_hand_boxes=tensor("rhand_bbox") if full else None,
        )
