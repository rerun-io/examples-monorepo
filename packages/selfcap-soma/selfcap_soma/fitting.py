"""Batched sparse-observation SOMA fit with one shared identity."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from soma.body import SOMALayer, SOMAPoseOutput
from torch import Tensor

from selfcap_soma.anthropometry import SCALE_MAX, SCALE_MIN, bounded_scale, usable_ears
from selfcap_soma.assets import soma_assets
from selfcap_soma.catalog import Observations
from selfcap_soma.hand_pose import HandOrientations, HandOrientationSource, frame, hand_limit_penalty, orientation_targets
from selfcap_soma.landmarks import EAR_VERTICES, ear_vertices, rotation_delta
from selfcap_soma.losses import body_acceleration_penalty, body_lengths, robust_points, rotation_temporal_penalty, symmetry_penalty
from selfcap_soma.observations import coco_joint_names
from selfcap_soma.shape_frames import InsufficientBodyObservation
from selfcap_soma.upper_body import lower_body_mask, mesh_selection, visible_joint_mask


@dataclass(frozen=True, slots=True)
class FitResult:
    poses: Float32[ndarray, "n 77 3"]
    """Joint-local axis angles; first entry is global pelvis rotation."""
    translations: Float32[ndarray, "n 3"]
    """World root translations in metres."""
    identity: Float32[ndarray, "1 128"]
    """One shared native SOMA identity; active coefficients are selected by FitOptions."""
    scale: float
    """One shared global scale."""
    joints: Float32[ndarray, "n 77 3"]
    """Fitted world joints."""
    vertices: Float32[ndarray, "n v 3"]
    """Low-LOD world surface, generated only after optimization."""
    faces: Int32[ndarray, "f 3"]
    """Shared triangle topology."""
    names: tuple[str, ...]
    """Nonvirtual joint names, matching joints."""
    parents: Int64[ndarray, "78"]
    """Public joint parent indices including virtual root."""
    seconds: float
    """Optimization and mesh evaluation wall time, excluding asset load."""
    peak_bytes: int
    """Peak PyTorch allocated GPU bytes, not driver-wide memory."""
    head_angle_deg: Float64[ndarray, "n"]
    """Rotation error against the shared camera-to-skull attachment."""
    head_attachment: Float32[ndarray, "3 3"]
    """Shared tracked-camera to skull rotation."""
    ear_landmarks: Float32[ndarray, "n 2 3"]
    """Skinned tragus-region surface landmarks."""
    head_anchor: Literal["ears", "ears-rotation"]
    """Ear landmarks, with optional tracked-head rotation."""
    head_offset: Float32[ndarray, "3"]
    """Tracked-camera to the ear anchor, in metres in the skull frame."""
    bone_scales: Float32[ndarray, "1 60"]
    """Native SOMA local bone-length ratios, shared over all frames."""
    upper_body_vertices: Int64[ndarray, "v"]
    """Required rest-pose vertex selection for the upper-body display surface."""
    visible_joints: Int64[ndarray, "j"]
    """Required public joint indices shown in upper-body displays."""
    display_twists: bool = False
    """Whether the output surface uses procedural twist distribution."""
    hand_orientations: bool = False
    """Whether raw Quest bone rotations constrained this fit."""
    hand_orientation_source: HandOrientationSource | None = None
    """Per-hand measured orientation availability; none denotes position-derived fallback."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FitOptions:
    leg_support_weight: float = 1.0
    """Relative knee/ankle observation weight; final upper-body runs use zero."""
    hip_landmark_weight: float = 1.0
    """Relative COCO hip weight; these landmarks differ from anatomical joint centres."""
    freeze_legs: bool = False
    """Keep hip-to-knee and descendant local rotations at the rest pose."""
    torso_temporal_weight: float = 0.0
    """World-space pelvis/chest angular-velocity-change penalty."""
    body_acceleration_weight: float = 0.0
    """World position acceleration on root, spine, neck and shoulder joints only."""
    final_learning_rate: float = 0.025
    """Final pose-pass Adam rate, reached by decay during the second half."""
    limb_symmetry_weight: float = 0.0
    """Shared-shape penalty on left/right arm, forearm, thigh and shin lengths."""
    head_mode: Literal["ears", "ears-rotation"] = "ears-rotation"
    """Triangulated ears, with optional soft Quest orientation."""
    head_rotation_weight: float = 1.0
    """Rotation weight, ignored by ears-only mode."""
    orientation_weight: float = 0.0
    """Raw Quest world bone-orientation weight, after constant frame alignment."""
    display_twists: bool = False
    """Enable procedural twist distribution for final display geometry."""
    learn_head_rotation: bool = False
    """Fit one shared camera-to-skull rotation during calibration."""
    shape_components: int = 16
    """Number of active native SOMA identity coefficients."""
    identity_prior: float = 1.0
    """Mean squared identity penalty."""
    scale_prior: float = 20.0
    """Squared log global-scale penalty."""
    learn_bone_scales: bool = False
    """Fit native body bone-length controls during the shape stage."""
    bone_length_weight: float = 0.0
    """Calibration-only loss matching neutral widths and limb lengths to observed medians."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Calibration:
    identity: Float32[ndarray, "1 128"]
    """Shared statistical body identity."""
    scale: float
    """Shared global scale."""
    head_offset: Float32[ndarray, "3"]
    """Camera-to-anchor translation in metres, expressed in skull frame for head mode."""
    bone_scales: Float32[ndarray, "1 60"]
    """Shared native bone ratios in SOMA's public control order."""

    head_attachment: Float32[ndarray, "3 3"]
    """Camera-to-skull rotation fitted during shape calibration."""
    head_anchor: Literal["ears", "ears-rotation"]
    """Ear-calibrated head position anchor."""


def pose_gradient_mask(
    names: tuple[str, ...], leg_indices: Int64[ndarray, "k"], freeze_legs: bool, device: torch.device
) -> Float32[Tensor, "1 j 1"]:
    """Select pose gradients for frozen-leg fitting."""
    mask = torch.ones((1, len(names), 1), device=device)
    if freeze_legs:
        mask[:, leg_indices] = 0.0
    return mask


@cache
def soma_layer(assets: Path | None, procedural: bool) -> SOMALayer:
    """Reuse the two model variants in this process; each fit resets their identity state."""
    model: SOMALayer = SOMALayer(
        data_root=soma_assets(assets),
        device="cuda",
        identity_model_type="soma",
        mode="torch",
        lod="low",
        enable_procedural_transforms=procedural,
        correctives_model_path=None,
    )
    model.requires_grad_(False)
    return model


@dataclass(frozen=True, slots=True)
class FitTargets:
    """Observation tensors shared by shape calibration and pose fitting."""

    exo: Float32[Tensor, "n j 3"]
    """Triangulated COCO body and hand positions, in metres."""
    confidence: Float32[Tensor, "n j"]
    """Per-joint triangulation confidence."""
    hands: Float32[Tensor, "n j 3"]
    """Quest world hand positions; missing joints are NaN."""
    body_confidence: Float32[Tensor, "n j"]
    """Triangulation confidence after leg and hip weighting."""
    hand_weights: Float32[Tensor, "n j"]
    """One for each finite Quest joint, zero for missing observations."""
    head_valid: Float32[Tensor, "n"]
    """One for each finite observed head rotation, zero otherwise."""
    head_frames: Float32[Tensor, "n 3 3"]
    """Observed head rotations, with identity placeholders for missing rows."""
    dt: Float32[Tensor, "interval"]
    """Positive timestamp differences in seconds."""
    ears_target: Float32[Tensor, "n 2 3"]
    """Triangulated left and right tragus-region landmarks, in metres."""
    ears_confidence: Float32[Tensor, "n 2"]
    """Ear confidence after the span and minimum-frame gates."""


@dataclass(frozen=True, slots=True)
class FitLayout:
    """Public skeleton indices used by the shared fit objective."""

    names: tuple[str, ...]
    """Nonvirtual SOMA joint names in public order."""
    soma: list[int]
    """SOMA indices matching the selected COCO body and hand targets."""
    body_columns: list[int]
    """Target columns for the COCO body landmarks."""
    hand_columns: list[int]
    """Target columns for the COCO hand landmarks."""
    head_index: int
    """Skull index in the nonvirtual public layout."""
    body_chain: list[int]
    """Pelvis, spine, neck and shoulder indices for temporal regularization."""


@dataclass(frozen=True, slots=True)
class HandConstraints:
    """Aligned hand targets and limits for position-derived rotations."""

    orientation_indices: Int64[Tensor, "j"] | None
    """Constrained transform indices, including the virtual root offset."""
    orientation_rotations: Float32[Tensor, "n j 3 3"] | None
    """Aligned world hand rotations."""
    orientation_valid: Bool[Tensor, "n j"] | None
    """Finite aligned rotation targets per frame and joint."""
    rotation_weights: Float32[Tensor, "j"] | None
    """Per-joint weights derived from the actual target source."""
    fallback_ids: list[int]
    """Position-only hand joint indices whose rotation limits need a prior."""
    fallback_basis_tensor: Float32[Tensor, "j 3 3"] | Float32[Tensor, "0"]
    """Bone-aligned frames for position-only hand limits; empty when unused."""
    fallback_limit_tensor: Float32[Tensor, "j 4"] | Float32[Tensor, "0"]
    """Flexion, abduction and twist limits in radians; empty when unused."""
    fallback_sign_tensor: Float32[Tensor, "j"]
    """Flexion signs chosen from the initial hand poses."""


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Objective total and the residual groups printed during optimization."""

    total: Float32[Tensor, ""]
    """Weighted fitting objective."""
    body: Float32[Tensor, ""]
    """Body observation term."""
    hands: Float32[Tensor, ""]
    """Quest hand observation term."""
    ears: Float32[Tensor, ""]
    """Ear landmark observation term."""


@dataclass(frozen=True, slots=True)
class FitObjective:
    """The same observation objective for shared-shape and fixed-shape fitting."""

    model: SOMALayer
    """Prepared differentiable body model."""
    options: FitOptions
    """Objective weights."""
    targets: FitTargets
    """Observed positions, scores, clocks and head frames."""
    layout: FitLayout
    """Public skeleton indexing."""
    hands: HandConstraints
    """Aligned hand rotations and fallback limits."""
    base_attachment: Float32[Tensor, "3 3"]
    """Initial camera-to-skull rotation."""

    def __call__(self, poses: Float32[Tensor, "n 77 3"], translations: Float32[Tensor, "n 3"], rotation_raw: Float32[Tensor, "3"]) -> LossTerms:
        """Evaluate the shared objective without changing optimizer or shape state."""
        posed: SOMAPoseOutput = self.model.pose(poses, transl=translations, fk_only=True, apply_correctives=False)
        selected: Float32[Tensor, "n j 3"] = posed.joints[:, self.layout.soma]
        body_loss = robust_points(selected[:, self.layout.body_columns], self.targets.exo[:, self.layout.body_columns], self.targets.body_confidence[:, self.layout.body_columns], 0.02)
        exo_hand_loss = robust_points(selected[:, self.layout.hand_columns], self.targets.exo[:, self.layout.hand_columns], self.targets.confidence[:, self.layout.hand_columns], 0.02)
        hand_loss = robust_points(selected, self.targets.hands, self.targets.hand_weights, 0.01)
        head_attachment = self.base_attachment @ rotation_delta(rotation_raw)
        target_rotation = self.targets.head_frames @ head_attachment
        skin = self.model.batched_skinning
        ears = ear_vertices(
            skin.bind_shapes,
            self.model.skinning_weights,
            skin.inverse_bind_transform,
            posed.transforms[:, torch.argsort(self.model.public_transform_joint_indices)],
        )
        ear_loss = robust_points(ears, self.targets.ears_target, self.targets.ears_confidence, 0.02)
        head_rotation = posed.transforms[:, self.layout.head_index + 1, :3, :3]
        rotation_loss = torch.zeros((), device="cuda")
        if self.options.head_mode != "ears":
            rotation_loss = (
                ((head_rotation - target_rotation).square().mean(dim=(-1, -2)) * self.targets.head_valid).sum() / self.targets.head_valid.sum().clamp_min(1.0) / 0.03
            )
        velocity = (translations[1:] - translations[:-1]) / self.targets.dt[:, None]
        angular_velocity = (poses[1:] - poses[:-1]) / self.targets.dt[:, None, None]
        smooth = (velocity[1:] - velocity[:-1]).square().mean() + 0.1 * (angular_velocity[1:] - angular_velocity[:-1]).square().mean()
        prior = poses[:, 1:].square().mean() + 0.5 * torch.relu(poses[:, 1:].norm(dim=-1) - 2.6).square().mean()
        loss = (
            body_loss
            + 0.25 * exo_hand_loss
            + 3.0 * hand_loss
            + (0.0 if self.options.head_mode == "ears" else self.options.head_rotation_weight) * rotation_loss
            + 3.0 * ear_loss
            + 0.005 * smooth
            + 0.03 * prior
        )
        if self.options.body_acceleration_weight > 0.0:
            loss = loss + self.options.body_acceleration_weight * body_acceleration_penalty(posed.joints[:, self.layout.body_chain], self.targets.dt)
        if self.options.torso_temporal_weight > 0.0:
            loss = loss + self.options.torso_temporal_weight * rotation_temporal_penalty(
                posed.transforms[:, [self.layout.names.index("Hips") + 1, self.layout.names.index("Chest") + 1], :3, :3], self.targets.dt
            )
        if self.hands.orientation_valid is not None:
            assert self.hands.rotation_weights is not None and self.hands.orientation_indices is not None and self.hands.orientation_rotations is not None
            orientation_error = (posed.transforms[:, self.hands.orientation_indices, :3, :3] - self.hands.orientation_rotations).square().mean((-1, -2))
            loss = loss + self.options.orientation_weight * (orientation_error * self.hands.orientation_valid * self.hands.rotation_weights).sum() / self.hands.orientation_valid.sum().clamp_min(1) / 0.03
        if self.hands.fallback_ids:
            loss = loss + hand_limit_penalty(poses[:, self.hands.fallback_ids], self.hands.fallback_basis_tensor, self.hands.fallback_limit_tensor, self.hands.fallback_sign_tensor)
        loss = loss + 0.02 * rotation_raw.square().mean()
        return LossTerms(loss, body_loss, hand_loss, ear_loss)


@dataclass(frozen=True, slots=True)
class FittedParameters:
    """Optimized numeric state before display geometry is built."""

    model: SOMALayer
    """Model retained for display evaluation."""
    names: tuple[str, ...]
    """Nonvirtual SOMA joint names, in public joint order."""
    poses: Float32[Tensor, "n 77 3"]
    """Optimized joint-local axis angles; the first joint is the global pelvis."""
    translations: Float32[Tensor, "n 3"]
    """Optimized world root positions, in metres."""
    identity: Float32[Tensor, "1 128"]
    """All 128 identity coefficients, padded after fitting the active components."""
    log_scale: Float32[Tensor, ""]
    """Optimized logarithm of global body scale."""
    bone_scales: Float32[Tensor, "1 60"]
    """Shared local bone-length ratios in public control order."""
    head_attachment: Float32[Tensor, "3 3"]
    """Fitted camera-to-skull rotation."""
    head_frames: Float32[Tensor, "n 3 3"]
    """Observed head-to-world rotations, with identity placeholders for missing rows."""
    head_index: int
    """Skull joint index in the nonvirtual public layout."""
    head_offset: Float32[Tensor, "3"]
    """Tracked-camera to head-anchor translation, in metres."""
    started: float
    """Monotonic start time for optimization and display evaluation."""


def fit_parameters(
    data: Observations,
    assets: Path | None,
    iterations: int = 400,
    shape_iterations: int = 60,
    options: FitOptions | None = None,
    fixed: Calibration | None = None,
    initial: FitResult | None = None,
    orientations: HandOrientations | None = None,
) -> FittedParameters:
    """Prepare and optimize the shared objective; freeze identity after the shape stage."""
    options = options or FitOptions()
    head_mode = options.head_mode
    seed = fixed if fixed is not None else initial
    active_components = options.shape_components if fixed is None else 128
    torch.cuda.reset_peak_memory_stats()
    model: SOMALayer = soma_layer(assets, False)
    names: tuple[str, ...] = tuple(model.public_joint_names[1:])
    mapping: dict[int, str] = coco_joint_names()
    coco: list[int] = list(mapping)
    soma: list[int] = [names.index(mapping[k]) for k in coco]
    neutral_identity: Float32[Tensor, "1 128"] = torch.zeros(1, 128, device="cuda") if seed is None else torch.as_tensor(seed.identity, device="cuda")
    with torch.no_grad():
        model.prepare_identity(
            neutral_identity,
            scale_params=None if seed is None else torch.as_tensor(seed.bone_scales, device="cuda"),
            global_scale=1.0 if seed is None else bounded_scale(seed.scale),
        )
        neutral: SOMAPoseOutput = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
    neutral_joints: Float32[ndarray, "77 3"] = neutral.joints[0].detach().cpu().numpy()
    neutral_transforms: Float32[Tensor, "1 78 4 4"] = neutral.transforms.detach()
    # RDF frame: anatomical right, down, and forward. The eyes and cranial
    # axis initialize the attachment; ear mode learns a segment-wide correction.
    right: Float32[ndarray, "3"] = neutral_joints[names.index("RightEye")] - neutral_joints[names.index("LeftEye")]
    up: Float32[ndarray, "3"] = neutral_joints[names.index("HeadEnd")] - neutral_joints[names.index("Head")]
    right /= np.linalg.norm(right)
    up -= right * (up @ right)
    up /= np.linalg.norm(up)
    camera_basis: Float32[ndarray, "3 3"] = np.column_stack((right, -up, np.cross(right, -up)))
    head_index = names.index("Head")
    attachment: Float32[ndarray, "3 3"] = camera_basis.T @ neutral_transforms[0, head_index + 1, :3, :3].cpu().numpy()
    if seed is not None:
        attachment = seed.head_attachment
    if head_mode == "ears":
        attachment = np.eye(3, dtype=np.float32)
    base_attachment = torch.as_tensor(attachment, dtype=torch.float32, device="cuda")
    rotation_raw = torch.nn.Parameter(
        torch.zeros(3, device="cuda"), requires_grad=options.learn_head_rotation and fixed is None and head_mode != "ears"
    )

    # Per-frame Kabsch initialization from shoulders and hips avoids a
    # convention-dependent global yaw guess.
    anchors: list[int] = [key for key in (5, 6, 11, 12) if np.isfinite(data.exo[:, key]).all(1).sum() >= 2]
    if len(anchors) < 3:
        raise InsufficientBodyObservation("initialization needs at least three observed torso anchors")
    initial_poses: Float32[ndarray, "n 77 3"]
    initial_translation: Float32[ndarray, "n 3"]
    if initial is not None:
        initial_poses = initial.poses.copy()
        initial_translation = initial.translations.copy()
    else:
        source: Float32[ndarray, "a 3"] = neutral_joints[[names.index(mapping[k]) for k in anchors]]
        source_center: Float32[ndarray, "3"] = source.mean(axis=0)
        initial_poses = np.zeros((len(data.times), 77, 3), dtype=np.float32)
        initial_translation = np.zeros((len(data.times), 3), dtype=np.float32)
        initial_anchors: Float64[ndarray, "n a 3"] = data.exo[:, anchors].copy()
        for joint in range(len(anchors)):
            valid = np.isfinite(initial_anchors[:, joint]).all(-1)
            for axis in range(3):
                initial_anchors[:, joint, axis] = np.interp(data.times, data.times[valid], initial_anchors[valid, joint, axis])
        for i, target in enumerate(initial_anchors):
            rotation: Rotation = Rotation.align_vectors(target - target.mean(axis=0), source - source_center)[0]
            initial_poses[i, 0] = rotation.as_rotvec()
            initial_translation[i] = target.mean(axis=0) - rotation.apply(source_center)
    leg_indices = np.flatnonzero(lower_body_mask(tuple(model.public_joint_names), model.public_joint_parent_ids.cpu().numpy().astype(np.int64))[1:])
    if options.freeze_legs:
        initial_poses[:, leg_indices] = 0.0
    aligned = None
    if orientations is not None:
        aligned = orientation_targets(orientations, data.times, names, neutral_joints, neutral_transforms[0, :, :3, :3].cpu().numpy())
        if initial is not None or "none" in orientations.sources:
            # Initialize rotations from measured world frames, rather than the
            # position-only solution's arbitrary roll local minimum.
            with torch.no_grad():
                start_frames = (
                    model.pose(torch.as_tensor(initial_poses, device="cuda"), fk_only=True, apply_correctives=False)
                    .transforms[:, :, :3, :3]
                    .cpu()
                    .numpy()
                )
            desired = start_frames.copy()
            for column, index in enumerate(aligned.joint_indices):
                desired[aligned.valid[:, column], index + 1] = aligned.rotations[aligned.valid[:, column], column]
            rest_frames = neutral_transforms[0, :, :3, :3].cpu().numpy()
            parents = model.public_joint_parent_ids.cpu().numpy()
            for column, index in enumerate(aligned.joint_indices):
                parent = parents[index + 1]
                local = rest_frames[parent] @ desired[:, parent].transpose(0, 2, 1) @ desired[:, index + 1] @ rest_frames[index + 1].T
                valid = aligned.valid[:, column]
                initial_poses[valid, index] = Rotation.from_matrix(local[valid]).as_rotvec()
    poses: Float32[Tensor, "n 77 3"] = torch.nn.Parameter(torch.as_tensor(initial_poses, device="cuda"))
    translations: Float32[Tensor, "n 3"] = torch.nn.Parameter(
        torch.as_tensor(initial_translation, device="cuda")
    )
    if not 1 <= options.shape_components <= 128:
        raise ValueError("shape components must be in [1,128]")
    identity = torch.nn.Parameter(neutral_identity[:, :active_components].clone(), requires_grad=fixed is None)
    log_scale = torch.nn.Parameter(
        torch.tensor(0.0 if seed is None else np.log(bounded_scale(seed.scale)), device="cuda", dtype=torch.float32), requires_grad=fixed is None
    )
    if model.num_scale_params != 60:
        raise ValueError("expected 60 native SOMA bone-scale controls")
    seed_bones = torch.ones(1, 60, device="cuda") if seed is None else torch.as_tensor(seed.bone_scales, device="cuda")
    bone_raw = torch.nn.Parameter(
        torch.atanh(((seed_bones - 1.0) / 0.3).clamp(-0.999, 0.999)), requires_grad=options.learn_bone_scales and fixed is None
    )
    body_bone_names = {
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
        "LeftShin",
        "RightShin",
        "LeftFoot",
        "RightFoot",
    }
    bone_mask = torch.tensor([[float(name in body_bone_names) for name in model.scale_param_names]], device="cuda")
    bone_scales = seed_bones
    head_offset = torch.zeros(3, device="cuda")
    parameters = (
        [poses, translations]
        + ([identity, log_scale] if fixed is None else [])
        + ([bone_raw] if options.learn_bone_scales and fixed is None else [])
        + ([rotation_raw] if options.learn_head_rotation and fixed is None else [])
    )
    optimizer: torch.optim.Adam = torch.optim.Adam(parameters, lr=0.025)
    exo: Float32[Tensor, "n j 3"] = torch.as_tensor(data.exo[:, coco], dtype=torch.float32, device="cuda")
    confidence: Float32[Tensor, "n j"] = torch.as_tensor(data.confidence[:, coco], dtype=torch.float32, device="cuda")
    hands: Float32[Tensor, "n j 3"] = torch.as_tensor(data.hands[:, coco], dtype=torch.float32, device="cuda")
    body_confidence = confidence.clone()
    body_confidence[:, [i for i, key in enumerate(coco) if 13 <= key <= 16]] *= options.leg_support_weight
    body_confidence[:, [i for i, key in enumerate(coco) if key in (11, 12)]] *= options.hip_landmark_weight
    hand_weights: Float32[Tensor, "n j"] = torch.isfinite(hands).all(-1).float()
    head_valid = torch.as_tensor(np.isfinite(data.head_rotations).all(axis=(1, 2)), dtype=torch.float32, device="cuda")
    head_frames = (
        torch.eye(3, device="cuda").expand(len(data.times), -1, -1)
        if head_mode == "ears"
        else torch.as_tensor(
            np.where(np.isfinite(data.head_rotations).all(axis=(1, 2))[:, None, None], data.head_rotations, np.eye(3)),
            device="cuda",
            dtype=torch.float32,
        )
    )
    dt: Float32[Tensor, "interval"] = torch.as_tensor(np.diff(data.times), dtype=torch.float32, device="cuda")
    ears_target = torch.as_tensor(data.exo[:, 3:5], dtype=torch.float32, device="cuda")
    ears_confidence = torch.as_tensor(data.confidence[:, 3:5], dtype=torch.float32, device="cuda")
    ear_pairs = usable_ears(data.exo, data.confidence)
    if ear_pairs.sum() < 3:
        ear_pairs[:] = False
    ears_confidence = ears_confidence * torch.as_tensor(ear_pairs[:, None], device="cuda")
    warmup: int = min(100, iterations // 3)
    shape_end: int = min(warmup + shape_iterations, iterations - 1)
    target_lengths = torch.nanmedian(body_lengths(torch.as_tensor(data.exo[:, 5:17], device="cuda", dtype=torch.float32)), dim=0).values
    soma_body = [names.index(mapping[key]) for key in range(5, 17)]
    body_columns: list[int] = [i for i, key in enumerate(coco) if key < 17]
    hand_columns: list[int] = [i for i, key in enumerate(coco) if key >= 91]
    orientation_indices: Int64[Tensor, "j"] | None = None
    orientation_rotations: Float32[Tensor, "n j 3 3"] | None = None
    orientation_valid: Bool[Tensor, "n j"] | None = None
    if aligned is not None:
        orientation_indices = torch.as_tensor(aligned.joint_indices + 1, device="cuda")
        orientation_rotations = torch.as_tensor(aligned.rotations, device="cuda")
        orientation_valid = torch.as_tensor(aligned.valid, device="cuda")
        print("Quest-to-SOMA alignment mean spread deg:", float(np.median(aligned.spread_deg)), flush=True)
    pose_mask = pose_gradient_mask(names, leg_indices, options.freeze_legs, poses.device)
    body_chain = [
        names.index(name) for name in ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm")
    ]
    fallback_ids = []
    fallback_bases = []
    fallback_limits = []
    fallback_signs = []
    if orientations is not None:
        for side_index, side in enumerate(("Left", "Right")):
            if orientations.sources[side_index] != "none":
                continue
            wrist = names.index(side + "Hand")
            normal = np.cross(
                neutral_joints[names.index(side + "HandIndex2")] - neutral_joints[wrist],
                neutral_joints[names.index(side + "HandPinky2")] - neutral_joints[wrist],
            ).astype(np.float64)
            for index, name in enumerate(names):
                if name == side + "Hand":
                    direction = neutral_joints[wrist] - neutral_joints[names.index(side + "ForeArm")]
                    limit = [1.2, 1.2, 0.6, 0.8]
                elif name.startswith(side + "Hand") and name[-1:].isdigit():
                    child = name[:-1] + str(int(name[-1]) + 1)
                    if child not in names:
                        child = name[:-1] + "End"
                    direction = neutral_joints[names.index(child)] - neutral_joints[index]
                    limit = [1.4, 1.4, 0.8, 0.3] if "Thumb" in name else [0.35, 2.0, 0.6 if name.endswith("1") or name.endswith("2") else 0.3, 0.25]
                else:
                    continue
                basis = frame(direction.astype(np.float64), normal).astype(np.float32)
                flex = initial_poses[:, index] @ basis[:, 1]
                fallback_ids.append(index)
                fallback_bases.append(basis)
                fallback_limits.append(limit)
                fallback_signs.append(-1.0 if np.median(flex) < 0 else 1.0)
    fallback_basis_tensor = torch.as_tensor(np.asarray(fallback_bases), dtype=torch.float32, device="cuda")
    fallback_limit_tensor = torch.as_tensor(fallback_limits, dtype=torch.float32, device="cuda")
    fallback_sign_tensor = torch.as_tensor(fallback_signs, dtype=torch.float32, device="cuda")
    rotation_weights = None
    if aligned is not None and orientations is not None:
        rotation_weights = torch.as_tensor(aligned.weights, device="cuda")
    targets_input = FitTargets(exo, confidence, hands, body_confidence, hand_weights, head_valid, head_frames, dt, ears_target, ears_confidence)
    layout_input = FitLayout(names, soma, body_columns, hand_columns, head_index, body_chain)
    hands_input = HandConstraints(orientation_indices, orientation_rotations, orientation_valid, rotation_weights, fallback_ids, fallback_basis_tensor, fallback_limit_tensor, fallback_sign_tensor)
    objective = FitObjective(model, options, targets_input, layout_input, hands_input, base_attachment)
    start: float = perf_counter()
    for iteration in range(iterations):
        if fixed is not None and iteration >= iterations // 2:
            fraction = (iteration - iterations // 2) / max(1, iterations - 1 - iterations // 2)
            for group in optimizer.param_groups:
                group["lr"] = 0.025 * (options.final_learning_rate / 0.025) ** fraction
        optimizer.zero_grad(set_to_none=True)
        if fixed is None and warmup <= iteration < shape_end:
            bone_scales = 1.0 + 0.3 * torch.tanh(bone_raw) * bone_mask
            model.prepare_identity(
                torch.nn.functional.pad(identity, (0, 128 - active_components)),
                scale_params=bone_scales if options.learn_bone_scales else None,
                global_scale=log_scale.exp(),
            )
        elif fixed is None and iteration == shape_end:
            # Upstream reposing includes its trainable bind rotations even
            # when identity inputs are detached. Freeze the entire cache.
            with torch.no_grad():
                bone_scales = 1.0 + 0.3 * torch.tanh(bone_raw) * bone_mask
                model.prepare_identity(
                    torch.nn.functional.pad(identity.detach(), (0, 128 - active_components)),
                    scale_params=bone_scales if options.learn_bone_scales else None,
                    global_scale=log_scale.detach().exp(),
                )
        terms = objective(poses, translations, rotation_raw)
        loss = terms.total
        if fixed is None and warmup <= iteration < shape_end:
            loss = loss + options.identity_prior * identity.square().mean() + options.scale_prior * log_scale.square()
            loss = loss + 0.1 * (((bone_scales - 1.0) / 0.15).square() * bone_mask).sum() / bone_mask.sum()
            if options.bone_length_weight > 0.0 or options.limb_symmetry_weight > 0.0:
                neutral_shape = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
                lengths = body_lengths(neutral_shape.joints[:, soma_body])[0]
                if options.limb_symmetry_weight > 0.0:
                    loss = loss + options.limb_symmetry_weight * symmetry_penalty(lengths[None, 3:])
                valid_lengths = torch.isfinite(target_lengths)
                length_weights = torch.ones_like(lengths)
                length_weights[7:] *= options.leg_support_weight
                length_weights[1:3] *= options.hip_landmark_weight
                length_cost = torch.nn.functional.smooth_l1_loss(
                    lengths[valid_lengths] / 0.02, target_lengths[valid_lengths] / 0.02, beta=2.0, reduction="none"
                )
                loss = loss + options.bone_length_weight * (length_cost * length_weights[valid_lengths]).sum() / length_weights[
                    valid_lengths
                ].sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite fit at iteration {iteration}")
        loss.backward()
        if options.freeze_legs:
            assert poses.grad is not None
            poses.grad.mul_(pose_mask)
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(np.log(SCALE_MIN), np.log(SCALE_MAX))
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(
                f"iteration={iteration} loss={loss.item():.3f} exo={terms.body.item():.3f} hands={terms.hands.item():.3f} ears={terms.ears.item():.3f}",
                flush=True,
            )
    with torch.no_grad():
        if fixed is None:
            bone_scales = 1.0 + 0.3 * torch.tanh(bone_raw) * bone_mask
        identity_full = torch.nn.functional.pad(identity, (0, 128 - active_components))
        model.prepare_identity(identity_full, scale_params=bone_scales, global_scale=log_scale.exp())
        head_attachment = base_attachment @ rotation_delta(rotation_raw)
    return FittedParameters(model, names, poses, translations, identity_full, log_scale, bone_scales, head_attachment, head_frames, head_index, head_offset, start)


def build_display_fit(data: Observations, assets: Path | None, fitted: FittedParameters, options: FitOptions, orientations: HandOrientations | None) -> FitResult:
    """Build the display mesh and joint arrays only after numerical optimization."""
    model = fitted.model
    names = fitted.names
    with torch.no_grad():
        if options.display_twists:
            model = soma_layer(assets, True)
            model.prepare_identity(
                fitted.identity, scale_params=fitted.bone_scales, global_scale=fitted.log_scale.exp()
            )
        rest_pose = model.pose(torch.zeros_like(fitted.poses[:1]), apply_correctives=False)
        waist_height = float(rest_pose.joints[0, [names.index("LeftLeg"), names.index("RightLeg")], 1].mean().item()) - 0.025
        selection = mesh_selection(rest_pose.vertices[0].cpu().numpy(), model.faces.cpu().numpy(), waist_height)
        rim = np.flatnonzero(np.any(selection.rest_offsets != 0.0, axis=1))
        rest_offsets = torch.as_tensor(selection.rest_offsets[rim], device="cuda")
        rim_skinning_weights = model.public_skinning_weights()[rim]
        inverse_bind_transforms = torch.linalg.inv(model.public_bind_transforms_world())
        meshes: list[Float32[ndarray, "b v 3"]] = []
        joints: list[Float32[ndarray, "b 77 3"]] = []
        for begin in range(0, len(data.times), 16):
            output: SOMAPoseOutput = model.pose(fitted.poses[begin : begin + 16], transl=fitted.translations[begin : begin + 16], apply_correctives=False)
            # The waist has no procedural twist weights; the public hierarchy
            # gives its exact skinning motion in both 78- and 110-joint modes.
            blended = torch.einsum(
                "vj,njkl->nvkl",
                rim_skinning_weights,
                output.transforms @ inverse_bind_transforms,
            )
            display_vertices = output.vertices.clone()
            display_vertices[:, rim] += torch.einsum("nvij,vj->nvi", blended[..., :3, :3], rest_offsets)
            meshes.append(display_vertices.cpu().numpy())
            joints.append(output.joints.cpu().numpy())
        final: SOMAPoseOutput = model.pose(fitted.poses, transl=fitted.translations, fk_only=True, apply_correctives=False)
        angles: Float64[ndarray, "n"] = np.degrees(
            Rotation.from_matrix(
                (fitted.head_frames @ fitted.head_attachment).cpu().numpy().transpose(0, 2, 1) @ final.transforms[:, fitted.head_index + 1, :3, :3].cpu().numpy()
            ).magnitude()
        )
    if options.head_mode == "ears":
        angles[:] = np.nan
    torch.cuda.synchronize()
    public_names = tuple(model.public_joint_names)
    public_parents = model.public_joint_parent_ids.cpu().numpy().astype(np.int64)
    visible = np.flatnonzero(visible_joint_mask(public_names, public_parents)[1:]).astype(np.int64)
    vertices: Float32[ndarray, "n v 3"] = np.concatenate(meshes)
    return FitResult(
        fitted.poses.detach().cpu().numpy(),
        fitted.translations.detach().cpu().numpy(),
        fitted.identity.detach().cpu().numpy(),
        float(fitted.log_scale.exp().item()),
        np.concatenate(joints),
        vertices,
        model.faces.cpu().numpy(),
        names,
        model.public_joint_parent_ids.cpu().numpy().astype(np.int64),
        perf_counter() - fitted.started,
        torch.cuda.max_memory_allocated(),
        angles,
        fitted.head_attachment.detach().cpu().numpy(),
        vertices[:, EAR_VERTICES].reshape(-1, 2, 3, 3).mean(axis=2),
        options.head_mode,
        fitted.head_offset.detach().cpu().numpy(),
        fitted.bone_scales.cpu().numpy(),
        selection.vertices,
        visible,
        options.display_twists,
        orientations is not None and "quest" in orientations.sources,
        HandOrientationSource(*orientations.sources) if orientations is not None else None,
    )


def calibrate_shape(data: Observations, assets: Path | None, iterations: int = 400, shape_iterations: int = 60, options: FitOptions | None = None) -> FitResult:
    """Calibrate shared identity, scale, bone lengths and camera-to-skull attachment."""
    resolved = options or FitOptions()
    fitted = fit_parameters(data, assets, iterations, shape_iterations, resolved)
    return build_display_fit(data, assets, fitted, resolved, None)


def fit_pose(data: Observations, calibration: Calibration, assets: Path | None, iterations: int = 400, options: FitOptions | None = None,
             initial: FitResult | None = None, orientations: HandOrientations | None = None) -> FitResult:
    """Fit poses against a required frozen shape calibration, then build display geometry."""
    resolved = options or FitOptions()
    fitted = fit_parameters(data, assets, iterations, 0, resolved, calibration, initial, orientations)
    return build_display_fit(data, assets, fitted, resolved, orientations)
