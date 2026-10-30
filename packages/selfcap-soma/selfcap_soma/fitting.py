"""Batched sparse-observation SOMA fit with one shared identity."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from soma.body import SOMALayer, SOMAPoseOutput
from torch import Tensor

from selfcap_soma.catalog import Observations
from selfcap_soma.landmarks import EAR_VERTICES, ear_vertices, rotation_delta
from selfcap_soma.losses import attached_eye, body_lengths, robust_points
from selfcap_soma.observations import coco_joint_names


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
    head_anchor: str
    """Position anchor: eye for legacy ablations, head for ear fitting."""
    head_offset: Float32[ndarray, "3"]
    """Tracked-camera to head-anchor translation, metres. Skull frame for ear mode; Quest frame for legacy eye mode."""
    bone_scales: Float32[ndarray, "1 60"]
    """Native SOMA local bone-length ratios, shared over all frames."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FitOptions:
    use_ears: bool = False
    """Fit COCO ears as surface landmarks and attach the headset to the skull."""
    learn_head_rotation: bool = False
    """Fit one shared camera-to-skull rotation during calibration."""
    ear_weight: float = 3.0
    """Confidence-weighted ear position term at 20 mm scale."""
    shape_components: int = 16
    """Number of active native SOMA identity coefficients."""
    identity_prior: float = 1.0
    """Mean squared identity penalty."""
    scale_prior: float = 20.0
    """Squared log global-scale penalty."""
    head_position_weight: float = 3.0
    """Position weight; rotation remains constrained in all ablations."""
    learn_head_offset: bool = False
    """Learn one attachment offset in the tracked head frame."""
    offset_prior: float = 0.2
    """Penalty on offset in units of 5 cm."""
    learn_bone_scales: bool = False
    """Fit native body bone-length controls during the shape stage."""
    bone_prior: float = 0.1
    """Penalty on body bone ratios in units of 15 percent."""
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

    head_attachment: Float32[ndarray, "3 3"] | None = None
    """Camera-to-skull rotation; absent for legacy calibration files."""
    head_anchor: str = "eye"
    """Legacy eye or ear-calibrated head position anchor."""


def fit(
    data: Observations,
    assets: Path,
    iterations: int = 400,
    shape_iterations: int = 60,
    options: FitOptions | None = None,
    fixed: FitResult | Calibration | None = None,
) -> FitResult:
    """Fit all frames together; freeze identity after the first stage."""
    options = options or FitOptions()
    use_ears = options.use_ears or (fixed is not None and fixed.head_anchor == "head")
    active_components = options.shape_components if fixed is None else 128
    torch.cuda.reset_peak_memory_stats()
    model: SOMALayer = SOMALayer(
        data_root=assets,
        device="cuda",
        identity_model_type="soma",
        mode="torch",
        lod="low",
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    model.requires_grad_(False)
    names: tuple[str, ...] = tuple(model.public_joint_names[1:])
    mapping: dict[int, str] = coco_joint_names()
    coco: list[int] = list(mapping)
    soma: list[int] = [names.index(mapping[k]) for k in coco]
    neutral_identity: Float32[Tensor, "1 128"] = (
        torch.zeros(1, 128, device="cuda") if fixed is None else torch.as_tensor(fixed.identity, device="cuda")
    )
    with torch.no_grad():
        model.prepare_identity(
            neutral_identity,
            scale_params=None if fixed is None else torch.as_tensor(fixed.bone_scales, device="cuda"),
            global_scale=1.0 if fixed is None else fixed.scale,
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
    head_index, eye_index = names.index("Head"), names.index("LeftEye")
    attachment: Float32[ndarray, "3 3"] = camera_basis.T @ neutral_transforms[0, head_index + 1, :3, :3].cpu().numpy()
    if fixed is not None and fixed.head_attachment is not None:
        attachment = fixed.head_attachment
    base_attachment = torch.as_tensor(attachment, dtype=torch.float32, device="cuda")
    rotation_raw = torch.nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=options.learn_head_rotation and fixed is None)

    # Per-frame Kabsch initialization from shoulders and hips avoids a
    # convention-dependent global yaw guess.
    anchors: list[int] = [5, 6, 11, 12]
    source: Float32[ndarray, "4 3"] = neutral_joints[[names.index(mapping[k]) for k in anchors]]
    source_center: Float32[ndarray, "3"] = source.mean(axis=0)
    initial_poses: Float32[ndarray, "n 77 3"] = np.zeros((len(data.times), 77, 3), dtype=np.float32)
    initial_translation: Float32[ndarray, "n 3"] = np.zeros((len(data.times), 3), dtype=np.float32)
    initial_anchors: Float64[ndarray, "n 4 3"] = data.exo[:, anchors].copy()
    for joint in range(4):
        valid = np.isfinite(initial_anchors[:, joint]).all(-1)
        if valid.sum() < 2:
            raise ValueError("prototype initialization needs two observations per torso anchor")
        for axis in range(3):
            initial_anchors[:, joint, axis] = np.interp(data.times, data.times[valid], initial_anchors[valid, joint, axis])
    for i, target in enumerate(initial_anchors):
        rotation: Rotation = Rotation.align_vectors(target - target.mean(axis=0), source - source_center)[0]
        initial_poses[i, 0] = rotation.as_rotvec()
        initial_translation[i] = target.mean(axis=0) - rotation.apply(source_center)
    poses: Float32[Tensor, "n 77 3"] = torch.nn.Parameter(torch.as_tensor(initial_poses, device="cuda"))
    translations: Float32[Tensor, "n 3"] = torch.nn.Parameter(torch.as_tensor(initial_translation, device="cuda"))
    if not 1 <= options.shape_components <= 128:
        raise ValueError("shape components must be in [1,128]")
    identity = torch.nn.Parameter(neutral_identity[:, :active_components].clone(), requires_grad=fixed is None)
    log_scale = torch.nn.Parameter(
        torch.tensor(0.0 if fixed is None else np.log(fixed.scale), device="cuda", dtype=torch.float32), requires_grad=fixed is None
    )
    offset_raw = torch.nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=options.learn_head_offset and fixed is None)
    if model.num_scale_params != 60:
        raise ValueError("expected 60 native SOMA bone-scale controls")
    bone_raw = torch.nn.Parameter(torch.zeros(1, 60, device="cuda"), requires_grad=options.learn_bone_scales and fixed is None)
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
    bone_scales = torch.ones_like(bone_raw) if fixed is None else torch.as_tensor(fixed.bone_scales, device="cuda")
    fixed_offset = torch.zeros(3, device="cuda") if fixed is None else torch.as_tensor(fixed.head_offset, device="cuda")
    parameters = (
        [poses, translations]
        + ([identity, log_scale] if fixed is None else [])
        + ([offset_raw] if options.learn_head_offset and fixed is None else [])
        + ([bone_raw] if options.learn_bone_scales and fixed is None else [])
        + ([rotation_raw] if options.learn_head_rotation and fixed is None else [])
    )
    optimizer: torch.optim.Adam = torch.optim.Adam(parameters, lr=0.025)
    exo: Float32[Tensor, "n j 3"] = torch.as_tensor(data.exo[:, coco], dtype=torch.float32, device="cuda")
    confidence: Float32[Tensor, "n j"] = torch.as_tensor(data.confidence[:, coco], dtype=torch.float32, device="cuda")
    hands: Float32[Tensor, "n j 3"] = torch.as_tensor(data.hands[:, coco], dtype=torch.float32, device="cuda")
    hand_weights: Float32[Tensor, "n j"] = torch.isfinite(hands).all(-1).float()
    head: Float32[Tensor, "n 1 3"] = torch.as_tensor(data.head_positions[:, None], dtype=torch.float32, device="cuda")
    head_valid = torch.isfinite(head[:, 0]).all(-1).float()
    head_frames = torch.as_tensor(data.head_rotations, device="cuda", dtype=torch.float32)
    dt: Float32[Tensor, "interval"] = torch.as_tensor(np.diff(data.times), dtype=torch.float32, device="cuda")
    ears_target = torch.as_tensor(data.exo[:, 3:5], dtype=torch.float32, device="cuda")
    ears_confidence = torch.as_tensor(data.confidence[:, 3:5], dtype=torch.float32, device="cuda")
    warmup: int = min(100, iterations // 3)
    shape_end: int = min(warmup + shape_iterations, iterations - 1)
    target_lengths = torch.nanmedian(body_lengths(torch.as_tensor(data.exo[:, 5:17], device="cuda", dtype=torch.float32)), dim=0).values
    soma_body = [names.index(mapping[key]) for key in range(5, 17)]
    body_columns: list[int] = [i for i, key in enumerate(coco) if key < 17]
    hand_columns: list[int] = [i for i, key in enumerate(coco) if key >= 91]
    start: float = perf_counter()
    for iteration in range(iterations):
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
        posed: SOMAPoseOutput = model.pose(poses, transl=translations, fk_only=True, apply_correctives=False)
        selected: Float32[Tensor, "n j 3"] = posed.joints[:, soma]
        body_loss = robust_points(selected[:, body_columns], exo[:, body_columns], confidence[:, body_columns], 0.02)
        exo_hand_loss = robust_points(selected[:, hand_columns], exo[:, hand_columns], confidence[:, hand_columns], 0.02)
        hand_loss = robust_points(selected, hands, hand_weights, 0.01)
        head_offset = (0.2 if use_ears else 0.1) * torch.tanh(offset_raw) if options.learn_head_offset and fixed is None else fixed_offset
        head_attachment = base_attachment @ rotation_delta(rotation_raw)
        target_rotation = head_frames @ head_attachment
        position_index = head_index if use_ears else eye_index
        anchor_target = attached_eye(head[:, 0], target_rotation if use_ears else head_frames, head_offset)[:, None]
        head_loss = robust_points(posed.joints[:, position_index : position_index + 1], anchor_target, torch.ones_like(head[..., 0]), 0.01)
        ear_loss = torch.zeros((), device="cuda")
        if use_ears:
            skin = model.batched_skinning
            ears = ear_vertices(
                skin.bind_shapes,
                model.skinning_weights,
                skin.inverse_bind_transform,
                posed.transforms[:, torch.argsort(model.public_transform_joint_indices)],
            )
            ear_loss = robust_points(ears, ears_target, ears_confidence, 0.02)
        head_rotation = posed.transforms[:, head_index + 1, :3, :3]
        rotation_loss = ((head_rotation - target_rotation).square().mean(dim=(-1, -2)) * head_valid).sum() / head_valid.sum().clamp_min(1.0) / 0.03
        velocity = (translations[1:] - translations[:-1]) / dt[:, None]
        angular_velocity = (poses[1:] - poses[:-1]) / dt[:, None, None]
        smooth = (velocity[1:] - velocity[:-1]).square().mean() + 0.1 * (angular_velocity[1:] - angular_velocity[:-1]).square().mean()
        prior = poses[:, 1:].square().mean() + 0.5 * torch.relu(poses[:, 1:].norm(dim=-1) - 2.6).square().mean()
        loss = (
            body_loss
            + 0.25 * exo_hand_loss
            + 3.0 * hand_loss
            + options.head_position_weight * head_loss
            + rotation_loss
            + options.ear_weight * ear_loss
            + 0.005 * smooth
            + 0.03 * prior
        )
        loss = loss + options.offset_prior * (head_offset / (0.1 if use_ears else 0.05)).square().mean()
        loss = loss + 0.02 * rotation_raw.square().mean()
        if fixed is None and warmup <= iteration < shape_end:
            loss = loss + options.identity_prior * identity.square().mean() + options.scale_prior * log_scale.square()
            loss = loss + options.bone_prior * (((bone_scales - 1.0) / 0.15).square() * bone_mask).sum() / bone_mask.sum()
            if options.bone_length_weight > 0.0:
                neutral_shape = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
                lengths = body_lengths(neutral_shape.joints[:, soma_body])[0]
                valid_lengths = torch.isfinite(target_lengths)
                loss = loss + options.bone_length_weight * torch.nn.functional.smooth_l1_loss(
                    lengths[valid_lengths] / 0.02, target_lengths[valid_lengths] / 0.02, beta=2.0
                )
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite fit at iteration {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(
                f"iteration={iteration} loss={loss.item():.3f} exo={body_loss.item():.3f} hands={hand_loss.item():.3f} head={head_loss.item():.3f}",
                flush=True,
            )
    with torch.no_grad():
        if fixed is None:
            bone_scales = 1.0 + 0.3 * torch.tanh(bone_raw) * bone_mask
        model.prepare_identity(
            torch.nn.functional.pad(identity, (0, 128 - active_components)), scale_params=bone_scales, global_scale=log_scale.exp()
        )
        head_offset = (0.2 if use_ears else 0.1) * torch.tanh(offset_raw) if options.learn_head_offset and fixed is None else fixed_offset
        head_attachment = base_attachment @ rotation_delta(rotation_raw)
        target_rotation = head_frames @ head_attachment
        meshes: list[Float32[ndarray, "b v 3"]] = []
        joints: list[Float32[ndarray, "b 77 3"]] = []
        for begin in range(0, len(data.times), 16):
            output: SOMAPoseOutput = model.pose(poses[begin : begin + 16], transl=translations[begin : begin + 16], apply_correctives=False)
            meshes.append(output.vertices.cpu().numpy())
            joints.append(output.joints.cpu().numpy())
        final: SOMAPoseOutput = model.pose(poses, transl=translations, fk_only=True, apply_correctives=False)
        angles: Float64[ndarray, "n"] = np.degrees(
            Rotation.from_matrix(
                target_rotation.cpu().numpy().transpose(0, 2, 1) @ final.transforms[:, head_index + 1, :3, :3].cpu().numpy()
            ).magnitude()
        )
    torch.cuda.synchronize()
    return FitResult(
        poses.detach().cpu().numpy(),
        translations.detach().cpu().numpy(),
        torch.nn.functional.pad(identity.detach(), (0, 128 - active_components)).cpu().numpy(),
        float(log_scale.exp().item()),
        np.concatenate(joints),
        np.concatenate(meshes),
        model.faces.cpu().numpy(),
        names,
        model.public_joint_parent_ids.cpu().numpy().astype(np.int64),
        perf_counter() - start,
        torch.cuda.max_memory_allocated(),
        angles,
        head_attachment.detach().cpu().numpy(),
        np.concatenate(meshes)[:, EAR_VERTICES].reshape(-1, 2, 3, 3).mean(axis=2),
        "head" if use_ears else "eye",
        head_offset.detach().cpu().numpy(),
        bone_scales.cpu().numpy(),
    )
