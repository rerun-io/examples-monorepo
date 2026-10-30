"""Batched sparse-observation SOMA fit with one shared identity."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from soma.body import SOMALayer, SOMAPoseOutput
from torch import Tensor

from selfcap_soma.catalog import Observations
from selfcap_soma.losses import robust_points
from selfcap_soma.observations import coco_joint_names


@dataclass(frozen=True, slots=True)
class FitResult:
    poses: Float32[ndarray, "n 77 3"]
    """Joint-local axis angles; first entry is global pelvis rotation."""
    translations: Float32[ndarray, "n 3"]
    """World root translations in metres."""
    identity: Float32[ndarray, "1 128"]
    """One shared native SOMA shape; only first 16 coefficients fitted."""
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
    """Rotation error against the fixed eye-camera attachment."""


def fit(data: Observations, assets: Path, iterations: int = 400, shape_iterations: int = 60) -> FitResult:
    """Fit all frames together; freeze identity after the first stage."""
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
    neutral_identity: Float32[Tensor, "1 128"] = torch.zeros(1, 128, device="cuda")
    with torch.no_grad():
        model.prepare_identity(neutral_identity)
        neutral: SOMAPoseOutput = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
    neutral_joints: Float32[ndarray, "77 3"] = neutral.joints[0].detach().cpu().numpy()
    neutral_transforms: Float32[Tensor, "1 78 4 4"] = neutral.transforms.detach()
    # RDF frame: anatomical right, down, and forward. The eyes and cranial
    # axis define a fixed camera attachment, not a fitted calibration change.
    right: Float32[ndarray, "3"] = neutral_joints[names.index("RightEye")] - neutral_joints[names.index("LeftEye")]
    up: Float32[ndarray, "3"] = neutral_joints[names.index("HeadEnd")] - neutral_joints[names.index("Head")]
    right /= np.linalg.norm(right)
    up -= right * (up @ right)
    up /= np.linalg.norm(up)
    camera_basis: Float32[ndarray, "3 3"] = np.column_stack((right, -up, np.cross(right, -up)))
    head_index, eye_index = names.index("Head"), names.index("LeftEye")
    attachment: Float32[ndarray, "3 3"] = camera_basis.T @ neutral_transforms[0, head_index + 1, :3, :3].cpu().numpy()
    target_rotation: Float32[Tensor, "n 3 3"] = torch.as_tensor(data.head_rotations @ attachment, dtype=torch.float32, device="cuda")

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
    identity: Float32[Tensor, "1 16"] = torch.nn.Parameter(torch.zeros(1, 16, device="cuda"))
    log_scale: Float32[Tensor, ""] = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer: torch.optim.Adam = torch.optim.Adam([poses, translations, identity, log_scale], lr=0.025)
    exo: Float32[Tensor, "n j 3"] = torch.as_tensor(data.exo[:, coco], dtype=torch.float32, device="cuda")
    confidence: Float32[Tensor, "n j"] = torch.as_tensor(data.confidence[:, coco], dtype=torch.float32, device="cuda")
    hands: Float32[Tensor, "n j 3"] = torch.as_tensor(data.hands[:, coco], dtype=torch.float32, device="cuda")
    hand_weights: Float32[Tensor, "n j"] = torch.isfinite(hands).all(-1).float()
    head: Float32[Tensor, "n 1 3"] = torch.as_tensor(data.head_positions[:, None], dtype=torch.float32, device="cuda")
    dt: Float32[Tensor, "interval"] = torch.as_tensor(np.diff(data.times), dtype=torch.float32, device="cuda")
    warmup: int = min(100, iterations // 3)
    shape_end: int = min(warmup + shape_iterations, iterations - 1)
    body_columns: list[int] = [i for i, key in enumerate(coco) if key < 17]
    hand_columns: list[int] = [i for i, key in enumerate(coco) if key >= 91]
    start: float = perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        if warmup <= iteration < shape_end:
            model.prepare_identity(torch.nn.functional.pad(identity, (0, 112)), global_scale=log_scale.exp())
        elif iteration == shape_end:
            # Upstream reposing includes its trainable bind rotations even
            # when identity inputs are detached. Freeze the entire cache.
            with torch.no_grad():
                model.prepare_identity(torch.nn.functional.pad(identity.detach(), (0, 112)), global_scale=log_scale.detach().exp())
        posed: SOMAPoseOutput = model.pose(poses, transl=translations, fk_only=True, apply_correctives=False)
        selected: Float32[Tensor, "n j 3"] = posed.joints[:, soma]
        body_loss = robust_points(selected[:, body_columns], exo[:, body_columns], confidence[:, body_columns], 0.02)
        exo_hand_loss = robust_points(selected[:, hand_columns], exo[:, hand_columns], confidence[:, hand_columns], 0.02)
        hand_loss = robust_points(selected, hands, hand_weights, 0.01)
        head_loss = robust_points(posed.joints[:, eye_index : eye_index + 1], head, torch.ones_like(head[..., 0]), 0.01)
        head_rotation = posed.transforms[:, head_index + 1, :3, :3]
        rotation_loss = (head_rotation - target_rotation).square().mean() / 0.03
        velocity = (translations[1:] - translations[:-1]) / dt[:, None]
        angular_velocity = (poses[1:] - poses[:-1]) / dt[:, None, None]
        smooth = (velocity[1:] - velocity[:-1]).square().mean() + 0.1 * (angular_velocity[1:] - angular_velocity[:-1]).square().mean()
        prior = poses[:, 1:].square().mean() + 0.5 * torch.relu(poses[:, 1:].norm(dim=-1) - 2.6).square().mean()
        loss = body_loss + 0.25 * exo_hand_loss + 3.0 * hand_loss + 3.0 * head_loss + rotation_loss + 0.005 * smooth + 0.03 * prior
        if warmup <= iteration < shape_end:
            loss = loss + identity.square().mean() + 20.0 * log_scale.square()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite fit at iteration {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([poses, translations, identity, log_scale], 100.0)
        optimizer.step()
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(
                f"iteration={iteration} loss={loss.item():.3f} exo={body_loss.item():.3f} hands={hand_loss.item():.3f} head={head_loss.item():.3f}",
                flush=True,
            )
    with torch.no_grad():
        model.prepare_identity(torch.nn.functional.pad(identity, (0, 112)), global_scale=log_scale.exp())
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
        torch.nn.functional.pad(identity.detach(), (0, 112)).cpu().numpy(),
        float(log_scale.exp().item()),
        np.concatenate(joints),
        np.concatenate(meshes),
        model.faces.cpu().numpy(),
        names,
        model.public_joint_parent_ids.cpu().numpy().astype(np.int64),
        perf_counter() - start,
        torch.cuda.max_memory_allocated(),
        angles,
    )
