"""Quest orientation constraints and anatomical hand diagnostics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from dataforge.quest_joint_poses import read_quest_joint_poses
from jaxtyping import Bool, Float, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from torch import Tensor

from selfcap_soma.observations import quest_sample_times


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HandOrientationSource:
    left: Literal["quest", "none"]
    """Whether the left raw hand supplies measured orientations."""
    right: Literal["quest", "none"]
    """Whether the right raw hand supplies measured orientations."""


@dataclass(frozen=True, slots=True)
class HandOrientations:
    positions: Float64[ndarray, "n 2 26 3"]
    """All raw hand positions sampled on the fit clock."""
    rotations: Float64[ndarray, "n 2 26 3 3"]
    """World rotations sampled with SLERP; invalid entries have identity placeholders."""
    valid: Bool[ndarray, "n 2 26"]
    """Finite observations with valid timestamps (position validity for xyz-only hands)."""
    sources: tuple[Literal["quest", "none"], Literal["quest", "none"]] = ("quest", "quest")
    """Per-hand raw orientation availability; derived frames are never called measured."""


def read_orientations(directory: Path, times: Float64[ndarray, "n"], offset_ms: float) -> HandOrientations:
    """Read optional raw rotations and positions; xyz-only hands use position-derived frames."""
    samples = quest_sample_times(times, offset_ms)[1]
    positions = np.full((len(times), 2, 26, 3), np.nan)
    rotations = np.tile(np.eye(3), (len(times), 2, 26, 1, 1))
    valid = np.zeros((len(times), 2, 26), dtype=bool)
    sources: list[Literal["quest", "none"]] = []
    for side, name in enumerate(("left", "right")):
        raw = read_quest_joint_poses(directory / f"{name}_hand_poses.csv")
        sources.append("none" if raw.orientations is None else "quest")
        stamps = raw.times_ns.astype(np.float64) / 1e9
        right = np.clip(np.searchsorted(stamps, samples), 1, len(stamps) - 1)
        left = right - 1
        fraction = ((samples - stamps[left]) / (stamps[right] - stamps[left]))[:, None, None]
        positions[:, side] = (1 - fraction) * raw.positions[left] + fraction * raw.positions[right]
        if raw.orientations is None:
            valid[:, side] = (
                np.isfinite(positions[:, side]).all(-1)
                & (samples[:, None] >= stamps[0])
                & (samples[:, None] <= stamps[-1])
                & ((stamps[right] - stamps[left])[:, None] < 0.04)
            )
            continue
        for joint in range(26):
            q = raw.orientations[:, joint].astype(np.float64)
            good = np.isfinite(q).all(-1) & (np.linalg.norm(q, axis=-1) > 0.5)
            if good.sum() < 2:
                continue
            rotations[:, side, joint] = Slerp(stamps[good], Rotation.from_quat(q[good]))(
                np.clip(samples, stamps[good][0], stamps[good][-1])
            ).as_matrix()
            valid[:, side, joint] = (
                good[left] & good[right] & (samples >= stamps[0]) & (samples <= stamps[-1]) & ((stamps[right] - stamps[left]) < 0.04)
            )
    return HandOrientations(positions, rotations, valid, (sources[0], sources[1]))


def frame(direction: Float64[ndarray, "... 3"], normal: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3 3"]:
    """Right-handed bone frame: x along the bone, z toward the palm normal."""
    x = direction / np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-8)
    z = normal - np.sum(normal * x, axis=-1, keepdims=True) * x
    z /= np.linalg.norm(z, axis=-1, keepdims=True).clip(1e-8)
    return np.stack((x, np.cross(z, x), z), axis=-1)


@dataclass(frozen=True, slots=True)
class OrientationTargets:
    joint_indices: Int64[ndarray, "j"]
    """SOMA joint indices constrained by the observed hand frames."""
    rotations: Float32[ndarray, "n j 3 3"]
    """Aligned world-space rotations, identity where invalid."""
    valid: Bool[ndarray, "n j"]
    """Availability of each aligned rotation."""
    spread_deg: Float64[ndarray, "j"]
    """Mean angular spread of each measured alignment."""


def orientation_targets(
    observed: HandOrientations,
    times: Float64[ndarray, "n"],
    names: tuple[str, ...],
    neutral: Float32[ndarray, "77 3"],
    neutral_rotations: Float32[ndarray, "78 3 3"],
) -> OrientationTargets:
    """Align measured Quest rotations, or derive bone frames from position-only hands."""
    indices = []
    targets = []
    masks = []
    spread = []
    for side, label in enumerate(("Left", "Right")):
        wrist = names.index(label + "Hand")
        normal0 = np.cross(
            neutral[names.index(label + "HandIndex2")] - neutral[wrist], neutral[names.index(label + "HandPinky2")] - neutral[wrist]
        ).astype(np.float64)
        p = observed.positions[:, side]
        normal = np.cross(p[:, 7] - p[:, 1], p[:, 22] - p[:, 1])
        entries = [(1, label + "Hand", 12, label + "HandMiddle2")]
        for digit, base in zip(("Thumb", "Index", "Middle", "Ring", "Pinky"), (2, 6, 11, 16, 21), strict=True):
            count = 3 if digit == "Thumb" else 4
            for j in range(count):
                current = label + "Hand" + digit + str(j + 1)
                child = label + "Hand" + digit + (str(j + 2) if j < count - 1 else "End")
                entries.append((base + j, current, base + j + 1, child))
        for qindex, name, childq, childname in entries:
            index = names.index(name)
            reference = frame((neutral[names.index(childname)] - neutral[index]).astype(np.float64), normal0)
            geometry = frame(p[:, childq] - p[:, qindex], normal)
            positional_target = geometry @ reference.T @ neutral_rotations[index + 1]
            position_valid = np.isfinite(positional_target).all((1, 2)) & (np.linalg.norm(normal, axis=-1) > 1e-5)
            position_valid &= np.linalg.norm(p[:, childq] - p[:, qindex], axis=-1) > 1e-5
            candidates = observed.rotations[:, side, qindex].transpose(0, 2, 1) @ positional_target
            valid = observed.valid[:, side, qindex] & np.isfinite(candidates).all((1, 2)) & position_valid
            calibration = valid & (times < 25.0)
            if observed.sources[side] == "none" or calibration.sum() < 10:
                target = positional_target
                valid = position_valid & observed.valid[:, side, qindex]
                spread.append(0.0)
            else:
                mean = Rotation.from_matrix(candidates[calibration]).mean().as_matrix()
                spread.append(float(np.degrees(Rotation.from_matrix(mean.T @ candidates[calibration]).magnitude()).mean()))
                target = observed.rotations[:, side, qindex] @ mean
            indices.append(index)
            targets.append(np.where(valid[:, None, None], target, np.eye(3)))
            masks.append(valid)
    return OrientationTargets(np.asarray(indices, dtype=np.int64), np.stack(targets, axis=1).astype(np.float32), np.stack(masks, axis=1), np.asarray(spread))


def hand_limit_penalty(
    poses: Float[Tensor, "n j 3"],
    bases: Float[Tensor, "j 3 3"],
    limits: Float[Tensor, "j 4"],
    signs: Float[Tensor, "j"],
) -> Float[Tensor, ""]:
    """Bound flexion, abduction and exact axial twist for position-only hand bones.

    Limits columns are hyperextension, flexion, sideways, twist (radians).
    """
    local = torch.einsum("nji,jik->njk", poses, bases)
    angle = torch.linalg.vector_norm(local, dim=-1)
    qx = 0.5 * local[..., 0] * torch.sinc(angle / (2 * torch.pi))
    qw = torch.cos(angle / 2)
    twist = 2 * torch.atan2(qx, qw)
    twist = torch.atan2(torch.sin(twist), torch.cos(twist))
    flex = local[..., 1] * signs
    excess = torch.relu(-limits[:, 0] - flex).square() + torch.relu(flex - limits[:, 1]).square()
    excess += torch.relu(local[..., 2].abs() - limits[:, 2]).square()
    excess += torch.relu(twist.abs() - limits[:, 3]).square()
    return (excess / 0.1**2 + 0.2 * (twist / limits[:, 3]).square()).mean()
