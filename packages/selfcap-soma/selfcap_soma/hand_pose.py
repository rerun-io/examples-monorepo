"""Quest orientation constraints and anatomical hand diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dataforge.quest_joint_poses import read_quest_joint_poses
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp

from selfcap_soma.observations import quest_sample_times


@dataclass(frozen=True, slots=True)
class HandOrientations:
    positions: Float64[ndarray, "n 2 26 3"]
    """All raw hand positions sampled on the fit clock."""
    rotations: Float64[ndarray, "n 2 26 3 3"]
    """World rotations sampled with SLERP; invalid entries have identity placeholders."""
    valid: Bool[ndarray, "n 2 26"]
    """Finite orientation observations with valid timestamps."""


def read_orientations(directory: Path, times: Float64[ndarray, "n"], offset_ms: float) -> HandOrientations:
    """Read raw CSV orientation channels only because the catalog ingest dropped them."""
    samples = quest_sample_times(times, offset_ms)[1]
    positions = np.full((len(times), 2, 26, 3), np.nan)
    rotations = np.tile(np.eye(3), (len(times), 2, 26, 1, 1))
    valid = np.zeros((len(times), 2, 26), dtype=bool)
    for side, name in enumerate(("left", "right")):
        raw = read_quest_joint_poses(directory / f"{name}_hand_poses.csv")
        if raw.orientations is None:
            raise ValueError(f"{name} CSV has no orientations")
        stamps = raw.times_ns.astype(np.float64) / 1e9
        right = np.clip(np.searchsorted(stamps, samples), 1, len(stamps) - 1)
        left = right - 1
        fraction = ((samples - stamps[left]) / (stamps[right] - stamps[left]))[:, None, None]
        positions[:, side] = (1 - fraction) * raw.positions[left] + fraction * raw.positions[right]
        for joint in range(26):
            q = raw.orientations[:, joint].astype(np.float64)
            good = np.isfinite(q).all(-1) & (np.linalg.norm(q, axis=-1) > 0.5)
            rotations[:, side, joint] = Slerp(stamps[good], Rotation.from_quat(q[good]))(
                np.clip(samples, stamps[good][0], stamps[good][-1])
            ).as_matrix()
            valid[:, side, joint] = (
                good[left] & good[right] & (samples >= stamps[0]) & (samples <= stamps[-1]) & ((stamps[right] - stamps[left]) < 0.04)
            )
    return HandOrientations(positions, rotations, valid)


def frame(direction: Float64[ndarray, "... 3"], normal: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3 3"]:
    """Right-handed bone frame: x along the bone, z toward the palm normal."""
    x = direction / np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-8)
    z = normal - np.sum(normal * x, axis=-1, keepdims=True) * x
    z /= np.linalg.norm(z, axis=-1, keepdims=True).clip(1e-8)
    return np.stack((x, np.cross(z, x), z), axis=-1)


def orientation_targets(
    observed: HandOrientations,
    times: Float64[ndarray, "n"],
    names: tuple[str, ...],
    neutral: Float32[ndarray, "77 3"],
    neutral_rotations: Float32[ndarray, "78 3 3"],
) -> tuple[Int64[ndarray, "j"], Float32[ndarray, "n j 3 3"], Bool[ndarray, "n j"], Float64[ndarray, "j"]]:
    """Estimate constant Quest-to-SOMA bone frame alignments from the first 25 s."""
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
            candidates = observed.rotations[:, side, qindex].transpose(0, 2, 1) @ geometry @ reference.T @ neutral_rotations[index + 1]
            valid = observed.valid[:, side, qindex] & np.isfinite(candidates).all((1, 2)) & (np.linalg.norm(normal, axis=-1) > 1e-5)
            calibration = valid & (times < 25.0)
            if calibration.sum() < 10:
                raise ValueError(f"not enough orientation calibration for {name}")
            mean = Rotation.from_matrix(candidates[calibration]).mean().as_matrix()
            spread.append(float(np.degrees(Rotation.from_matrix(mean.T @ candidates[calibration]).magnitude()).mean()))
            indices.append(index)
            targets.append(observed.rotations[:, side, qindex] @ mean)
            masks.append(valid)
    return np.asarray(indices, dtype=np.int64), np.stack(targets, axis=1).astype(np.float32), np.stack(masks, axis=1), np.asarray(spread)
