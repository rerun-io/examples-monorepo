"""Broad hand-angle and surface-fold proxies, evaluated in the model's rest frame."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from soma.body import SOMALayer

from selfcap_soma.fitting import FitResult
from selfcap_soma.hand_pose import frame


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Plausibility:
    wrist_twist_median_deg: float
    wrist_twist_p90_deg: float
    wrist_twist_max_deg: float
    finger_outside_percent: float
    finger_roll_p90_deg: float
    flipped_faces_p90_percent: float
    collapsed_faces_p90_percent: float
    worst_times_s: list[float]


def swing_twist_angles(rotations: Float64[ndarray, "n 3 3"]) -> Float64[ndarray, "n 3"]:
    """Separate longitudinal roll from flex/side swing without Euler singularities."""
    relative = Rotation.from_matrix(rotations)
    quaternion = relative.as_quat()
    twist = np.zeros_like(quaternion)
    twist[:, 0] = quaternion[:, 0]
    twist[:, 3] = quaternion[:, 3]
    norm = np.linalg.norm(twist, axis=-1, keepdims=True)
    twist = np.divide(twist, norm, out=np.tile([0.0, 0.0, 0.0, 1.0], (len(twist), 1)), where=norm > 1e-8)
    roll = np.degrees(2 * np.arctan2(abs(twist[:, 0]), abs(twist[:, 3])))
    swing = np.degrees((relative * Rotation.from_quat(twist).inv()).as_rotvec())
    return np.column_stack((roll, swing[:, 1], swing[:, 2]))


def measure(model: SOMALayer, result: FitResult, times: Float64[ndarray, "n"]) -> Plausibility:
    """Angle bounds are broad proxies: roll 45°, sideways 35°, flex magnitude 120°."""
    with torch.no_grad():
        rest = model.pose(torch.zeros(1, 77, 3, device="cuda"), apply_correctives=False)
    xyz = rest.joints[0].cpu().numpy()
    orientations = rest.transforms[0, :, :3, :3].cpu().numpy()
    twists = []
    bad = []
    rolls = []
    for side in ("Left", "Right"):
        wrist = result.names.index(side + "Hand")
        elbow = result.names.index(side + "ForeArm")
        axis = (xyz[wrist] - xyz[elbow]).astype(np.float64)
        axis /= np.linalg.norm(axis)
        q = Rotation.from_rotvec(result.poses[:, wrist].astype(np.float64)).as_quat()
        angle = 2 * np.arctan2(abs(q[:, :3] @ axis), abs(q[:, 3]))
        twists.append(np.degrees(angle))
        normal = np.cross(xyz[result.names.index(side + "HandIndex2")] - xyz[wrist], xyz[result.names.index(side + "HandPinky2")] - xyz[wrist])
        for index, name in enumerate(result.names):
            if name.startswith(side + "Hand") and name[-1:].isdigit():
                childname = name[:-1] + str(int(name[-1]) + 1)
                if childname not in result.names:
                    childname = name[:-1] + "End"
                basis = frame((xyz[result.names.index(childname)] - xyz[index]).astype(np.float64), normal.astype(np.float64))
                rotation = Rotation.from_rotvec(result.poses[:, index].astype(np.float64)).as_matrix()
                angles = swing_twist_angles(basis.T @ rotation @ basis)
                rolls.append(angles[:, 0])
                bad.append((angles[:, 0] > 45) | (abs(angles[:, 1]) > 120) | (abs(angles[:, 2]) > 35))
    weights = model.public_skinning_weights().cpu().numpy()
    vertices = rest.vertices[0].cpu().numpy()
    faces = model.faces.cpu().numpy()
    hand_ids = [i + 1 for i, n in enumerate(result.names) if "Hand" in n]
    region = weights[:, hand_ids].sum(1) > 0.15
    for side in ("Left", "Right"):
        center = xyz[result.names.index(side + "Hand")]
        region |= np.linalg.norm(vertices - center, axis=-1) < 0.09
    faces = faces[region[faces].all(-1)]
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    area = np.linalg.norm(normals, axis=-1)
    keep = area > 1e-9
    faces = faces[keep]
    normals = normals[keep]
    area = area[keep]
    normals /= area[:, None]
    face_weights = weights[faces].mean(1)
    flipped = []
    collapsed = []
    for start in range(0, len(times), 32):
        with torch.no_grad():
            output = model.pose(
                torch.tensor(result.poses[start : start + 32], device="cuda"),
                transl=torch.tensor(result.translations[start : start + 32], device="cuda"),
                apply_correctives=False,
            )
        v = output.vertices.cpu().numpy()
        delta = output.transforms[:, :, :3, :3].cpu().numpy() @ orientations.transpose(0, 2, 1)
        expected = np.einsum("fj,njkl,fl->nfk", face_weights, delta, normals, optimize=True)
        actual = np.cross(v[:, faces[:, 1]] - v[:, faces[:, 0]], v[:, faces[:, 2]] - v[:, faces[:, 0]])
        flipped.extend((np.sum(actual * expected, axis=-1) < 0).mean(1) * 100)
        collapsed.extend((np.linalg.norm(actual, axis=-1) / area < 0.2).mean(1) * 100)
    wrist = np.stack(twists, 1)
    roll = np.stack(rolls, 1)
    outside = np.stack(bad, 1)
    score = outside.mean(1) * 100 + np.asarray(collapsed) + wrist.max(1) / 10
    order = np.argsort(score)[::-1]
    worst = []
    for i in order:
        if all(abs(times[i] - t) > 2 for t in worst):
            worst.append(float(times[i]))
        if len(worst) == 2:
            break
    return Plausibility(
        float(np.median(wrist)),
        float(np.percentile(wrist, 90)),
        float(wrist.max()),
        float(outside.mean() * 100),
        float(np.percentile(roll, 90)),
        float(np.percentile(flipped, 90)),
        float(np.percentile(collapsed, 90)),
        worst,
    )
