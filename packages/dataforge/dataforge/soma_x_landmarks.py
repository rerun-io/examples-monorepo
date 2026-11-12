"""SOMA-X hand fits to 21 3D hand landmarks (COCO-WholeBody hand order), for sources without MANO parameters (UmeTrack, SHOW3D).

The fit has three parts, all batched over frames on the GPU:

- The skeleton. SOMA-X's hand (``SOMAHandLayer``, ``mano`` identity backend) poses 25 joints by forward kinematics from
  per-joint local translations, which depend on the identity (10 MANO betas), and absolute local rotations. This module runs
  its own differentiable FK (equal to ``SOMAHandLayer.pose(fk_only=True, absolute_pose=True)`` to 2e-9 m) on a linear model
  of the local translations in the betas, built by finite differences of ``prepare_identity``.
- The landmarks. Landmark k sits at SOMA joint ``SOMA_JOINT_OF_LANDMARK[k]`` plus a fixed offset in the frame of that joint's
  parent (the wrist's own frame for the wrist). The offsets carry the source's landmark definition (UmeTrack's fingertips are
  on the skin, its knuckle landmarks are not at SOMA's rotation centres); they are calibrated once per source on frames
  that are not evaluated, then held fixed (``calibrate_offsets``).
- The solve. Per frame: analytic initialisation (Kabsch of the palm, then each bone swung onto its child's landmark with the
  minimal rotation, so zero twist), then batched Levenberg-Marquardt on the joint rotations (relative to the bind pose) and
  the root translation. Per hand and segment: the identity is solved by alternating pose LM on a frame subsample with a
  Gauss-Newton step on the betas (``fit_identity``).

Twist about a bone moves no landmark unless the next joint bends, so 21 points leave it weakly observed: the initialisation
adds no twist, and LM carries a prior on each finger joint's rotation away from the bind pose that is ``TWIST_PRIOR`` times
stronger about the bone axis than across it. The finger tips' end joints carry no rotation.

Importable only where SOMA-X and CUDA torch are installed (the ``dataforge-soma-x`` environments).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from soma.hand import SOMAHandLayer
from torch import Tensor

from dataforge import schema, soma_x
from dataforge.soma_x_fit import Side, mano_model, rotvecs_from_matrices, soma_assets

SOMA_JOINT_OF_LANDMARK: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)
"""SOMA hand joint of each COCO-WholeBody hand landmark (0 wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).
SOMA's finger chains start at the metacarpal (Index1 ...), which no landmark observes; the thumb starts at the CMC (Thumb1)."""
FREE_JOINTS: tuple[int, ...] = tuple(j for j in range(25) if j not in (4, 9, 14, 19, 24))
"""Joints with a rotation unknown: every joint but the five finger-tip end joints (they move nothing)."""
COCO_HAND_SLOTS: dict[Side, slice] = {"left": slice(91, 112), "right": slice(112, 133)}
"""COCO-WholeBody 133 rows of each hand."""
TWIST_PRIOR: float = 4.0
"""How much stronger the rotation prior is about a bone's own axis than across it."""
ROTATION_PRIOR_M: float = 0.002
"""Rotation prior weight, metres of landmark residual per radian away from the bind pose (across the bone)."""
BETA_PRIOR_M: float = 0.001
"""Identity prior weight, metres of landmark residual per unit beta (betas are standard normal)."""
UMETRACK_LANDMARK_OFFSETS_M: dict[Side, tuple[tuple[float, float, float], ...]] = {
    "left": (
        (-0.0023, 0.0023, -0.0016),
        (0.0026, 0.0025, 0.0036),
        (-0.0013, 0.0037, 0.0033),
        (0.0066, 0.0003, -0.0021),
        (-0.0083, 0.0000, -0.0001),
        (-0.0040, -0.0022, -0.0020),
        (-0.0036, -0.0004, -0.0003),
        (0.0014, 0.0000, 0.0002),
        (-0.0012, 0.0000, 0.0001),
        (0.0014, -0.0001, -0.0003),
        (-0.0022, 0.0000, -0.0011),
        (-0.0005, -0.0001, -0.0007),
        (0.0017, 0.0002, 0.0002),
        (0.0027, 0.0001, -0.0009),
        (-0.0004, 0.0003, -0.0001),
        (-0.0024, -0.0003, -0.0014),
        (0.0025, 0.0002, 0.0002),
        (0.0050, 0.0001, -0.0004),
        (-0.0006, -0.0005, -0.0018),
        (0.0000, -0.0005, -0.0002),
        (0.0007, 0.0002, 0.0001),
    ),
    "right": (
        (0.0008, -0.0021, 0.0002),
        (-0.0036, -0.0004, -0.0004),
        (0.0060, -0.0056, -0.0017),
        (-0.0050, 0.0012, 0.0039),
        (0.0073, 0.0001, 0.0002),
        (0.0028, -0.0003, -0.0006),
        (0.0022, 0.0003, 0.0008),
        (-0.0018, -0.0001, -0.0001),
        (0.0003, 0.0000, 0.0000),
        (-0.0004, -0.0019, -0.0006),
        (0.0021, -0.0006, 0.0010),
        (0.0004, 0.0000, 0.0002),
        (-0.0021, -0.0001, -0.0001),
        (-0.0008, -0.0028, -0.0003),
        (0.0006, -0.0008, 0.0002),
        (0.0024, 0.0002, 0.0011),
        (-0.0023, -0.0001, -0.0002),
        (-0.0032, 0.0001, -0.0003),
        (0.0018, 0.0001, 0.0013),
        (-0.0002, 0.0001, -0.0004),
        (-0.0013, -0.0002, 0.0000),
    ),
}
"""UmeTrack's landmark definition (its hand model's 21 skinned landmarks; SHOW3D uses the same model): offset of each
landmark from its SOMA joint in the joint's parent frame, metres. Calibrated by ``calibrate_offsets`` on four UmeTrack
training recordings (users 00, 08, 15, 33; 60 s each), none of them evaluated; they lower UmeTrack's landmark MPJPE from
~2.7 to ~1.4 mm and leave the mesh distance as it was. Other sources start from zeros."""


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """One hand's 21 landmarks over a segment's rows, world metres."""

    times_ns: Int64[ndarray, "t"]
    """``video_time`` of each row."""
    frame_indices: Int64[ndarray, "t"]
    """``frame_index`` of each row."""
    positions: Float32[ndarray, "t 21 3"]
    """Landmarks in COCO-WholeBody hand order; NaN where absent."""
    confidence: Float32[ndarray, "t 21"]
    """Per-landmark confidence; 0 where absent."""

    @property
    def present(self) -> Bool[ndarray, "t"]:
        """Rows whose 21 landmarks are all finite with positive confidence: the rows the fit uses."""
        return np.isfinite(self.positions).all(axis=(1, 2)) & (self.confidence > 0).all(axis=1)


def read_hand_landmarks(table: pa.Table, side: Side) -> HandLandmarks:
    """One hand's landmarks from a ``video_time``-sorted table of ``/world/gt/coco133_xyz`` (``apis.soma_x.read_rows``)."""
    path: str = schema.coco133_xyz_path()
    cells: list[list[list[float]] | None] = table.column(f"{path}:Points3D:positions").to_pylist()
    scores: list[list[float] | None] = table.column(f"{path}:simplecv.KeypointConfidence3D:confidences").to_pylist()
    rows: int = len(cells)
    positions: Float32[ndarray, "t 21 3"] = np.full((rows, 21, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "t 21"] = np.zeros((rows, 21), dtype=np.float32)
    slots: slice = COCO_HAND_SLOTS[side]
    for row, (cell, score) in enumerate(zip(cells, scores, strict=True)):
        if cell and len(cell) == 133:
            positions[row] = np.asarray(cell[slots], dtype=np.float32)
        if score and len(score) == 133:
            confidence[row] = np.asarray([0.0 if value is None else value for value in score[slots]], dtype=np.float32)
    confidence[~np.isfinite(positions).all(axis=2)] = 0.0
    times_ns: Int64[ndarray, "t"] = table.column(schema.TIMELINE).cast(pa.int64()).to_numpy()
    frames: Int64[ndarray, "t"] = table.column(schema.FRAME_INDEX).to_numpy(zero_copy_only=False).astype(np.int64)
    return HandLandmarks(times_ns, frames, positions, confidence)


def so3_exp(rotvecs: Float32[Tensor, "*b 3"]) -> Float32[Tensor, "*b 3 3"]:
    """Rodrigues' formula, differentiable at zero (forward-mode Jacobians are taken there)."""
    squared: Tensor = (rotvecs * rotvecs).sum(-1, keepdim=True)
    angle: Tensor = torch.sqrt(squared + 1e-12)
    a: Tensor = torch.where(squared < 1e-8, 1 - squared / 6, torch.sin(angle) / angle)[..., None]
    b: Tensor = torch.where(squared < 1e-8, 0.5 - squared / 24, (1 - torch.cos(angle)) / (squared + 1e-12))[..., None]
    x, y, z = rotvecs.unbind(-1)
    zero: Tensor = torch.zeros_like(x)
    k: Tensor = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*rotvecs.shape[:-1], 3, 3)
    eye: Tensor = torch.eye(3, dtype=rotvecs.dtype, device=rotvecs.device).expand_as(k)
    return eye + a * k + b * (k @ k)


@dataclass(frozen=True, slots=True)
class HandSkeleton:
    """The SOMA hand's kinematic tree for one identity, plus the linear identity model it came from."""

    parents: tuple[int, ...]
    """Parent of each of the 25 joints; the wrist (0) is its own parent."""
    bind_rotations: Float32[Tensor, "25 3 3"]
    """Absolute local rotations of the bind pose at this identity."""
    translations: Float32[Tensor, "25 3"]
    """Local translations (parent frame) at this identity; the root's is unused (``global_translation`` places it)."""
    translation_basis: Float32[Tensor, "10 25 3"]
    """d translations / d beta, by central differences at the zero identity."""
    translation_mean: Float32[Tensor, "25 3"]
    """Local translations at the zero identity."""
    twist_axes: Float32[Tensor, "25 3"]
    """Unit bone axis of each joint in its own frame (towards its child); the root's is unused."""


class LandmarkHandModel:
    """SOMA-X hand of one side (XLO layer, whose ``prepare_identity`` gives the skeleton) and its differentiable joint FK."""

    def __init__(self, side: Side, *, device: str = "cuda") -> None:
        self.side: Side = side
        self.device: torch.device = torch.device(device)
        settings: dict[str, object] = {
            "data_root": str(soma_assets()),
            "hand_type": side,
            "device": device,
            "identity_model_type": "mano",
            "identity_model_kwargs": {"model_path": str(mano_model(side))},
            "correctives_model_path": None,
        }
        self.xlo: SOMAHandLayer = SOMAHandLayer(**settings, lod=soma_x.LOD).to(self.device)  # type: ignore[arg-type]
        self.parents: tuple[int, ...] = tuple(int(p) for p in torch.as_tensor(self.xlo.joint_parent_ids).tolist())
        with torch.no_grad():
            mean: Float32[Tensor, "25 3"] = self._translations(torch.zeros(10, device=self.device))[0]
            columns: list[Tensor] = []
            for k in range(10):
                step: Tensor = torch.zeros(10, device=self.device)
                step[k] = 0.5
                columns.append((self._translations(step)[0] - self._translations(-step)[0]) / 1.0)
        self.translation_mean: Float32[Tensor, "25 3"] = mean
        self.translation_basis: Float32[Tensor, "10 25 3"] = torch.stack(columns)

    def _translations(self, betas: Float32[Tensor, "10"]) -> tuple[Tensor, Tensor]:
        """Local translations and bind rotations of an identity, from ``prepare_identity``'s bind transforms."""
        self.xlo.prepare_identity(betas[None].float())
        cached: Tensor | None = self.xlo._cached_bind_transforms_world
        if cached is None:
            raise RuntimeError("prepare_identity cached no bind transforms")
        world: Tensor = cached[0]
        local: Tensor = torch.stack([world[j] if self.parents[j] == j else torch.linalg.inv(world[self.parents[j]]) @ world[j] for j in range(25)])
        return local[:, :3, 3].clone(), local[:, :3, :3].clone()

    def skeleton(self, betas: Float32[ndarray, "10"]) -> HandSkeleton:
        """The exact skeleton of an identity (``prepare_identity``), with the linear model attached."""
        with torch.no_grad():
            translations, rotations = self._translations(torch.as_tensor(betas, device=self.device))
        children: dict[int, int] = {}
        for joint, parent in enumerate(self.parents):
            if parent != joint and parent not in children:
                children[parent] = joint
        axes: Tensor = torch.zeros(25, 3, device=self.device)
        for joint in range(25):
            child: int = children.get(joint, joint)
            vector: Tensor = translations[child] if child != joint else translations[joint]
            axes[joint] = vector / vector.norm().clamp_min(1e-9)
        return HandSkeleton(self.parents, rotations, translations, self.translation_basis, self.translation_mean, axes)


def forward_kinematics(
    parents: tuple[int, ...], rotations: Float32[Tensor, "*b 25 3 3"], translations: Float32[Tensor, "*b 25 3"], root: Float32[Tensor, "*b 3"]
) -> tuple[Float32[Tensor, "*b 25 3 3"], Float32[Tensor, "*b 25 3"]]:
    """World rotations and positions of the 25 joints; equals ``SOMAHandLayer.pose(..., absolute_pose=True, fk_only=True)``."""
    world_rotations: list[Tensor] = []
    world_positions: list[Tensor] = []
    for joint, parent in enumerate(parents):
        if parent == joint:
            world_rotations.append(rotations[..., joint, :, :])
            world_positions.append(root)
        else:
            world_rotations.append(world_rotations[parent] @ rotations[..., joint, :, :])
            world_positions.append(world_positions[parent] + (world_rotations[parent] @ translations[..., joint, :, None])[..., 0])
    return torch.stack(world_rotations, -3), torch.stack(world_positions, -2)


def landmarks_from_fk(
    parents: tuple[int, ...],
    world_rotations: Float32[Tensor, "*b 25 3 3"],
    world_positions: Float32[Tensor, "*b 25 3"],
    offsets: Float32[Tensor, "21 3"],
) -> Float32[Tensor, "*b 21 3"]:
    """The 21 model landmarks: each mapped joint plus its offset in the frame of the joint's parent (the wrist's own frame)."""
    joints: list[int] = list(SOMA_JOINT_OF_LANDMARK)
    frames: list[int] = [parents[j] for j in joints]
    return world_positions[..., joints, :] + (world_rotations[..., frames, :, :] @ offsets[:, :, None])[..., 0]


def kabsch(source: Float32[Tensor, "b k 3"], target: Float32[Tensor, "b k 3"]) -> tuple[Float32[Tensor, "b 3 3"], Float32[Tensor, "b 3"]]:
    """Rotation and translation that best carry ``source`` onto ``target`` per batch row (no scale)."""
    source_mean: Tensor = source.mean(1, keepdim=True)
    target_mean: Tensor = target.mean(1, keepdim=True)
    covariance: Tensor = (target - target_mean).transpose(1, 2) @ (source - source_mean)
    u, _, vh = torch.linalg.svd(covariance.double())
    sign: Tensor = torch.sign(torch.linalg.det(u @ vh))
    fix: Tensor = torch.diag_embed(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], -1))
    rotation: Tensor = (u @ fix @ vh).float()
    return rotation, (target_mean - (rotation @ source_mean.transpose(1, 2)).transpose(1, 2))[:, 0]


def minimal_rotation(a: Float32[Tensor, "b 3"], b: Float32[Tensor, "b 3"]) -> Float32[Tensor, "b 3 3"]:
    """Smallest rotation taking direction ``a`` to direction ``b`` (no twist about either)."""
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axis: Tensor = torch.linalg.cross(a, b)
    sine: Tensor = axis.norm(dim=-1, keepdim=True)
    cosine: Tensor = (a * b).sum(-1, keepdim=True)
    angle: Tensor = torch.atan2(sine, cosine)
    return so3_exp(axis / sine.clamp_min(1e-9) * angle)


PALM_LANDMARKS: tuple[int, ...] = (0, 5, 9, 13, 17)
"""Wrist and the four finger MCP landmarks: the rigid palm the root initialisation aligns."""
SWING_CHILD: dict[int, int] = {
    1: 2,
    2: 3,
    3: 4,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
}
"""Joint -> the child whose landmark its swing aims at."""


def initial_pose(
    skeleton: HandSkeleton, translations: Float32[Tensor, "25 3"], offsets: Float32[Tensor, "21 3"], targets: Float32[Tensor, "b 21 3"]
) -> tuple[Float32[Tensor, "b 25 3 3"], Float32[Tensor, "b 3"]]:
    """Analytic start: Kabsch of the bind-pose palm onto the landmarks, then every bone swung (minimal rotation) onto its child's landmark."""
    count: int = len(targets)
    rotations: Tensor = skeleton.bind_rotations.expand(count, -1, -1, -1).clone()
    root: Tensor = torch.zeros(count, 3, device=targets.device)
    landmark_of: dict[int, int] = {joint: k for k, joint in enumerate(SOMA_JOINT_OF_LANDMARK)}
    world_r, world_p = forward_kinematics(skeleton.parents, rotations, translations.expand(count, -1, -1), root)
    model: Tensor = landmarks_from_fk(skeleton.parents, world_r, world_p, offsets)
    palm: list[int] = list(PALM_LANDMARKS)
    turn, shift = kabsch(model[:, palm], targets[:, palm])
    rotations[:, 0] = turn @ rotations[:, 0]
    root = shift
    for joint, child in SWING_CHILD.items():
        world_r, world_p = forward_kinematics(skeleton.parents, rotations, translations.expand(count, -1, -1), root)
        model = landmarks_from_fk(skeleton.parents, world_r, world_p, offsets)
        k: int = landmark_of[child]
        pivot: Tensor = world_p[:, joint]
        swing: Tensor = minimal_rotation(model[:, k] - pivot, targets[:, k] - pivot)
        frame: Tensor = world_r[:, joint]
        rotations[:, joint] = rotations[:, joint] @ (frame.transpose(1, 2) @ swing @ frame)
    return rotations, root


@dataclass(frozen=True, slots=True)
class PoseSolve:
    """Per-frame pose of one hand: absolute local rotations and root translation, world metres."""

    rotations: Float32[Tensor, "b 25 3 3"]
    """Absolute local rotations (SOMA ``pose`` input with ``absolute_pose=True``)."""
    root: Float32[Tensor, "b 3"]
    """Wrist position (SOMA ``pose``'s ``global_translation``)."""
    error_m: Float32[Tensor, "b 21"]
    """Per-landmark distance between model and target."""


def _rotation_prior_weights(skeleton: HandSkeleton) -> tuple[Tensor, Tensor]:
    """Per free joint: the swing and twist prior weights (none on the root, whose rotation is the hand's orientation)."""
    free: list[int] = list(FREE_JOINTS)
    swing: Tensor = torch.full((len(free),), ROTATION_PRIOR_M, device=skeleton.translations.device)
    swing[0] = 0.0
    return swing, swing * TWIST_PRIOR


def solve_poses(
    skeleton: HandSkeleton,
    offsets: Float32[Tensor, "21 3"],
    targets: Float32[Tensor, "b 21 3"],
    weights: Float32[Tensor, "b 21"],
    *,
    iterations: int = 12,
    start: tuple[Tensor, Tensor] | None = None,
) -> PoseSolve:
    """Batched Levenberg-Marquardt on the free joints' rotations (right-multiplied increments) and the root translation.

    Residuals: sqrt(weight) times the 63 landmark coordinates, plus the rotation prior on each finger joint's rotation away
    from the bind pose (stronger about the bone axis, ``TWIST_PRIOR``). Each frame keeps its own damping and accepts a step
    only when its cost drops. ``start`` is a warm start (rotations, root); the analytic ``initial_pose`` otherwise.
    """
    rotations, root = initial_pose(skeleton, skeleton.translations, offsets, targets) if start is None else start
    free: Tensor = torch.as_tensor(FREE_JOINTS, device=targets.device)
    swing_w, twist_w = _rotation_prior_weights(skeleton)
    axes: Tensor = skeleton.twist_axes[free]
    bind: Tensor = skeleton.bind_rotations[free]
    root_weights: Tensor = weights.sqrt()
    unknowns: int = len(FREE_JOINTS) * 3 + 3

    @no_type_check
    def residual(delta, rotations, root, target, sqrt_w):
        """Residual vector of one frame after an increment ``delta`` (60 rotation + 3 translation)."""
        step = so3_exp(delta[:-3].reshape(-1, 3))
        updated = rotations.clone()
        updated[free] = rotations[free] @ step
        world_r, world_p = forward_kinematics(skeleton.parents, updated, skeleton.translations, root + delta[-3:])
        model = landmarks_from_fk(skeleton.parents, world_r, world_p, offsets)
        data = ((model - target) * sqrt_w[:, None]).reshape(-1)
        relative = rotvec_small(bind.transpose(-1, -2) @ updated[free])
        along = (relative * axes).sum(-1)
        across = relative - along[:, None] * axes
        prior = torch.cat([(across * swing_w[:, None]).reshape(-1), along * twist_w])
        return torch.cat([data, prior])

    jacobian: Callable[..., Tensor] = torch.func.vmap(torch.func.jacfwd(residual, argnums=0))
    evaluate: Callable[..., Tensor] = torch.func.vmap(residual)
    count: int = len(targets)
    damping: Tensor = torch.full((count,), 1e-3, device=targets.device)
    zero: Tensor = torch.zeros(count, unknowns, device=targets.device)
    cost: Tensor = (evaluate(zero, rotations, root, targets, root_weights) ** 2).sum(-1)
    eye: Tensor = torch.eye(unknowns, device=targets.device)
    for _ in range(iterations):
        r: Tensor = evaluate(zero, rotations, root, targets, root_weights)
        j: Tensor = jacobian(zero, rotations, root, targets, root_weights)
        jtj: Tensor = j.transpose(1, 2) @ j
        jtr: Tensor = (j.transpose(1, 2) @ r[..., None])[..., 0]
        system: Tensor = jtj + damping[:, None, None] * (torch.diag_embed(jtj.diagonal(dim1=1, dim2=2)) + 1e-9 * eye)
        delta: Tensor = -torch.linalg.solve(system, jtr)
        delta = torch.nan_to_num(delta)
        trial_r: Tensor = rotations.clone()
        trial_r[:, free] = rotations[:, free] @ so3_exp(delta[:, :-3].reshape(count, -1, 3))
        trial_root: Tensor = root + delta[:, -3:]
        trial_cost: Tensor = (evaluate(zero, trial_r, trial_root, targets, root_weights) ** 2).sum(-1)
        better: Tensor = trial_cost < cost
        rotations = torch.where(better[:, None, None, None], trial_r, rotations)
        root = torch.where(better[:, None], trial_root, root)
        cost = torch.where(better, trial_cost, cost)
        damping = torch.where(better, damping * 0.3, damping * 10.0).clamp(1e-7, 1e7)
    world_r, world_p = forward_kinematics(skeleton.parents, rotations, skeleton.translations.expand(count, -1, -1), root)
    error: Tensor = (landmarks_from_fk(skeleton.parents, world_r, world_p, offsets) - targets).norm(dim=-1)
    return PoseSolve(rotations, root, error)


@no_type_check
def rotvec_small(matrices):
    """Axis-angle of rotations well inside (-pi, pi): the skew part over sinc (the prior's argument; differentiable at identity)."""
    skew = (
        torch.stack(
            [matrices[..., 2, 1] - matrices[..., 1, 2], matrices[..., 0, 2] - matrices[..., 2, 0], matrices[..., 1, 0] - matrices[..., 0, 1]], -1
        )
        / 2
    )
    cosine = ((matrices[..., 0, 0] + matrices[..., 1, 1] + matrices[..., 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cosine)
    return skew * torch.where(angle < 1e-3, 1 + angle * angle / 6, angle / torch.sin(angle))[..., None]


def skeleton_at(skeleton: HandSkeleton, betas: Float32[Tensor, "10"]) -> HandSkeleton:
    """The linear identity model's skeleton at ``betas`` (translations only; bind rotations kept)."""
    translations: Tensor = skeleton.translation_mean + torch.einsum("k,kjc->jc", betas, skeleton.translation_basis)
    return HandSkeleton(
        skeleton.parents, skeleton.bind_rotations, translations, skeleton.translation_basis, skeleton.translation_mean, skeleton.twist_axes
    )


def solve_shared(
    skeleton: HandSkeleton,
    rotations: Float32[Tensor, "b 25 3 3"],
    root: Float32[Tensor, "b 3"],
    targets: Float32[Tensor, "b 21 3"],
    weights: Float32[Tensor, "b 21"],
    betas: Float32[Tensor, "10"],
    offsets: Float32[Tensor, "21 3"],
    *,
    solve_betas: bool,
    solve_offsets: bool,
    offset_prior_m: float = 0.0,
) -> tuple[Float32[Tensor, "10"], Float32[Tensor, "21 3"]]:
    """One Gauss-Newton step on the shared unknowns (betas, landmark offsets) with every frame's pose held; the landmarks are
    linear in both, so one step solves them."""
    count: int = len(targets)
    root_weights: Tensor = weights.sqrt()

    @no_type_check
    def residual(shared):
        b, o = shared[:10], shared[10:].reshape(21, 3)
        translations = skeleton.translation_mean + torch.einsum("k,kjc->jc", b, skeleton.translation_basis)
        world_r, world_p = forward_kinematics(skeleton.parents, rotations, translations.expand(count, -1, -1), root)
        model = landmarks_from_fk(skeleton.parents, world_r, world_p, o)
        data = ((model - targets) * root_weights[..., None]).reshape(-1)
        scale = (count * 21) ** 0.5
        return torch.cat([data, BETA_PRIOR_M * scale * b, offset_prior_m * scale * o.reshape(-1)])

    shared: Tensor = torch.cat([betas, offsets.reshape(-1)])
    r: Tensor = residual(shared)
    j: Tensor = torch.func.jacfwd(residual)(shared)
    mask: Tensor = torch.zeros(73, dtype=torch.bool, device=targets.device)
    mask[:10] = solve_betas
    mask[10:] = solve_offsets
    jm: Tensor = j[:, mask]
    step: Tensor = -torch.linalg.lstsq(jm.double(), r.double()[:, None]).solution[:, 0].float()
    shared = shared.clone()
    shared[mask] += step
    return shared[:10], shared[10:].reshape(21, 3)


@dataclass(frozen=True, slots=True)
class LandmarkFit:
    """One hand's SOMA-X fit to landmarks over the rows it used, world metres."""

    betas: Float32[ndarray, "10"]
    """SOMA-X identity (MANO backend betas), shared by every row."""
    rotations: Float32[ndarray, "n 25 3"]
    """Absolute local axis-angle rotations (``soma_x.ROTATION_ENCODING``)."""
    root_translation: Float32[ndarray, "n 3"]
    """SOMA ``pose``'s ``global_translation`` (the wrist joint)."""
    joints: Float32[ndarray, "n 25 3"]
    """SOMA-X joints posed by SOMA-X itself (XLO layer)."""
    landmarks: Float32[ndarray, "n 21 3"]
    """Model landmarks (joints plus the source's offsets)."""
    error_mm: Float32[ndarray, "n 21"]
    """Per-landmark distance to the target landmarks."""


def fit_identity(
    model: LandmarkHandModel,
    targets: Float32[Tensor, "s 21 3"],
    weights: Float32[Tensor, "s 21"],
    offsets: Float32[Tensor, "21 3"],
    *,
    rounds: int = 6,
    solve_offsets: bool = False,
    betas: Float32[Tensor, "10"] | None = None,
) -> tuple[Float32[Tensor, "10"], Float32[Tensor, "21 3"]]:
    """Betas (and, when ``solve_offsets``, the landmark offsets) by alternating pose LM on the sample frames and a shared GN step."""
    base: HandSkeleton = model.skeleton(np.zeros(10, dtype=np.float32))
    betas = torch.zeros(10, device=targets.device) if betas is None else betas
    start: tuple[Tensor, Tensor] | None = None
    for _ in range(rounds):
        skeleton: HandSkeleton = skeleton_at(base, betas)
        solve: PoseSolve = solve_poses(skeleton, offsets, targets, weights, iterations=8, start=start)
        start = (solve.rotations, solve.root)
        betas, offsets = solve_shared(
            base, solve.rotations, solve.root, targets, weights, betas, offsets, solve_betas=True, solve_offsets=solve_offsets
        )
    return betas, offsets


def fit_hand(
    model: LandmarkHandModel,
    targets: Float32[ndarray, "n 21 3"],
    weights: Float32[ndarray, "n 21"],
    offsets: Float32[ndarray, "21 3"],
    *,
    identity_frames: int = 256,
    batch: int = 2048,
    iterations: int = 12,
) -> LandmarkFit:
    """Fit one hand of one segment: the identity on an evenly spaced subsample, then every frame's pose with the exact skeleton.

    Args:
        model: the side's SOMA-X hand.
        targets: Float32[ndarray, "n 21 3"] landmarks of the rows to fit (all finite), world metres.
        weights: Float32[ndarray, "n 21"] per-landmark confidence.
        offsets: Float32[ndarray, "21 3"] the source's landmark offsets (``calibrate_offsets``), metres.
    """
    device: torch.device = model.device
    target_t: Tensor = torch.as_tensor(targets, device=device)
    weight_t: Tensor = torch.as_tensor(weights, device=device)
    offsets_t: Tensor = torch.as_tensor(offsets, device=device)
    sample: Int64[ndarray, "s"] = np.unique(np.linspace(0, len(targets) - 1, min(identity_frames, len(targets))).round().astype(np.int64))
    betas_t, _ = fit_identity(model, target_t[sample], weight_t[sample], offsets_t)
    betas: Float32[ndarray, "10"] = betas_t.cpu().numpy().astype(np.float32)
    skeleton: HandSkeleton = model.skeleton(betas)
    rotations: list[Tensor] = []
    roots: list[Tensor] = []
    errors: list[Tensor] = []
    for first in range(0, len(targets), batch):
        part: slice = slice(first, first + batch)
        solve: PoseSolve = solve_poses(skeleton, offsets_t, target_t[part], weight_t[part], iterations=iterations)
        rotations.append(solve.rotations)
        roots.append(solve.root)
        errors.append(solve.error_m)
    rotation_t: Tensor = torch.cat(rotations)
    root_t: Tensor = torch.cat(roots)
    with torch.no_grad():
        model.xlo.prepare_identity(betas_t[None])
        joints: Tensor = model.xlo.pose(rotation_t, pose2rot=False, absolute_pose=True, global_translation=root_t, fk_only=True)["joints"]
        world_r, world_p = forward_kinematics(skeleton.parents, rotation_t, skeleton.translations.expand(len(root_t), -1, -1), root_t)
        landmarks: Tensor = landmarks_from_fk(skeleton.parents, world_r, world_p, offsets_t)
    return LandmarkFit(
        betas=betas,
        rotations=rotvecs_from_matrices(rotation_t).float().cpu().numpy(),
        root_translation=root_t.cpu().numpy().astype(np.float32),
        joints=joints.cpu().numpy().astype(np.float32),
        landmarks=landmarks.cpu().numpy().astype(np.float32),
        error_mm=(torch.cat(errors) * 1000.0).cpu().numpy().astype(np.float32),
    )


def calibrate_offsets(
    model: LandmarkHandModel,
    hands: list[tuple[Float32[ndarray, "n 21 3"], Float32[ndarray, "n 21"]]],
    *,
    frames_per_hand: int = 128,
    rounds: int = 4,
) -> Float32[ndarray, "21 3"]:
    """One side's landmark offsets shared by several hands (segments): alternate each hand's identity fit with a joint
    least-squares step on the offsets over every hand's sample frames. Calibrate on segments that are not evaluated."""
    device: torch.device = model.device
    base: HandSkeleton = model.skeleton(np.zeros(10, dtype=np.float32))
    samples: list[tuple[Tensor, Tensor]] = []
    for targets, weights in hands:
        pick: Int64[ndarray, "s"] = np.unique(np.linspace(0, len(targets) - 1, min(frames_per_hand, len(targets))).round().astype(np.int64))
        samples.append((torch.as_tensor(targets[pick], device=device), torch.as_tensor(weights[pick], device=device)))
    offsets: Tensor = torch.zeros(21, 3, device=device)
    betas: list[Tensor] = [torch.zeros(10, device=device) for _ in samples]
    for _ in range(rounds):
        poses: list[PoseSolve] = []
        for index, (targets_t, weights_t) in enumerate(samples):
            betas[index], _ = fit_identity(model, targets_t, weights_t, offsets, rounds=3, betas=betas[index])
            poses.append(solve_poses(skeleton_at(base, betas[index]), offsets, targets_t, weights_t, iterations=8))
        # Offsets: the landmarks are linear in them; stack every hand's normal equations (betas held).
        normal: Tensor = torch.zeros(63, 63, device=device, dtype=torch.float64)
        rhs: Tensor = torch.zeros(63, device=device, dtype=torch.float64)
        for index, (targets_t, weights_t) in enumerate(samples):
            skeleton: HandSkeleton = skeleton_at(base, betas[index])
            pose: PoseSolve = poses[index]
            world_r, world_p = forward_kinematics(skeleton.parents, pose.rotations, skeleton.translations.expand(len(targets_t), -1, -1), pose.root)
            joints_only: Tensor = landmarks_from_fk(skeleton.parents, world_r, world_p, torch.zeros(21, 3, device=device))
            frames: Tensor = world_r[:, [skeleton.parents[j] for j in SOMA_JOINT_OF_LANDMARK]]  # (s, 21, 3, 3)
            gap: Tensor = targets_t - joints_only
            w: Tensor = weights_t.double()
            for k in range(21):
                a: Tensor = frames[:, k].double()  # landmark_k = joint + a @ o_k
                normal[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] += torch.einsum("s,sji,sjk->ik", w[:, k], a, a)
                rhs[3 * k : 3 * k + 3] += torch.einsum("s,sji,sj->i", w[:, k], a, gap[:, k].double())
        offsets = torch.linalg.solve(normal + 1e-6 * torch.eye(63, device=device, dtype=torch.float64), rhs).float().reshape(21, 3)
    return offsets.cpu().numpy().astype(np.float32)
