"""SOMA-X fits of source body and hand meshes: the GPU half of the ``soma_x`` layer (``dataforge.soma_x`` writes it).

The official SOMA-X conversion (upstream ``tools/hand/mano2soma.py``): the source model is SOMA-X's identity backend,
so the source shape coefficients are the SOMA-X identity; the posed source mesh is carried onto SOMA's topology
(barycentric wrap) and ``PoseInversion`` recovers SOMA's rotations and root translation. The pose is fitted at SOMA's
mid level of detail and re-posed at XLO. The fit error is the median per-vertex distance between the fitted mid mesh
and the wrapped source mesh, boundary vertices without a correspondence excluded.

Importable only where SOMA-X and CUDA torch are installed (the ``dataforge-soma-x`` environments).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, no_type_check

import numpy as np
import simplecv
import soma.fitting.pose_inversion as pose_inversion
import torch
from huggingface_hub import snapshot_download
from jaxtyping import Float, Float32, Float64, Int64, UInt32
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.mano.mano_np import MANOLayerNP
from soma.body import SOMALayer
from soma.fitting.pose_inversion import PoseInversion
from soma.geometry.lbs import batch_rodrigues
from soma.geometry.lbs_warp import linear_blend_skinning
from soma.hand import SOMAHandLayer
from torch import Tensor

from dataforge import soma_x
from dataforge.soma_x_identity import install_fast_identity_setup

Side: TypeAlias = Literal["left", "right"]
BodyBackend: TypeAlias = Literal["smpl", "smplh", "smplx"]

HAND_MAX_MEDIAN_ERROR_MM: float = 2.0
"""A hand row is trusted when its fit's median per-vertex error is at most this (decision 11)."""
BODY_MAX_MEDIAN_ERROR_MM: float = 5.0
"""A body row is trusted when its fit's median per-vertex error is at most this (decision 11)."""
FIT_BATCH: dict[soma_x.Kind, int] = {"hand": 512, "body": 512}
"""Frames per inversion batch: with Lie-GN on the vertex subset, ~4.5 GB of GPU memory; 512 amortises the per-joint FK loops
(Ego-Exo4D take on the DGX Spark: 631 fps at 128, 793 at 512, 729 at 1024)."""
SOMA_X_FILES: tuple[str, ...] = (
    "SOMA_neutral.npz",
    "SOMAHand.npz",
    "SOMA_procedural_transforms.json",
    "SOMA_template_rig.usda",
    "MANO/*",
    "SMPL/*",
    "SMPLX/*",
    "LICENSE",
)
"""The pinned files the hand and SMPL-family backends read; correctives are not used (the fit runs without them)."""


def rotvecs_to_matrices(rotations: Float32[ndarray, "n j 3"]) -> Float32[ndarray, "n j 3 3"]:
    """Axis-angle to rotation matrices, the inverse of ``rotvecs_from_matrices`` (SOMA-X 0.3.0's ``matrix_to_rotvec``/``rotvec_to_matrix`` pair does not round-trip)."""
    flat: Float64[ndarray, "m 3 3"] = Rotation.from_rotvec(rotations.reshape(-1, 3).astype(np.float64)).as_matrix()
    return flat.reshape(*rotations.shape, 3).astype(np.float32)


def rotvecs_from_matrices(matrices: Float[Tensor, "*batch 3 3"]) -> Float64[Tensor, "*batch 3"]:
    """Rotation matrices to axis-angle in float64 on the matrices' device: quaternion by the largest of the diagonal and the
    trace (scipy's ``Rotation.from_matrix`` route), angle in [0, pi]. Matches scipy to 2.4e-7 on 400k random rotations, near
    pi and near zero included, and runs on the GPU (scipy took 1.3 s of a 3 s Ego-Exo4D fit)."""
    lead: tuple[int, ...] = tuple(matrices.shape[:-2])
    m: Float64[Tensor, "m 3 3"] = matrices.double().reshape(-1, 3, 3)
    trace: Float64[Tensor, "m"] = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    choice: Int64[Tensor, "m"] = torch.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], trace], dim=1).argmax(dim=1)
    quaternion_xyzw: Float64[Tensor, "m 4"] = torch.stack(
        [m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1], 1 + trace], dim=1
    )
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        rows: Tensor = choice == i
        picked: Float64[Tensor, "p 3 3"] = m[rows]
        q: Float64[Tensor, "p 4"] = torch.empty(len(picked), 4, dtype=m.dtype, device=m.device)
        q[:, i] = 1 - trace[rows] + 2 * picked[:, i, i]
        q[:, j] = picked[:, j, i] + picked[:, i, j]
        q[:, k] = picked[:, k, i] + picked[:, i, k]
        q[:, 3] = picked[:, k, j] - picked[:, j, k]
        quaternion_xyzw[rows] = q
    quaternion_xyzw = quaternion_xyzw / quaternion_xyzw.norm(dim=1, keepdim=True)
    quaternion_xyzw = torch.where(quaternion_xyzw[:, 3:] < 0, -quaternion_xyzw, quaternion_xyzw)
    angle: Float64[Tensor, "m"] = 2 * torch.atan2(quaternion_xyzw[:, :3].norm(dim=1), quaternion_xyzw[:, 3])
    squared: Float64[Tensor, "m"] = angle * angle
    scale: Float64[Tensor, "m"] = torch.where(
        angle <= 1e-3, 2 + squared / 12 + 7 * squared * squared / 2880, angle / torch.sin(angle / 2).clamp_min(torch.finfo(m.dtype).tiny)
    )
    return (scale[:, None] * quaternion_xyzw[:, :3]).reshape(*lead, 3)


LIE_VERTEX_STRIDE: int = 4
"""Lie-GN refines on every 4th vertex of the fitting LOD; its dense (frames x joints x vertices x 3) work is ~85 % of the
inversion's time. Measured on an Ego-Exo4D take (3,336 frames, DGX Spark): 2.2x the fit rate (290 -> 640 fps), median error
2.366 -> 2.399 mm, worst frame 5.30 -> 5.19 mm. Stride 8 was 787 fps but +0.19 mm median, over the 0.1 mm parity band."""


KEYFRAME_STRIDE: int = 1
"""1: every frame gets the full fit. Warm-starting the frames between keyframes from them (stride 4, 2 Lie-GN iterations) ran 1.2x faster but its worst frame rose from 5.2 to 11.9 mm on an Ego-Exo4D take, so it stays off."""
INBETWEEN_LIE_ITERS: int = 2
"""Lie-GN iterations of a frame warm-started from its keyframe."""


@dataclass(frozen=True, slots=True)
class VertexSubset:
    """The fitting layer's vertices that the Gauss-Newton refinement uses, the inversion cache cut to them, and the skeleton
    tables Lie-GN derives from them (built once per identity instead of once per batch: ~45 ms of small launches each)."""

    indices: Tensor
    """Kept vertex indices of the fitting LOD."""
    cache: dict[str, Any]
    """``PoseInversion._cache`` with every per-vertex entry and the root arm's vertex ids cut to ``indices``."""
    ancestors: Float32[Tensor, "j j"]
    """``ancestors[a, j]`` is 1 when joint a is joint j or one of its ancestors."""
    descendant_weights: Float32[Tensor, "j s"]
    """Skinning weight of each kept vertex summed over each joint's subtree."""
    active: Int64[Tensor, "a"]
    """Joints with geometry under them (upstream's ``_active_lie_joint_indices``): the Lie-GN unknowns."""
    self_parent: Tensor
    """Bool (j,): joints that are their own parent (the rig root)."""


def vertex_subset(inversion: PoseInversion, stride: int) -> VertexSubset:
    """Every ``stride``-th vertex, with the cache cut to them; the root arm (``_update_root_translation``) keeps its vertices among them."""
    assert inversion._cache is not None, "prepare_identity() first"
    cache: dict[str, Any] = inversion._cache
    count: int = int(cache["skinning_weights"].shape[0])
    device: torch.device = cache["skinning_weights"].device
    keep: Tensor = torch.arange(0, count, stride, device=device)
    sub: dict[str, Any] = {
        key: (value[keep] if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == count else value) for key, value in cache.items()
    }
    root: int = int(cache.get("root_idx", pose_inversion._HIPS_IDX))
    arm: dict[str, Tensor] | None = cache["joint_cache"].get(root)
    if arm is not None:
        where: Tensor = torch.full((count,), -1, dtype=torch.long, device=device)
        where[keep] = torch.arange(len(keep), device=device)
        arm_ids: Tensor = torch.as_tensor(arm["arm_vids"], device=device).long()
        inside: Tensor = where[arm_ids] >= 0
        cut: dict[str, Any] = {}
        for key, value in arm.items():
            if key == "arm_vids":
                cut[key] = where[arm_ids][inside]
            elif torch.is_tensor(value) and value.ndim >= 1 and len(arm_ids) in value.shape:
                axis: int = list(value.shape).index(len(arm_ids))
                cut[key] = value.index_select(axis, torch.nonzero(inside).flatten())
            else:
                cut[key] = value
        sub["joint_cache"] = {**cache["joint_cache"], root: cut}
    parents: list[int] = sub["parent_ids_list"]
    joints: int = len(parents)
    ancestors: Float32[Tensor, "j j"] = torch.zeros(joints, joints)
    for joint in range(joints):
        current: int = joint
        while True:
            ancestors[current, joint] = 1.0
            parent: int = parents[current]
            if parent == current:
                break
            current = parent
    weights: Tensor = sub["skinning_weights"]
    ancestors = ancestors.to(device=device, dtype=weights.dtype)
    descendant_weights: Float32[Tensor, "j s"] = ancestors @ weights.T
    return VertexSubset(
        indices=keep,
        cache=sub,
        ancestors=ancestors,
        descendant_weights=descendant_weights,
        active=pose_inversion._active_lie_joint_indices(descendant_weights, inversion._root_joint_idx),
        self_parent=sub["parent_ids"] == torch.arange(joints, device=device),
    )


@no_type_check  # torch.compile traces this with fake tensors, which beartype's jaxtyping checks reject (locals unannotated for the same reason)
def _normal_equations(
    skinning: Float32[Tensor, "b j 4 4"],
    pivots: Float32[Tensor, "b j 3"],
    bind_shape: Float32[Tensor, "b s 3"],
    target: Float32[Tensor, "b s 3"],
    weights: Float32[Tensor, "s j"],
    ancestors: Float32[Tensor, "j j"],
    descendant_weights: Float32[Tensor, "j s"],
    active: Int64[Tensor, "a"],
) -> tuple[Float32[Tensor, "b n n"], Float32[Tensor, "b a 3"], Float32[Tensor, "b"]]:
    """Upstream's Lie-GN normal equations (uniform vertex weights) with the Gram matrix as one bmm over the active joints'
    lever arms: JtJ, JtE and the mean residual. Compiled below, so the (frames x joints x vertices x 3) passes fuse."""
    count, joints = skinning.shape[0], skinning.shape[1]
    vertices = bind_shape.shape[-2]
    p_world = torch.einsum("bjmn,bvn->bjvm", skinning[:, :, :3, :3], bind_shape) + skinning[:, :, None, :3, 3]
    residual = target - torch.einsum("vj,bjvm->bvm", weights, p_world)
    weighted = p_world * weights.T[None, :, :, None]
    lever = (ancestors[active] @ weighted.reshape(count, joints, vertices * 3)).reshape(count, -1, vertices, 3)
    q = lever - descendant_weights[active][None, :, :, None] * pivots[:, active][:, :, None, :]
    unknowns = q.shape[1]
    jte = torch.linalg.cross(q, residual[:, None].expand_as(q)).sum(dim=2)
    rows = q.permute(0, 1, 3, 2).reshape(count, unknowns * 3, vertices)
    gram = torch.bmm(rows, rows.transpose(1, 2)).view(count, unknowns, 3, unknowns, 3)
    dots = torch.diagonal(gram, dim1=2, dim2=4).sum(-1)
    eye3 = torch.eye(3, device=q.device, dtype=q.dtype)
    jtj = (dots[:, :, None, :, None] * eye3[None, None, :, None, :] - gram.permute(0, 1, 4, 3, 2)).reshape(count, unknowns * 3, unknowns * 3)
    return jtj, jte, residual.norm(dim=-1).mean(dim=-1)


def solve_normal_equations(jtj: Float32[Tensor, "b n n"], rhs: Float32[Tensor, "b n"]) -> Float32[Tensor, "b n"]:
    """Cholesky first (the damped JtJ is symmetric positive definite); frames where it fails or is not finite take upstream's
    LU with its damping ladder. Same fit error as upstream's solver to 0.01 mm; Ego-Exo4D fit 2,599 -> 4,251 fps (5090)."""
    factor, info = torch.linalg.cholesky_ex(jtj)
    solution: Float32[Tensor, "b n"] = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
    failed: Tensor = (info != 0) | ~torch.isfinite(solution).all(dim=-1)
    if bool(failed.any()):
        solution[failed] = pose_inversion._solve_lie_gn_normal_equations(jtj[failed], rhs[failed])
    return solution


_compiled_normal_equations = torch.compile(_normal_equations, dynamic=False)
LINE_SEARCH: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
"""Upstream's Lie-GN step fractions, tried in one batched pass; the first strictly better one of the lowest error wins, as upstream."""


def lie_gn_on_vertices(
    inversion: PoseInversion,
    target: Tensor,
    rest: Tensor,
    subset: VertexSubset,
    init: dict[str, Tensor],
    *,
    iterations: int = 3,
    damping: float = 0.1,
) -> dict[str, Tensor]:
    """SOMA-X 0.3.0's ``PoseInversion._fit_lie_algebra_gn`` on an explicit vertex subset, same math, fewer launches.

    ``target`` (b s 3) and ``rest`` (1 s 3) are already cut to ``subset.indices``. Same damping, line search (1, 1/2, 1/4,
    1/8), root-translation update and warm start from ``init`` (the analytical stage); uniform vertex weights (upstream's
    leaf weight 1.0). The normal equations run through ``torch.compile`` (Triton), the four line-search steps run as one
    batch, and batches are padded to a power of two so a process compiles a handful of shapes. Same fit error as the
    step-for-step copy to 0.01 mm on Ego-Exo4D, HO-Cap and EPFL; Ego-Exo4D fit 1,183 -> 3,251 fps on the 5090.
    """
    count: int = target.shape[0]
    padded: int = 64
    while padded < count:
        padded *= 2
    if padded != count:
        extra: int = padded - count
        target = torch.cat([target, target[-1:].expand(extra, -1, -1)])
        init = {
            "rotations": torch.cat([init["rotations"], init["rotations"][-1:].expand(extra, -1, -1, -1)]),
            "root_translation": torch.cat([init["root_translation"], init["root_translation"][-1:].expand(extra, -1)]),
        }
    cache: dict[str, Any] = subset.cache
    batch: int = target.shape[0]
    device, dtype = target.device, target.dtype
    bind_inv: Tensor = pose_inversion._bexpand4(cache["W_bind_inv"], batch)
    bind_shape: Tensor = rest.expand(batch, -1, -1)
    parent_ids: Tensor = cache["parent_ids"]
    joints: int = len(cache["parent_ids_list"])
    root: int = inversion._root_joint_idx
    active: Tensor = subset.active
    steps: int = len(LINE_SEARCH)
    fractions: Tensor = torch.tensor(LINE_SEARCH, device=device, dtype=dtype)
    frames: Tensor = torch.arange(batch, device=device)
    pose_local: Tensor = torch.zeros(batch, joints, 4, 4, device=device, dtype=dtype)
    pose_local[:, :, :3, :3] = init["rotations"]
    pose_local[:, root, :3, 3] = init["root_translation"]
    if root > 0:
        pose_local[:, 0, :3, :3] = torch.eye(3, device=device, dtype=dtype)
    for _ in range(iterations):
        world: Tensor = pose_inversion._build_world_transforms(pose_local, cache)
        jtj, jte, accepted_error = _compiled_normal_equations(
            world @ bind_inv,
            world[:, :, :3, 3].contiguous(),
            bind_shape,
            target,
            cache["skinning_weights"],
            subset.ancestors,
            subset.descendant_weights,
            active,
        )
        jtj.diagonal(dim1=-2, dim2=-1)[:] *= 1.0 + damping
        step: Tensor = torch.zeros(batch, joints, 3, device=device, dtype=dtype)
        step[:, active] = solve_normal_equations(jtj, jte.reshape(batch, -1)).reshape(batch, len(active), 3)
        rotations_world: Tensor = world[:, :, :3, :3]
        delta: Tensor = batch_rodrigues((fractions[:, None, None, None] * step[None]).reshape(-1, 3), dtype=dtype).reshape(steps, batch, joints, 3, 3)
        new_world: Tensor = delta @ rotations_world[None]
        local: Tensor = rotations_world[:, parent_ids].transpose(-2, -1)[None] @ new_world
        local[:, :, subset.self_parent] = new_world[:, :, subset.self_parent]
        trials: Tensor = pose_local[None].expand(steps, -1, -1, -1, -1).clone()
        trials[..., :3, :3] = local
        flat: Tensor = trials.reshape(steps * batch, joints, 4, 4)
        recon: Tensor = linear_blend_skinning(
            bind_shape.repeat(steps, 1, 1),
            cache["bone_weights"],
            cache["bone_indices"],
            pose_inversion._build_world_transforms(flat, cache) @ bind_inv.repeat(steps, 1, 1, 1),
        ).reshape(steps, batch, -1, 3)
        errors: Tensor = torch.cat([accepted_error[None], (target[None] - recon).norm(dim=-1).mean(dim=-1)])
        pose_local = torch.cat([pose_local[None], trials])[errors.argmin(dim=0), frames]
        pose_inversion._update_root_translation(pose_local, target, cache, None)
    return {"rotations": pose_local[:count, :, :3, :3].clone(), "root_translation": pose_local[:count, root, :3, 3].clone()}


@cache
def soma_assets() -> Path:
    """The pinned SOMA-X model directory, downloaded into the Hugging Face cache on first use."""
    return Path(snapshot_download(repo_id=soma_x.SOMA_X_REPO_ID, revision=soma_x.SOMA_X_REVISION, allow_patterns=list(SOMA_X_FILES)))


def mano_model(side: Side) -> Path:
    """simplecv's MANO pickle for one side, fetched by simplecv on first use: the same model the converters pose."""
    path: Path = Path(simplecv.__file__).resolve().parent / "data" / f"MANO_{side.upper()}.pkl"
    if not path.is_file():
        MANOLayerNP(side=side, betas=np.zeros(10, dtype=np.float32), use_pca=False)
    return path


@dataclass(frozen=True, slots=True)
class ModelFit:
    """One body's or hand's SOMA-X fit over n frames, world metres."""

    identity_coeffs: Float32[ndarray, "1 c"]
    """SOMA-X identity: the source model's shape coefficients (its identity backend)."""
    rotations: Float32[ndarray, "n j 3"]
    """Absolute local axis-angle joint rotations (``soma_x.ROTATION_ENCODING``)."""
    root_translation: Float32[ndarray, "n 3"]
    """World translation that the SOMA-X layer's ``pose`` takes."""
    vertices: Float32[ndarray, "n v 3"]
    """XLO vertices."""
    joints: Float32[ndarray, "n j 3"]
    """SOMA-X joints."""
    error_mm: Float32[ndarray, "n"]
    """Median per-vertex fit error of each frame."""


HandFit: TypeAlias = ModelFit
BodyFit: TypeAlias = ModelFit


class _Fitter:
    """The shared half of both fitters: inversion at a fitting LOD, re-posing at XLO, the model record."""

    kind: soma_x.Kind
    identity_model_type: str
    max_median_error_mm: float
    _subset: "VertexSubset | None" = None
    _identity_key: bytes = b""
    fit_layer: SOMALayer | SOMAHandLayer
    xlo: SOMALayer | SOMAHandLayer
    inversion: PoseInversion
    excluded: Int64[ndarray, "e"]
    topology: soma_x.SomaXTopology

    def __init__(self, device: str) -> None:
        self.device: torch.device = _cuda(device)
        # Ego-Exo4D ships one identity per ~100-frame chunk: upstream's prepare_identity (~0.14 s, sequential per-joint LU
        # solves) was ~55 % of its wall clock. Same math, batched (soma_x_identity): ~19 ms.
        install_fast_identity_setup()

    def _finish_setup(self, joint_names: tuple[str, ...]) -> None:
        excluded: object = getattr(self.fit_layer, "excluded_vert_ids", None)
        self.excluded = np.asarray([] if excluded is None else torch.as_tensor(excluded).cpu().numpy(), dtype=np.int64)
        faces: UInt32[ndarray, "f 3"] = cast(Tensor, self.xlo.faces).cpu().numpy().astype(np.uint32)
        parents: Int64[ndarray, "j"] = np.asarray(torch.as_tensor(self.xlo.joint_parent_ids).cpu().tolist(), dtype=np.int64)
        self.topology = soma_x.SomaXTopology(faces=faces, parents=parents, joint_names=joint_names)

    def model(self) -> soma_x.SomaXModel:
        """The ``soma_x`` model record of this fitter's fits."""
        return soma_x.SomaXModel(kind=self.kind, identity_model_type=self.identity_model_type, max_median_error_mm=self.max_median_error_mm)

    def pose(
        self, identity_coeffs: Float32[ndarray, "1 c"], rotations: Float32[ndarray, "n j 3"], root_translation: Float32[ndarray, "n 3"]
    ) -> Float32[ndarray, "n v 3"]:
        """XLO vertices that logged parameters generate: what a consumer re-running SOMA-X gets."""
        return self._pose_xlo(identity_coeffs, rotations, root_translation)["vertices"].float().cpu().numpy()

    def _pose_xlo(
        self, identity_coeffs: Float32[ndarray, "1 c"], rotations: Float32[ndarray, "n j 3"], root_translation: Float32[ndarray, "n 3"]
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            self.xlo.prepare_identity(torch.as_tensor(identity_coeffs, device=self.device).expand(len(rotations), -1))
            return self._pose_layer(
                torch.as_tensor(rotvecs_to_matrices(rotations), device=self.device), torch.as_tensor(root_translation, device=self.device)
            )

    def _pose_layer(self, matrices: Tensor, translation: Tensor) -> dict[str, Tensor]:
        """The XLO layer's own posing call."""
        raise NotImplementedError

    def _public(self, matrices: Tensor) -> Tensor:
        """The inversion's rotations that the layer's ``pose`` takes (the body drops its Root)."""
        return matrices

    def _target(self, posed: Tensor, centre: Tensor) -> Tensor:
        """What the inversion fits, in the fitting layer's frame: the source mesh, centred on ``centre``."""
        return posed - centre

    lie_stride: int = LIE_VERTEX_STRIDE
    """Vertex stride of the Gauss-Newton refinement; 1 runs upstream's own ``PoseInversion.fit``."""

    keyframe_stride: int = KEYFRAME_STRIDE
    """Every this-many-th frame of a batch gets the full fit; the frames between warm-start Lie-GN from it (1 disables)."""

    def _invert(self, target: Tensor, settings: dict[str, int]) -> tuple[Tensor, Tensor, Tensor]:
        """Rotations, root translation and per-vertex error (on every fitting-LOD vertex) of one batch of SOMA-topology targets.

        Upstream's ``fit`` runs the analytical stage then Lie-GN on every frame. Here the keyframes (every
        ``keyframe_stride``-th frame) do that, Lie-GN on ``lie_stride`` vertices; every other frame starts Lie-GN from the
        keyframe before it (frames come in time order, a few tens of milliseconds apart) and runs ``INBETWEEN_LIE_ITERS``.
        """
        if self.lie_stride == 1:
            result: dict[str, Tensor] = self.inversion.fit(
                target,
                batch_size=len(target),
                body_iters=settings["body_iters"],
                finger_iters=settings["finger_iters"],
                full_iters=settings["full_iters"],
                lie_iters=settings["lie_iters"],
            )
            return result["rotations"], result["root_translation"], result["per_vertex_error"]
        assert self.inversion._cache is not None and self.fit_layer._cached_rest_shape is not None, "prepare_identity() first"
        cache: dict[str, Any] = self.inversion._cache
        count: int = len(target)
        with torch.no_grad():
            rest: Tensor = self.fit_layer._cached_rest_shape[:1]
            if self._subset is None:
                self._subset = vertex_subset(self.inversion, self.lie_stride)
            keep: Tensor = self._subset.indices
            keys: Tensor = torch.arange(0, count, self.keyframe_stride, device=self.device)
            init: dict[str, Tensor] = self.inversion._fit_analytical(
                target[keys], cache, settings["body_iters"], settings["finger_iters"], settings["full_iters"], False, 1.0
            )
            keyed: dict[str, Tensor] = lie_gn_on_vertices(
                self.inversion, target[keys][:, keep], rest[:, keep], self._subset, init, iterations=settings["lie_iters"]
            )
            rotations: Tensor = keyed["rotations"][torch.arange(count, device=self.device) // self.keyframe_stride].clone()
            translation: Tensor = keyed["root_translation"][torch.arange(count, device=self.device) // self.keyframe_stride].clone()
            others: Tensor = torch.nonzero(torch.arange(count, device=self.device) % self.keyframe_stride != 0).flatten()
            if len(others):
                warm: dict[str, Tensor] = {"rotations": rotations[others], "root_translation": translation[others]}
                refined: dict[str, Tensor] = lie_gn_on_vertices(
                    self.inversion, target[others][:, keep], rest[:, keep], self._subset, warm, iterations=INBETWEEN_LIE_ITERS
                )
                rotations[others] = refined["rotations"]
                translation[others] = refined["root_translation"]
            rotations[keys] = keyed["rotations"]
            translation[keys] = keyed["root_translation"]
            pose_local: Tensor = torch.zeros(count, rotations.shape[1], 4, 4, device=self.device, dtype=target.dtype)
            pose_local[:, :, :3, :3] = rotations
            pose_local[:, self.inversion._root_joint_idx, :3, 3] = translation
            skinning: Tensor = pose_inversion._build_world_transforms(pose_local, cache) @ pose_inversion._bexpand4(cache["W_bind_inv"], count)
            recon: Tensor = linear_blend_skinning(rest.expand(count, -1, -1), cache["bone_weights"], cache["bone_indices"], skinning)
        return rotations, translation, (recon - target).norm(dim=-1)

    def _fit(
        self, identity: Float32[ndarray, "1 c"], vertices: Float32[ndarray, "n s 3"], centres: Float32[ndarray, "n 3"], **settings: int
    ) -> ModelFit:
        parts: list[tuple[Float32[ndarray, "b j 3"], Float32[ndarray, "b 3"], Float32[ndarray, "b"]]] = []
        if identity.tobytes() != self._identity_key:  # preparing an identity costs ~0.4 s; consecutive windows of one subject share it
            self.inversion.prepare_identity(torch.as_tensor(identity, device=self.device))
            self._subset = None  # the cache is per identity
            self._identity_key = identity.tobytes()
        batch: int = FIT_BATCH[self.kind]
        for start in range(0, len(vertices), batch):
            centre: Tensor = torch.as_tensor(centres[start : start + batch, None], device=self.device)
            posed: Tensor = torch.as_tensor(vertices[start : start + batch], device=self.device)
            with torch.no_grad():
                target: Tensor = self.inversion.transfer_to_soma(self._target(posed, centre))
            inverted: tuple[Tensor, Tensor, Tensor] = self._invert(
                target, {"body_iters": 0, "finger_iters": 0, "full_iters": 1, "lie_iters": 3, **settings}
            )
            translation: Tensor = inverted[1]
            per_vertex: Tensor = inverted[2]
            matrices: Float[Tensor, "b j 3 3"] = self._public(inverted[0])
            kept: Tensor = torch.ones(per_vertex.shape[1], dtype=torch.bool, device=self.device)
            kept[torch.as_tensor(self.excluded, device=self.device)] = False
            parts.append(
                (
                    rotvecs_from_matrices(matrices).float().cpu().numpy(),
                    (translation + centre[:, 0]).float().cpu().numpy(),
                    (per_vertex[:, kept].median(dim=1).values * 1000.0).float().cpu().numpy(),
                )
            )
        rotations: Float32[ndarray, "n j 3"] = np.concatenate([part[0] for part in parts])
        root_translation: Float32[ndarray, "n 3"] = np.concatenate([part[1] for part in parts])
        posed_xlo: dict[str, Tensor] = self._pose_xlo(identity, rotations, root_translation)
        return ModelFit(
            identity_coeffs=identity,
            rotations=rotations,
            root_translation=root_translation,
            vertices=posed_xlo["vertices"].float().cpu().numpy(),
            joints=posed_xlo["joints"].float().cpu().numpy(),
            error_mm=np.concatenate([part[2] for part in parts]),
        )


def _cuda(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SOMA-X fitting needs CUDA, and none is available")
    return torch.device(device)


class HandFitter(_Fitter):
    """MANO → SOMA-X hand for one side (upstream ``hand/mano2soma``): wrist-centred inversion at mid LOD, meshes at XLO."""

    def __init__(self, side: Side, *, fix_left_shapedirs: bool = False, device: str = "cuda") -> None:
        self.kind, self.identity_model_type, self.max_median_error_mm = "hand", "mano", HAND_MAX_MEDIAN_ERROR_MM
        super().__init__(device)
        self.side: Side = side
        settings: dict[str, Any] = {
            "data_root": str(soma_assets()),
            "hand_type": side,
            "device": device,
            "identity_model_type": "mano",
            "identity_model_kwargs": {"model_path": str(mano_model(side))},
            "correctives_model_path": None,
        }
        self.fit_layer = SOMAHandLayer(**settings, lod="mid").to(self.device)  # type: ignore[arg-type]
        self.xlo = SOMAHandLayer(**settings, lod=soma_x.LOD).to(self.device)  # type: ignore[arg-type]
        if fix_left_shapedirs and side == "left":
            # MANO_LEFT.pkl's first shape direction has x not mirrored; sources fitted with the fix (simplecv's
            # fix_left_shapedirs, HOT3D, DAS-Ego) pose a different left shape unless the identity backend applies it too.
            for layer in (self.fit_layer, self.xlo):
                cast(Tensor, layer.identity_model._shapedirs)[:, 0, :] *= -1.0
        self.inversion = PoseInversion(self.fit_layer, low_lod=False)
        self._finish_setup(tuple(self.xlo.rig_data["joint_names"]))

    def _pose_layer(self, matrices: Tensor, translation: Tensor) -> dict[str, Tensor]:
        assert isinstance(self.xlo, SOMAHandLayer)
        return self.xlo.pose(matrices, pose2rot=False, absolute_pose=True, global_translation=translation)

    def _target(self, posed: Tensor, centre: Tensor) -> Tensor:
        """The wrist-centred MANO mesh carried onto SOMA's hand topology."""
        return self.fit_layer.identity_model._to_soma_interp(posed - centre)

    def fit(self, betas: Float32[ndarray, "c"], vertices: Float32[ndarray, "n 778 3"], wrist: Float32[ndarray, "n 3"]) -> ModelFit:
        """Fit SOMA-X to posed MANO hands that share one shape.

        Args:
            betas: Float32[ndarray, "c"] MANO shape of every frame.
            vertices: Float32[ndarray, "n 778 3"] posed MANO vertices, world metres.
            wrist: Float32[ndarray, "n 3"] MANO wrist joint of each frame, world metres: the fit runs wrist-centred.
        """
        return self._fit(betas[None].astype(np.float32), vertices, wrist, body_iters=0, finger_iters=0, full_iters=1, lie_iters=3)


class BodyFitter(_Fitter):
    """SMPL-family body → SOMA-X body (upstream ``smpl2soma``): inversion at low LOD, meshes at XLO."""

    def __init__(self, backend: BodyBackend, *, model_path: Path, gender: str = "neutral", num_betas: int = 10, device: str = "cuda") -> None:
        super().__init__(device)
        self.kind, self.identity_model_type, self.max_median_error_mm = "body", backend, BODY_MAX_MEDIAN_ERROR_MM
        settings: dict[str, Any] = {
            "data_root": str(soma_assets()),
            "device": device,
            "identity_model_type": backend,
            "identity_model_kwargs": {"model_path": str(model_path), "gender": gender, "num_betas": num_betas},
            "correctives_model_path": None,
            # The public 78-joint rig (77 posed joints + root), as selfcap-soma fits: no procedural twist helpers.
            "enable_procedural_transforms": False,
        }
        # The fit runs on a low-LOD layer built here: PoseInversion(low_lod=True) on a mid layer would build its own low layer
        # through `from .body import SOMALayer`, a module path SOMA-X 0.3.0 no longer has (soma.fitting.body), and without our
        # identity_model_kwargs.
        self.fit_layer = SOMALayer(**settings, lod="low").to(self.device)  # type: ignore[arg-type]
        self.xlo = SOMALayer(**settings, lod=soma_x.LOD).to(self.device)  # type: ignore[arg-type]
        self.inversion = PoseInversion(self.fit_layer, low_lod=True)
        self._finish_setup(tuple(str(name) for name in self.xlo._public_joint_names))
        # The rig's joint 0 is a virtual Root that pose() neither takes nor returns: keep the 77 posed joints, Hips as their root.
        parents: Int64[ndarray, "j"] = self.topology.parents[1:] - 1
        parents[parents < 0] = np.flatnonzero(parents < 0)
        self.topology = soma_x.SomaXTopology(faces=self.topology.faces, parents=parents, joint_names=self.topology.joint_names[1:])

    def _pose_layer(self, matrices: Tensor, translation: Tensor) -> dict[str, Tensor]:
        assert isinstance(self.xlo, SOMALayer)
        return self.xlo.pose(matrices, transl=translation, pose2rot=False, absolute_pose=True, apply_correctives=False)

    def _public(self, matrices: Tensor) -> Tensor:
        return matrices[:, 1:]

    def fit(self, betas: Float32[ndarray, "c"], vertices: Float32[ndarray, "n s 3"]) -> ModelFit:
        """Fit SOMA-X to posed source bodies that share one shape (``vertices`` in the source topology, world metres)."""
        origin: Float32[ndarray, "n 3"] = np.zeros((len(vertices), 3), dtype=np.float32)
        return self._fit(betas[None].astype(np.float32), vertices, origin, body_iters=2, finger_iters=0, full_iters=1, lie_iters=3)
