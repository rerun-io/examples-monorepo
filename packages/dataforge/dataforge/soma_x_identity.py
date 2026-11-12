"""Fast per-identity setup for SOMA-X 0.3.0's ``PoseInversion.prepare_identity``: same math, structure built once.

``PoseInversion.prepare_identity`` rebuilds two things every time the identity (betas) changes:

* ``SkeletonTransfer.update_bind`` (``soma/geometry/skeleton_transfer.py``) re-derives the per-joint regressor masks,
  builds one ``RadialBasisFunction`` per joint (77 for the body: a pairwise kernel matrix, a ``lu_factor`` and a
  ``lu_solve`` each), and rebuilds every Warp index buffer with Python loops and ``torch.where``/``.tolist()`` syncs.
* ``_precompute_refit_cache`` (``soma/fitting/pose_inversion.py``) re-derives the skeleton levels, top-k skinning,
  per-joint subtree vertex sets and sparse weights, joint groups and the fused-kernel level data with Python loops
  and ``.item()`` syncs.

Almost all of that depends only on the skinning weights, the joint hierarchy and the excluded vertices, which never
change between identities. What depends on the identity is small: the RBF systems (their control points are rest-shape
vertices) and their basis weights at the bind joint positions, the bind-centred vertex and child offsets of the Warp
rotation fit, the bind transforms, and the rest-shape vertices each refit joint gathers.

The replacements here capture the structure once, from upstream's own output, and recompute only those pieces:

* :func:`fast_update_bind` replaces ``SkeletonTransfer.update_bind``. It solves all joints' RBF systems in a few
  batched ``lu_factor``/``lu_solve`` calls: joints are grouped by system size (power-of-two buckets) and each system
  ``A_i`` is padded to the bucket size as ``diag(A_i, I)``. Partial pivoting never picks a padding row (it is exactly
  zero in every column of ``A_i``), so the leading block of the batched factorisation is an LU factorisation of
  ``A_i`` and the solution's leading entries are ``A_i^-1 b_i``. Matrices and right-hand sides are assembled with the
  same elementwise operations as upstream (bit-identical); batched LU uses a different GPU kernel than the single
  one, so weights agree to float32 round-off. Each joint still gets a ``RadialBasisFunction`` (built without its
  ``__init__``) whose ``A`` and ``_lu`` are views of the batched ones, so the dense ``fit_joint_positions`` path works.
* :func:`fast_precompute_refit_cache` replaces ``pose_inversion._precompute_refit_cache``. The first call for a rig
  runs upstream and keeps its result as the structure; later calls copy it and replace the identity-dependent
  entries (``bind_local_t``, ``W_bind_inv``, each joint's ``bind_verts_arm``, each level's ``bind_verts_cat``, the
  1-DOF constraint orientations, ``t_pose_orient``), computed with upstream's own expressions: identical bits.

Only ``soma.fitting.pose_inversion`` is patched: ``soma.fitting.pose_inversion_mhr`` imports
``_precompute_refit_cache`` by name and keeps upstream's. ``SkeletonTransfer`` instances whose dtype is not
float32/float64 fall back to upstream.
"""

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from jaxtyping import Bool, Float, Int, Int64
from soma.fitting import pose_inversion
from soma.geometry.interpolate import RadialBasisFunction
from soma.geometry.rig_utils import joint_world_to_local
from soma.geometry.skeleton_transfer import SkeletonTransfer
from soma.geometry.transforms import SE3_inverse
from torch import Tensor

UPSTREAM_UPDATE_BIND: Callable[[SkeletonTransfer, Tensor, Tensor], None] = SkeletonTransfer.update_bind
"""``SkeletonTransfer.update_bind`` as SOMA-X ships it."""
UPSTREAM_REFIT_CACHE: Callable[..., dict[str, Any]] = pose_inversion._precompute_refit_cache
"""``pose_inversion._precompute_refit_cache`` as SOMA-X ships it."""

_WARP_BIND_BUFFERS: tuple[str, ...] = (
    "_warp_stage1_skinned_orig_centered",
    "_warp_stage2_n1_children_orig_centered",
    "_warp_stage2_n2_children_orig_centered",
)
"""The Warp rotation-fit buffers that depend on the bind (all the other ``_warp_*`` state is structural)."""


@dataclass(slots=True)
class _RbfBucket:
    """Distinct RBF systems that share one padded size, and the joints that use them.

    Joints whose regressor masks are equal (the body has three such 1,037-point joints) have the same control points,
    hence the same system: it is factorised once.
    """

    vertex_ids: Int64[Tensor, "s n"]
    """Each system's control-point vertex ids, padded with 0."""
    is_point: Bool[Tensor, "s m"]
    """Rows ``< n_i`` (kernel rows); ``m = n + 4`` (affine rows ``n_i .. n_i + 3``, then padding)."""
    is_pad: Bool[Tensor, "s m"]
    """Rows ``>= n_i + 4``: the padding identity block."""
    poly_rows: Int64[Tensor, "s 4"]
    """Row ``n_i + q`` of affine term ``q`` (1, x, y, z)."""
    joints: list[int]
    """Joint ids, in joint order."""
    joint_system: Int64[Tensor, "b"]
    """Each joint's system."""
    joint_system_list: list[int]
    """``joint_system`` on the host."""
    sizes: list[int]
    """Each joint's control-point count ``n_i``, on the host."""


@dataclass(slots=True)
class _TransferStructure:
    """What ``SkeletonTransfer.update_bind`` rebuilds that does not depend on the bind."""

    skinning_weights: Tensor
    """The weights this structure was captured from (the cache key, by identity)."""
    regressor_mask: Bool[Tensor, "v j"]
    """Upstream's per-joint regressor vertex mask."""
    buckets: list[_RbfBucket]
    """The joints' RBF systems grouped by padded size."""
    value_order: Int64[Tensor, "nnz"]
    """Gathers the bucket-ordered concatenated weights into joint-major CSR order."""
    crow_indices: Int64[Tensor, "jp1"]
    """CSR row pointers of the sparse regressor matrix."""
    col_indices: Int64[Tensor, "nnz"]
    """CSR column (vertex) indices of the sparse regressor matrix."""
    warp: dict[str, Any]
    """Every structural ``_warp_*`` attribute, as upstream built it."""


def _capture_transfer(transfer: SkeletonTransfer) -> _TransferStructure:
    """Structure of ``transfer`` from its current (upstream-built) state."""
    device: torch.device = transfer.device
    mask: Bool[Tensor, "v j"] = transfer.regressor_mask
    joints: list[int] = list(range(transfer._first_joint, transfer.num_joints))
    vertex_lists: dict[int, Int64[Tensor, "n"]] = {j: torch.where(mask[:, j])[0] for j in joints}
    systems: dict[bytes, list[int]] = {}  # distinct control-point sets -> their joints
    for j in joints:
        systems.setdefault(vertex_lists[j].cpu().numpy().tobytes(), []).append(j)
    by_bucket: dict[int, list[list[int]]] = {}
    for users in systems.values():
        by_bucket.setdefault(max(32, 1 << (len(vertex_lists[users[0]]) + 4 - 1).bit_length()), []).append(users)
    buckets: list[_RbfBucket] = []
    for members in by_bucket.values():
        n: int = max(len(vertex_lists[users[0]]) for users in members)
        rows: Int64[Tensor, "m"] = torch.arange(n + 4, device=device)
        counts: Int64[Tensor, "s"] = torch.tensor([len(vertex_lists[users[0]]) for users in members], device=device)
        vertex_ids: Int64[Tensor, "s n"] = torch.zeros(len(members), n, dtype=torch.long, device=device)
        for s, users in enumerate(members):
            vertex_ids[s, : len(vertex_lists[users[0]])] = vertex_lists[users[0]]
        user_system: list[tuple[int, int]] = sorted((j, s) for s, users in enumerate(members) for j in users)
        buckets.append(
            _RbfBucket(
                vertex_ids=vertex_ids,
                is_point=rows[None] < counts[:, None],
                is_pad=rows[None] >= counts[:, None] + 4,
                poly_rows=counts[:, None] + torch.arange(4, device=device)[None],
                joints=[j for j, _ in user_system],
                joint_system=torch.tensor([s for _, s in user_system], dtype=torch.long, device=device),
                joint_system_list=[s for _, s in user_system],
                sizes=[len(vertex_lists[j]) for j, _ in user_system],
            )
        )
    starts: dict[int, int] = {}
    offset: int = 0
    for bucket in buckets:
        for j, size in zip(bucket.joints, bucket.sizes, strict=True):
            starts[j] = offset
            offset += size
    value_order: Int64[Tensor, "nnz"] = torch.cat([torch.arange(starts[j], starts[j] + len(vertex_lists[j]), device=device) for j in joints])
    sparse: Tensor | None = transfer.sparse_rbf_matrix
    warp_names: list[str] = [name for name in vars(transfer) if name.startswith("_warp_") and name not in _WARP_BIND_BUFFERS]
    warp_names += [name for name in transfer._buffers if name.startswith("_warp_") and name not in _WARP_BIND_BUFFERS]
    return _TransferStructure(
        skinning_weights=cast(Tensor, transfer.skinning_weights),
        regressor_mask=mask,
        buckets=buckets,
        value_order=value_order,
        crow_indices=sparse.crow_indices() if sparse is not None else torch.empty(0, dtype=torch.long, device=device),
        col_indices=sparse.col_indices() if sparse is not None else torch.cat(list(vertex_lists.values())),
        warp={name: getattr(transfer, name) for name in warp_names},
    )


@dataclass(slots=True)
class _BucketSolve:
    """One bucket's systems, their LU factors, and each joint's basis weights."""

    points: Float[Tensor, "s n 3"]
    """Control points per system, zero-padded."""
    systems: Float[Tensor, "s m m"]
    """``diag(A_i, I)`` per system."""
    lu: Float[Tensor, "s m m"]
    """LU factors of ``systems``."""
    pivots: Int[Tensor, "s m"]
    """Pivots of ``systems``."""
    info: Int[Tensor, "s"]
    """``lu_factor_ex``'s per-system status (0: factorised)."""
    weights: Float[Tensor, "nnz"]
    """Every joint's basis weights, joint after joint."""


def _solve_bucket(transfer: SkeletonTransfer, bucket: _RbfBucket, bind_shape: Float[Tensor, "v 3"]) -> _BucketSolve:
    """Matches ``RadialBasisFunction._precompute_system_matrix`` and ``get_basis_weights`` (polynomial included)."""
    dtype: torch.dtype = bind_shape.dtype
    device: torch.device = bind_shape.device
    zero: Tensor = torch.zeros((), dtype=dtype, device=device)
    kernel: Callable[..., Tensor] = RadialBasisFunction.KERNELS[transfer.rbf_kernel]
    count: int = bucket.vertex_ids.shape[0]
    m: int = bucket.is_point.shape[1]
    n: int = m - 4
    points: Float[Tensor, "s n 3"] = bind_shape[bucket.vertex_ids]
    diff: Float[Tensor, "s n n 3"] = points[:, :, None, :] - points[:, None, :, :]
    gram: Float[Tensor, "s n n"] = kernel(torch.sqrt((diff * diff).sum(dim=-1))).to(dtype)
    gram = gram + torch.eye(n, dtype=dtype, device=device) * 1e-8  # upstream's eps for float32/float64
    pair: Bool[Tensor, "s n n"] = bucket.is_point[:, :n, None] & bucket.is_point[:, None, :n]
    systems: Float[Tensor, "s m m"] = torch.zeros(count, m, m, dtype=dtype, device=device)
    systems[:, :n, :n] = torch.where(pair, gram, zero)
    affine: Float[Tensor, "s n 4"] = torch.where(
        bucket.is_point[:, :n, None], torch.cat([torch.ones(count, n, 1, dtype=dtype, device=device), points], dim=2), zero
    )
    batch: Int64[Tensor, "s 1 1"] = torch.arange(count, device=device)[:, None, None]
    rows: Int64[Tensor, "1 n 1"] = torch.arange(n, device=device)[None, :, None]
    poly: Int64[Tensor, "s 1 4"] = bucket.poly_rows[:, None, :]
    systems[batch, rows, poly] = affine
    systems[batch, poly, rows] = affine
    diagonal: Int64[Tensor, "m"] = torch.arange(m, device=device)
    systems[:, diagonal, diagonal] += bucket.is_pad.to(dtype)
    factored: tuple[Tensor, Tensor, Tensor] = torch.linalg.lu_factor_ex(systems)

    used: Int64[Tensor, "b"] = bucket.joint_system
    joint_points: Float[Tensor, "b n 3"] = points[used]
    is_point: Bool[Tensor, "b m"] = bucket.is_point[used]
    queries: Float[Tensor, "b 3"] = transfer.bind_world_transforms[bucket.joints, :3, 3].to(dtype)
    distances: Float[Tensor, "b n"] = kernel(torch.norm(joint_points - queries[:, None, :], dim=2)).to(dtype)
    rhs: Float[Tensor, "b m"] = torch.zeros(len(bucket.joints), m, dtype=dtype, device=device)
    rhs[:, :n] = torch.where(is_point[:, :n], distances, zero)
    rhs.scatter_(1, bucket.poly_rows[used], torch.cat([torch.ones(len(bucket.joints), 1, dtype=dtype, device=device), queries], dim=1))
    solution: Float[Tensor, "b m"] = torch.linalg.lu_solve(factored[0][used], factored[1][used], rhs[:, :, None])[:, :, 0]
    return _BucketSolve(points=points, systems=systems, lu=factored[0], pivots=factored[1], info=factored[2], weights=solution[is_point])


def _regressor(points: Float[Tensor, "n 3"], system: Float[Tensor, "m m"], factors: tuple[Tensor, Tensor], kernel: str) -> RadialBasisFunction:
    """A ``RadialBasisFunction`` over ``points`` whose system and LU factors are views of a batched solve."""
    rbf: RadialBasisFunction = RadialBasisFunction.__new__(RadialBasisFunction)
    rbf.dtype, rbf.device, rbf.source_control_points = points.dtype, points.device, points
    rbf.n_control, rbf.dim = points.shape
    rbf.kernel_name, rbf.kernel_params, rbf.include_polynomial = kernel, {}, True
    rbf._rbf_func = RadialBasisFunction.KERNELS[kernel]
    rbf.A, rbf._lu = system, factors
    return rbf


_TRANSFER_STRUCTURES: "weakref.WeakKeyDictionary[SkeletonTransfer, _TransferStructure]" = weakref.WeakKeyDictionary()
"""The captured structure of every ``SkeletonTransfer`` the fast path has updated."""


def fast_update_bind(self: SkeletonTransfer, bind_world_transforms: Tensor, bind_shape: Tensor) -> None:
    """Drop-in for ``SkeletonTransfer.update_bind``: batched RBF solves on structure captured from upstream's state.

    Args:
        bind_world_transforms: (J, 4, 4) new bind transforms in world space.
        bind_shape: (V, 3) new rest-shape vertices.
    """
    if self.dtype not in (torch.float32, torch.float64):
        UPSTREAM_UPDATE_BIND(self, bind_world_transforms, bind_shape)
        return
    structure: _TransferStructure | None = _TRANSFER_STRUCTURES.get(self)
    if structure is None or structure.skinning_weights is not self.skinning_weights:
        structure = _capture_transfer(self)  # the mask and Warp indices depend on the weights only: valid before the update
        _TRANSFER_STRUCTURES[self] = structure

    self.bind_world_transforms = bind_world_transforms.detach()
    self.bind_local_transforms = joint_world_to_local(bind_world_transforms, self.joint_parent_ids)
    self.bind_shape = bind_shape.detach()
    self.regressor_mask = structure.regressor_mask

    regressors: list[RadialBasisFunction | None] = [None] * self.num_joints
    solves: list[_BucketSolve] = [_solve_bucket(self, bucket, self.bind_shape) for bucket in structure.buckets]
    if bool(torch.cat([solve.info for solve in solves]).any()):  # what upstream's lu_factor raises, with one sync
        raise torch.linalg.LinAlgError("SkeletonTransfer: a joint regressor's RBF system is singular")
    for bucket, solve in zip(structure.buckets, solves, strict=True):
        for j, s, n in zip(bucket.joints, bucket.joint_system_list, bucket.sizes, strict=True):
            factors: tuple[Tensor, Tensor] = (solve.lu[s, : n + 4, : n + 4], solve.pivots[s, : n + 4])
            regressors[j] = _regressor(solve.points[s, :n], solve.systems[s, : n + 4, : n + 4], factors, self.rbf_kernel)
    self.joint_pos_regressors = regressors
    if self.use_sparse_rbf_matrix:
        self.sparse_rbf_matrix = torch.sparse_csr_tensor(
            crow_indices=structure.crow_indices,
            col_indices=structure.col_indices,
            values=torch.cat([solve.weights for solve in solves])[structure.value_order].to(self.dtype),
            size=(len(self.bind_world_transforms), len(self.bind_shape)),
            device=self.device,
            dtype=self.dtype,
        )
    else:
        self.sparse_rbf_matrix = None

    if self.use_warp_for_rotations:
        for name, value in structure.warp.items():
            setattr(self, name, value)
        joints: Float[Tensor, "j 3"] = self.bind_world_transforms[:, :3, 3]
        empty: Tensor = torch.tensor([], dtype=self.dtype, device=self.device)
        if len(self._warp_stage1_joint_indices):
            self._warp_stage1_skinned_orig_centered = (
                self.bind_shape[self._warp_stage1_skinned_vids_flat] - joints[self._warp_stage1_joint_indices][self._warp_stage1_repeat_indices]
            )
        else:
            self._warp_stage1_skinned_orig_centered = empty
        if len(self._warp_stage2_n1_joint_indices):
            self._warp_stage2_n1_children_orig_centered = joints[self._warp_stage2_n1_child_indices] - joints[self._warp_stage2_n1_joint_indices]
        else:
            self._warp_stage2_n1_children_orig_centered = empty
        if len(self._warp_stage2_n2_joint_indices):
            self._warp_stage2_n2_children_orig_centered = (
                joints[self._warp_stage2_n2_child_flat] - joints[self._warp_stage2_n2_joint_indices][self._warp_stage2_n2_repeat_indices]
            )
        else:
            self._warp_stage2_n2_children_orig_centered = empty


@dataclass(slots=True)
class _RefitStructure:
    """An upstream refit cache with its identity-dependent entries still in, kept as the template of a rig."""

    key: tuple[Any, ...]
    """Joint names, parents, root, device, dtype, fused-kernel availability."""
    skinning_weights: Tensor
    """The weights it was built from (compared by identity, then by value)."""
    cache: dict[str, Any]
    """Upstream's cache, its mutable containers copied."""
    constrained_parents: Int64[Tensor, "c"] | None
    """Parent of each 1-DOF constrained joint."""


_REFIT_STRUCTURES: list[_RefitStructure] = []
"""One template per rig seen (a handful per process: body, hands)."""
_LEVEL_KEYS: tuple[str, ...] = ("body_level_data", "finger_level_data", "all_level_data")


def _copy_containers(cache: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``cache`` that shares its tensors but none of the dicts, lists and sets ``prepare_identity`` may change."""
    copied: dict[str, Any] = dict(cache)
    copied["joint_cache"] = {j: dict(entry) for j, entry in cache["joint_cache"].items()}
    copied["body_groups"] = [list(group) for group in cache["body_groups"]]
    copied["finger_groups"] = [list(group) for group in cache["finger_groups"]]
    copied["levels"] = list(cache["levels"])
    copied["constrained_set"] = set(cache["constrained_set"])
    if cache["constrained_data"] is not None:
        copied["constrained_data"] = dict(cache["constrained_data"])
    for name in _LEVEL_KEYS:
        if cache[name] is not None:
            copied[name] = [None if level is None else dict(level) for level in cache[name]]
    return copied


def fast_precompute_refit_cache(
    joint_names: Sequence[str],
    joint_parent_ids: Any,
    bind_world: Float[Tensor, "j 4 4"],
    bind_shape: Float[Tensor, "v 3"],
    skinning_weights: Float[Tensor, "v j"],
    t_pose_world: Float[Tensor, "j 4 4"],
    root_idx: int = 1,
) -> dict[str, Any]:
    """Drop-in for ``pose_inversion._precompute_refit_cache``: upstream's cache for a rig's first identity, then a copy of
    it with only the identity-dependent entries recomputed (same expressions, so the same bits)."""
    parents: list[int] = joint_parent_ids.tolist() if hasattr(joint_parent_ids, "tolist") else list(joint_parent_ids)
    key: tuple[Any, ...] = (
        tuple(joint_names),
        tuple(parents),
        root_idx,
        bind_shape.device,
        bind_shape.dtype,
        pose_inversion._fused_refit_level is not None,
    )
    template: _RefitStructure | None = next(
        (
            entry
            for entry in _REFIT_STRUCTURES
            if entry.key == key
            and (
                entry.skinning_weights is skinning_weights
                or (entry.skinning_weights.shape == skinning_weights.shape and torch.equal(entry.skinning_weights, skinning_weights))
            )
        ),
        None,
    )
    if template is None:
        cache: dict[str, Any] = UPSTREAM_REFIT_CACHE(
            joint_names, joint_parent_ids, bind_world, bind_shape, skinning_weights, t_pose_world, root_idx=root_idx
        )
        constrained: dict[str, Any] | None = cache["constrained_data"]
        _REFIT_STRUCTURES.append(
            _RefitStructure(
                key=key,
                skinning_weights=skinning_weights,
                cache=_copy_containers(cache),
                constrained_parents=None
                if constrained is None
                else torch.tensor([parents[j] for j in constrained["indices"].tolist()], dtype=torch.long, device=bind_shape.device),
            )
        )
        return cache

    fresh: dict[str, Any] = _copy_containers(template.cache)
    fresh["joint_names"], fresh["parent_ids"], fresh["skinning_weights"] = joint_names, joint_parent_ids, skinning_weights
    bind_local: Float[Tensor, "j 4 4"] = cast(Tensor, joint_world_to_local(bind_world, joint_parent_ids))
    fresh["bind_local_t"] = bind_local[:, :3, 3]
    fresh["W_bind_inv"] = SE3_inverse(bind_world)
    for entry in fresh["joint_cache"].values():
        entry["bind_verts_arm"] = bind_shape[entry["arm_vids"]]
    for name in _LEVEL_KEYS:
        for level in fresh[name] or []:
            if level is not None:
                level["bind_verts_cat"] = bind_shape[level["arm_vids_cat"]]
    t_orient: Float[Tensor, "j 3 3"] = t_pose_world[:, :3, :3]
    fresh["t_pose_orient"] = t_orient
    if fresh["constrained_data"] is not None:
        assert template.constrained_parents is not None
        fresh["constrained_data"]["orient_j"] = t_orient[fresh["constrained_data"]["indices"]]
        fresh["constrained_data"]["orient_p"] = t_orient[template.constrained_parents]
    return fresh


def install_fast_identity_setup() -> None:
    """Patch SOMA-X so ``PoseInversion.prepare_identity`` uses :func:`fast_update_bind` and :func:`fast_precompute_refit_cache`.

    Idempotent; affects every ``PoseInversion`` in the process. Call it once before fitting.
    """
    SkeletonTransfer.update_bind = fast_update_bind  # type: ignore[method-assign]
    pose_inversion._precompute_refit_cache = fast_precompute_refit_cache


def uninstall_fast_identity_setup() -> None:
    """Restore upstream's ``update_bind`` and ``_precompute_refit_cache``. Idempotent."""
    SkeletonTransfer.update_bind = UPSTREAM_UPDATE_BIND  # type: ignore[method-assign]
    pose_inversion._precompute_refit_cache = UPSTREAM_REFIT_CACHE
