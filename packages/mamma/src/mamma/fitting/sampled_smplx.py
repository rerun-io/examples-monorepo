"""SMPL-X forward restricted to the 512 sampled landmark vertices.

The fit losses only ever consume ``verts_512 @ vertices``, yet the full model
deforms all 10,475 vertices each of the 16 Adam iterations (~67 ms GPU per
optimize call — the profiled bottleneck). Linear blend skinning is linear in
the vertex dimension, so the sampling matrix folds into the template, shape
dirs, pose dirs, and skinning weights once at construction; each iteration
then deforms exactly 512 vertices. Rest-pose joints (which drive the rigid
transforms) are computed exactly via the pre-multiplied joint regressor.

The full model remains the source of truth for emission (one forward per tick).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from smplx.lbs import batch_rodrigues as _batch_rodrigues

# smplx's own `Tensor` alias confuses pyrefly; the function takes/returns torch tensors.
batch_rodrigues = cast(Any, _batch_rodrigues)


def rigid_transform_chain(
    rot_mats: Float32[torch.Tensor, "t j 3 3"],
    rest_joints: Float32[torch.Tensor, "t j 3"],
    parents: list[int],
    parent_index: torch.Tensor,
) -> Float32[torch.Tensor, "t j 4 4"]:
    """Relative rigid transforms along the kinematic chain (smplx's
    ``batch_rigid_transform``), CUDA-graph capturable.

    smplx indexes ``transform_chain[parents[i]]`` with a 0-dim GPU tensor —
    an implicit ``.item()`` host sync 54x per forward that serializes the
    fit loop and invalidates capture; ``parents`` as plain ints avoids both.
    ``parent_index`` is the same chain as a device tensor (list-indexing a
    CUDA tensor builds a CPU index tensor = H2D copy, illegal under capture).
    """
    rel_joints: Float32[torch.Tensor, "t j 3"] = rest_joints.clone()
    rel_joints[:, 1:] -= rest_joints[:, parent_index]
    transforms_mat: Float32[torch.Tensor, "t j 4 4"] = torch.cat(
        [F.pad(rot_mats, (0, 0, 0, 1)), F.pad(rel_joints.unsqueeze(-1), (0, 0, 0, 1), value=1.0)], dim=-1
    )

    chain: list[torch.Tensor] = [transforms_mat[:, 0]]
    for i in range(1, len(parents)):
        chain.append(chain[parents[i]] @ transforms_mat[:, i])
    transforms: Float32[torch.Tensor, "t j 4 4"] = torch.stack(chain, dim=1)

    posed_joints: Float32[torch.Tensor, "t j 3"] = transforms[..., :3, 3]
    joints_homogen: Float32[torch.Tensor, "t j 4 1"] = F.pad(rest_joints.unsqueeze(-1), (0, 0, 0, 1))
    rel_transforms: Float32[torch.Tensor, "t j 4 4"] = transforms - F.pad(
        transforms @ joints_homogen, (3, 0, 0, 0, 0, 0, 0, 0)
    )
    del posed_joints
    return rel_transforms


def kinematic_levels(parents: list[int]) -> list[tuple[list[int], list[int]]]:
    """Group joints by tree depth: ``[(joints, parent_pos_in_prev_level), ...]``.

    Every joint's parent sits exactly one level up, so a level composes with
    one gather + one batched 3x3 product instead of one matmul per joint.
    """
    depth: list[int] = [0] * len(parents)
    for i in range(1, len(parents)):
        depth[i] = depth[parents[i]] + 1
    levels: list[list[int]] = [[] for _ in range(max(depth) + 1)]
    for i, d in enumerate(depth):
        levels[d].append(i)
    out: list[tuple[list[int], list[int]]] = [(levels[0], [])]
    for d in range(1, len(levels)):
        prev_pos: dict[int, int] = {j: k for k, j in enumerate(levels[d - 1])}
        out.append((levels[d], [prev_pos[parents[j]] for j in levels[d]]))
    return out


def rigid_transform_levels(
    rot_mats: Float32[torch.Tensor, "t j 3 3"],
    rest_joints: Float32[torch.Tensor, "t j 3"],
    parent_index: torch.Tensor,
    levels: list[tuple[torch.Tensor, torch.Tensor]],
    level_inverse: torch.Tensor,
) -> tuple[Float32[torch.Tensor, "t j 3 3"], Float32[torch.Tensor, "t j 3"]]:
    """Same result as :func:`rigid_transform_chain` (rotation + translation of
    the relative transforms; the constant [0 0 0 1] row is dropped), composed
    one tree level at a time with elementwise 3x3 products.

    54 per-joint 4x4 cuBLAS matmuls (x3 with backward) were ~60% of a compiled
    fit iteration; per-level elementwise products are ~depth kernels that
    inductor fuses. ``levels`` holds device index tensors (capture-safe).
    """
    rel_joints: Float32[torch.Tensor, "t j 3"] = torch.cat(
        [rest_joints[:, :1], rest_joints[:, 1:] - rest_joints[:, parent_index]], dim=1
    )
    root_idx, _ = levels[0]
    rot_lv: list[torch.Tensor] = [rot_mats[:, root_idx]]
    trans_lv: list[torch.Tensor] = [rel_joints[:, root_idx]]
    for joint_idx, parent_pos in levels[1:]:
        rot_p: Float32[torch.Tensor, "t k 3 3"] = rot_lv[-1][:, parent_pos]
        trans_p: Float32[torch.Tensor, "t k 3"] = trans_lv[-1][:, parent_pos]
        rot_l: Float32[torch.Tensor, "t k 3 3"] = rot_mats[:, joint_idx]
        trans_l: Float32[torch.Tensor, "t k 3"] = rel_joints[:, joint_idx]
        rot_lv.append((rot_p.unsqueeze(-1) * rot_l.unsqueeze(-3)).sum(-2))
        trans_lv.append((rot_p * trans_l.unsqueeze(-2)).sum(-1) + trans_p)
    rot_world: Float32[torch.Tensor, "t j 3 3"] = torch.cat(rot_lv, dim=1)[:, level_inverse]
    trans_world: Float32[torch.Tensor, "t j 3"] = torch.cat(trans_lv, dim=1)[:, level_inverse]
    rel_trans: Float32[torch.Tensor, "t j 3"] = trans_world - (rot_world * rest_joints.unsqueeze(-2)).sum(-1)
    return rot_world, rel_trans


class SampledSmplx:
    """Exact SMPL-X sampled-vertex forward (matches ``sampling @ model.vertices``)."""

    def __init__(self, model, sampling: Float32[torch.Tensor, "n v"]) -> None:
        """Args:
        model: A constructed ``smplx.SMPLX`` (neutral) on the target device.
        sampling: ``[n_landmarks, n_vertices]`` vertex subsampling matrix.
        """
        device: torch.device = model.v_template.device
        sampling = sampling.to(device=device, dtype=torch.float32)
        self.parents: list[int] = [int(p) for p in model.parents.tolist()]
        self.parent_index: torch.Tensor = torch.tensor(self.parents[1:], device=device, dtype=torch.long)
        self.pose_mean: Float32[torch.Tensor, "165"] = model.pose_mean.float()
        v_template: Float32[torch.Tensor, "v 3"] = model.v_template.float()
        self.v_template_s: Float32[torch.Tensor, "n 3"] = sampling @ v_template
        self.shapedirs_s: Float32[torch.Tensor, "n 3 nb"] = torch.einsum("nv,vck->nck", sampling, model.shapedirs.float())
        posedirs: Float32[torch.Tensor, "p v3"] = model.posedirs.float()
        posedirs_v: Float32[torch.Tensor, "p v 3"] = posedirs.view(posedirs.shape[0], -1, 3)
        self.posedirs_s: Float32[torch.Tensor, "p n3"] = torch.einsum("nv,pvc->pnc", sampling, posedirs_v).reshape(
            posedirs.shape[0], -1
        )
        self.lbs_weights_s: Float32[torch.Tensor, "n 55"] = sampling @ model.lbs_weights.float()
        j_regressor: Float32[torch.Tensor, "55 v"] = model.J_regressor.float()
        self.joints_template: Float32[torch.Tensor, "55 3"] = j_regressor @ v_template
        self.joints_shapedirs: Float32[torch.Tensor, "55 3 nb"] = torch.einsum("jv,vck->jck", j_regressor, model.shapedirs.float())
        levels: list[tuple[list[int], list[int]]] = kinematic_levels(self.parents)
        self.levels: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.tensor(j, device=device, dtype=torch.long), torch.tensor(p, device=device, dtype=torch.long)) for j, p in levels
        ]
        order: list[int] = [j for joints, _ in levels for j in joints]
        inverse: list[int] = [0] * len(order)
        for pos, j in enumerate(order):
            inverse[j] = pos
        self.level_inverse: torch.Tensor = torch.tensor(inverse, device=device, dtype=torch.long)

    def forward(
        self,
        global_orient: Float32[torch.Tensor, "t 3"],
        body_pose: Float32[torch.Tensor, "t 63"],
        left_hand_pose: Float32[torch.Tensor, "t 45"],
        right_hand_pose: Float32[torch.Tensor, "t 45"],
        jaw_pose: Float32[torch.Tensor, "t 3"],
        betas: Float32[torch.Tensor, "1 nb"],
        transl: Float32[torch.Tensor, "t 3"],
    ) -> Float32[torch.Tensor, "t n 3"]:
        """World-space sampled vertices (same args as ``smplx_forward_per_parts``)."""
        t: int = body_pose.shape[0]
        device: torch.device = body_pose.device
        eyes: Float32[torch.Tensor, "t 6"] = torch.zeros(t, 6, device=device, dtype=body_pose.dtype)
        full_pose: Float32[torch.Tensor, "t 165"] = (
            torch.cat([global_orient, body_pose, jaw_pose, eyes, left_hand_pose, right_hand_pose], dim=-1) + self.pose_mean
        )
        rot_mats: Float32[torch.Tensor, "t 55 3 3"] = batch_rodrigues(full_pose.reshape(-1, 3)).view(t, 55, 3, 3)

        betas_t: Float32[torch.Tensor, "t nb"] = betas.expand(t, -1)
        rest_joints: Float32[torch.Tensor, "t 55 3"] = self.joints_template + torch.einsum(
            "jck,bk->bjc", self.joints_shapedirs, betas_t
        )
        identity: Float32[torch.Tensor, "3 3"] = torch.eye(3, device=device, dtype=rot_mats.dtype)
        pose_feature: Float32[torch.Tensor, "t p"] = (rot_mats[:, 1:] - identity).reshape(t, -1)
        pose_offsets: Float32[torch.Tensor, "t n 3"] = (pose_feature @ self.posedirs_s).view(t, -1, 3)
        v_posed: Float32[torch.Tensor, "t n 3"] = (
            self.v_template_s + torch.einsum("nck,bk->bnc", self.shapedirs_s, betas_t) + pose_offsets
        )

        rel_rot: Float32[torch.Tensor, "t 55 3 3"]
        rel_trans: Float32[torch.Tensor, "t 55 3"]
        rel_rot, rel_trans = rigid_transform_levels(rot_mats, rest_joints, self.parent_index, self.levels, self.level_inverse)
        # Skinning: blend the 3x4 [R|t] per vertex as ONE (n,55)@(55,t*12) GEMM.
        rel_34: Float32[torch.Tensor, "t 55 12"] = torch.cat([rel_rot.reshape(t, 55, 9), rel_trans], dim=-1)
        skin: Float32[torch.Tensor, "n t 12"] = (self.lbs_weights_s @ rel_34.permute(1, 0, 2).reshape(55, t * 12)).view(-1, t, 12)
        skin = skin.permute(1, 0, 2)
        skin_rot: Float32[torch.Tensor, "t n 3 3"] = skin[..., :9].reshape(t, -1, 3, 3)
        skin_trans: Float32[torch.Tensor, "t n 3"] = skin[..., 9:]
        verts: Float32[torch.Tensor, "t n 3"] = (skin_rot * v_posed.unsqueeze(-2)).sum(-1) + skin_trans
        return verts + transl.unsqueeze(1)
