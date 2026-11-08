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


def ancestor_jumps(parents: list[int]) -> list[list[int]]:
    """Pointer-jumping ancestor tables for :func:`rigid_transform_jump`.

    Joint ``len(parents)`` is a virtual identity root (its own ancestor).
    Step k's table maps each joint to its 2^k-th ancestor; ceil(log2(depth))
    steps reach the virtual root from every joint.
    """
    root: int = len(parents)
    anc: list[int] = [root] + [parents[j] for j in range(1, len(parents))] + [root]
    tables: list[list[int]] = []
    while any(a != root for a in anc):
        tables.append(anc)
        anc = [anc[a] for a in anc]
    return tables


def rigid_transform_jump(
    rot_mats: Float32[torch.Tensor, "t j 3 3"],
    rest_joints: Float32[torch.Tensor, "t j 3"],
    parent_index: torch.Tensor,
    jumps: list[torch.Tensor],
) -> tuple[Float32[torch.Tensor, "t j 3 3"], Float32[torch.Tensor, "t j 3"]]:
    """Relative rigid transforms along the kinematic chain (smplx's
    ``batch_rigid_transform``: rotation + translation; the constant [0 0 0 1]
    row is dropped), composed by pointer jumping. CUDA-graph capturable:
    ``parent_index``/``jumps`` are device index tensors built once.

    Each step composes every joint's partial chain with its 2^k-th ancestor's
    (one gather + one elementwise 3x3 product over all joints), so the
    kinematic tree's 11-level chain takes 4 steps. The original 54 per-joint
    4x4 cuBLAS matmuls (x3 with backward) were ~60% of a compiled fit
    iteration; these few elementwise steps fuse under inductor.
    """
    t: int = rot_mats.shape[0]
    rel_joints: Float32[torch.Tensor, "t j 3"] = torch.cat(
        [rest_joints[:, :1], rest_joints[:, 1:] - rest_joints[:, parent_index]], dim=1
    )
    eye: Float32[torch.Tensor, "t 1 3 3"] = torch.eye(3, device=rot_mats.device, dtype=rot_mats.dtype).expand(t, 1, 3, 3)
    rot: Float32[torch.Tensor, "t j1 3 3"] = torch.cat([rot_mats, eye], dim=1)
    trans: Float32[torch.Tensor, "t j1 3"] = torch.cat([rel_joints, rel_joints.new_zeros(t, 1, 3)], dim=1)
    for anc in jumps:
        rot_a: Float32[torch.Tensor, "t j1 3 3"] = rot[:, anc]
        trans = (rot_a * trans.unsqueeze(-2)).sum(-1) + trans[:, anc]
        rot = (rot_a.unsqueeze(-1) * rot.unsqueeze(-3)).sum(-2)
    rot_world: Float32[torch.Tensor, "t j 3 3"] = rot[:, :-1]
    rel_trans: Float32[torch.Tensor, "t j 3"] = trans[:, :-1] - (rot_world * rest_joints.unsqueeze(-2)).sum(-1)
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
        self.jumps: list[torch.Tensor] = [torch.tensor(a, device=device, dtype=torch.long) for a in ancestor_jumps(self.parents)]

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
        rel_rot, rel_trans = rigid_transform_jump(rot_mats, rest_joints, self.parent_index, self.jumps)
        # Skinning: blend the 3x4 [R|t] per vertex as ONE (n,55)@(55,t*12) GEMM.
        rel_34: Float32[torch.Tensor, "t 55 12"] = torch.cat([rel_rot.reshape(t, 55, 9), rel_trans], dim=-1)
        skin: Float32[torch.Tensor, "n t 12"] = (self.lbs_weights_s @ rel_34.permute(1, 0, 2).reshape(55, t * 12)).view(-1, t, 12)
        skin = skin.permute(1, 0, 2)
        skin_rot: Float32[torch.Tensor, "t n 3 3"] = skin[..., :9].reshape(t, -1, 3, 3)
        skin_trans: Float32[torch.Tensor, "t n 3"] = skin[..., 9:]
        verts: Float32[torch.Tensor, "t n 3"] = (skin_rot * v_posed.unsqueeze(-2)).sum(-1) + skin_trans
        return verts + transl.unsqueeze(1)
