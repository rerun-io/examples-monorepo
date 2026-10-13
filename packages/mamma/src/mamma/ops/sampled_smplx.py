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

        rel_transforms: Float32[torch.Tensor, "t 55 4 4"] = rigid_transform_chain(
            rot_mats, rest_joints, self.parents, self.parent_index
        )
        skin_t: Float32[torch.Tensor, "t n 4 4"] = torch.einsum("nj,bjxy->bnxy", self.lbs_weights_s, rel_transforms)
        v_h: Float32[torch.Tensor, "t n 4"] = torch.cat([v_posed, torch.ones_like(v_posed[..., :1])], dim=-1)
        verts: Float32[torch.Tensor, "t n 3"] = torch.einsum("bnxy,bny->bnx", skin_t, v_h)[..., :3]
        return verts + transl.unsqueeze(1)
