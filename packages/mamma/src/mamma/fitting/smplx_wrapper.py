"""SMPL-X model construction + per-part forward, mirroring the golden run setup
(neutral body for all subjects, 16 betas, MANO-mean hands, no PCA, no v_template).

The golden DAG builds its NEUTRAL model with ``flat_hand_mean=False``
(``optimization/utils_smplx.py`` default ``flat_hand=False``; only the unused
male/female models are ``True``). A zero hand pose therefore rests at the
natural MANO mean, not splayed-flat — we never fit fingers, so this rest pose
IS the hand output, and matching the golden's convention removes a systematic
hand-region vertex error.
"""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float32

NUM_BETAS: int = 16
"""Shape coefficients optimized per body (golden ``n_betas: 16``)."""


def _patch_smplx_rigid_transform() -> None:
    """Replace ``smplx.lbs.batch_rigid_transform`` with a sync-free version.

    Upstream indexes ``transform_chain[parents[i]]`` with a 0-dim GPU tensor —
    an implicit ``.item()`` host sync 54x per forward (~10 ms of stalls per
    full-model call, paid on every emitted tick).
    """
    from typing import Any

    import torch.nn.functional as F

    _lbs: Any = __import__("smplx.lbs", fromlist=["lbs"])  # loose-typed: we monkeypatch it

    if getattr(_lbs, "_mamma_sync_free", False):
        return

    def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
        parent_ints: list[int] = [int(p) for p in parents.tolist()]
        joints = torch.unsqueeze(joints, dim=-1)
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parent_ints[1:]]
        transforms_mat = _lbs.transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(
            -1, joints.shape[1], 4, 4
        )
        chain = [transforms_mat[:, 0]]
        for i in range(1, len(parent_ints)):
            chain.append(torch.matmul(chain[parent_ints[i]], transforms_mat[:, i]))
        transforms = torch.stack(chain, dim=1)
        posed_joints = transforms[:, :, :3, 3]
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
        return posed_joints, rel_transforms

    _lbs.batch_rigid_transform = batch_rigid_transform
    _lbs._mamma_sync_free = True


def build_smplx_neutral(model_folder: Path, device: str = "cuda"):
    """Neutral SMPL-X model matching ``utils_smplx.get_smplx_models``."""
    import smplx

    _patch_smplx_rigid_transform()
    return smplx.create(
        str(model_folder),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        num_betas=NUM_BETAS,
        flat_hand_mean=False,  # MANO-mean rest hands, matching the golden neutral model
        use_pca=False,
    ).to(device)


def smplx_forward_per_parts(
    model,
    global_orient: Float32[torch.Tensor, "t 3"],
    body_pose: Float32[torch.Tensor, "t 63"],
    left_hand_pose: Float32[torch.Tensor, "t 45"],
    right_hand_pose: Float32[torch.Tensor, "t 45"],
    jaw_pose: Float32[torch.Tensor, "t 3"],
    betas: Float32[torch.Tensor, "1 nb"],
    transl: Float32[torch.Tensor, "t 3"],
):
    """Forward with per-part pose blocks (original ``get_smplx_forward_per_parts``)."""
    t: int = body_pose.shape[0]
    device: torch.device = body_pose.device
    zeros3: Float32[torch.Tensor, "t 3"] = torch.zeros(t, 3, device=device, dtype=body_pose.dtype)
    expression: Float32[torch.Tensor, "t ne"] = torch.zeros(t, model.num_expression_coeffs, device=device, dtype=body_pose.dtype)
    return model(
        betas=betas.expand(t, -1),
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=zeros3,
        reye_pose=zeros3,
        transl=transl,
        expression=expression,
    )
