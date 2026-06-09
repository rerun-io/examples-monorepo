"""SMPL-X model construction + per-part forward, mirroring the golden run setup
(neutral body for all subjects, 16 betas, flat hand mean, no PCA, no v_template).
"""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float32

NUM_BETAS: int = 16
"""Shape coefficients optimized per body (golden ``n_betas: 16``)."""


def build_smplx_neutral(model_folder: Path, device: str = "cuda"):
    """Neutral SMPL-X model matching ``utils_smplx.get_smplx_models``."""
    import smplx

    return smplx.create(
        str(model_folder),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        num_betas=NUM_BETAS,
        flat_hand_mean=True,
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
