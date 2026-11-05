"""Export adapter for EgoHand's pose and direct-keypoint heads on virtual crops."""

import torch
from jaxtyping import Float32
from torch import Tensor, nn

from egohand.models.egohand import EgoHand
from egohand.utils.ray_pe import fourier_encode_angles


class EgoHandHead(nn.Module):
    """The released heads with analytic rays for the 256-pixel virtual camera."""

    patch_xy: Float32[Tensor, "1 256 2"]

    def __init__(self, model: EgoHand) -> None:
        super().__init__()
        if model.kp2d_topology != "separate" or model.kp2d_head is None or not model.ray_condition or not model.cliff_condition:
            raise ValueError("The fast head requires EgoHand-v1's separate keypoint, ray and CLIFF heads.")
        self.hand_head = model.hand_head
        self.kp2d_head = model.kp2d_head
        self.num_bands: int = model.ray_pe_num_bands
        centers: Float32[Tensor, "16"] = (torch.arange(16, device=next(model.parameters()).device).float() + 0.5) * 16.0 - 128.0
        rows, columns = torch.meshgrid(centers, centers, indexing="ij")
        self.register_buffer("patch_xy", torch.stack((columns, rows), dim=-1).reshape(1, 256, 2), persistent=False)

    def forward(
        self, features: Float32[Tensor, "b 1280 16 16"], focal: Float32[Tensor, "b"]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return rotations, identity, scales, camera, direct pixels and presence."""
        cliff: Float32[Tensor, "b 3"] = torch.stack((torch.zeros_like(focal), torch.zeros_like(focal), 256.0 / focal), dim=-1)
        angles: Float32[Tensor, "b 256 2"] = torch.atan(self.patch_xy / focal[:, None, None])
        rays: Float32[Tensor, "b 256 32"] = fourier_encode_angles(angles, num_bands=self.num_bands)
        parameters, camera, _ = self.hand_head(features, cliff_cond=cliff, ray_feats=rays)
        direct = self.kp2d_head(features)
        return (
            torch.cat((parameters["wrist_orient"], parameters["hand_pose"]), dim=1).float(),
            parameters["identity_coeffs"].float(),
            parameters["scale_params"].float(),
            camera.float(),
            direct["mu"].float(),
            direct["presence"].float(),
        )


def solve_virtual_translation(
    joints: Float32[Tensor, "b 25 3"], camera: Float32[Tensor, "b 3"], direct: Float32[Tensor, "b 25 2"],
    presence: Float32[Tensor, "b 25"], focal: Float32[Tensor, "b"],
) -> Float32[Tensor, "b 3"]:
    """Solve the released fixed-depth fit for an isotropic, centered virtual camera.

    Its normal matrix is a scalar times the identity, so its two equal
    eigenvalues and damped solve have a closed form. The visibility, depth,
    finite-value and four-point gates retain the general solver's fallback.
    """
    scale = 2.0 * torch.exp(-camera[:, 0].clamp(-3.5, 3.0))
    depth = 2.0 * focal / (scale * 256.0)
    pixels = direct * 256.0 + 127.5
    finite_joints = torch.isfinite(joints).all(-1)
    usable = (
        (presence >= 0.5) & torch.isfinite(direct).all(-1) & finite_joints
        & (direct >= -0.5).all(-1) & (direct < 0.5).all(-1)
        & (pixels >= 0.0).all(-1) & (pixels < 256.0).all(-1)
        & ((joints[..., 2] + depth[:, None]) > 1e-4)
    )
    clean_joints = torch.where(finite_joints[..., None], joints, 0.0)
    absolute_depth = torch.where(usable, clean_joints[..., 2] + depth[:, None], 1.0)
    design = torch.where(usable, focal[:, None] / absolute_depth, 0.0)
    residual = torch.where(usable[..., None], pixels - 127.5 - design[..., None] * clean_joints[..., :2], 0.0)
    normal = design.square().sum(-1)
    rhs = (design[..., None] * residual).sum(-2)
    valid = (usable.sum(-1) >= 4) & (normal > 0.0) & torch.isfinite(normal) & torch.isfinite(rhs).all(-1)
    xy = rhs / torch.where(valid, normal * (1.0 + 1e-6), 1.0)[:, None]
    xy = torch.where((valid & torch.isfinite(xy).all(-1))[:, None], xy, camera[:, 1:])
    return torch.cat((xy, depth[:, None]), dim=-1)
