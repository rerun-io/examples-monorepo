"""Device-resident SAM body estimates and the released body/hand fusion rules."""

from dataclasses import dataclass, fields

import numpy as np
import roma
import torch
from jaxtyping import Bool, Float32
from sam_3d_body.models.modules.mhr_utils import fix_wrist_euler, rotation_angle_difference
from torch import Tensor

from posekit.models.sam3d_body_geometry import rotmat_to_euler
from posekit.models.sam3d_body_head import CaptureMhrHead


@dataclass(slots=True)
class BodyEstimate:
    """The tensor part of a decoder result, before camera translation is applied."""

    pred_pose_raw: Float32[Tensor, "n 266"]
    """Continuous root and body rotation predictions before compact conversion."""
    global_rot: Float32[Tensor, "n 3"]
    """MHR root Euler parameters, in radians."""
    body_pose: Float32[Tensor, "n 133"]
    """Compact MHR body pose, including the wrist parameters used by fusion."""
    shape: Float32[Tensor, "n 45"]
    """Identity blend-shape coefficients."""
    scale: Float32[Tensor, "n 28"]
    """Predicted segment-scale coefficients."""
    hand: Float32[Tensor, "n 108"]
    """Left and right hand pose coefficients, 54 each."""
    face: Float32[Tensor, "n 72"]
    """Face expression coefficients; the released inference head sets them to zero."""
    pred_keypoints_3d: Float32[Tensor, "n 70 3"]
    """Camera-axis MHR keypoints in metres, before camera translation."""
    pred_vertices: Float32[Tensor, "n v 3"]
    """Camera-axis mesh vertices in metres; intermediate decoders retain only keypoint support vertices."""
    pred_joint_coords: Float32[Tensor, "n 127 3"]
    """Camera-axis skeleton centres in metres, before camera translation."""
    joint_global_rots: Float32[Tensor, "n 127 3 3"]
    """Global MHR skeleton rotations from the last decoder prediction."""
    pred_cam: Float32[Tensor, "n 3"]
    """Network camera parameters, retained for the re-prompt input."""
    pred_keypoints_2d: Float32[Tensor, "n 70 2"]
    """Projected keypoints in source-image pixels."""
    pred_cam_t: Float32[Tensor, "n 3"]
    """Camera translation in metres."""
    focal_length: Float32[Tensor, "n"]
    """Projection focal length in pixels."""
    pred_keypoints_2d_cropped: Float32[Tensor, "n 70 2"]
    """Keypoints in normalized crop coordinates, with crop edges at -0.5 and 0.5."""

    @classmethod
    def from_decoder(cls, output: dict[str, Tensor | np.ndarray | None]) -> "BodyEstimate":
        """Leave the vendor dictionary at the decoder boundary."""
        values: list[Tensor] = []
        for item in fields(cls):
            value = output[item.name]
            if not isinstance(value, Tensor):
                raise TypeError(f"Decoder field {item.name} must be a tensor.")
            values.append(value)
        return cls(*values)

    def select(self, rows: slice) -> "BodyEstimate":
        """Own the selected rows so that hand unmirroring cannot affect another estimate."""
        return BodyEstimate(*(getattr(self, item.name)[rows].clone() for item in fields(self)))

    def choose(self, use: Bool[Tensor, "n"], other: "BodyEstimate") -> "BodyEstimate":
        """Keep the initial body when all four re-prompt points are invalid."""
        return BodyEstimate(*(
            torch.where(use.reshape((-1,) + (1,) * (getattr(self, item.name).ndim - 1)), getattr(self, item.name), getattr(other, item.name))
            for item in fields(self)
        ))


class BodyHandFusion:
    """The original wrist gates and MHR merge, using only device values."""

    def __init__(self, head: CaptureMhrHead) -> None:
        self.head: CaptureMhrHead = head
        device: torch.device = head.scale_mean.device
        self.wrist_parameters: Tensor = torch.tensor([41, 43, 42, 31, 33, 32], device=device)
        self.lowarm: Tensor = torch.tensor([76, 40], device=device)
        self.wrist_twist: Tensor = torch.tensor([77, 41], device=device)
        self.sides: Tensor = torch.tensor([1, 0, 1, 0], device=device)
        self.labels: Tensor = torch.tensor([41, 62, 8, 7], device=device)

    def prepare_prompts(
        self, body: BodyEstimate, left: BodyEstimate, right: BodyEstimate,
        batch: dict[str, Tensor], hand_batch: dict[str, Tensor], width: int, multiple_people: Bool[Tensor, "n"],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Unmirror the left hand, check the four original gates, and make four padded prompts."""
        head: CaptureMhrHead = self.head
        count: int = len(body.global_rot)
        left.scale[:, 9] = (head.scale_mean[8] + head.scale_comps[8, 8] * left.scale[:, 8] - head.scale_mean[9]) / head.scale_comps[9, 9]
        left.joint_global_rots[:, 78] = left.joint_global_rots[:, 42] * head.axes[None, :, None]
        left.hand[:, :54] = left.hand[:, 54:].clone()
        original: Tensor = roma.euler_to_rotmat("XZY", body.body_pose[:, self.wrist_parameters].unflatten(1, (2, 3)))
        global_wrist: Tensor = torch.stack((left.joint_global_rots[:, 78], right.joint_global_rots[:, 42]), dim=1)
        zero_wrist: Tensor = body.joint_global_rots[:, self.lowarm] @ head.joint_rotation[self.wrist_twist]
        local_wrist: Tensor = torch.einsum("kabc,kabd->kadc", global_wrist, zero_wrist)
        valid: Tensor = rotation_angle_difference(original, local_wrist) < 1.4
        scale: Tensor = hand_batch["bbox_scale"][0]
        valid = valid & torch.stack(((scale[:count] > 64).all(-1), (scale[count:] > 64).all(-1)), -1)
        valid = valid & torch.stack((left.pred_keypoints_2d_cropped.abs().amax((1, 2)) < 0.5, right.pred_keypoints_2d_cropped.abs().amax((1, 2)) < 0.5), -1)
        left_wrist: Tensor = left.pred_keypoints_2d[:, 41:42].clone()
        left_wrist[:, :, 0] = width - left_wrist[:, :, 0] - 1
        right_wrist: Tensor = right.pred_keypoints_2d[:, 41:42]
        # The released code uses the opposite hand's crop scale for these two gates.
        right_distance: Tensor = (right_wrist - body.pred_keypoints_2d[:, 41:42]).flatten(0, 1).norm(dim=-1) / scale[:count, 0]
        left_distance: Tensor = (left_wrist - body.pred_keypoints_2d[:, 62:63]).flatten(0, 1).norm(dim=-1) / scale[count:, 0]
        valid = valid & torch.stack((left_distance < 0.25, right_distance < 0.25), -1)
        points: Tensor = torch.cat((right_wrist, left_wrist, body.pred_keypoints_2d[:, 8:9], body.pred_keypoints_2d[:, 7:8]), 1)
        homogeneous: Tensor = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        crop: Tensor = homogeneous @ batch["affine_trans"][0].mT / batch["img_size"][0, :, None] - 0.5
        inside: Tensor = torch.where(multiple_people[:, None], ((crop >= -0.5) & (crop <= 0.5)).all(-1), ((crop > -0.5) & (crop < 0.5)).all(-1))
        prompt_valid: Tensor = inside & valid[:, self.sides]
        # The reference removes invalid prompts for one person, but retains
        # learned dummy prompts when an input frame contains several people.
        # Preserve that per-frame rule even when several frames share a batch.
        labels: Tensor = torch.where(prompt_valid, self.labels[None], torch.where(multiple_people[:, None], -2, -3))
        coordinates: Tensor = torch.where((labels != -2)[..., None], (crop + 0.5).clamp(0.0, 1.0), 0.0)
        prompts: Tensor = torch.cat((coordinates, labels[..., None]), -1)
        return prompts, valid, original, global_wrist

    def merge(
        self, body: BodyEstimate, left: BodyEstimate, right: BodyEstimate,
        hand_valid: Tensor, original: Tensor, global_wrist: Tensor, image_size: Tensor,
    ) -> BodyEstimate:
        """Apply the released hand replacement gates and reconstruct the complete final mesh."""
        head: CaptureMhrHead = self.head
        updated_hand: Tensor = torch.cat((left.hand[:, :54], right.hand[:, 54:]), 1)
        updated_scale: Tensor = body.scale.clone()
        updated_scale[:, 9] = left.scale[:, 9]
        updated_scale[:, 8] = right.scale[:, 8]
        updated_scale[:, 18:] = (left.scale[:, 18:] + right.scale[:, 18:]) / 2
        updated_shape: Tensor = body.shape.clone()
        updated_shape[:, 40:] = (left.shape[:, 40:] + right.shape[:, 40:]) / 2
        reconstruction = head.mhr_forward(
            body.global_rot * 0, body.global_rot, body.body_pose, updated_hand, updated_scale, updated_shape, body.face,
            return_joint_rotations=True, keypoints_only=True,
        )
        assert isinstance(reconstruction, tuple)
        zero_wrist: Tensor = reconstruction[1][:, self.lowarm] @ head.joint_rotation[self.wrist_twist]
        local_wrist: Tensor = torch.einsum("kabc,kabd->kadc", global_wrist, zero_wrist)
        wrist: Tensor = fix_wrist_euler(rotmat_to_euler("XZY", local_wrist))
        valid: Tensor = ((rotation_angle_difference(original, local_wrist) < 1.4) & hand_valid)[..., None]
        current: Tensor = body.body_pose[:, self.wrist_parameters].unflatten(1, (2, 3))
        body.body_pose = body.body_pose.index_copy(1, self.wrist_parameters, torch.where(valid, wrist, current).flatten(1))
        body.hand = torch.where(valid, updated_hand.unflatten(1, (2, 54)), body.hand.unflatten(1, (2, 54))).flatten(1)
        body.scale[:, 9] = torch.where(valid[:, 0, 0], updated_scale[:, 9], body.scale[:, 9])
        body.scale[:, 8] = torch.where(valid[:, 1, 0], updated_scale[:, 8], body.scale[:, 8])
        number: Tensor = valid.squeeze(-1).sum(1, keepdim=True)
        body.scale[:, 18:] = torch.where(number > 0, (left.scale[:, 18:] * valid[:, 0] + right.scale[:, 18:] * valid[:, 1]) / (number + 1e-8), body.scale[:, 18:])
        body.shape[:, 40:] = torch.where(number > 0, (left.shape[:, 40:] * valid[:, 0] + right.shape[:, 40:] * valid[:, 1]) / (number + 1e-8), body.shape[:, 40:])
        final = head.mhr_forward(
            body.global_rot * 0, body.global_rot, body.body_pose, body.hand, body.scale, body.shape, body.face,
            return_keypoints=True, return_joint_coords=True, return_model_params=True, return_joint_rotations=True,
        )
        assert isinstance(final, tuple)
        body.pred_vertices = final[0] * head.axes
        body.pred_keypoints_3d = final[1][:, :70] * head.axes
        body.pred_joint_coords = final[2] * head.axes
        # Preserve the released output's decoder rotations (the final merge does not replace them).
        projected: Tensor = body.pred_keypoints_3d + body.pred_cam_t[:, None]
        body.pred_keypoints_2d = (projected[..., :2] * body.focal_length[:, None, None] + image_size[None, None] / 2 * projected[..., 2:]) / projected[..., 2:]
        return body
