"""The released pose head with capture-safe rotations and dynamic MHR."""

import numpy as np
import roma
import torch
from jaxtyping import Float32, Int64
from sam_3d_body.models.heads.mhr_head import MHRHead
from sam_3d_body.models.modules import rot6d_to_rotmat
from torch import Tensor, nn

from posekit.models.sam3d_body_geometry import CompactPose, rotmat_to_euler
from posekit.models.sam3d_body_mhr import BodySkinning, MhrOutput
from posekit.ops.compile import compile_unless_dev


class CaptureMhrHead(nn.Module):
    """Reuse every learned parameter, replacing only the geometry implementation."""

    scale_mean: Float32[Tensor, "68"]
    scale_comps: Float32[Tensor, "28 68"]
    hand_pose_mean: Float32[Tensor, "54"]
    hand_pose_comps: Float32[Tensor, "54 54"]
    hand_joint_idxs_left: Int64[Tensor, "27"]
    hand_joint_idxs_right: Int64[Tensor, "27"]
    keypoint_mapping: Float32[Tensor, "308 v"]
    local_to_world_wrist: Float32[Tensor, "3 3"]
    right_wrist_coords: Float32[Tensor, "3"]
    root_coords: Float32[Tensor, "3"]
    joint_rotation: Float32[Tensor, "127 3 3"]

    def __init__(self, source: MHRHead) -> None:
        nn.Module.__init__(self)
        for name, parameter in source.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        self.proj: nn.Module = source.proj
        self.enable_hand_model: bool = source.enable_hand_model
        self.mhr: BodySkinning = BodySkinning(source.mhr)
        vertices: Tensor = torch.nonzero(source.keypoint_mapping[:70, :18439].ne(0).any(dim=0)).flatten()
        self.keypoint_mhr: BodySkinning = BodySkinning(source.mhr, vertices)
        self.compact_mapping: Tensor = torch.cat((source.keypoint_mapping[:70, vertices], source.keypoint_mapping[:70, 18439:]), dim=1)
        self.body_conversion: CompactPose = CompactPose("body", source.scale_mean.device)
        self.hand_conversion: CompactPose = CompactPose("hand", source.scale_mean.device)
        self.face_array: Int64[np.ndarray, "f 3"] = source.faces.cpu().numpy()
        self.axes: Float32[Tensor, "3"] = torch.tensor([1.0, -1.0, -1.0], device=source.scale_mean.device)
        self.body_mask: Float32[Tensor, "133"] = torch.ones(133, device=source.scale_mean.device)
        self.body_mask[62:116] = 0
        self.body_mask[130:] = 0
        self.hand_model_mask: Float32[Tensor, "204"] = torch.ones(204, device=source.scale_mean.device)
        self.hand_model_mask[source.nonhand_param_idxs] = 0

    def mhr_forward(
        self, global_trans: Tensor, global_rot: Tensor, body_pose_params: Tensor,
        hand_pose_params: Tensor | None, scale_params: Tensor, shape_params: Tensor,
        expr_params: Tensor | None = None, return_keypoints: bool = False, do_pcblend: bool = True,
        return_joint_coords: bool = False, return_model_params: bool = False,
        return_joint_rotations: bool = False, scale_offsets: Tensor | None = None,
        vertex_offsets: Tensor | None = None,
        *, keypoints_only: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Full MHR reconstruction in the original model's centimetre-to-metre convention."""
        if scale_params.ndim == 1:
            scale_params = scale_params[None]
        if shape_params.ndim == 1:
            shape_params = shape_params[None]
        if self.enable_hand_model:
            original_trans: Tensor = global_trans
            global_rot = rotmat_to_euler("xyz", roma.euler_to_rotmat("xyz", global_rot) @ self.local_to_world_wrist)
            global_trans = -(
                roma.euler_to_rotmat("xyz", global_rot) @ (self.right_wrist_coords - self.root_coords) + self.root_coords
            ) + original_trans
        scales: Tensor = self.scale_mean[None] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets
        pose: Tensor = torch.cat((global_trans * 10, global_rot, body_pose_params[..., :130]), dim=1)
        if hand_pose_params is not None:
            left: Tensor = self.hand_conversion(self.hand_pose_mean + hand_pose_params[:, :54] @ self.hand_pose_comps)
            right: Tensor = self.hand_conversion(self.hand_pose_mean + hand_pose_params[:, 54:] @ self.hand_pose_comps)
            pose = pose.index_copy(1, self.hand_joint_idxs_left, left).index_copy(1, self.hand_joint_idxs_right, right)
        model_parameters: Tensor = torch.cat((pose, scales), dim=1)
        if self.enable_hand_model:
            model_parameters = model_parameters * self.hand_model_mask
        if expr_params is None:
            expr_params = shape_params.new_zeros((len(shape_params), 72))
        output: MhrOutput = (self.keypoint_mhr if keypoints_only else self.mhr)(shape_params, model_parameters, expr_params)
        vertices: Tensor = output.vertices / 100
        joints: Tensor = output.state[..., :3] / 100
        result: list[Tensor] = [vertices]
        if return_keypoints:
            combined: Tensor = torch.cat((vertices, joints), dim=1)
            mapping: Tensor = self.compact_mapping if keypoints_only else self.keypoint_mapping
            keypoints: Tensor = (mapping @ combined.permute(1, 0, 2).flatten(1, 2)).reshape(-1, len(pose), 3).permute(1, 0, 2)
            if self.enable_hand_model:
                keypoints = keypoints.clone()
                keypoints[:, :21] = 0
                keypoints[:, 42:] = 0
            result.append(keypoints)
        if return_joint_coords:
            result.append(joints)
        if return_model_params:
            result.append(model_parameters)
        if return_joint_rotations:
            result.append(roma.unitquat_to_rotmat(output.state[..., 3:7]))
        return result[0] if len(result) == 1 else tuple(result)

    def forward(
        self, x: Tensor, init_estimate: Tensor | None = None, do_pcblend: bool = True, slim_keypoints: bool = False,
    ) -> dict[str, Tensor | np.ndarray | None]:
        """Regress the original parameters and keypoints; reconstruct the final complete mesh once later."""
        # This head already uses device indices. The released decoder's list
        # indexing adapter must not be traced into compiled geometry functions.
        with torch._C.DisableTorchFunction():
            output: dict[str, Tensor | np.ndarray | None] = dict(self._forward_geometry(x, init_estimate))
        output["faces"] = self.face_array
        return output

    @compile_unless_dev
    def _forward_geometry(self, x: Tensor, init_estimate: Tensor | None) -> dict[str, Tensor | None]:
        """Geometry below the decoder's temporary TorchFunctionMode boundary."""
        prediction: Tensor = self.proj(x)
        if init_estimate is not None:
            prediction = prediction + init_estimate
        rotation: Tensor = rotmat_to_euler("ZYX", rot6d_to_rotmat(prediction[:, :6]))
        body: Tensor = self.body_conversion(prediction[:, 6:266]) * self.body_mask
        shape: Tensor = prediction[:, 266:311]
        scale: Tensor = prediction[:, 311:339]
        hand: Tensor = prediction[:, 339:447]
        face: Tensor = prediction[:, 447:519] * 0
        result = self.mhr_forward(
            torch.zeros_like(rotation), rotation, body, hand, scale, shape, face,
            return_keypoints=True, return_joint_coords=True, return_model_params=True, return_joint_rotations=True,
            keypoints_only=True,
        )
        assert isinstance(result, tuple)
        vertices, keypoints, joints, parameters, rotations = result
        return {
            "pred_pose_raw": prediction[:, :266], "pred_pose_rotmat": None,
            "global_rot": rotation, "body_pose": body, "shape": shape, "scale": scale,
            "hand": hand, "face": face, "pred_keypoints_3d": keypoints[:, :70] * self.axes,
            "pred_vertices": vertices * self.axes, "pred_joint_coords": joints * self.axes,
            "joint_global_rots": rotations, "mhr_model_params": parameters,
        }

    def complete_mesh(self, output: dict[str, Tensor | np.ndarray | None]) -> None:
        """Reconstruct the final body mesh once, after the decoder's keypoint-only layers."""
        shape = output["shape"]
        parameters = output["mhr_model_params"]
        expression = output["face"]
        assert isinstance(shape, Tensor) and isinstance(parameters, Tensor) and isinstance(expression, Tensor)
        output["pred_vertices"] = self.mhr(shape, parameters, expression).vertices / 100 * self.axes
