"""Scale-only MHR rest skinning with the released shape and skinning coefficients.

EgoHand supplies zero body pose to MHR. Joint orientations therefore stay fixed;
translations and scales still change with all 26 predicted hand parameters.
This exposes those operations to Torch compilation without the TorchScript body
model's general rotation and pose-corrective work.
"""

import torch
from jaxtyping import Float32, Float64, Int64
from soma.geometry.transforms import quaternion_multiply_xyzw, quaternion_normalize_xyzw, quaternion_xyzw_to_matrix
from torch import Tensor, nn


class MhrHandRestSkinning(nn.Module):
    """Released MHR shape, double-precision scale FK, and all contributing skin influences."""

    def __init__(
        self, character: torch.jit.ScriptModule, source_vertices: Int64[Tensor, "m"], scale_indices: Int64[Tensor, "26"],
        neutral_correctives: Float32[Tensor, "1 v 3"],
    ) -> None:
        super().__init__()
        self.shape_vectors: Float32[Tensor, "5 m 3"] = character.blend_shape.shape_vectors[40:45, source_vertices].contiguous()
        self.mean_shape: Float32[Tensor, "1 m 3"] = character.blend_shape.base_shape[None, source_vertices] + neutral_correctives[:, source_vertices]
        self.parameter_transform: Float32[Tensor, "127 7 26"] = character.parameter_transform.parameter_transform[:, 136 + scale_indices].reshape(127, 7, 26).contiguous()
        if torch.count_nonzero(self.parameter_transform[:, 3:6]).item():
            raise ValueError("The released MHR hand scales must not change joint rotations.")
        self.translation_offsets: Float32[Tensor, "127 3"] = character.skeleton.joint_translation_offsets
        quaternion: Float64[Tensor, "127 4"] = character.skeleton.joint_prerotations.double().clone()
        self.prefix_indices: list[Int64[Tensor, "2 k"]] = list(character.skeleton.pmi.split(character.skeleton._pmi_buffer_sizes, dim=1))
        self.prefix_rotations: list[Float64[Tensor, "k 3 3"]] = []
        for indices in self.prefix_indices:
            source, target = indices.unbind(0)
            self.prefix_rotations.append(quaternion_xyzw_to_matrix(quaternion[target]))
            quaternion[source] = quaternion_multiply_xyzw(quaternion_normalize_xyzw(quaternion[target]), quaternion_normalize_xyzw(quaternion[source]))
        global_quaternion: Float32[Tensor, "127 4"] = quaternion.float()
        inverse_bind: Float32[Tensor, "127 8"] = character.linear_blend_skinning.inverse_bind_pose
        self.global_rotation: Float32[Tensor, "127 3 3"] = quaternion_xyzw_to_matrix(global_quaternion)
        self.inverse_translation: Float32[Tensor, "127 3"] = inverse_bind[:, :3]
        self.inverse_scale: Float32[Tensor, "127"] = inverse_bind[:, 7]
        skin_quaternion: Float32[Tensor, "127 4"] = quaternion_multiply_xyzw(
            quaternion_normalize_xyzw(global_quaternion), quaternion_normalize_xyzw(inverse_bind[:, 3:7]),
        )
        self.skin_rotation: Float32[Tensor, "127 3 3"] = quaternion_xyzw_to_matrix(skin_quaternion)
        skinning = character.linear_blend_skinning
        selected: Int64[Tensor, "s"] = torch.nonzero(torch.isin(skinning.vert_indices_flattened, source_vertices)).flatten()
        self.skin_vertices: Int64[Tensor, "s"] = torch.searchsorted(source_vertices, skinning.vert_indices_flattened[selected])
        self.skin_joints: Int64[Tensor, "s"] = skinning.skin_indices_flattened[selected].long()
        self.skin_weights: Float32[Tensor, "s"] = skinning.skin_weights_flattened[selected]

    def forward(self, identity: Float32[Tensor, "b 5"], scales: Float32[Tensor, "b 26"]) -> Float32[Tensor, "b m 3"]:
        """Compute each frame's shaped, scaled source vertices; no learned value is cached."""
        parameters: Float32[Tensor, "b 127 7"] = torch.einsum("jpn,bn->bjp", self.parameter_transform, scales)
        translation: Float64[Tensor, "b 127 3"] = (parameters[..., :3] + self.translation_offsets).double()
        scale: Float64[Tensor, "b 127"] = torch.exp(parameters[..., 6] * 0.69314718246459961).double()
        for indices, rotation in zip(self.prefix_indices, self.prefix_rotations, strict=True):
            source, target = indices.unbind(0)
            updated_translation: Float64[Tensor, "b k 3"] = translation[:, target] + scale[:, target, None] * torch.einsum("kij,bkj->bki", rotation, translation[:, source])
            updated_scale: Float64[Tensor, "b k"] = scale[:, target] * scale[:, source]
            translation = translation.index_copy(1, source, updated_translation)
            scale = scale.index_copy(1, source, updated_scale)
        world_translation: Float32[Tensor, "b 127 3"] = translation.float()
        world_scale: Float32[Tensor, "b 127"] = scale.float()
        skin_translation: Float32[Tensor, "b 127 3"] = world_translation + world_scale[..., None] * torch.einsum("jik,jk->ji", self.global_rotation, self.inverse_translation)
        skin_scale: Float32[Tensor, "b 127"] = world_scale * self.inverse_scale
        rest: Float32[Tensor, "b m 3"] = torch.einsum("nvd,bn->bvd", self.shape_vectors, identity) + self.mean_shape
        transformed: Float32[Tensor, "b s 3"] = skin_translation[:, self.skin_joints] + torch.einsum(
            "sij,bsj->bsi", self.skin_rotation[self.skin_joints], skin_scale[:, self.skin_joints, None] * rest[:, self.skin_vertices],
        )
        return torch.zeros_like(rest).index_add(1, self.skin_vertices, transformed * self.skin_weights[None, :, None])
