"""Dynamic MHR forward kinematics and skinning exposed to Torch compilation.

The math follows the released MHR TorchScript (pymomentum): XYZ local rotations,
double-precision prefix products, learned pose correctives, then all skin weights.
Only constant weights and the sparse matrix layout are prepared at construction.
"""

from typing import NamedTuple

import roma
import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Int64
from torch import Tensor, nn

from posekit.ops.compile import compile_unless_dev


class MhrOutput(NamedTuple):
    """MHR vertices in centimetres and skeleton states (translation, XYZW, scale)."""

    vertices: Float32[Tensor, "b v 3"]
    """Selected mesh vertices in the MHR model's centimetre convention."""
    state: Float32[Tensor, "b 127 8"]
    """Global joint translation, XYZW quaternion, and scale."""


class ConstantSparseLinear(nn.Module):
    """A fixed sparse matrix with differentiable, batched dense inputs."""

    def __init__(self, weights: Float32[Tensor, "o i"]) -> None:
        super().__init__()
        rows, columns = weights.nonzero(as_tuple=True)
        self.columns: Int64[Tensor, "s"] = columns
        self.values: Float32[Tensor, "s"] = weights[rows, columns].detach()
        self.lengths: Int64[Tensor, "1 o"] = rows.bincount(minlength=weights.shape[0])[None]

    @compile_unless_dev
    def forward(self, values: Float32[Tensor, "b i"]) -> Float32[Tensor, "b o"]:
        """Multiply only nonzero weights, retaining the original row order."""
        products: Tensor = values[:, self.columns] * self.values
        # Lengths come from this matrix's row indices. They are checked once by
        # construction, so the reduction needs no host-side value check.
        return torch.segment_reduce(products, "sum", lengths=self.lengths.expand(len(values), -1), axis=1, unsafe=True)


def _rotate(quaternion: Float[Tensor, "... 4"], vector: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
    """Rotate by a normalized XYZW quaternion, in pymomentum's operation order."""
    cross: Tensor = torch.cross(quaternion[..., :3], vector, dim=-1)
    double_cross: Tensor = torch.cross(quaternion[..., :3], cross, dim=-1)
    return vector + 2 * (cross * quaternion[..., 3:] + double_cross)


def _multiply(first: Float[Tensor, "... 8"], second: Float[Tensor, "... 8"]) -> Float[Tensor, "... 8"]:
    """Compose normalized translation/quaternion/scale skeleton states."""
    q1: Tensor = F.normalize(first[..., 3:7], dim=-1)
    q2: Tensor = F.normalize(second[..., 3:7], dim=-1)
    translation: Tensor = first[..., :3] + first[..., 7:] * _rotate(q1, second[..., :3])
    quaternion: Tensor = roma.quat_product(q1, q2)
    return torch.cat((translation, quaternion, first[..., 7:] * second[..., 7:]), dim=-1)


class BodySkinning(nn.Module):
    """The released MHR with dynamic body pose, identity, scale and correctives."""

    def __init__(self, model: torch.jit.ScriptModule, vertices: Int64[Tensor, "v"] | None = None) -> None:
        super().__init__()
        character = model.character_torch
        full_mesh: bool = vertices is None
        if vertices is None:
            vertices = torch.arange(character.blend_shape.base_shape.shape[0], device=character.blend_shape.base_shape.device)
        self.shape_vectors: Float32[Tensor, "45 v 3"] = character.blend_shape.shape_vectors[:, vertices].contiguous()
        self.base_shape: Float32[Tensor, "v 3"] = character.blend_shape.base_shape[vertices].contiguous()
        self.face_vectors: Float32[Tensor, "72 v 3"] = model.face_expressions_model.shape_vectors[:, vertices].contiguous()
        self.parameter_transform: Float32[Tensor, "889 204"] = character.parameter_transform.parameter_transform[:, :204].contiguous()
        self.translation_offsets: Float32[Tensor, "127 3"] = character.skeleton.joint_translation_offsets
        self.prerotations: Float32[Tensor, "127 4"] = character.skeleton.joint_prerotations
        self.prefix_indices: list[Int64[Tensor, "2 k"]] = list(character.skeleton.pmi.split(character.skeleton._pmi_buffer_sizes, dim=1))
        self.inverse_bind: Float32[Tensor, "127 8"] = character.linear_blend_skinning.inverse_bind_pose
        sparse = getattr(model.pose_correctives_model.pose_dirs_predictor, "0")
        self.corrective_input: Float32[Tensor, "3000 750"] = torch.sparse_coo_tensor(
            sparse.sparse_indices, sparse.sparse_weight, sparse.sparse_shape,
        ).to_dense()
        linear = getattr(model.pose_correctives_model.pose_dirs_predictor, "2")
        # The full released matrix is 97.3% zero. Sparse reduction saves its
        # large memory read; the small keypoint support remains faster dense.
        self.sparse_correctives: ConstantSparseLinear | None = ConstantSparseLinear(linear.weight) if full_mesh else None
        self.corrective_output: Float32[Tensor, "v 3 3000"] | None = (
            None if full_mesh else linear.weight.unflatten(0, (-1, 3))[vertices].contiguous()
        )
        skinning = character.linear_blend_skinning
        selected: Tensor = torch.isin(skinning.vert_indices_flattened, vertices)
        self.skin_vertices: Int64[Tensor, "s"] = torch.searchsorted(vertices, skinning.vert_indices_flattened[selected])
        self.skin_joints: Int64[Tensor, "s"] = skinning.skin_indices_flattened[selected].long()
        self.skin_weights: Float32[Tensor, "s"] = skinning.skin_weights_flattened[selected]

    @compile_unless_dev
    def forward(
        self, identity: Float32[Tensor, "b 45"], pose: Float32[Tensor, "b 204"], expression: Float32[Tensor, "b 72"],
    ) -> MhrOutput:
        """Reconstruct selected vertices and every joint, with no per-frame value cached."""
        parameters: Tensor = F.linear(pose, self.parameter_transform).unflatten(-1, (127, 7))
        angles: Tensor = parameters[..., 3:6]
        half: Tensor = angles * 0.5
        cosine: Tensor = half.cos()
        sine: Tensor = half.sin()
        cr, cp, cy = cosine.unbind(-1)
        sr, sp, sy = sine.unbind(-1)
        local_quaternion: Tensor = torch.stack((
            sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy,
        ), -1)
        local_quaternion = roma.quat_product(self.prerotations, local_quaternion)
        local_translation: Tensor = parameters[..., :3] + self.translation_offsets
        local_scale: Tensor = torch.exp(parameters[..., 6:] * 0.69314718246459961)
        state: Tensor = torch.cat((local_translation, local_quaternion, local_scale), -1).double()
        for indices in self.prefix_indices:
            source, target = indices.unbind(0)
            state = state.index_copy(1, source, _multiply(state[:, target], state[:, source]))
        state = state.float()
        # The learned corrective MLP consumes the first two columns of each
        # non-root joint's XYZ rotation, with the identity diagonal subtracted.
        cosine = angles[:, 2:].cos()
        sine = angles[:, 2:].sin()
        cx, cy, cz = cosine.unbind(-1)
        sx, sy, sz = sine.unbind(-1)
        features: Tensor = torch.stack((
            cy * cz - 1, cy * sz, -sy,
            -cx * sz + sx * sy * cz, cx * cz + sx * sy * sz - 1, sx * cy,
        ), -1).flatten(1)
        hidden: Tensor = F.linear(features, self.corrective_input).relu()
        if self.sparse_correctives is not None:
            correctives: Tensor = self.sparse_correctives(hidden).unflatten(-1, (-1, 3))
        else:
            assert self.corrective_output is not None
            correctives = F.linear(hidden, self.corrective_output.flatten(0, 1)).unflatten(-1, (-1, 3))
        rest: Tensor = torch.einsum("nvd,bn->bvd", self.shape_vectors, identity) + self.base_shape
        rest = rest + torch.einsum("nvd,bn->bvd", self.face_vectors, expression) + correctives
        skin: Tensor = _multiply(state, self.inverse_bind[None])
        skin = skin[:, self.skin_joints]
        quaternion: Tensor = F.normalize(F.normalize(skin[..., 3:7], dim=-1), dim=-1)
        transformed: Tensor = skin[..., :3] + _rotate(quaternion, skin[..., 7:] * rest[:, self.skin_vertices])
        vertices: Tensor = torch.zeros_like(rest).index_add(1, self.skin_vertices, transformed * self.skin_weights[None, :, None])
        return MhrOutput(vertices, state)
