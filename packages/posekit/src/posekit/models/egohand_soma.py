"""Inference-only SOMA adapters with static data prepared before CUDA capture.

The released MHR parameters remain dynamic: all five identity coefficients,
26 segment scales, 25 joint rotations and every mesh vertex are evaluated for
each hand. Only the neutral pose corrective and topology tables are cached.
"""

from collections.abc import Mapping
from typing import Any

import torch
from jaxtyping import Float32, Int64
from soma.geometry.batched_skinning import BatchedSkinning
from soma.geometry.transforms import SE3_inverse
from soma.hand.identity_model import MHRHandIdentityModel
from torch import Tensor

from egohand.models.soma_hand_wrapper import SomaHandWrapper
from posekit.models.egohand_mhr import MHR_JOINT_PARAMETERS, MHR_JOINTS, MhrHandRestSkinning
from posekit.ops.compile import compile_unless_dev


class _CudaMHRHand(MHRHandIdentityModel):
    """Reuse the loaded MHR asset with its fixed neutral corrective prepared once."""

    def __init__(self, source: MHRHandIdentityModel) -> None:
        # Transfer an already loaded backend; do not reload the multi-GB body asset.
        self.__dict__.update(source.__dict__)
        self._modules = dict(source._modules)
        self._buffers = dict(source._buffers)
        self._parameters = dict(source._parameters)
        scale_indices: Int64[Tensor, "26"] = torch.tensor(self._scale_mhr_indices, device=self.device, dtype=torch.int64)
        self._hand_vert_ids = torch.as_tensor(self._hand_vert_ids, device=self.device)
        self._wrist_boundary_global = torch.as_tensor(self._wrist_boundary_global, device=self.device)
        neutral_correctives: Float32[Tensor, "1 v 3"] = self._mhr_model.module.pose_correctives_model(
            torch.zeros((1, MHR_JOINTS * MHR_JOINT_PARAMETERS), device=self.device)
        )
        interpolation = self._to_soma_interp
        if self._laplacian_mesh is not None or interpolation.tet_normal_scale != "area":
            raise ValueError("EgoHand requires the released area-based hand topology transfer.")
        source_faces: Int64[Tensor, "f 3"] = interpolation.get_buffer("F_src_torch")
        face_ids: Int64[Tensor, "v"] = interpolation.get_buffer("face_ids")
        source_hand_faces: Int64[Tensor, "v 3"] = source_faces[face_ids[self._hand_vert_ids]]
        self.hand_barycentric: Float32[Tensor, "v 4"] = interpolation.bary_coords[self._hand_vert_ids].float()
        self.wrist_boundary: Int64[Tensor, "w"] = torch.searchsorted(self._hand_vert_ids, self._wrist_boundary_global)
        if not torch.equal(self._hand_vert_ids[self.wrist_boundary], self._wrist_boundary_global):
            raise ValueError("The released hand vertex ids must be sorted and contain the wrist boundary.")
        source_vertices, local_faces = torch.unique(source_hand_faces, sorted=True, return_inverse=True)
        self.hand_faces: Int64[Tensor, "v 3"] = local_faces
        character = self._mhr_model.module.character_torch
        self.rest_model = compile_unless_dev(MhrHandRestSkinning(character, source_vertices, scale_indices, neutral_correctives))

    def _get_shaped_verts(self, identity_coeffs: Float32[Tensor, "b 5"]) -> Float32[Tensor, "b v 3"]:
        """Evaluate the released shape, scale and FK for every required source vertex."""
        scales: Float32[Tensor, "b 26"] = self._current_scale_params if self._current_scale_params is not None else torch.zeros((len(identity_coeffs), 26), device=identity_coeffs.device)
        return self.rest_model(identity_coeffs, scales)

    def forward(
        self, identity_coeffs: Float32[Tensor, "b 5"], scale_params: Float32[Tensor, "b 26"] | None = None,
        kwargs: Mapping[str, Any] | None = None, global_scale: float | Float32[Tensor, "b"] = 1.0,
    ) -> Float32[Tensor, "b v 3"]:
        """Transfer all hand vertices directly, without materializing the body mesh in SOMA topology."""
        shaped: Float32[Tensor, "b m 3"] = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        triangles: Float32[Tensor, "b v 3 3"] = shaped[:, self.hand_faces]
        p0, p1, p2 = triangles.unbind(-2)
        p3: Float32[Tensor, "b v 3"] = p0 + torch.cross(p1 - p0, p2 - p0, dim=-1)
        tetrahedra: Float32[Tensor, "b v 4 3"] = torch.cat((triangles, p3.unsqueeze(-2)), dim=-2)
        hand: Float32[Tensor, "b v 3"] = (tetrahedra * self.hand_barycentric[None, :, :, None]).sum(-2)
        hand = self._apply_coord_transform(hand)
        hand = (hand - hand[:, self.wrist_boundary].mean(1, keepdim=True)) * self._unit_conversion
        if isinstance(global_scale, Tensor):
            return hand * global_scale[:, None, None]
        return hand * global_scale


class _CudaHandSkinning(BatchedSkinning):
    """Keep the hand's static hierarchy on CUDA when rebinding changing shapes."""

    def __init__(self, source: BatchedSkinning) -> None:
        self.__dict__.update(source.__dict__)
        if source.source_joint_parent_ids is not None:
            raise ValueError("The EgoHand CUDA adapter requires the released single hand topology.")
        self.cuda_parents: Int64[Tensor, "25"] = torch.tensor(self.joint_parent_ids, device=self.device, dtype=torch.int64)
        self.roots: tuple[int, ...] = tuple(index for index, parent in enumerate(self.joint_parent_ids) if index == parent)

    def rebind(self, bind_world_transforms: Float32[Tensor, "b 25 4 4"], bind_shapes: Float32[Tensor, "b v 3"]) -> None:
        """Rebind with the same rigid inverse and root convention as SOMA."""
        self.bind_batched = True
        self.bind_world_transforms = bind_world_transforms
        self.inverse_bind_transform = SE3_inverse(bind_world_transforms)
        local: Float32[Tensor, "b 25 4 4"] = self.inverse_bind_transform[:, self.cuda_parents] @ bind_world_transforms
        for root in self.roots:
            local[:, root] = bind_world_transforms[:, root]
        self.local_rotations = local[..., :3, :3]
        self.local_translations = local[..., :3, 3]
        self.bind_shapes = bind_shapes


def prepare_cuda_soma(model: SomaHandWrapper) -> None:
    """Prepare inference backends without changing the role or its full-mesh contract."""
    for hand in (model.left, model.right):
        if not isinstance(hand.identity_model, MHRHandIdentityModel):
            raise ValueError("EgoHand CUDA execution requires its released MHR identity model.")
        hand.identity_model = _CudaMHRHand(hand.identity_model)
        hand.batched_skinning = _CudaHandSkinning(hand.batched_skinning)
        # RBF interpolation is linear in vertex positions. The library precomputes
        # the same weights as a sparse matrix; its default loops over joint masks.
        hand.skeleton_transfer.use_sparse_rbf_matrix = True
        hand.skeleton_transfer._precompute_regressors()
        # Kabsch avoids the auto method's host-side validity check/fallback.
        hand.skeleton_transfer.rotation_method = "kabsch"
