"""CUDA execution of EgoHand's released network and full SOMA hand reconstruction."""

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from jaxtyping import Float32
from soma.geometry.rig_utils import precompute_joint_orient, remove_joint_orient_local
from torch import Tensor

from egohand_inference.pipeline.predictor import EgoHandPredictor
from posekit.models.egohand_cuda_crops import CudaHandCropper, PreparedHandCrops
from posekit.models.egohand_head import EgoHandHead, solve_virtual_translation
from posekit.models.egohand_soma import prepare_cuda_soma
from posekit.predictions import HandSide
from posekit.runtimes import TensorRuntime


@dataclass(frozen=True, slots=True)
class EgoHandTensors:
    """Reconstruction tensors in source-camera axes, with the full SOMA mesh."""

    joints: Float32[Tensor, "n 25 3"]
    """Camera-space joint positions, metres."""
    pixels: Float32[Tensor, "n 25 2"]
    """Source-image projections, pixels."""
    vertices: Float32[Tensor, "n v 3"]
    """Camera-space vertices, metres."""
    rotations: Float32[Tensor, "n 25 3 3"]
    """SOMA local rotations, with source-camera wrist orientation."""
    identity: Float32[Tensor, "n 5"]
    """MHR identity coefficients."""
    scale: Float32[Tensor, "n"]
    """Uniform scale."""
    translation: Float32[Tensor, "n 3"]
    """Source-camera wrist position, metres."""
    scale_parameters: Float32[Tensor, "n 26"]
    """MHR segment scales."""


class EgoHandTensorExecutor:
    """Keep inference tensors on the device through restoration to the source camera."""

    def __init__(self, predictor: EgoHandPredictor, head_runtime: TensorRuntime | None = None) -> None:
        self.predictor = predictor
        self.head_runtime = head_runtime
        self.head = EgoHandHead(predictor.model).eval()
        prepare_cuda_soma(predictor.model.soma_hand)
        right = predictor.model.soma_hand.right
        self.right_parents = torch.as_tensor(right.joint_parent_ids, device="cuda", dtype=torch.int64)
        self.right_orient, self.right_parent_orient = precompute_joint_orient(right.t_pose_world, right.joint_parent_ids)
        self.reflection = torch.tensor([-1.0, 1.0, 1.0], device="cuda")
        self.graphs = {}
        self.stage_events = {}
        self.stage_ms: dict[str, float] = {}
        self.batch_size: int = min(predictor.batch_size, head_runtime.spec.max_batch_size) if head_runtime is not None else predictor.batch_size

    def __del__(self) -> None:
        """Destroy captured work while its external events and model buffers are still alive."""
        for state in getattr(self, "graphs", {}).values():
            state[0].reset()

    def _run(self, crops, focal, right_indices, events):
        model = self.predictor.model
        model.soma_hand.profile_events = None
        events["backbone"][0].record()
        features = model.backbone(crops)
        events["backbone"][1].record()
        events["head"][0].record()
        if self.head_runtime is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                poses, identity, scales, camera, direct, presence = self.head(features, focal)
        else:
            output = self.head_runtime({"features": features, "focal": focal})
            poses, identity, scales, camera, direct, presence = (output[k] for k in ("poses", "identity", "scales", "camera", "direct", "presence"))
        events["head"][1].record()
        poses = poses.clone()
        events["soma"][0].record()
        left = model.soma_hand(poses, identity, "left", pose2rot=False, global_scale=scales[:, 0], scale_params=scales[:, 1:])
        joints = left["joints"].clone()
        vertices = left["vertices"].clone()
        translation = solve_virtual_translation(joints, camera, direct, presence, focal)
        if len(right_indices):
            # Reuse left FK already computed for the shape and translation solve.
            rotations = left.transforms[right_indices, :, :3, :3] * -self.reflection[None, None, :, None]
            local = rotations[:, self.right_parents].transpose(-1, -2) @ rotations
            local[:, 0] = rotations[:, 0]
            right_poses = remove_joint_orient_local(local, self.right_orient, self.right_parent_orient)
            right = model.soma_hand(right_poses, identity[right_indices], "right", pose2rot=False,
                                    global_scale=scales[right_indices, 0], scale_params=scales[right_indices, 1:])
            joints[right_indices] = right["joints"]
            vertices[right_indices] = right["vertices"]
            translation[right_indices] = (left.transforms[right_indices, 0, :3, 3] + translation[right_indices]) * self.reflection
            poses[right_indices] = right_poses
        events["soma"][1].record()
        return poses, identity, scales, translation, joints, vertices

    @torch.inference_mode()
    def __call__(self, prepared: PreparedHandCrops, sides: tuple[HandSide, ...], cropper: CudaHandCropper) -> EgoHandTensors:
        """Run calibrated crops and return a source-camera reconstruction on CUDA."""
        self.stage_ms.clear()
        batches: list[EgoHandTensors] = []
        for start in range(0, len(sides), self.batch_size):
            stop: int = min(start + self.batch_size, len(sides))
            chunk = PreparedHandCrops(prepared.images[start:stop], prepared.focal[start:stop], prepared.rotation[start:stop])
            batches.append(self._run_chunk(chunk, sides[start:stop], cropper))
            # Read capture events before a repeated graph reuses them on the next chunk.
            torch.cuda.synchronize()
            for stage, (begin, end) in self.stage_events.items():
                self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + begin.elapsed_time(end)
        if len(batches) == 1:
            return batches[0]
        return EgoHandTensors(
            torch.cat([batch.joints for batch in batches]), torch.cat([batch.pixels for batch in batches]),
            torch.cat([batch.vertices for batch in batches]), torch.cat([batch.rotations for batch in batches]),
            torch.cat([batch.identity for batch in batches]), torch.cat([batch.scale for batch in batches]),
            torch.cat([batch.translation for batch in batches]), torch.cat([batch.scale_parameters for batch in batches]),
        )

    def _run_chunk(self, prepared: PreparedHandCrops, sides: tuple[HandSide, ...], cropper: CudaHandCropper) -> EgoHandTensors:
        """Replay one bounded batch; retained graph buffers remain live until the executor is released."""
        crops = prepared.images
        device = crops.device
        focal = prepared.focal.float()
        if sides not in self.graphs:
            static_crops = crops.clone()
            static_focal = focal.clone()
            right_indices = torch.tensor([i for i, side in enumerate(sides) if side == "right"], device=device, dtype=torch.int64)
            events = {name: (torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True))
                      for name in ("backbone", "head", "soma")}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
                for _ in range(3):
                    self._run(static_crops, static_focal, right_indices, events)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with wp.ScopedMempool(device=str(device), enable=True), wp.ScopedStream(wp.stream_from_torch(stream)), torch.cuda.graph(graph, stream=stream), wp.ScopedCapture(external=True):
                outputs = self._run(static_crops, static_focal, right_indices, events)
            self.graphs[sides] = (graph, static_crops, static_focal, outputs, events, right_indices)
        graph, static_crops, static_focal, outputs, events, _right_indices = self.graphs[sides]
        static_crops.copy_(crops)
        static_focal.copy_(focal)
        graph.replay()
        self.stage_events = events
        poses, identity, scales, translation, joints, vertices = (tensor.clone() for tensor in outputs)
        scale = scales[:, 0]
        scale_parameters = scales[:, 1:]
        rotation = prepared.rotation.float()
        wrist_orients = self.predictor._model_wrist_orients
        assert wrist_orients is not None
        wrist_orientation = torch.from_numpy(np.stack([wrist_orients[side] for side in sides])).to(device=device, dtype=torch.float32)
        joints = joints @ rotation.transpose(-1, -2)
        vertices = vertices @ rotation.transpose(-1, -2)
        translation = (rotation @ translation.unsqueeze(-1)).squeeze(-1)
        poses[:, 0] = wrist_orientation @ rotation @ wrist_orientation.transpose(-1, -2) @ poses[:, 0]
        joints = joints + translation[:, None]
        vertices = vertices + translation[:, None]
        pixels = cropper.project(joints)
        return EgoHandTensors(joints, pixels, vertices, poses, identity, scale, translation, scale_parameters)
