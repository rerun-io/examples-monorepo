"""CUDA execution of EgoHand's released network and full SOMA hand reconstruction."""

from dataclasses import dataclass, fields

import torch
import warp as wp
from jaxtyping import Float, Float32, Int64
from soma.geometry.rig_utils import precompute_joint_orient, remove_joint_orient_local
from torch import Tensor

from egohand.models.egohand import EgoHand
from egohand_inference.pipeline.predictor import EgoHandPredictor
from posekit.models.egohand_capture import capture_hand_graph
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


@dataclass(frozen=True, slots=True)
class _VirtualHand:
    """Captured full-hand state before restoring source-camera axes."""

    poses: Float32[Tensor, "n 25 3 3"]
    """SOMA rotations in virtual-camera wrist axes."""
    identity: Float32[Tensor, "n 5"]
    """Predicted identity coefficients."""
    scales: Float32[Tensor, "n 27"]
    """Uniform scale followed by 26 MHR scales."""
    translation: Float32[Tensor, "n 3"]
    """Virtual-camera metric translation."""
    joints: Float32[Tensor, "n 25 3"]
    """Untranslated full SOMA joints."""
    vertices: Float32[Tensor, "n v 3"]
    """Untranslated full SOMA vertices."""


@dataclass(frozen=True, slots=True)
class _CapturedHand:
    """Graph inputs, outputs and external resources with one lifetime."""

    graph: torch.cuda.CUDAGraph
    """Captured backbone, heads and SOMA work."""
    crops: Float32[Tensor, "n 3 256 256"]
    """Stable image input storage."""
    focal: Float32[Tensor, "n"]
    """Stable virtual-camera focal input."""
    output: _VirtualHand
    """Captured model output storage."""
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]
    """External timing events retained until graph destruction."""
    right_indices: Int64[Tensor, "r"]
    """Captured indexing storage for mirrored hands."""
    wrist_orientation: Float32[Tensor, "n 3 3"]
    """Static hand-dependent orientation for restoring source-camera axes."""


class EgoHandTensorExecutor:
    """Keep inference tensors on the device through restoration to the source camera."""

    def __init__(self, predictor: EgoHandPredictor, head_runtime: TensorRuntime | None = None) -> None:
        self.predictor: EgoHandPredictor = predictor
        self.head_runtime: TensorRuntime | None = head_runtime
        self.head: EgoHandHead = EgoHandHead(predictor.model).eval()
        prepare_cuda_soma(predictor.model.soma_hand)
        right = predictor.model.soma_hand.right
        self.right_parents: Int64[Tensor, "25"] = torch.as_tensor(right.joint_parent_ids, device="cuda", dtype=torch.int64)
        self.right_orient, self.right_parent_orient = precompute_joint_orient(right.t_pose_world, right.joint_parent_ids)
        self.reflection: Float32[Tensor, "3"] = torch.tensor([-1.0, 1.0, 1.0], device="cuda")
        self.graphs: dict[tuple[int, int], _CapturedHand] = {}
        """Captured hands by hand count and right-hand count; left hands come first in every graph."""
        self.stage_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        assert predictor._model_wrist_orients is not None
        self.wrist_orientations: dict[HandSide, Float32[Tensor, "3 3"]] = {
            side: torch.as_tensor(orientation, device="cuda", dtype=torch.float32)
            for side, orientation in predictor._model_wrist_orients.items()
        }
        self.stage_ms: dict[str, float] = {}
        self.batch_size: int = min(predictor.batch_size, head_runtime.spec.max_batch_size) if head_runtime is not None else predictor.batch_size

    def __del__(self) -> None:
        """Destroy captured work while its external events and model buffers are still alive."""
        for state in getattr(self, "graphs", {}).values():
            state.graph.reset()

    def _run(
        self, crops: Float32[Tensor, "n 3 256 256"], focal: Float32[Tensor, "n"], right_indices: Int64[Tensor, "r"],
        events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> _VirtualHand:
        """Execute the released heads and full mesh using retained device inputs."""
        model: EgoHand = self.predictor.model
        model.soma_hand.profile_events = None
        events["backbone"][0].record()
        features: Float[Tensor, "n 1280 16 16"] = model.backbone(crops)
        events["backbone"][1].record()
        events["head"][0].record()
        if self.head_runtime is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                poses, identity, scales, camera, direct, presence = self.head(features, focal)
        else:
            output: dict[str, Tensor] = self.head_runtime({"features": features, "focal": focal})
            poses, identity, scales, camera, direct, presence = (output[k] for k in ("poses", "identity", "scales", "camera", "direct", "presence"))
        events["head"][1].record()
        poses = poses.clone()
        events["soma"][0].record()
        left = model.soma_hand(poses, identity, "left", pose2rot=False, global_scale=scales[:, 0], scale_params=scales[:, 1:])
        joints: Float32[Tensor, "n 25 3"] = left["joints"].clone()
        vertices: Float32[Tensor, "n v 3"] = left["vertices"].clone()
        translation: Float32[Tensor, "n 3"] = solve_virtual_translation(joints, camera, direct, presence, focal)
        if len(right_indices):
            # Reuse left FK already computed for the shape and translation solve.
            rotations: Float32[Tensor, "r 25 3 3"] = left.transforms[right_indices, :, :3, :3] * -self.reflection[None, None, :, None]
            local: Float32[Tensor, "r 25 3 3"] = rotations[:, self.right_parents].transpose(-1, -2) @ rotations
            local[:, 0] = rotations[:, 0]
            right_poses: Float32[Tensor, "r 25 3 3"] = remove_joint_orient_local(local, self.right_orient, self.right_parent_orient)
            right = model.soma_hand(right_poses, identity[right_indices], "right", pose2rot=False,
                                    global_scale=scales[right_indices, 0], scale_params=scales[right_indices, 1:])
            joints[right_indices] = right["joints"]
            vertices[right_indices] = right["vertices"]
            translation[right_indices] = (left.transforms[right_indices, 0, :3, 3] + translation[right_indices]) * self.reflection
            poses[right_indices] = right_poses
        events["soma"][1].record()
        return _VirtualHand(poses, identity, scales, translation, joints, vertices)

    @torch.inference_mode()
    def __call__(self, prepared: PreparedHandCrops, sides: tuple[HandSide, ...], cropper: CudaHandCropper) -> EgoHandTensors:
        """Run calibrated crops and return a source-camera reconstruction on CUDA."""
        self.stage_ms.clear()
        batches: list[EgoHandTensors] = []
        for start in range(0, len(sides), self.batch_size):
            stop: int = min(start + self.batch_size, len(sides))
            chunk: PreparedHandCrops = PreparedHandCrops(prepared.images[start:stop], prepared.focal[start:stop], prepared.rotation[start:stop])
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
        """Replay one bounded batch with its left hands first, then restore the caller's hand order.

        Retained graph buffers remain live until the executor is released.
        """
        right_count: int = sum(side == "right" for side in sides)
        left_first: list[int] = sorted(range(len(sides)), key=lambda index: sides[index] == "right")
        if left_first == list(range(len(sides))):
            return self._run_left_first(prepared, right_count, cropper)
        order: Int64[Tensor, "n"] = torch.tensor(left_first, device=prepared.images.device)
        reordered: PreparedHandCrops = PreparedHandCrops(prepared.images[order], prepared.focal[order], prepared.rotation[order])
        result: EgoHandTensors = self._run_left_first(reordered, right_count, cropper)
        restore: Int64[Tensor, "n"] = torch.argsort(order)
        return EgoHandTensors(*(getattr(result, item.name)[restore] for item in fields(EgoHandTensors)))

    def _run_left_first(self, prepared: PreparedHandCrops, right_count: int, cropper: CudaHandCropper) -> EgoHandTensors:
        """Replay the graph for this many hands, the last ``right_count`` of them right hands."""
        crops: Float32[Tensor, "n 3 256 256"] = prepared.images
        device: torch.device = crops.device
        focal: Float32[Tensor, "n"] = prepared.focal.float()
        count: int = len(crops)
        key: tuple[int, int] = (count, right_count)
        if key not in self.graphs:
            static_crops: Float32[Tensor, "n 3 256 256"] = crops.clone()
            static_focal: Float32[Tensor, "n"] = focal.clone()
            right_indices: Int64[Tensor, "r"] = torch.arange(count - right_count, count, device=device, dtype=torch.int64)
            events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {name: (torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True))
                      for name in ("backbone", "head", "soma")}
            stream: torch.cuda.Stream = torch.cuda.Stream()
            warp_stream: wp.Stream = wp.stream_from_torch(stream)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
                for _ in range(3):
                    self._run(static_crops, static_focal, right_indices, events)
            torch.cuda.current_stream().wait_stream(stream)
            graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
            with capture_hand_graph(graph, stream, warp_stream):
                outputs: _VirtualHand = self._run(static_crops, static_focal, right_indices, events)
            wrist_orientation: Float32[Tensor, "n 3 3"] = torch.stack(
                [self.wrist_orientations["left"]] * (count - right_count) + [self.wrist_orientations["right"]] * right_count
            )
            self.graphs[key] = _CapturedHand(graph, static_crops, static_focal, outputs, events, right_indices, wrist_orientation)
        state: _CapturedHand = self.graphs[key]
        state.crops.copy_(crops)
        state.focal.copy_(focal)
        state.graph.replay()
        self.stage_events = state.events
        poses: Float32[Tensor, "n 25 3 3"] = state.output.poses.clone()
        identity: Float32[Tensor, "n 5"] = state.output.identity.clone()
        scales: Float32[Tensor, "n 27"] = state.output.scales.clone()
        translation: Float32[Tensor, "n 3"] = state.output.translation.clone()
        joints: Float32[Tensor, "n 25 3"] = state.output.joints.clone()
        vertices: Float32[Tensor, "n v 3"] = state.output.vertices.clone()
        scale: Float32[Tensor, "n"] = scales[:, 0]
        scale_parameters: Float32[Tensor, "n 26"] = scales[:, 1:]
        rotation: Float32[Tensor, "n 3 3"] = prepared.rotation.float()
        joints = joints @ rotation.transpose(-1, -2)
        vertices = vertices @ rotation.transpose(-1, -2)
        translation = (rotation @ translation.unsqueeze(-1)).squeeze(-1)
        poses[:, 0] = state.wrist_orientation @ rotation @ state.wrist_orientation.transpose(-1, -2) @ poses[:, 0]
        joints = joints + translation[:, None]
        vertices = vertices + translation[:, None]
        pixels: Float32[Tensor, "n 25 2"] = cropper.project(joints)
        return EgoHandTensors(joints, pixels, vertices, poses, identity, scale, translation, scale_parameters)
