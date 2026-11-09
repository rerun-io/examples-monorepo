"""GPU body crops, batched hand inference and complete SAM body reconstruction."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody
from torch import Tensor, nn

from posekit.models.sam3d_body import Sam3dBodyStage, _StageClock
from posekit.models.sam3d_body_capture import _DecoderTensorMode
from posekit.models.sam3d_body_crops import BodyCropper
from posekit.models.sam3d_body_head import CaptureMhrHead
from posekit.models.sam3d_body_tensors import BodyEstimate, BodyHandFusion
from posekit.ops.cuda_graph import CapturedGraph, GraphPool
from posekit.predictions import BoxDetections

FrameOutput: TypeAlias = tuple[BodyEstimate, Tensor | None, Tensor | None]
FrameKey: TypeAlias = tuple[tuple[int, ...], int, bool, bool, bool, bool]
StageEvent: TypeAlias = tuple[Sam3dBodyStage, torch.cuda.Event, torch.cuda.Event]


@dataclass(slots=True)
class CapturedBodyFrame:
    """One whole-frame graph and the optional timing events recorded inside it."""

    owner: CapturedGraph[FrameOutput]
    """Owns input/output storage and all captured execution resources."""
    events: tuple[StageEvent, ...]
    """Empty for ordinary inference; retained for profiling graph replays."""


class BodyTensorExecutor:
    """Run body first, then both dependent hand crops in one backbone/decoder batch."""

    def __init__(
        self, model: SAM3DBody, head: CaptureMhrHead, clock: _StageClock, *,
        capture: bool = False, max_people: int = 8, hand_backbone: nn.Module | None = None,
    ) -> None:
        self.model: SAM3DBody = model
        self.clock: _StageClock = clock
        self.cropper: BodyCropper = BodyCropper(head.scale_mean.device)
        self.fusion: BodyHandFusion = BodyHandFusion(head)
        self.image_mean: Tensor = model.get_buffer("image_mean")
        self.image_std: Tensor = model.get_buffer("image_std")
        self.grid: Tensor = torch.stack(torch.meshgrid(torch.arange(512, device=head.scale_mean.device), torch.arange(512, device=head.scale_mean.device), indexing="xy"), 0).float()
        self.body_backbone: nn.Module = model.backbone
        self.hand_backbone: nn.Module = model.backbone if hand_backbone is None else hand_backbone
        self.capture: bool = capture
        self.max_people: int = max_people
        self.pool: GraphPool = torch.cuda.graph_pool_handle()
        self.graphs: dict[FrameKey, CapturedBodyFrame] = {}
        self.condition_indices: _DecoderTensorMode = _DecoderTensorMode()

    def _branch(self, batch: dict[str, Tensor], *, hand: bool) -> tuple[Tensor, Tensor, Tensor, BodyEstimate]:
        model: SAM3DBody = self.model
        model._initialize_batch(batch)
        count: int = batch["img"].shape[1]
        model.body_batch_idx = [] if hand else list(range(count))
        model.hand_batch_idx = list(range(count)) if hand else []
        images: Tensor = batch["img"][0]
        normalized: Tensor = (images - self.image_mean) / self.image_std
        affine: Tensor = batch["affine_trans"][0]
        diagonal: Tensor = affine.diagonal(dim1=-2, dim2=-1)[..., None, None]
        offset: Tensor = affine[..., 2, None, None]
        camera: Tensor = batch["cam_int"][0]
        principal: Tensor = camera[:2, 2, None, None]
        focal: Tensor = camera.diagonal()[:2, None, None]
        rays: Tensor = ((self.grid[None] / diagonal - offset / diagonal) - principal) / focal
        batch["ray_cond_hand" if hand else "ray_cond"] = rays
        self.clock.current = "hands" if hand else "body"
        backbone: nn.Module = self.hand_backbone if hand else self.body_backbone
        features: Tensor = backbone(normalized.to(model.backbone_dtype)).float()
        features = features + model._get_mask_prompt(batch, features)
        with self.condition_indices:
            condition: Tensor | None = model._get_decoder_condition(batch)
        assert condition is not None
        prompts: Tensor = images.new_zeros((count, 1, 3))
        prompts[..., 2] = -2
        tokens, outputs = (model.forward_decoder_hand if hand else model.forward_decoder)(
            features, keypoints=prompts, condition_info=condition, batch=batch,
        )
        estimate: BodyEstimate = BodyEstimate.from_decoder(outputs[-1])
        return features, condition, tokens, estimate

    def run(
        self, frames: Tensor, detections: BoxDetections, camera: Tensor | None, *, full: bool,
    ) -> FrameOutput:
        """Capture by shape, then replay with owned outputs and no data-dependent host checks."""
        if not self.capture or detections.num_detections > self.max_people:
            return self._run(frames, detections, camera, full=full)
        key: FrameKey = (tuple(frames.shape), detections.num_detections, detections.masks is not None, camera is not None, full, self.clock.enabled)
        values: tuple[Tensor, ...] = (frames, detections.xyxy, detections.scores, detections.frame_indices)
        if detections.masks is not None:
            values += (detections.masks,)
        if camera is not None:
            values += (camera,)
        if key not in self.graphs:
            storage: tuple[Tensor, ...] = tuple(value.clone() for value in values)
            boxes = BoxDetections(storage[1], storage[2], storage[3], None if detections.masks is None else storage[4])
            recorded: list[StageEvent] = []

            def execute() -> FrameOutput:
                self.clock.events.clear()
                output = self._run(storage[0], boxes, None if camera is None else storage[-1], full=full)
                recorded[:] = self.clock.events
                return output

            owner = CapturedGraph(execute, storage, pool=self.pool, keep=(self, boxes))
            self.graphs[key] = CapturedBodyFrame(owner, tuple(recorded))
        captured: CapturedBodyFrame = self.graphs[key]
        output = captured.owner.replay(*values)
        self.clock.events[:] = captured.events
        body, left, right = output
        return body.select(slice(None)), None if left is None else left.clone(), None if right is None else right.clone()

    def _run(
        self, frames: Tensor, detections: BoxDetections, camera: Tensor | None, *, full: bool,
    ) -> FrameOutput:
        """Return untranslated estimates and expanded source-space hand crop boxes."""
        count: int = detections.num_detections
        width: int = frames.shape[2]
        batch: dict[str, Tensor] = self.cropper.prepare(
            frames, detections.xyxy, detections.frame_indices, padding=1.25,
            masks=None if detections.masks is None else detections.masks.float(), scores=detections.scores, camera=camera,
        )
        features, condition, tokens, body = self._branch(batch, hand=False)
        if not full:
            return body, None, None
        hand_boxes: Tensor = self.model.bbox_embed(tokens).sigmoid() * 512
        affine: Tensor = batch["affine_trans"][0]
        centers: Tensor = (hand_boxes[..., :2] - affine[:, None, :, 2]) / affine[:, None, 0, 0, None]
        sizes: Tensor = hand_boxes[..., 2:].amax(-1, keepdim=True) / affine[:, None, 0, 0, None]
        boxes: Tensor = torch.cat((centers - sizes / 2, centers + sizes / 2), -1)
        left_box: Tensor = boxes[:, 0].clone()
        left_box[:, 0] = width - boxes[:, 0, 2] - 1
        left_box[:, 2] = width - boxes[:, 0, 0] - 1
        hand_batch: dict[str, Tensor] = self.cropper.prepare(
            frames, torch.cat((left_box, boxes[:, 1])), detections.frame_indices.repeat(2), padding=0.9,
            camera=camera, mirrored=torch.cat((torch.ones(count, dtype=torch.bool, device=frames.device), torch.zeros(count, dtype=torch.bool, device=frames.device))),
        )
        _, _, _, hands = self._branch(hand_batch, hand=True)
        left: BodyEstimate = hands.select(slice(0, count))
        right: BodyEstimate = hands.select(slice(count, None))
        multiple_people: Tensor = (detections.frame_indices[:, None] == detections.frame_indices[None, :]).sum(1) > 1
        prompts, valid, original, global_wrist = self.fusion.prepare_prompts(body, left, right, batch, hand_batch, width, multiple_people)
        self.model._initialize_batch(batch)
        self.model.body_batch_idx = list(range(count))
        self.model.hand_batch_idx = []
        self.clock.current = "prompt"
        previous: Tensor = torch.cat((body.pred_pose_raw, body.shape, body.scale, body.hand, body.face, body.pred_cam), 1)[:, None]
        _, outputs = self.model.forward_decoder(features, keypoints=prompts, prev_estimate=previous, condition_info=condition, batch=batch)
        refined: BodyEstimate = BodyEstimate.from_decoder(outputs[-1])
        body = refined.choose((prompts[..., 2] != -3).any(1), body)
        body = self.fusion.merge(body, left, right, valid, original, global_wrist, batch["ori_img_size"][0, 0])
        hand_centers: Tensor = hand_batch["bbox_center"][0].clone()
        hand_centers[:count, 0] = width - hand_centers[:count, 0] - 1
        hand_scales: Tensor = hand_batch["bbox_scale"][0]
        expanded: Tensor = torch.cat((hand_centers - hand_scales / 2, hand_centers + hand_scales / 2), -1)
        return body, expanded[:count], expanded[count:]
