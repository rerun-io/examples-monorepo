"""Bystanders on FullCircle's level faces: the people that are not the operator, in the same detection pass.

FullCircle masks only the operator. The other people come from the faces :meth:`FullCircleMasker.detect` already
renders and runs YOLOX on, instead of a second detector on the squashed raw fisheye:

1. The YOLOX boxes of a frameset (score >= ``box_score``) that the tracked operator mask does not cover (less than
   ``operator_cover`` of the box, looked up in the synthetic view) are bystander candidates.
2. Only the faces that keep a candidate are rendered again at SAM 3's input size from the full-resolution sources,
   and SAM 3 segments ``person`` on them.
3. A SAM 3 instance is kept when its box matches a candidate box (IoU >= ``match_iou``): both detectors must agree,
   which drops SAM 3's small false positives (chairs, a jacket) and YOLOX's (desks).
4. The kept masks map back to each source's mask grid through the face rays (nearest lookup) and are dilated.

No tracking: each tracked frameset is detected on its own (bystanders are re-found every frameset, at 3 Hz). On the
Mission office X5 clip most framesets keep no candidate, so SAM 3 runs on few faces.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.models.base import SegmentationPrompts
from posekit.models.fullcircle import FACE_ROTATIONS, FaceBoxes, FullCircleMasker, OperatorMasks, disk_dilate
from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig
from posekit.ops.sphere import PinholeCamera, SphereView, pixel_grid, sample_sphere, view_from_rays
from posekit.predictions import BoxDetections

COVER_SAMPLES: int = 16
"""Samples across each axis of a box when its operator cover is measured."""


@dataclass(frozen=True, slots=True)
class FacePeopleConfig:
    """Bystander masks from FullCircle's level faces."""

    enabled: bool = True
    """Mask bystanders as well as the operator (``False``: the operator only, and SAM 3 is not loaded)."""
    box_score: float = 0.25
    """Lowest YOLOX score of a bystander candidate (FullCircle's SAM 3 threshold); the operator direction keeps its
    own, higher threshold (:attr:`~posekit.models.fullcircle.FullCircleConfig.operator_score`)."""
    operator_cover: float = 0.25
    """A box whose area the tracked operator mask covers at least this much is the operator, not a bystander."""
    match_iou: float = 0.3
    """Box IoU (face pixels) at which a SAM 3 instance is confirmed by a candidate box."""
    face_size: int = 1008
    """Size of the faces SAM 3 sees (its input size), rendered from the full-resolution sources."""
    sam3: Sam3SegmenterConfig = field(default_factory=lambda: Sam3SegmenterConfig(score_threshold=0.5, dtype="float16", backend="tensorrt"))
    """SAM 3 options: the people stage's threshold and fp16 TensorRT engines."""
    prompt: str = "person"
    """SAM 3 text prompt."""
    dilate_px: float = 8.0
    """Dilation radius of the bystander masks in source-image pixels (the people stage's 8 px)."""


@dataclass(frozen=True, slots=True)
class FacePeopleMasks:
    """One frameset's bystanders."""

    source_masks: list[Bool[Tensor, "mh mw"]]
    """Per source camera: the bystander mask at its ``mask_hw``, dilated (all false without bystanders)."""
    faces: Int64[ndarray, " k"]
    """The faces SAM 3 ran on."""
    face_images: UInt8[Tensor, "k s s 3"]
    """Those faces at ``face_size``."""
    face_masks: Bool[Tensor, "k s s"]
    """The kept instances on them."""
    candidates: Float32[ndarray, "c 4"]
    """Candidate boxes, face pixels of the detection faces (``xyxy``)."""
    candidate_faces: Int64[ndarray, " c"]
    """Face of each candidate."""


def match_instances(candidates: Float32[ndarray, "c 4"], instances: Float32[ndarray, "m 4"], min_iou: float) -> Bool[ndarray, " m"]:
    """Which instance boxes match a candidate box with an IoU of at least ``min_iou`` (all ``xyxy``, same pixels)."""
    if not len(candidates) or not len(instances):
        return np.zeros(len(instances), dtype=bool)
    low: Float32[ndarray, "m c 2"] = np.maximum(instances[:, None, :2], candidates[None, :, :2])
    high: Float32[ndarray, "m c 2"] = np.minimum(instances[:, None, 2:], candidates[None, :, 2:])
    overlap: Float32[ndarray, "m c"] = np.clip(high - low, 0, None).prod(axis=2)
    area_i: Float32[ndarray, " m"] = np.clip(instances[:, 2:] - instances[:, :2], 0, None).prod(axis=1)
    area_c: Float32[ndarray, " c"] = np.clip(candidates[:, 2:] - candidates[:, :2], 0, None).prod(axis=1)
    iou: Float32[ndarray, "m c"] = overlap / np.maximum(area_i[:, None] + area_c[None, :] - overlap, 1e-6)
    return (iou >= min_iou).any(axis=1)


def faces_to_source_mask(
    face_masks: Bool[Tensor, "k s s"], rig_R_faces: Float32[Tensor, "k 3 3"], face: PinholeCamera, mask_rays: Float32[Tensor, "n 3"]
) -> Bool[Tensor, " n"]:
    """Per source mask cell (its ray in the rig frame), whether a face mask holds it: nearest lookup, OR over the faces."""
    size: int = face.size
    hit: Bool[Tensor, " n"] = torch.zeros(mask_rays.shape[0], dtype=torch.bool, device=mask_rays.device)
    for masks, rig_R_face in zip(face_masks, rig_R_faces, strict=True):
        pixels, valid = face.project(mask_rays @ rig_R_face)
        index: Int64[Tensor, "n 2"] = pixels.nan_to_num(0.0).round().long().clamp(0, size - 1)
        hit |= valid & masks[index[:, 1], index[:, 0]]
    return hit


class FacePeople:
    """Bystander masks for the frames a :class:`~posekit.models.fullcircle.FullCircleMasker` tracks."""

    def __init__(self, config: FacePeopleConfig, masker: FullCircleMasker) -> None:
        """Load SAM 3 and plan the face rays at its input size.

        Args:
            config: Bystander options.
            masker: The operator masker whose faces, sources and synthetic view are used.
        """
        self.config: FacePeopleConfig = config
        self.masker: FullCircleMasker = masker
        device: torch.device = masker.device
        self.face: PinholeCamera = PinholeCamera(config.face_size)
        face_rays, _ = self.face.unproject(pixel_grid(config.face_size, config.face_size, device))
        self.face_rays: Float32[Tensor, "n 3"] = face_rays.float()
        self.sam3: Sam3Segmenter = config.sam3.setup()
        self.sam3.prepare(config.prompt)
        self.mask_radius: list[int] = [math.ceil(config.dilate_px / s.mask_scale[0] - 1e-9) for s in masker.sources]
        steps: Float32[Tensor, " q"] = (torch.arange(COVER_SAMPLES, device=device, dtype=torch.float32) + 0.5) / COVER_SAMPLES
        self.cover_grid: Float32[Tensor, "q2 2"] = torch.stack(torch.meshgrid(steps, steps, indexing="xy"), dim=-1).reshape(-1, 2)
        self.sam3_faces: int = 0
        """Faces SAM 3 ran on so far."""
        self.seconds: float = 0.0
        """Wall time of :meth:`__call__` so far (synchronised)."""

    def operator_cover(self, boxes: FaceBoxes, rig_R_level: Float64[ndarray, "3 3"], operator: OperatorMasks) -> Float32[Tensor, " n"]:
        """The fraction of each detection box that the tracked operator mask covers (sampled on a grid in the box)."""
        xyxy: Float32[Tensor, "n 4"] = torch.from_numpy(boxes.xyxy).to(self.masker.device)
        pixels: Float32[Tensor, "n q2 2"] = xyxy[:, None, :2] + self.cover_grid[None] * (xyxy[:, None, 2:] - xyxy[:, None, :2])
        rays, _ = self.masker.face.unproject(pixels.reshape(-1, 2))
        rig_R_faces: Float32[Tensor, "n 3 3"] = torch.from_numpy(rig_R_level @ FACE_ROTATIONS[boxes.faces]).to(self.masker.device, torch.float32)
        rig_rays: Float32[Tensor, "n q2 3"] = torch.einsum("nij,nqj->nqi", rig_R_faces, rays.float().reshape(len(xyxy), -1, 3))
        view_R_rig: Float32[Tensor, "3 3"] = torch.from_numpy(operator.rig_R_view.T).to(self.masker.device, torch.float32)
        synthetic_px, valid = self.masker.synthetic.project(rig_rays.reshape(-1, 3) @ view_R_rig.T)
        size: int = self.masker.config.synthetic_size
        index: Int64[Tensor, "nq 2"] = synthetic_px.nan_to_num(0.0).round().long().clamp(0, size - 1)
        covered: Bool[Tensor, " nq"] = valid & operator.synthetic_mask[index[:, 1], index[:, 0]]
        return covered.reshape(len(xyxy), -1).float().mean(dim=1)

    @torch.inference_mode()
    def __call__(
        self, frames: list[UInt8[Tensor, "h w 3"]], boxes: FaceBoxes, rig_R_level: Float64[ndarray, "3 3"], operator: OperatorMasks
    ) -> FacePeopleMasks:
        """The bystanders of one tracked frameset: ``boxes`` from its detection pass, ``operator`` from its tracking step."""
        start: float = time.perf_counter()
        device: torch.device = self.masker.device
        size: int = self.config.face_size
        is_candidate: Bool[ndarray, " n"] = boxes.scores >= self.config.box_score
        if is_candidate.any():
            is_candidate &= (self.operator_cover(boxes, rig_R_level, operator) < self.config.operator_cover).cpu().numpy()
        candidates: Float32[ndarray, "c 4"] = boxes.xyxy[is_candidate]
        candidate_faces: Int64[ndarray, " c"] = boxes.faces[is_candidate]
        faces: Int64[ndarray, " k"] = np.unique(candidate_faces)
        empty: list[Bool[Tensor, "mh mw"]] = [torch.zeros(s.mask_hw, dtype=torch.bool, device=device) for s in self.masker.sources]
        if not len(faces):
            return FacePeopleMasks(
                empty,
                faces,
                torch.zeros((0, size, size, 3), dtype=torch.uint8, device=device),
                torch.zeros((0, size, size), dtype=torch.bool, device=device),
                candidates,
                candidate_faces,
            )
        rig_R_faces: Float32[Tensor, "k 3 3"] = torch.from_numpy(rig_R_level @ FACE_ROTATIONS[faces]).to(device, torch.float32)
        rays: Float32[Tensor, "k n 3"] = self.face_rays @ rig_R_faces.transpose(1, 2)
        view: SphereView = view_from_rays(
            rays.reshape(-1, 3), len(faces) * size, size, self.masker.cameras, self.masker.rig_R_cams, self.masker.full_hw
        )
        stacked: Float[Tensor, "3 ks s"] = sample_sphere(self.masker.face_images(frames), view)
        face_images: UInt8[Tensor, "k s s 3"] = rearrange(
            stacked.round().clamp(0, 255).to(torch.uint8), "c (k h) w -> k h w c", k=len(faces)
        ).contiguous()
        detections: BoxDetections = self.sam3(
            face_images, SegmentationPrompts(frame_indices=torch.arange(len(faces), device=device), text=self.config.prompt)
        )
        self.sam3_faces += len(faces)
        face_masks: Bool[Tensor, "k s s"] = torch.zeros((len(faces), size, size), dtype=torch.bool, device=device)
        if detections.num_detections:
            assert detections.masks is not None
            # Instance boxes into the detection faces' pixels (same field of view, another size).
            scale: float = self.masker.config.face_size / size
            instance_boxes: Float32[ndarray, "m 4"] = (detections.xyxy.float() * scale).cpu().numpy()
            instance_faces: Int64[ndarray, " m"] = faces[detections.frame_indices.cpu().numpy()]
            keep: Bool[ndarray, " m"] = np.zeros(len(instance_boxes), dtype=bool)
            for face in faces:
                on_face: Bool[ndarray, " m"] = instance_faces == face
                keep[on_face] = match_instances(candidates[candidate_faces == face], instance_boxes[on_face], self.config.match_iou)
            for row in np.flatnonzero(keep):
                face_masks[int(detections.frame_indices[row])] |= detections.masks[row]
        source_masks: list[Bool[Tensor, "mh mw"]] = []
        for index, source in enumerate(self.masker.sources):
            hit: Bool[Tensor, " n"] = faces_to_source_mask(face_masks, rig_R_faces, self.face, self.masker.mask_rays[index])
            mask: Bool[Tensor, "mh mw"] = (hit & self.masker.mask_ray_valid[index]).reshape(source.mask_hw)
            source_masks.append(disk_dilate(mask, self.mask_radius[index]))
        torch.cuda.synchronize(device)
        self.seconds += time.perf_counter() - start
        return FacePeopleMasks(source_masks, faces, face_images, face_masks, candidates, candidate_faces)


__all__ = ("FacePeople", "FacePeopleConfig", "FacePeopleMasks", "faces_to_source_mask", "match_instances")
