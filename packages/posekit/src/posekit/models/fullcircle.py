"""FullCircle operator masks: find the person holding a 360 camera, track them in a view centred on them, map back.

FullCircle (Foroutan et al., "FullCircle: Effortless 3D Reconstruction from Casual 360 Captures", arXiv 2603.22572;
code theialab/fullcircle ``masking/`` at f8941d3 and theialab/sam-ui at c38098f) masks the camera operator with no
training and no manual prompt. Their eight steps, and where each lives here:

1. Omni frame -> the six 90-degree cube faces (``omni2perspectives.py``): :meth:`FullCircleMasker.detect` renders the
   faces straight from the source cameras through their rays (no intermediate panorama resample), in a gravity-level
   frame given per frameset (FullCircle's omni frames are Insta360 Studio panoramas, which are horizon-levelled).
2. A person detector on every face (``mask_perspectives.py``: SAM 3, text ``person``, conf 0.25; the paper used
   YOLOv8): posekit's YOLOX-HumanArt (TensorRT) by default, SAM 3 ``person`` as the other choice.
3. The detections -> one operator direction (``perspectives2omni.py`` -> ``centers/<frame>.csv``, read by
   ``omni2synthetic.py``): the area-weighted mean of the detection centres on the sphere. Their per-face module
   ``perspectives2omni_two_cubes`` is missing from the release, so the centre rows are taken as one row per detection
   (centre, area), which is what ``omni2synthetic.py`` consumes. A frame with no detection reuses the nearest frame
   that has one (earlier first at equal distance), as ``omni2synthetic.py`` does: :func:`fill_nearest`.
4. A synthetic fisheye looking along that direction (``omni2synthetic.py``: ``look_at_camZ`` with up ``+Y``, the
   rear lens of ``mapping.txt`` as an OPENCV_FISHEYE at 2880 px with a 107.5 px border): the same lens scaled to the
   tracker's input size, with one up hint per sequence (:func:`view_up`).
5. SAM 2 on the synthetic fisheye sequence, one positive click at the image centre on the first frame, propagated
   forward (sam-ui ``tracking_gui.py --headless``, SAM 2.1 Hiera-L at 512 px): posekit's streaming EfficientTAM
   (``Sam2VideoSegmenter``), prompted with the detector's box around the view centre on the first frameset and
   again every ``reprompt_every`` framesets (:meth:`FullCircleMasker._operator_prompt`). sam-ui clicks
   ``(1440, 1440)`` -- the centre of a 2880 px frame -- on frames it has resized to 512 px, so its click lies outside
   the image; the paper's intent is the image centre.
6.-7. Synthetic mask -> omni -> raw fisheye masks (``synthetic2omni.py``, ``omni2fisheyes.py``): every source mask
   pixel looks its ray up in the synthetic mask (nearest), in one step.
8. Dilation (``dilate.py``: 9x9 ellipse, 1 iteration at 2880 px, "about 4 pixels" in the paper): a disk of
   ``dilate_px`` image pixels, applied at mask resolution.

Four changes, each forced by a measured failure (FullCircle's room3 scene: IoU against the masks FullCircle released;
clip 007: a 17.8 min Insta360 X4 Air walk with the camera on a stick above the operator):

- Box prompt, not the centre click: EfficientTAM (FullCircle used SAM 2.1 Hiera-L) segments only the shirt on a torso
  click (room3 IoU 0.20); with the box, 0.958 (0.945 with the re-prompts and memory bound below).
- Re-prompts: a single prompt drifted onto other people within 10 s of clip 007.
- One up hint per sequence: with the operator below the camera (near +Y) the look-at fallback flipped the view.
- Level faces: lens 0's +Y is not down on the X4 Air, and people lying sideways in the faces were missed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor
from trtkit.backends import TensorRtBackendConfig

from posekit.models.base import SegmentationPrompts
from posekit.models.sam2_video import Sam2VideoSegmenter, Sam2VideoSegmenterConfig
from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig
from posekit.models.yolox import YoloxDetector, YoloxDetectorConfig
from posekit.ops.sphere import (
    Kb4Camera,
    PinholeCamera,
    SphereCamera,
    SphereView,
    cube_face_rotations,
    look_at,
    pixel_grid,
    sample_sphere,
    view_from_rays,
)
from posekit.predictions import BoxDetections

OperatorDetector: TypeAlias = Literal["yolox", "sam3"]

FULLCIRCLE_SYNTHETIC_LENS: Kb4Camera = Kb4Camera(
    774.971, 775.267, 1440.0, 1440.0, (0.0908432, -0.0450016, 0.017596, -0.00323419), radius_px=1440 - 107.5
)
"""FullCircle's synthetic fisheye: the rear lens of ``masking/mapping.txt`` (2880 px) inside ``radius_r = 107``."""
FULLCIRCLE_SYNTHETIC_SIZE: int = 2880
"""Image size the synthetic lens is stated at."""
FACE_ROTATIONS: Float64[ndarray, "6 3 3"] = cube_face_rotations()
"""``rig_R_face`` of the six detection faces."""
VIEW_UP: Float64[ndarray, "3"] = np.array([0.0, 1.0, 0.0])
"""The up hint of the synthetic view (+Y of the rig frame, as in ``omni2synthetic.py``)."""


@dataclass(frozen=True, slots=True)
class SphereSource:
    """One source camera of the 360 rig, all sharing one centre."""

    camera: SphereCamera
    """Its ray <-> pixel model at full image resolution."""
    rig_R_cam: Float64[ndarray, "3 3"]
    """Rotation of camera rays into the rig frame."""
    image_hw: tuple[int, int]
    """Full image height and width, pixels."""
    mask_hw: tuple[int, int]
    """Height and width of the mask written for this camera (a whole-number fraction of the image is not required)."""

    @property
    def mask_scale(self) -> tuple[float, float]:
        """Image pixels per mask cell ``(sx, sy)``: the grid the masks are sampled on and the scale written with them."""
        return (self.image_hw[1] / self.mask_hw[1], self.image_hw[0] / self.mask_hw[0])


@dataclass(frozen=True, slots=True)
class FullCircleConfig:
    """FullCircle operator masking options."""

    detector: OperatorDetector = "yolox"
    """Person detector on the cube faces: posekit YOLOX-HumanArt (TensorRT) or SAM 3 ``person``."""
    yolox: YoloxDetectorConfig = field(default_factory=lambda: YoloxDetectorConfig(backend=TensorRtBackendConfig()))
    """YOLOX variant, backend and thresholds."""
    sam3: Sam3SegmenterConfig = field(default_factory=lambda: Sam3SegmenterConfig(score_threshold=0.25))
    """SAM 3 options (FullCircle runs it at conf 0.25)."""
    sam3_prompt: str = "person"
    """SAM 3 text prompt."""
    tracker: Sam2VideoSegmenterConfig = field(
        default_factory=lambda: Sam2VideoSegmenterConfig(variant="efficienttam-s-512", max_conditional_memories=4, single_click_multimask=True)
    )
    """The video tracker on the synthetic fisheye; it attends to the 4 nearest re-prompted frames (all of them made the
    17.8 min clip 007 slow down along the clip), and the fallback centre click takes SAM 2's best-of-three mask."""
    reprompt_every: int = 6
    """Tracked framesets between two re-prompts of the tracker with the detector's box (0: FullCircle's single prompt)."""
    reprompt_radius: float = 0.15
    """A re-prompt box that does not hold the view centre must have its centre within this fraction of the view size."""
    face_size: int = 640
    """Cube face size, pixels (YOLOX's input size)."""
    synthetic_size: int = 512
    """Synthetic fisheye size, pixels (sam-ui resizes to 512)."""
    work_size: int = 1280
    """Source images are area-downscaled by a whole factor to at most this size before resampling (anti-aliasing)."""
    dilate_px: float = 4.0
    """Dilation radius in source-image pixels (FullCircle: about 4 px at 2880 px)."""
    device: str = "cuda"
    """Inference device."""


@dataclass(frozen=True, slots=True)
class OperatorEvidence:
    """What the face detector saw in one frameset."""

    direction: Float64[ndarray, "3"] | None
    """Area-weighted mean direction of the detections in the rig frame (unit), ``None`` without detections."""
    boxes: Float32[ndarray, "n 4"]
    """Each detection's face-pixel box ``xyxy``."""
    faces: Int64[ndarray, " n"]
    """Face index of each detection."""
    face_images: UInt8[Tensor, "6 s s 3"]
    """The six faces as the detector saw them."""


@dataclass(frozen=True, slots=True)
class OperatorMasks:
    """One tracked frameset's output."""

    source_masks: list[Bool[Tensor, "mh mw"]]
    """Per source camera: the operator mask at its ``mask_hw``, dilated."""
    synthetic_rgb: UInt8[Tensor, "s s 3"]
    """The synthetic fisheye the tracker saw."""
    synthetic_mask: Bool[Tensor, "s s"]
    """The tracker's mask on it."""


def fill_nearest(directions: list[Float64[ndarray, "3"] | None]) -> list[Float64[ndarray, "3"]]:
    """Give every frame without a direction the nearest frame's direction (earlier first at equal distance).

    Raises:
        ValueError: If no frame has a direction.
    """
    known: list[int] = [i for i, d in enumerate(directions) if d is not None]
    if not known:
        raise ValueError("the detector found no person in any frame, so there is no operator direction")
    filled: list[Float64[ndarray, "3"]] = []
    for i, direction in enumerate(directions):
        if direction is not None:
            filled.append(direction)
            continue
        nearest: int = min(known, key=lambda j: (abs(j - i), j > i))
        found: Float64[ndarray, "3"] | None = directions[nearest]
        assert found is not None
        filled.append(found)
    return filled


def level_rotation(down: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """``rig_R_level`` of the level frame: +Y along ``down`` (gravity, rig frame), +Z the rig's +Z made horizontal."""
    y: Float64[ndarray, "3"] = down / np.linalg.norm(down)
    forward: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ y)) > 0.99:
        forward = np.array([1.0, 0.0, 0.0])
    z: Float64[ndarray, "3"] = forward - (forward @ y) * y
    z = z / np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)


def view_up(directions: list[Float64[ndarray, "3"]]) -> Float64[ndarray, "3"]:
    """The synthetic view's up hint for a sequence (level frame): FullCircle's +Y, or +Z when the operator sits near +Y.

    FullCircle's look-at falls back to +X only within about 8 degrees of the up hint, so an operator below a camera
    held on a stick (direction near the nadir, +Y) flips the view's roll from frame to frame. When the median
    operator direction lies within 45 degrees of +Y, the up hint is +Z (lens 0's axis) for the whole sequence.
    """
    median: Float64[ndarray, "3"] = np.median(np.stack(directions), axis=0)
    median = median / np.linalg.norm(median)
    return np.array([0.0, 0.0, 1.0]) if abs(float(median @ VIEW_UP)) > math.cos(math.pi / 4) else VIEW_UP


def disk_dilate(mask: Bool[Tensor, "h w"], radius: int) -> Bool[Tensor, "h w"]:
    """Binary dilation by a disk of ``radius`` pixels (0: unchanged)."""
    if radius <= 0:
        return mask
    offsets: Float32[Tensor, " k"] = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    disk: Float32[Tensor, "k k"] = ((offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius * radius + 1e-6).float()
    hits: Float32[Tensor, "1 1 h w"] = F.conv2d(mask[None, None].float(), disk[None, None], padding=radius)
    return hits[0, 0] > 0.5


class FullCircleMasker:
    """FullCircle's operator masking over a set of co-centred source cameras (two fisheye lenses or one panorama)."""

    def __init__(self, config: FullCircleConfig, sources: list[SphereSource]) -> None:
        """Load the detector and the tracker and plan every static resampling.

        Args:
            config: Model and resampling options.
            sources: The rig's cameras.
        """
        self.config: FullCircleConfig = config
        self.sources: list[SphereSource] = sources
        device: torch.device = torch.device(config.device)
        self.device: torch.device = device
        # Sources see the work images, whose pixel (i, j) covers full pixels [f i - 0.5, f (i + 1) - 0.5): grids are
        # normalised by the full size, which grid_sample maps onto any resolution of the same image.
        self.downscale: list[int] = [max(1, math.ceil(max(s.image_hw) / config.work_size)) for s in sources]
        self.full_hw: list[tuple[int, int]] = [s.image_hw for s in sources]
        # Ray tables are solved in float64 and kept in float32, as is every per-frame projection through them.
        rig_R_cams: Float64[Tensor, "n_src 3 3"] = torch.from_numpy(np.stack([s.rig_R_cam for s in sources])).to(device)
        self.rig_R_cams: Float32[Tensor, "n_src 3 3"] = rig_R_cams.float()
        self.cameras: list[SphereCamera] = [s.camera for s in sources]
        self.face: PinholeCamera = PinholeCamera(config.face_size)
        face_rays, _ = self.face.unproject(pixel_grid(config.face_size, config.face_size, device))
        self.face_rays: Float32[Tensor, "n 3"] = face_rays.float()

        self.synthetic: Kb4Camera = FULLCIRCLE_SYNTHETIC_LENS.scaled(config.synthetic_size / FULLCIRCLE_SYNTHETIC_SIZE)
        synthetic_rays, synthetic_valid = self.synthetic.unproject(pixel_grid(config.synthetic_size, config.synthetic_size, device))
        self.synthetic_rays: Float32[Tensor, "n 3"] = synthetic_rays.float()
        self.synthetic_valid: Bool[Tensor, "s s"] = synthetic_valid.reshape(config.synthetic_size, config.synthetic_size)

        # Each mask pixel's ray in the rig frame (static), and the dilation radius at mask resolution (across).
        self.mask_rays: list[Float32[Tensor, "n 3"]] = []
        self.mask_ray_valid: list[Bool[Tensor, " n"]] = []
        self.mask_radius: list[int] = []
        for index, source in enumerate(sources):
            rays, valid = source.camera.unproject(pixel_grid(source.mask_hw[0], source.mask_hw[1], device, scale=source.mask_scale))
            self.mask_rays.append((rays @ rig_R_cams[index].T).float())
            self.mask_ray_valid.append(valid)
            self.mask_radius.append(math.ceil(config.dilate_px / source.mask_scale[0] - 1e-9))

        self.yolox: YoloxDetector | None = config.yolox.setup() if config.detector == "yolox" else None
        self.sam3: Sam3Segmenter | None = config.sam3.setup() if config.detector == "sam3" else None
        self.tracker: Sam2VideoSegmenter = config.tracker.setup()
        self.steps: int = 0
        """Tracked framesets so far."""
        self.prompts: int = 0
        """Tracker prompts given so far (the first one included)."""
        self.seconds: dict[str, float] = {"faces": 0.0, "detect": 0.0, "synthetic": 0.0, "track": 0.0, "backmap": 0.0}
        """Synchronised wall time per step since construction."""

    def _work_images(self, frames: list[UInt8[Tensor, "h w 3"]]) -> list[Float[Tensor, "3 hw ww"]]:
        images: list[Tensor] = []
        for frame, factor in zip(frames, self.downscale, strict=True):
            chw: Float[Tensor, "3 h w"] = rearrange(frame, "h w c -> c h w").float()
            images.append(F.avg_pool2d(chw[None], factor)[0] if factor > 1 else chw)
        return images

    def _detect_people(self, images: UInt8[Tensor, "b s s 3"]) -> BoxDetections:
        if self.yolox is not None:
            return self.yolox(images)
        assert self.sam3 is not None
        return self.sam3(images, SegmentationPrompts(frame_indices=torch.arange(len(images), device=self.device), text=self.config.sam3_prompt))

    def _tick(self, key: str, start: float) -> float:
        torch.cuda.synchronize(self.device)
        now: float = time.perf_counter()
        self.seconds[key] += now - start
        return now

    @torch.inference_mode()
    def detect(self, frames: list[UInt8[Tensor, "h w 3"]], rig_R_level: Float64[ndarray, "3 3"]) -> OperatorEvidence:
        """Steps 1-3 on one frameset: cube faces, person detection, the operator direction (rig frame).

        The faces are those of the level frame (``rig_R_level``, +Y along gravity), so people stand upright in them, as
        they do in the horizon-levelled panoramas FullCircle starts from (identity for an already level source).
        """
        start: float = time.perf_counter()
        size: int = self.config.face_size
        images: list[Tensor] = self._work_images(frames)
        rig_R_faces: Float64[Tensor, "6 3 3"] = torch.from_numpy(rig_R_level @ FACE_ROTATIONS).to(self.device)
        # The six faces stacked into one 6s x s view: one sampling plan per frameset.
        rays: Float32[Tensor, "6 n 3"] = self.face_rays @ rig_R_faces.float().transpose(1, 2)
        view: SphereView = view_from_rays(rays.reshape(-1, 3), 6 * size, size, self.cameras, self.rig_R_cams, self.full_hw)
        stacked: UInt8[Tensor, "3 f s"] = sample_sphere(images, view).round().clamp(0, 255).to(torch.uint8)
        faces: UInt8[Tensor, "6 s s 3"] = rearrange(stacked, "c (f h) w -> f h w c", f=6).contiguous()
        start = self._tick("faces", start)
        detections: BoxDetections = self._detect_people(faces)
        boxes: Float32[Tensor, "n 4"] = detections.xyxy
        if self.yolox is not None:
            centres_px: Float[Tensor, "n 2"] = (boxes[:, :2] + boxes[:, 2:]) / 2
            areas: Float[Tensor, " n"] = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
        else:
            assert detections.masks is not None
            masks: Float[Tensor, "n s s"] = detections.masks.float()
            areas = masks.sum(dim=(1, 2))
            coords: Float[Tensor, " s"] = torch.arange(self.config.face_size, device=self.device, dtype=torch.float32)
            centres_px = (
                torch.stack([(masks.sum(dim=1) * coords).sum(dim=1), (masks.sum(dim=2) * coords).sum(dim=1)], dim=1) / areas.clamp_min(1)[:, None]
            )
        rays, _ = self.face.unproject(centres_px.double())
        rotations: Float64[Tensor, "n 3 3"] = rig_R_faces[detections.frame_indices]
        centres: Float64[ndarray, "n 3"] = torch.einsum("nij,nj->ni", rotations, rays).cpu().numpy()
        weights: Float64[ndarray, " n"] = areas.double().cpu().numpy()
        direction: Float64[ndarray, "3"] | None = None
        if len(weights):
            mean: Float64[ndarray, "3"] = np.average(centres, axis=0, weights=weights) if weights.sum() > 0 else centres.mean(axis=0)
            direction = mean / np.linalg.norm(mean)
        self._tick("detect", start)
        return OperatorEvidence(direction, boxes.float().cpu().numpy(), detections.frame_indices.cpu().numpy(), faces)

    def _operator_prompt(self, synthetic: UInt8[Tensor, "s s 3"], first: bool) -> SegmentationPrompts | None:
        """A tracker prompt for the operator: the detector's box around the synthetic view's centre.

        FullCircle clicks the centre once. With EfficientTAM a lone click on the torso selects the shirt (the
        best-IoU mask of three), so the box of the person the view is centred on -- the operator -- replaces it: a box
        holding the centre (the largest), else the box whose centre is nearest. Without a detection the first prompt
        falls back to FullCircle's centre click; a re-prompt is skipped, as it is when the nearest box centre lies
        further than ``reprompt_radius`` of the view size from the centre.
        """
        size: int = self.config.synthetic_size
        centre: float = (size - 1) / 2
        zero: Tensor = torch.zeros(1, dtype=torch.long, device=self.device)
        detections: BoxDetections = self._detect_people(synthetic[None])
        if detections.num_detections == 0:
            if not first:
                return None
            return SegmentationPrompts(
                frame_indices=zero, points_xy=torch.tensor([[centre, centre]], dtype=torch.float32, device=self.device), track_ids=zero
            )
        boxes: Float32[Tensor, "n 4"] = detections.xyxy.float()
        centres: Float[Tensor, "n 2"] = (boxes[:, :2] + boxes[:, 2:]) / 2
        contains: Bool[Tensor, " n"] = (boxes[:, 0] <= centre) & (boxes[:, 2] >= centre) & (boxes[:, 1] <= centre) & (boxes[:, 3] >= centre)
        distance: Float[Tensor, " n"] = (centres - centre).norm(dim=1)
        area: Float[Tensor, " n"] = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
        best: int = int(torch.where(contains, area, -distance).argmax())
        if not first and not bool(contains[best]) and float(distance[best]) > self.config.reprompt_radius * size:
            return None
        return SegmentationPrompts(frame_indices=zero, boxes_xyxy=boxes[best : best + 1], track_ids=zero)

    @torch.inference_mode()
    def track(self, frames: list[UInt8[Tensor, "h w 3"]], direction: Float64[ndarray, "3"], up: Float64[ndarray, "3"]) -> OperatorMasks:
        """Steps 4-8 on the next frameset: the synthetic fisheye along ``direction``, the tracker, the source masks.

        The first call prompts the tracker (:meth:`_operator_prompt`), every ``reprompt_every``-th call re-prompts it,
        the others propagate. ``up`` is the view's up hint (:func:`view_up`), one for the whole sequence.
        """
        start: float = time.perf_counter()
        size: int = self.config.synthetic_size
        rig_R_view: Float64[ndarray, "3 3"] = look_at(direction, up)
        rotation: Float32[Tensor, "3 3"] = torch.from_numpy(rig_R_view).to(self.device, torch.float32)
        view: SphereView = view_from_rays(self.synthetic_rays @ rotation.T, size, size, self.cameras, self.rig_R_cams, self.full_hw)
        rendered: Float[Tensor, "3 s s"] = sample_sphere(self._work_images(frames), view) * self.synthetic_valid[None]
        synthetic: UInt8[Tensor, "s s 3"] = rearrange(rendered.round().clamp(0, 255).to(torch.uint8), "c h w -> h w c").contiguous()
        start = self._tick("synthetic", start)
        prompts: SegmentationPrompts | None = None
        every: int = self.config.reprompt_every
        if self.steps == 0 or (every > 0 and self.steps % every == 0):
            prompts = self._operator_prompt(synthetic, first=self.steps == 0)
            self.prompts += prompts is not None
        self.steps += 1
        tracks: BoxDetections = self.tracker.step(synthetic[None], prompts)
        assert tracks.masks is not None
        synthetic_mask: Bool[Tensor, "s s"] = (
            tracks.masks.any(dim=0) if tracks.num_detections else torch.zeros((size, size), dtype=torch.bool, device=self.device)
        )
        synthetic_mask = synthetic_mask & self.synthetic_valid
        start = self._tick("track", start)
        source_masks: list[Bool[Tensor, "mh mw"]] = []
        for index, source in enumerate(self.sources):
            pixels, valid = self.synthetic.project(self.mask_rays[index] @ rotation)
            grid: Float[Tensor, "1 mh mw 2"] = ((pixels + 0.5) / size * 2 - 1).nan_to_num(0.0).float().reshape(1, *source.mask_hw, 2)
            hit: Float[Tensor, "1 1 mh mw"] = F.grid_sample(
                synthetic_mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False
            )
            mask: Bool[Tensor, "mh mw"] = (hit[0, 0] > 0.5) & (valid & self.mask_ray_valid[index]).reshape(source.mask_hw)
            source_masks.append(disk_dilate(mask, self.mask_radius[index]))
        self._tick("backmap", start)
        return OperatorMasks(source_masks, synthetic, synthetic_mask)


__all__ = (
    "FULLCIRCLE_SYNTHETIC_LENS",
    "FullCircleConfig",
    "FullCircleMasker",
    "OperatorDetector",
    "OperatorEvidence",
    "OperatorMasks",
    "SphereSource",
    "disk_dilate",
    "fill_nearest",
    "level_rotation",
    "view_up",
)
