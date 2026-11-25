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
   (centre, area), which is what ``omni2synthetic.py`` consumes. Here the mean takes only the detections near the
   previous direction (:func:`operator_directions`). A frame with no direction reuses the nearest frame that has one
   (earlier first at equal distance), as ``omni2synthetic.py`` does: :func:`fill_nearest`.
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
8. Dilation (``dilate.py``: a 9x9 ellipse at 2880 px, 1 iteration -> ``masks-1``, "about 4 pixels" in the paper, and 5
   iterations -> ``masks-5``, about 20 px; training reads ``masks-5``, ``threedgrut/datasets/dataset_colmap.py``): a disk
   of ``dilate_px`` image pixels, applied at mask resolution, 12 px by default (``masks-5`` is about 13 px at 1920 px).

Five changes, each forced by a measured failure (FullCircle's room3 scene: IoU against the masks FullCircle released;
clip 007: a 17.8 min Insta360 X4 Air walk with the camera on a stick above the operator):

- Box prompt, not the centre click: EfficientTAM (FullCircle used SAM 2.1 Hiera-L) segments only the shirt on a torso
  click (room3 IoU 0.20); with the box, 0.958 (0.945 with the re-prompts and memory bound below).
- Re-prompts: a single prompt drifted onto other people within 10 s of clip 007.
- One up hint per sequence: with the operator below the camera (near +Y) the look-at fallback flipped the view.
- Level faces: lens 0's +Y is not down on the X4 Air, and people lying sideways in the faces were missed.
- Gated direction: FullCircle averages all detections and assumes few false positives. On the Mission office X5 clip
  a YOLOX box on a desk with a jacket (7.0 s) moved the view about 50 degrees off the operator, and 6 of 250 tracked
  framesets ended more than 40 px (of 512) off centre. Only detections within ``direction_gate_deg`` of the previous
  direction count; the first frame and a lost operator start from the heaviest cluster of detections.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
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
from posekit.models.yolox import YoloxDetector, YoloxDetectorConfig, fp16_dynamic_yolox
from posekit.ops.people_masks import dilate_by_footprint
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

    def mask_radius(self, dilate_px: float) -> int:
        """A dilation radius of ``dilate_px`` source-image pixels in mask cells (across), rounded up."""
        return math.ceil(dilate_px / self.mask_scale[0] - 1e-9)


@dataclass(frozen=True, slots=True)
class FullCircleConfig:
    """FullCircle operator masking options."""

    detector: OperatorDetector = "yolox"
    """Person detector on the cube faces: posekit YOLOX-HumanArt (TensorRT) or SAM 3 ``person`` (operator only: the
    bystanders confirm YOLOX boxes with SAM 3)."""
    yolox: YoloxDetectorConfig = field(
        default_factory=lambda: YoloxDetectorConfig(
            backend=TensorRtBackendConfig(max_batch_size=48, opt_batch_size=48), onnx_transform=fp16_dynamic_yolox, score_thr=0.25
        )
    )
    """YOLOX variant, backend and thresholds: a dynamic-batch engine with an fp16 trunk that takes the faces of 8
    framesets per call. Boxes from 0.25 are kept for the bystanders
    (:class:`~posekit.models.fullcircle_people.FacePeopleConfig`); the operator uses those from ``operator_score``."""
    operator_score: float = 0.5
    """Lowest YOLOX score of a detection that counts for the operator direction (YOLOX's default threshold); SAM 3
    detections all count, at ``sam3.score_threshold``."""
    sam3: Sam3SegmenterConfig = field(default_factory=lambda: Sam3SegmenterConfig(score_threshold=0.25))
    """SAM 3 options (FullCircle runs it at conf 0.25)."""
    sam3_prompt: str = "person"
    """SAM 3 text prompt."""
    tracker: Sam2VideoSegmenterConfig = field(
        default_factory=lambda: Sam2VideoSegmenterConfig(
            variant="efficienttam-s-512", memory_on_device=True, bf16=True, max_conditional_memories=4, single_click_multimask=True
        )
    )
    """The video tracker on the synthetic fisheye; it attends to the 4 nearest re-prompted frames (all of them made the
    17.8 min clip 007 slow down along the clip), and the fallback centre click takes SAM 2's best-of-three mask. Memories
    stay on the GPU and the forward runs under bf16 autocast, as the people tracker does: room3 IoU 0.9453 (fp32 0.9449);
    fp16 autocast lost the operator on clip B frames."""
    reprompt_every: int = 3
    """Tracked framesets between two re-prompts of the tracker with the detector's box (0: FullCircle's single prompt);
    3 at 3 Hz (every 10th frameset at 30 fps) is once a second. The Insta360 stage sets it from
    ``reprompt_every_frames`` and ``stride`` (:class:`~posekit.apis.fullcircle_masks.FullCircleMasksConfig`)."""
    direction_gate_deg: float = 30.0
    """Detections whose centre lies within this angle of the previous operator direction make the next direction
    (:func:`operator_directions`); also the radius of a cluster of detections on the sphere."""
    lost_after: int = 3
    """Framesets in a row with detections, none of them within the gate, after which the operator is taken as lost and
    the heaviest cluster starts the gate again (3 at 3 Hz: one second)."""
    reprompt_radius: float = 0.15
    """A re-prompt box that does not hold the view centre must have its centre within this fraction of the view size."""
    face_size: int = 640
    """Cube face size, pixels (YOLOX's input size)."""
    synthetic_size: int = 512
    """Synthetic fisheye size, pixels (sam-ui resizes to 512)."""
    work_size: int = 1280
    """Source images are area-downscaled by a whole factor to at most this size before the synthetic view is resampled
    from them (anti-aliasing)."""
    face_work_size: int = 2048
    """The same for the faces. A 640 px face holds about 7 px per degree, the 1920 px X5 lens about 10, so the faces
    sample the full lens: from the 960 px work image YOLOX missed the people behind the glass of the Mission office clip
    (46.7 s), from the full lens it boxes them."""
    dilate_px: float = 12.0
    """Dilation radius in source-image pixels. FullCircle trains on its ``masks-5`` (a 9x9 ellipse 5 times at 2880 px,
    about 13 px at 1920 px); 12 px is 3 cells of the 480 px mask grid of a 1920 px lens (4 px, FullCircle's ``masks-1``,
    was 1 cell). On the Mission office X5 clip 12 px costs about 0.6 % of the lens disc more than 4 px."""
    device: str = "cuda"
    """Inference device."""
    step_timing: bool = False
    """Synchronise the device after each step to time it (:attr:`FullCircleMasker.seconds`); off, the steps queue freely."""


@dataclass(frozen=True, slots=True)
class OperatorDetections:
    """The person detections of one frameset on the sphere."""

    centres: Float64[ndarray, "n 3"]
    """Unit direction of each detection's centre in the rig frame."""
    weights: Float64[ndarray, " n"]
    """Each detection's weight: its area in face pixels."""


@dataclass(frozen=True, slots=True)
class FaceBoxes:
    """The detector's boxes on the six faces of one frameset."""

    xyxy: Float32[ndarray, "n 4"]
    """Each detection's face-pixel box."""
    scores: Float32[ndarray, " n"]
    """Each detection's score."""
    faces: Int64[ndarray, " n"]
    """Face index of each detection."""


@dataclass(frozen=True, slots=True)
class OperatorEvidence:
    """What the face detector saw in one frameset."""

    detections: OperatorDetections
    """The detections of the operator score as directions and weights, input of :func:`operator_directions`."""
    boxes: FaceBoxes
    """All the detections' boxes (from YOLOX's threshold, for the bystanders)."""
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
    rig_R_view: Float64[ndarray, "3 3"]
    """Rotation of the synthetic view's rays into the rig frame."""


def _mean_direction(detections: OperatorDetections, keep: Bool[ndarray, " n"]) -> Float64[ndarray, "3"]:
    """Weighted mean of the kept detection centres, made unit (plain mean when their weights sum to zero)."""
    weights: Float64[ndarray, " k"] = detections.weights[keep]
    mean: Float64[ndarray, "3"] = (
        np.average(detections.centres[keep], axis=0, weights=weights) if weights.sum() > 0 else detections.centres[keep].mean(axis=0)
    )
    return mean / np.linalg.norm(mean)


def _heaviest_cluster(detections: OperatorDetections, cos_gate: float) -> Float64[ndarray, "3"]:
    """The mean direction of the heaviest cluster: the detections within the gate of one detection, largest weight sum."""
    near: Bool[ndarray, "n n"] = detections.centres @ detections.centres.T >= cos_gate
    seed: int = int(np.argmax(near.astype(np.float64) @ detections.weights))
    return _mean_direction(detections, near[seed])


def operator_directions(frames: Sequence[OperatorDetections], gate_deg: float, lost_after: int) -> list[Float64[ndarray, "3"] | None]:
    """Per frameset, the operator direction from the detections near the previous one (``None`` without such detections).

    FullCircle's area-weighted mean over all detections follows any false positive; here only detections within
    ``gate_deg`` of the last direction count. The first frameset with detections, and the first after ``lost_after``
    framesets in a row whose detections all lie outside the gate, start from the heaviest cluster of detections
    (:func:`_heaviest_cluster`). A frameset without detections leaves the state unchanged.
    """
    cos_gate: float = math.cos(math.radians(gate_deg))
    previous: Float64[ndarray, "3"] | None = None
    outside: int = 0
    directions: list[Float64[ndarray, "3"] | None] = []
    for detections in frames:
        direction: Float64[ndarray, "3"] | None = None
        if len(detections.weights):
            if previous is None:
                direction = _heaviest_cluster(detections, cos_gate)
            else:
                keep: Bool[ndarray, " n"] = detections.centres @ previous >= cos_gate
                if keep.any():
                    direction = _mean_direction(detections, keep)
                    outside = 0
                else:
                    outside += 1
                    if outside >= lost_after:
                        direction = _heaviest_cluster(detections, cos_gate)
                        outside = 0
        if direction is not None:
            previous = direction
        directions.append(direction)
    return directions


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
    offsets: Float32[Tensor, " k"] = torch.arange(-radius, radius + 1, dtype=torch.float32)
    disk: Bool[Tensor, "k k"] = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius * radius + 1e-6
    return dilate_by_footprint(mask[None], disk)[0]


def sample_faces(
    images: list[Float[Tensor, "3 hw ww"]],
    face_rays: Float32[Tensor, "n 3"],
    rig_R_faces: Float32[Tensor, "k 3 3"],
    size: int,
    cameras: list[SphereCamera],
    rig_R_cams: Float32[Tensor, "n_src 3 3"],
    full_hw: list[tuple[int, int]],
) -> UInt8[Tensor, "k s s 3"]:
    """Render ``k`` square faces of ``size`` pixels (rays ``face_rays`` turned by each ``rig_R_face``) from the sources.

    The faces are stacked into one ``k size x size`` view, so one sampling plan serves them all.
    """
    rays: Float32[Tensor, "k n 3"] = face_rays @ rig_R_faces.transpose(1, 2)
    view: SphereView = view_from_rays(rays.reshape(-1, 3), len(rig_R_faces) * size, size, cameras, rig_R_cams, full_hw)
    stacked: UInt8[Tensor, "3 ks s"] = sample_sphere(images, view).round().clamp(0, 255).to(torch.uint8)
    return rearrange(stacked, "c (k h) w -> k h w c", k=len(rig_R_faces)).contiguous()


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
        self.face_downscale: list[int] = [max(1, math.ceil(max(s.image_hw) / config.face_work_size)) for s in sources]
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
        self.mask_radius: list[int] = [source.mask_radius(config.dilate_px) for source in sources]
        for index, source in enumerate(sources):
            rays, valid = source.camera.unproject(pixel_grid(source.mask_hw[0], source.mask_hw[1], device, scale=source.mask_scale))
            self.mask_rays.append((rays @ rig_R_cams[index].T).float())
            self.mask_ray_valid.append(valid)

        self.yolox: YoloxDetector | None = config.yolox.setup() if config.detector == "yolox" else None
        self.sam3: Sam3Segmenter | None = config.sam3.setup() if config.detector == "sam3" else None
        self.tracker: Sam2VideoSegmenter = config.tracker.setup()
        self.steps: int = 0
        """Tracked framesets so far."""
        self.prompts: int = 0
        """Tracker prompts given so far (the first one included)."""
        self.seconds: dict[str, float] = {"faces": 0.0, "detect": 0.0, "synthetic": 0.0, "track": 0.0, "backmap": 0.0}
        """Synchronised wall time per step since construction (with ``config.step_timing``)."""

    def _work_images(self, frames: list[UInt8[Tensor, "h w 3"]], downscale: list[int]) -> list[Float[Tensor, "3 hw ww"]]:
        images: list[Tensor] = []
        for frame, factor in zip(frames, downscale, strict=True):
            chw: Float[Tensor, "3 h w"] = rearrange(frame, "h w c -> c h w").float()
            images.append(F.avg_pool2d(chw[None], factor)[0] if factor > 1 else chw)
        return images

    def render_faces(
        self, frames: list[UInt8[Tensor, "h w 3"]], rig_R_faces: Float32[Tensor, "k 3 3"], face_rays: Float32[Tensor, "n 3"], size: int
    ) -> UInt8[Tensor, "k s s 3"]:
        """Faces of ``size`` pixels along ``rig_R_faces`` from one frameset, sampled at ``face_work_size`` (:func:`sample_faces`)."""
        images: list[Float[Tensor, "3 hw ww"]] = self._work_images(frames, self.face_downscale)
        return sample_faces(images, face_rays, rig_R_faces, size, self.cameras, self.rig_R_cams, self.full_hw)

    def _detect_people(self, images: UInt8[Tensor, "b s s 3"]) -> BoxDetections:
        if self.yolox is not None:
            return self.yolox(images)
        assert self.sam3 is not None
        return self.sam3(images, SegmentationPrompts(frame_indices=torch.arange(len(images), device=self.device), text=self.config.sam3_prompt))

    def _tick(self, key: str, start: float) -> float:
        if not self.config.step_timing:
            return start
        torch.cuda.synchronize(self.device)
        now: float = time.perf_counter()
        self.seconds[key] += now - start
        return now

    @torch.inference_mode()
    def detect(self, framesets: list[list[UInt8[Tensor, "h w 3"]]], rig_R_levels: list[Float64[ndarray, "3 3"]]) -> list[OperatorEvidence]:
        """Steps 1-3 on a batch of framesets: cube faces, person detection (one detector call), the detections on the sphere.

        The faces are those of each frameset's level frame (``rig_R_levels``, +Y along gravity), so people stand upright
        in them, as they do in the horizon-levelled panoramas FullCircle starts from (identity for an already level source).
        """
        start: float = time.perf_counter()
        size: int = self.config.face_size
        rig_R_faces: Float64[Tensor, "b 6 3 3"] = torch.from_numpy(np.stack([level @ FACE_ROTATIONS for level in rig_R_levels])).to(self.device)
        faces: UInt8[Tensor, "b6 s s 3"] = torch.empty((6 * len(framesets), size, size, 3), dtype=torch.uint8, device=self.device)
        for index, frames in enumerate(framesets):
            faces[6 * index : 6 * index + 6] = self.render_faces(frames, rig_R_faces[index].float(), self.face_rays, size)
        start = self._tick("faces", start)
        detections: BoxDetections = self._detect_people(faces)
        boxes: Float32[Tensor, "n 4"] = detections.xyxy.float()
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
        rotations: Float64[Tensor, "n 3 3"] = rig_R_faces.reshape(-1, 3, 3)[detections.frame_indices]
        centres: Float64[ndarray, "n 3"] = torch.einsum("nij,nj->ni", rotations, rays).cpu().numpy()
        weights: Float64[ndarray, " n"] = areas.double().cpu().numpy()
        all_boxes: Float32[ndarray, "n 4"] = boxes.cpu().numpy()
        scores: Float32[ndarray, " n"] = detections.scores.float().cpu().numpy()
        face_ids: Int64[ndarray, " n"] = detections.frame_indices.cpu().numpy()
        operator: Bool[ndarray, " n"] = scores >= self.config.operator_score if self.yolox is not None else np.ones(len(scores), dtype=bool)
        evidence: list[OperatorEvidence] = []
        for index in range(len(framesets)):
            keep: Bool[ndarray, " n"] = face_ids // 6 == index
            detections_here: OperatorDetections = OperatorDetections(centres[keep & operator], weights[keep & operator])
            evidence.append(
                OperatorEvidence(detections_here, FaceBoxes(all_boxes[keep], scores[keep], face_ids[keep] % 6), faces[6 * index : 6 * index + 6])
            )
        self._tick("detect", start)
        return evidence

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
        boxes: Float32[Tensor, "n 4"] = detections.xyxy.float()
        if self.yolox is not None:
            boxes = boxes[detections.scores >= self.config.operator_score]
        if len(boxes) == 0:
            if not first:
                return None
            return SegmentationPrompts(
                frame_indices=zero, points_xy=torch.tensor([[centre, centre]], dtype=torch.float32, device=self.device), track_ids=zero
            )
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
        rendered: Float[Tensor, "3 s s"] = sample_sphere(self._work_images(frames, self.downscale), view) * self.synthetic_valid[None]
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
        return OperatorMasks(source_masks, synthetic, synthetic_mask, rig_R_view)


__all__ = (
    "FULLCIRCLE_SYNTHETIC_LENS",
    "FaceBoxes",
    "FullCircleConfig",
    "FullCircleMasker",
    "OperatorDetections",
    "OperatorDetector",
    "OperatorEvidence",
    "OperatorMasks",
    "SphereSource",
    "disk_dilate",
    "sample_faces",
    "fill_nearest",
    "level_rotation",
    "operator_directions",
    "view_up",
)
