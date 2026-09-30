"""Detection-by-tracking (MEgATrack §3.3 and §5.1): one frame of the pipeline per ``Tracker.step``.

Per hand the state is θ(t−1) and θ(t−2) (none while untracked). On each frame:

- **Tracked hand:** extrapolate θ̂ = 2θ(t−1) − θ(t−2) (``pose.extrapolate``; after the first tracked frame θ(t−1) as it is),
  project its 21 keypoints into every camera; the hand box in a camera is the smallest enclosing circle of the keypoints
  in front of it, squared. KeyNet runs on at most two views: the cameras with the most keypoints inside the image
  (ties go to the lower camera index; a camera with none inside never runs). The keypoint input is θ̂ projected into
  the crop. A tracked hand with no keypoint inside any image is dropped without running KeyNet.
- **Untracked hands:** DetNet runs on ONE camera per frame, cycling through the cameras each time it runs, and only
  while at least one hand is untracked (a hand dropped on this frame is looked for from the next frame on). A hand
  with presence > 0.5 gets its box there and KeyNet with the zero keypoint input, then the fit from the neutral
  initialiser; a fit with ``converged=False`` does not acquire the hand.
- **Track end (our addition):** KeyNet presence below 0.5 in every view of a hand drops the track and clears its history;
  views below 0.5 stay out of the fit. A DetNet detection that KeyNet rejects is not tracked. A fit that puts the wrist
  out of reach (farther than ``max_reach_m`` from the headset) or is not finite also ends the track.
- **Fit:** ``fit_pose`` with θ(t−1) as the start and the temporal prior; a keypoint weighs 1, or 0 when its heatmap peak is
  below ``min_keypoint_confidence`` (an empty heatmap decodes to the crop corner). The headset pose is the
  dataset's ``world_from_rig`` (as UmeTrack's tracker uses its camera poses); a frame without one resets both tracks.

- **Crop refinement (our addition, off by default, ``refine_shift``):** when the fitted pose moved off its crops, cut them
  again around the fitted pose and run KeyNet and the fit once more on the same frame.

All KeyNet crops of a pass (at most four) go through one batched call. The networks sit behind two small protocols so
tests and the oracle mode can replace them; the fit and all state live on the CPU.
"""

import math
import time
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int8, Int64, UInt8
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import MAX_VIEWS, HandObservation, ViewObservation
from handtrack.fit.pose_fit import FitConfig, FitResult, fit_pose
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox
from handtrack.hand.pose import HandPose, Side, extrapolate, landmarks, mesh_vertices
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import apply_affine, crop_boxes, crop_from_net, cut_crops
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.labels.keypoint_input import keypoint_input, relative_distances
from handtrack.labels.visibility import flesh_margin, keypoints_hidden
from handtrack.models.detnet import Detections, DetNetF, decode_detections
from handtrack.models.keynet import KeyNetF, KeyNetOutput
from handtrack.results import BoxSource

MIN_BOX_RADIUS: float = 8.0
"""Floor on a hand circle's radius (net px) before it becomes a crop, so a degenerate circle still gives a valid crop (as the stream)."""


@dataclass(frozen=True, slots=True)
class CropRequest:
    """The KeyNet crops of one frame, on the CPU: n views (camera, hand) with their crop maps and keypoint inputs."""

    camera: Int64[Tensor, "n"]
    side: Int64[Tensor, "n"]
    crop_from_net: Float32[Tensor, "n 3 3"]
    """Net frame to crop pixels (``labels.crops.crop_from_net``), right hands mirrored."""
    keypoint_input: Float32[Tensor, "n 63"]
    """21 x (u, v, d) of θ̂ in the crop; all zeros for an acquisition."""
    poses: tuple[HandPose | None, HandPose | None] = (None, None)
    """Our crop-planning pose per hand; None on acquisition. Used by pose-backed estimators."""
    world_from_rig: Float32[Tensor, "4 4"] | None = None
    """Current headset transform, world metres."""
    circles: Float32[Tensor, "n 3"] | None = None
    """Requested circles in net pixels, before crop enlargement and right-hand mirroring."""
    native_images: tuple[UInt8[Tensor, "h w"], ...] = ()
    """Original camera frames, retained for perspective crops without SHOW3D downsampling."""


@dataclass(frozen=True, slots=True)
class KeypointEstimate:
    """KeyNet's answer per requested view, on the CPU."""

    points_net: Float32[Tensor, "n 21 2"]
    """Keypoints in the net frame (crop map and mirror undone)."""
    d_rel_mm: Float32[Tensor, "n 21"]
    """Relative distances in millimetres of the generic hand."""
    presence: Float32[Tensor, "n"]
    """Probability that the hand is in the crop."""
    confidence: Float32[Tensor, "n 21"]
    """Per keypoint: the heatmap's peak value (1 for a clean unit Gaussian; about 0 when the heatmap is empty)."""
    uses_detnet_presence: bool = False
    """No learned presence head: presence marks usable views; Tracker checks DetNet on those views."""
    visibility: Float32[Tensor, "n 21"] | None = None
    """Per keypoint: the probability that it is visible (KeyNet's visibility head, or the ground truth in oracle mode); None without."""
    pinch: Float32[Tensor, "n"] | None = None
    """Per view: the probability that thumb and index touch (KeyNet's pinch head); None without."""


@runtime_checkable
class Detector(Protocol):
    """DetNet: net-frame images of some cameras of frame ``frame`` -> CPU detections, left slot first."""

    def __call__(self, images: UInt8[Tensor, "b 480 640"], frame: int, cameras: Int64[Tensor, "b"]) -> Detections: ...


@runtime_checkable
class KeypointEstimator(Protocol):
    """KeyNet: the crops of ``request`` cut from the frame's net-frame images (one per camera) -> CPU estimates."""

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate: ...


class DetNetDetector:
    """A trained ``DetNetF`` as the tracker's detector, in eval mode, fp32."""

    def __init__(self, model: DetNetF, threshold: float = 0.5) -> None:
        self.model: DetNetF = model.eval()
        self.threshold: float = threshold

    def __call__(self, images: UInt8[Tensor, "b 480 640"], frame: int, cameras: Int64[Tensor, "b"]) -> Detections:
        with torch.inference_mode():
            detections: Detections = decode_detections(self.model(images[:, None].float() / 255.0), self.threshold)
        return Detections(
            circle=detections.circle.cpu(), probability=detections.probability.cpu(), present=detections.present.cpu(), box=detections.box.cpu()
        )


class KeyNetEstimator:
    """A trained ``KeyNetF`` as the tracker's keypoint estimator: cut the crops on the images' device, run, decode, undo the crop."""

    def __init__(self, model: KeyNetF) -> None:
        self.model: KeyNetF = model.eval()

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        device: torch.device = images.device
        crop_from_net_map: Float32[Tensor, "n 3 3"] = request.crop_from_net.to(device)
        with torch.inference_mode():
            crops: Float32[Tensor, "n 1 96 96"] = cut_crops(images, request.camera.to(device), crop_from_net_map)
            output: KeyNetOutput = self.model(crops, request.keypoint_input.to(device))
            decoded: tuple[Float32[Tensor, "n 21 2"], Float32[Tensor, "n 21"]] = decode_heatmaps(output.heatmaps)
            points_net: Float32[Tensor, "n 21 2"] = apply_affine(torch.linalg.inv(crop_from_net_map), decoded[0])
            return KeypointEstimate(
                points_net=points_net.cpu(),
                d_rel_mm=decode_distance(output.distance).cpu(),
                presence=output.presence_logit.sigmoid().cpu(),
                confidence=decoded[1].cpu(),
            )


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """The tracker's thresholds and the fit's settings."""

    detnet_threshold: float = 0.5
    """DetNet reports a hand when its presence exceeds this (§3.3)."""
    presence_threshold: float = 0.5
    """KeyNet presence below this keeps a view out of the fit; below it in every view ends the track (our addition)."""
    max_views: int = MAX_VIEWS
    """KeyNet views per tracked hand (§5.1: at most two)."""
    extrapolate: bool = True
    """Boxes and keypoint input from θ̂ = 2θ(t−1) − θ(t−2) (§3.3); False uses θ(t−1) as it is (a diagnostic, not the paper)."""
    refine_shift: float | None = None
    """Our addition, off by default: re-cut a hand's crops around its fitted pose and run KeyNet and the fit again when the
    pose's circle moved by more than this many crop radii in one of its views."""
    max_reach_m: float = 1.0
    """A fitted wrist farther than this from the headset ends the track (our addition; ground-truth wrists stay within 0.77 m)."""
    min_keypoint_confidence: float = 0.05
    """A keypoint whose heatmap peak is below this gets weight 0 in the fit (our choice): an empty heatmap decodes to the crop corner."""
    umetrack_presence_threshold: float = 0.5
    """DetNet must exceed this in at least one requested view to confirm a UmeTrack hand."""
    umetrack_drift_factor: float = 0.8
    """End a tracked UmeTrack hand whose confirmed views ALL disagree with DetNet: DetNet's circle centre lies farther than this
    times the larger radius from the circle projected from the tracked pose. Without it a pose that slid off the hand (often onto
    the other hand) keeps feeding its own crops for tens of frames; 0 disables the check. UmeTrack synthetic user_12/rec_09:
    off 73.9 mm, 1.0 30.0 mm, 0.8 25.6 mm (90 % tracked), 0.6 25.5 mm (83 % tracked)."""
    umetrack_miss_frames: int = 3
    """End a UmeTrack track after this many consecutive frames without DetNet confirmation in its requested views.
    During the grace frames, fit all usable network views; a confirmed frame fits only confirmed views.
    Acquisition requires confirmation immediately. Missing/invalid network output ends the track immediately."""
    visibility_weights: Literal["off", "soft", "hard"] = "off"
    """Weight each keypoint in the fit by its visibility: ``soft`` by the probability (floored at ``visibility_floor``), ``hard`` by
    probability >= 0.5 (hidden keypoints leave the fit and the temporal term carries them). Needs ``KeypointEstimate.visibility``."""
    visibility_floor: float = 0.0
    min_visible_keypoints: int = 0
    """Visibility as presence: a view with fewer keypoints of probability >= 0.5 is rejected like a low presence (0 disables)."""
    predicted_occlusion: bool = False
    """Our addition: a keypoint that the OTHER hand's predicted mesh covers in a view (both hands' planning poses, ray cast as
    ``labels.visibility``) gets weight 0 in the fit, whatever KeyNet says; needs no network output."""
    tip_weight: float = 1.0
    """Our addition for pinch: weight of the thumb tip and index fingertip keypoints in the fit (1 = as every keypoint)."""
    mask_out_of_image: bool = False
    """Our addition: a keypoint that the planning pose projects outside the camera image (or within ``image_margin_px`` of its edge)
    gets weight 0 in that view: KeyNet cannot see it and squeezes it into the visible part of a hand leaving the image."""
    image_margin_px: float = 0.0
    rejection_patience: int = 1
    """With ``end_on_view_rejection``: end the track only after this many consecutive frames with a rejected view."""
    acquire_clear_of_other: float = 1.0
    """Skip an acquisition whose DetNet circle has more than this share of its area inside the other (tracked) hand's circle in that
    camera (1.0 never skips): a hand under the other hand starts off the hand."""
    acquire_recrop: int = 0
    """Our addition: before an acquisition's fit, this many passes that run KeyNet on the acquisition crop and re-cut it around the
    enclosing circle of KeyNet's own keypoints (as MediaPipe re-crops from landmarks). On unseen HOT3D DetNet's circles were 2.8x too
    large (median IoU 0 with the hand), so the first crop showed a tiny hand and the first fit was 170 mm off."""
    acquire_refine: int = 0
    """Our addition: after an acquisition's first fit (one DetNet view, zero prior), this many same-frame passes that cut crops around the
    fitted pose in its best cameras (at most ``max_views``, prior = the fit), run KeyNet and refit from the first fit without the temporal
    term: the second view fixes the monocular depth before the track's history exists."""
    young_frames: int = 0
    """For the first this-many frames of a track, scale the temporal weight by ``young_temporal_scale`` (a poor start is left quickly)."""
    young_temporal_scale: float = 1.0
    detnet_all_cameras: bool = False
    """Our change: while a hand is untracked, run DetNet on every camera (not the paper's one round-robin camera per frame), so an
    acquisition starts from stereo. On the RK3588 NPU DetNet-F costs about 1 ms per camera."""
    warm_restart_frames: int = 0
    """Our addition: a hand re-acquired within this many frames of losing its track is also fitted from its last pose (no temporal
    term); the lower-energy fit of that and the neutral start wins."""
    acquire_max_rms_px: float = 1e9
    """Reject an acquisition whose fit leaves an RMS 2D residual above this (pixels of the camera images)."""
    confirm_frames: int = 0
    """A new track is reported (tracked, landmarks) only from its (confirm_frames+1)-th frame; it is tracked internally from the first."""
    end_on_view_rejection: bool = False
    """Our addition, off by default: end a tracked hand when KeyNet rejects some of its requested views and one view is left.
    On UmeTrack synthetic user_12/rec_09 those one-view frames averaged ~170 mm (their last view is off the hand as well)."""
    fit: FitConfig = field(default_factory=FitConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.max_views <= MAX_VIEWS:
            raise ValueError(f"max_views must be between 1 and {MAX_VIEWS}, got {self.max_views}")
        if self.umetrack_miss_frames < 1 or not 0.0 <= self.umetrack_presence_threshold <= 1.0:
            raise ValueError("UmeTrack miss frames must be positive and presence threshold must be in [0, 1]")


DEFAULT_TRACKER_CONFIG: TrackerConfig = TrackerConfig()


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One frame of tracker output on the CPU, boxes and keypoints in the net frame (c cameras, slot 0 = left)."""

    tracked: Bool[Tensor, "2"]
    """The hand has a fitted pose on this frame."""
    poses: tuple[HandPose | None, HandPose | None]
    """θ(t) per hand, None when untracked."""
    landmarks: Float32[Tensor, "2 21 3"]
    """World metres, NaN when untracked."""
    circle: Float32[Tensor, "c 2 3"]
    """The hand circle (cx, cy, r) in the net frame per camera; its square is the box. NaN when no box."""
    box_source: Int8[Tensor, "c 2"]
    """``BoxSource`` per camera and hand."""
    keypoints: Float32[Tensor, "c 2 21 2"]
    """KeyNet's keypoints in the net frame, NaN where KeyNet did not run."""
    presence: Float32[Tensor, "c 2"]
    """KeyNet presence, NaN where KeyNet did not run."""
    detnet_camera: int
    """-1 when DetNet did not run."""
    detnet_presence: Float32[Tensor, "2"]
    """NaN when DetNet did not run."""
    fit_energy: Float32[Tensor, "2"]
    """NaN when untracked."""
    observations: tuple[HandObservation | None, HandObservation | None]
    """What each fitted hand's fit saw (for the scale calibration)."""
    visibility: Float32[Tensor, "c 2 21"] | None = None
    """KeyNet's per-keypoint visibility per camera and hand (NaN where it did not run); None without a visibility output."""
    pinch: Float32[Tensor, "c 2"] | None = None
    """KeyNet's pinch probability per camera and hand (NaN where it did not run); None without a pinch head."""


@dataclass(slots=True)
class _History:
    previous: HandPose | None = None
    """θ(t−1); None while the hand is untracked."""
    before: HandPose | None = None
    """θ(t−2); None after the first tracked frame."""
    detnet_misses: int = 0
    """Consecutive UmeTrack frames without detector support."""
    rejections: int = 0
    """Consecutive frames with a rejected view (``rejection_patience``)."""
    age: int = 0
    """Frames this track has been fitted (``confirm_frames``)."""
    lost_pose: HandPose | None = None
    """The pose when the track was last lost (``warm_restart_frames``)."""
    lost_frame: int = -1


@dataclass(frozen=True, slots=True)
class _View:
    side: Side
    camera: int
    circle: Float32[Tensor, "3"]
    """The hand circle the crop was cut around (net frame)."""
    crop_from_net: Float32[Tensor, "3 3"]
    keypoint_input: Float32[Tensor, "63"]
    pose: HandPose | None = None
    """Pose that planned this crop, including extrapolation or refinement."""


@dataclass(frozen=True, slots=True)
class _Projection:
    """One hand pose's 21 keypoints in every camera."""

    points_cam: Float32[Tensor, "c 21 3"]
    net_xy: Float32[Tensor, "c 21 2"]
    inside: Int64[Tensor, "c"]
    """Keypoints in front and inside the image."""
    circles: Float32[Tensor, "c 3"]
    """Smallest enclosing circle of the in-front keypoints in the net frame; NaN without one."""


def _crop_map(circle: Float32[Tensor, "3"], side: Side) -> Float32[Tensor, "3 3"]:
    floored: Float32[Tensor, "1 3"] = torch.cat([circle[:2], circle[2:].clamp_min(MIN_BOX_RADIUS)])[None]
    return crop_from_net(crop_boxes(floored), torch.tensor([side == Side.RIGHT]))[0]


def _circle_share(circle: Float32[Tensor, "3"], other: Float32[Tensor, "3"]) -> float:
    """Share of ``circle``'s area inside ``other`` (0 when ``other`` is NaN): the lens area of two circles over the first's area."""
    if not bool(torch.isfinite(other).all()) or not bool(torch.isfinite(circle).all()):
        return 0.0
    r1, r2 = float(circle[2]), float(other[2])
    d: float = float((circle[:2] - other[:2]).norm())
    if r1 <= 0:
        return 0.0
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r2 - r1):
        return 1.0 if r2 >= r1 else (r2 * r2) / (r1 * r1)
    a1: float = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2: float = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    a3: float = 0.5 * math.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0))
    return (a1 + a2 - a3) / (math.pi * r1 * r1)


def _finite(pose: HandPose) -> bool:
    return bool(torch.isfinite(pose.rotation).all() and torch.isfinite(pose.translation).all() and torch.isfinite(pose.joint_angles).all())


class Tracker:
    """Detection-by-tracking for one rig and one hand model; call ``step`` on every frame in order."""

    def __init__(
        self,
        rig: CameraRig,
        letterboxes: tuple[Letterbox, ...],
        model: HandModelTorch,
        phi: float,
        detector: Detector,
        keypoints: KeypointEstimator,
        config: TrackerConfig = DEFAULT_TRACKER_CONFIG,
    ) -> None:
        """
        Args:
            rig: The headset cameras, on the CPU.
            letterboxes: Each camera's map into the net frame.
            model: The hand model at the subject's size (the profile, or the generic model scaled by ϕ), on the CPU.
            phi: ϕ of ``model`` relative to the generic hand (KeyNet's d_rel unit).
            detector: DetNet (or a stand-in).
            keypoints: KeyNet (or a stand-in).
            config: Thresholds and fit settings.
        """
        self.rig: CameraRig = rig
        self.letterboxes: tuple[Letterbox, ...] = letterboxes
        self.cameras: tuple[CameraRig, ...] = tuple(rig.select([camera]) for camera in range(len(rig.names)))
        """Each camera as a one-camera rig (the fit's view)."""
        self.model: HandModelTorch = model
        self.phi: float = phi
        self.detector: Detector = detector
        self.keypoints: KeypointEstimator = keypoints
        self.config: TrackerConfig = config
        self.history: tuple[_History, _History] = (_History(), _History())
        self.next_detnet_camera: int = 0
        self._margin: Float32[Tensor, "21"] = flesh_margin(model)
        self._frame: int = 0
        self.timings_s: dict[str, float] = {"detnet": 0.0, "keynet": 0.0, "fit": 0.0, "tracker": 0.0}
        """Seconds per stage, summed over frames (``tracker`` is the rest: projection, circles, bookkeeping)."""

    def _drop(self, side: Side) -> None:
        if self.history[side].previous is not None:
            self.history[side].lost_pose = self.history[side].previous
            self.history[side].lost_frame = self._frame
        self.history[side].previous = None
        self.history[side].before = None
        self.history[side].detnet_misses = 0
        self.history[side].rejections = 0
        self.history[side].age = 0

    def _project(self, pose: HandPose, side: Side, world_from_rig: Float32[Tensor, "4 4"]) -> _Projection:
        points_cam: Float32[Tensor, "c 21 3"] = world_to_cameras(self.rig, world_from_rig, landmarks(self.model, pose, side))
        pixels: Float32[Tensor, "c 21 2"] = project(self.rig, points_cam)
        front: Bool[Tensor, "c 21"] = in_front(points_cam)
        net_xy: Float32[Tensor, "c 21 2"] = torch.stack([letterbox.to_net(pixels[camera]) for camera, letterbox in enumerate(self.letterboxes)])
        return _Projection(
            points_cam=points_cam,
            net_xy=net_xy,
            inside=(front & inside_image(self.rig, pixels)).sum(dim=-1),
            circles=torch.from_numpy(enclosing_circles(net_xy.numpy(), front.numpy())),
        )

    def _view(self, side: Side, camera: int, projection: _Projection, pose: HandPose) -> _View:
        """A KeyNet view around the projected pose, with the pose as the keypoint input."""
        crop_map: Float32[Tensor, "3 3"] = _crop_map(projection.circles[camera], side)
        features: Float32[Tensor, "1 63"] = keypoint_input(
            apply_affine(crop_map[None], projection.net_xy[camera][None]),
            relative_distances(projection.points_cam[camera][None], torch.tensor([self.phi])),
        )
        return _View(side, camera, projection.circles[camera], crop_map, features[0], pose)

    def _plan_tracked(self, side: Side, world_from_rig: Float32[Tensor, "4 4"], out: "_FrameOutput") -> list[_View]:
        """Boxes from θ̂ in every camera that sees it, and the KeyNet views; drops a hand that no camera sees."""
        history: _History = self.history[side]
        assert history.previous is not None
        guess: HandPose = history.previous if history.before is None or not self.config.extrapolate else extrapolate(history.previous, history.before)
        projection: _Projection = self._project(guess, side, world_from_rig)
        seen: Bool[Tensor, "c"] = projection.inside > 0
        out.circle[seen, side] = projection.circles[seen]
        out.box_source[seen, side] = int(BoxSource.TRACKED)
        order: list[int] = sorted(
            (camera for camera in range(len(self.cameras)) if bool(seen[camera])), key=lambda camera: -int(projection.inside[camera])
        )
        if not order:
            self._drop(side)
            return []
        return [self._view(side, camera, projection, guess) for camera in order[: self.config.max_views]]

    def step(self, frame: int, images: UInt8[Tensor, "c 480 640"], world_from_rig: Float32[Tensor, "4 4"],
             native_images: tuple[UInt8[Tensor, "h w"], ...] = ()) -> FrameResult:
        """Track frame ``frame``: its net-frame images (one per camera, any device) and the headset pose."""
        start: float = time.perf_counter()
        self._frame = frame
        out: _FrameOutput = _FrameOutput.empty(len(self.cameras))
        spent: dict[str, float] = {"detnet": 0.0, "keynet": 0.0, "fit": 0.0}
        if bool(torch.isfinite(world_from_rig).all()):
            untracked: list[Side] = [side for side in Side if self.history[side].previous is None]
            views: list[_View] = [
                view for side in Side if self.history[side].previous is not None for view in self._plan_tracked(side, world_from_rig, out)
            ]
            if untracked:
                begin: float = time.perf_counter()
                views += self._detect(frame, images, untracked, out)
                spent["detnet"] = time.perf_counter() - begin
            if views:
                self._keypoints_and_fit(frame, images, world_from_rig, views, out, spent, native_images)
        else:
            for side in Side:
                self._drop(side)
        for stage, seconds in spent.items():
            self.timings_s[stage] += seconds
        self.timings_s["tracker"] += time.perf_counter() - start - sum(spent.values())
        return out.result()

    def _detect(self, frame: int, images: UInt8[Tensor, "c 480 640"], untracked: list[Side], out: "_FrameOutput") -> list[_View]:
        """DetNet on this frame's round-robin camera; an untracked hand it reports gets an acquisition view (zero keypoint input).
        With ``detnet_all_cameras`` DetNet runs on every camera and the hand gets a view in each camera that reports it (best first, at
        most ``max_views``), so the acquisition fit sees the hand in stereo."""
        if self.config.detnet_all_cameras:
            return self._detect_all(frame, images, untracked, out)
        camera: int = self.next_detnet_camera
        self.next_detnet_camera = (camera + 1) % len(self.cameras)
        detections: Detections = self.detector(images[camera : camera + 1], frame, torch.tensor([camera]))
        out.detnet_camera = camera
        out.detnet_presence = detections.probability[0].clone()
        views: list[_View] = []
        for side in untracked:
            if self.config.acquire_clear_of_other < 1.0 and _circle_share(detections.circle[0, side], out.circle[camera, 1 - side]) > self.config.acquire_clear_of_other:
                continue
            if float(detections.probability[0, side]) > self.config.detnet_threshold:
                out.circle[camera, side] = detections.circle[0, side]
                out.box_source[camera, side] = int(BoxSource.DETNET)
                views.append(_View(side, camera, detections.circle[0, side], _crop_map(detections.circle[0, side], side), torch.zeros(63)))
        return views

    def _detect_all(self, frame: int, images: UInt8[Tensor, "c 480 640"], untracked: list[Side], out: "_FrameOutput") -> list[_View]:
        cameras: int = len(self.cameras)
        detections: Detections = self.detector(images, frame, torch.arange(cameras))
        out.detnet_camera = cameras  # all of them
        out.detnet_presence = detections.probability.max(dim=0).values.clone()
        views: list[_View] = []
        for side in untracked:
            found: list[int] = [
                camera for camera in range(cameras)
                if float(detections.probability[camera, side]) > self.config.detnet_threshold
                and not (self.config.acquire_clear_of_other < 1.0
                         and _circle_share(detections.circle[camera, side], out.circle[camera, 1 - side]) > self.config.acquire_clear_of_other)
            ]
            found.sort(key=lambda camera: -float(detections.probability[camera, side]))
            for camera in found[: self.config.max_views]:
                out.circle[camera, side] = detections.circle[camera, side]
                out.box_source[camera, side] = int(BoxSource.DETNET)
                views.append(_View(side, camera, detections.circle[camera, side], _crop_map(detections.circle[camera, side], side), torch.zeros(63)))
        return views

    def _keypoints_and_fit(
        self,
        frame: int,
        images: UInt8[Tensor, "c 480 640"],
        world_from_rig: Float32[Tensor, "4 4"],
        views: list[_View],
        out: "_FrameOutput",
        spent: dict[str, float],
        native_images: tuple[UInt8[Tensor, "h w"], ...],
    ) -> None:
        """KeyNet on every view in one call, the presence rules, one batched fit of the hands that remain, and (optionally) one refinement pass."""
        if self.config.acquire_recrop:
            for _ in range(self.config.acquire_recrop):
                views = self._recrop_acquisitions(frame, images, world_from_rig, views, spent, native_images)
        observed: tuple[list[HandObservation], list[Side]] = self._observe(frame, images, world_from_rig, views, out, spent, native_images)
        for side in observed[1]:
            self._drop(side)
        hands: list[HandObservation] = observed[0]
        if not hands:
            return
        results: list[FitResult] = self._fit(hands, spent)
        if self.config.acquire_refine:
            for _ in range(self.config.acquire_refine):
                hands, results = self._refine_acquisitions(frame, images, world_from_rig, hands, results, out, spent, native_images)
        if self.config.refine_shift is not None and not any(self.history[hand.side].detnet_misses for hand in hands):
            refined: dict[Side, tuple[HandObservation, FitResult]] = self._refine(frame, images, world_from_rig, views, hands, results, out, spent, native_images)
            chosen: list[tuple[HandObservation, FitResult]] = [
                refined.get(hand.side, (hand, result)) for hand, result in zip(hands, results, strict=True)
            ]
            hands = [hand for hand, _ in chosen]
            results = [result for _, result in chosen]
        for hand, result in zip(hands, results, strict=True):
            reach: float = float((result.pose.translation - world_from_rig[:3, 3]).norm())
            acquiring: bool = self.history[hand.side].previous is None
            weighted: float = float(sum(float(view.weights.sum()) for view in hand.views))
            rms: float = math.sqrt(result.e_2d / max(weighted, 1.0)) if math.isfinite(result.e_2d) else math.inf
            if acquiring and rms > self.config.acquire_max_rms_px:
                self._drop(hand.side)
                continue
            if (acquiring and not result.converged) or not _finite(result.pose) or reach > self.config.max_reach_m:
                self._drop(hand.side)
                continue
            history: _History = self.history[hand.side]
            history.before = history.previous
            history.previous = result.pose
            history.age += 1
            if history.age <= self.config.confirm_frames:
                continue  # tentative: tracked internally, not reported yet
            out.poses[hand.side] = result.pose
            out.observations[hand.side] = hand
            out.fit_energy[hand.side] = result.energy
            out.landmarks[hand.side] = landmarks(self.model, result.pose, hand.side)

    def _observe(
        self,
        frame: int,
        images: UInt8[Tensor, "c 480 640"],
        world_from_rig: Float32[Tensor, "4 4"],
        views: list[_View],
        out: "_FrameOutput",
        spent: dict[str, float],
        native_images: tuple[UInt8[Tensor, "h w"], ...],
    ) -> tuple[list[HandObservation], list[Side]]:
        """KeyNet on ``views``; returns the hands with at least one view above the presence threshold, and the hands whose every view fell below it."""
        poses: list[HandPose | None] = [next((view.pose for view in views if view.side == side), None) for side in Side]
        request: CropRequest = CropRequest(
            camera=torch.tensor([view.camera for view in views]),
            side=torch.tensor([int(view.side) for view in views]),
            crop_from_net=torch.stack([view.crop_from_net for view in views]),
            keypoint_input=torch.stack([view.keypoint_input for view in views]),
            poses=(poses[0], poses[1]),
            world_from_rig=world_from_rig,
            circles=torch.stack([view.circle for view in views]),
            native_images=native_images,
        )
        begin: float = time.perf_counter()
        estimate: KeypointEstimate = self.keypoints(images, frame, request)
        spent["keynet"] += time.perf_counter() - begin
        detector_presence: Float32[Tensor, "n"] | None = None
        detector_circles: Float32[Tensor, "n 3"] | None = None
        if estimate.uses_detnet_presence:
            begin = time.perf_counter()
            cameras: Int64[Tensor, "k"] = request.camera.unique(sorted=True)
            detections: Detections = self.detector(images[cameras.to(images.device)], frame, cameras)
            detector_presence = detections.probability[torch.searchsorted(cameras, request.camera), request.side]
            detector_circles = detections.circle[torch.searchsorted(cameras, request.camera), request.side]
            spent["detnet"] += time.perf_counter() - begin
        hands: list[HandObservation] = []
        rejected: list[Side] = []
        for side in Side:
            mine: list[int] = [index for index, view in enumerate(views) if view.side == side]
            for index in mine:
                out.keypoints[views[index].camera, side] = estimate.points_net[index]
                out.presence[views[index].camera, side] = estimate.presence[index] if detector_presence is None else detector_presence[index]
                if estimate.visibility is not None:
                    if out.visibility is None:
                        out.visibility = torch.full((out.presence.shape[0], 2, 21), torch.nan)
                    out.visibility[views[index].camera, side] = estimate.visibility[index]
                if estimate.pinch is not None:
                    if out.pinch is None:
                        out.pinch = torch.full((out.presence.shape[0], 2), torch.nan)
                    out.pinch[views[index].camera, side] = estimate.pinch[index]
            good: list[int] = [index for index in mine if float(estimate.presence[index]) >= self.config.presence_threshold]
            if self.config.min_visible_keypoints and estimate.visibility is not None:
                good = [index for index in good if int((estimate.visibility[index] >= 0.5).sum()) >= self.config.min_visible_keypoints]
            if self.config.end_on_view_rejection and self.history[side].previous is not None and len(mine) >= 2:
                if len(good) == 1:
                    self.history[side].rejections += 1
                    if self.history[side].rejections >= self.config.rejection_patience:
                        good = []  # the pose is slipping off the hand: a one-view fit from here is poor; DetNet re-acquires next frame
                else:
                    self.history[side].rejections = 0
            if detector_presence is not None:
                confirmed: list[int] = [index for index in good if float(detector_presence[index]) > self.config.umetrack_presence_threshold]
                history: _History = self.history[side]
                history.detnet_misses = 0 if confirmed else history.detnet_misses + 1
                if confirmed and history.previous is not None and self.config.umetrack_drift_factor > 0 and detector_circles is not None:
                    tracked_circles: Float32[Tensor, "k 3"] = torch.stack([views[index].circle for index in confirmed]).to(detector_circles.device)
                    detected: Float32[Tensor, "k 3"] = detector_circles[torch.tensor(confirmed, device=detector_circles.device)]
                    distance: Float32[Tensor, "k"] = (tracked_circles[:, :2] - detected[:, :2]).norm(dim=-1)
                    limit: Float32[Tensor, "k"] = self.config.umetrack_drift_factor * torch.maximum(tracked_circles[:, 2], detected[:, 2])
                    if bool((distance > limit).all()):
                        confirmed = []
                        history.detnet_misses = self.config.umetrack_miss_frames  # drifted: drop now, DetNet re-acquires
                if confirmed:
                    good = confirmed
                elif history.previous is None or history.detnet_misses >= self.config.umetrack_miss_frames:
                    good = []
            if mine and not good:
                rejected.append(side)
            if good:
                views_seen: tuple[ViewObservation, ...] = tuple(
                    ViewObservation(
                        camera=self.cameras[views[index].camera],
                        world_from_rig=world_from_rig,
                        keypoints_px=self.letterboxes[views[index].camera].from_net(estimate.points_net[index]),
                        weights=self._weights(estimate, index) * self._clear(views[index], poses, world_from_rig),
                        d_rel_mm=estimate.d_rel_mm[index],
                    )
                    for index in good
                )
                hands.append(HandObservation(side=side, views=views_seen))
        return hands, rejected

    def _clear(self, view: "_View", poses: list[HandPose | None], world_from_rig: Float32[Tensor, "4 4"]) -> Float32[Tensor, "21"]:
        """1 per keypoint, or 0 where the other hand's planning pose covers it in this view (``predicted_occlusion``)."""
        own: HandPose | None = poses[view.side]
        other: HandPose | None = poses[1 - view.side]
        clear: Float32[Tensor, "21"] = torch.ones(21)
        if own is None or not (self.config.mask_out_of_image or self.config.predicted_occlusion):
            return clear
        camera: CameraRig = self.cameras[view.camera]
        points: Float32[Tensor, "1 1 21 3"] = world_to_cameras(camera, world_from_rig, landmarks(self.model, own, view.side))[:, None]
        if self.config.mask_out_of_image:
            pixels: Float32[Tensor, "21 2"] = project(camera, points[:, 0])[0]
            size: Float32[Tensor, "2"] = camera.image_size[0].to(pixels.dtype)
            margin: float = self.config.image_margin_px
            inside: Bool[Tensor, "21"] = (pixels >= margin - 0.5).all(-1) & (pixels < size - 0.5 - margin).all(-1) & (points[0, 0, :, 2] > 0)
            clear = clear * inside.to(torch.float32)
        if not self.config.predicted_occlusion or other is None:
            return clear
        sides: tuple[Side, Side] = (view.side, Side(1 - view.side))
        meshes: Float32[Tensor, "2 v 3"] = torch.stack([mesh_vertices(self.model, pose, side) for pose, side in ((own, sides[0]), (other, sides[1]))])
        in_camera: Float32[Tensor, "1 2 v 3"] = world_to_cameras(camera, world_from_rig, meshes.reshape(-1, 3)).reshape(1, 2, -1, 3)
        pair: Float32[Tensor, "1 2 21 3"] = torch.cat([points, torch.full_like(points, torch.nan)], dim=1)
        hidden: Bool[Tensor, "21"] = keypoints_hidden(pair, in_camera, self.model.mesh_triangles, self._margin)[0, 0]
        return clear * (~hidden).to(torch.float32)

    def _weights(self, estimate: KeypointEstimate, index: int) -> Float32[Tensor, "21"]:
        """A view's keypoint weights: 0 for an empty heatmap, times the visibility weighting when it is on."""
        weights: Float32[Tensor, "21"] = (estimate.confidence[index] >= self.config.min_keypoint_confidence).to(torch.float32)
        if self.config.tip_weight != 1.0:
            weights = weights.clone()
            weights[:2] *= self.config.tip_weight  # LANDMARK 0 = thumb tip, 1 = index fingertip
        if self.config.visibility_weights == "off" or estimate.visibility is None:
            return weights
        probability: Float32[Tensor, "21"] = estimate.visibility[index].to(torch.float32)
        if self.config.visibility_weights == "hard":
            return weights * (probability >= 0.5).to(torch.float32)
        return weights * probability.clamp_min(self.config.visibility_floor)

    def _fit(self, hands: list[HandObservation], spent: dict[str, float]) -> list[FitResult]:
        begin: float = time.perf_counter()
        young: list[bool] = [self.history[hand.side].previous is not None and self.history[hand.side].age < self.config.young_frames for hand in hands]
        if any(young) and self.config.young_temporal_scale != 1.0:
            relaxed: FitConfig = replace(self.config.fit, temporal_weight=self.config.fit.temporal_weight * self.config.young_temporal_scale)
            results: list[FitResult | None] = [None] * len(hands)
            for flag, config in ((True, relaxed), (False, self.config.fit)):
                chosen: list[int] = [index for index, value in enumerate(young) if value == flag]
                if chosen:
                    fitted: list[FitResult] = fit_pose(self.model, self.phi, [hands[i] for i in chosen], [self.history[hands[i].side].previous for i in chosen], config)
                    for i, result in zip(chosen, fitted, strict=True):
                        results[i] = result
            spent["fit"] += time.perf_counter() - begin
            return [result for result in results if result is not None]
        results_all: list[FitResult] = fit_pose(self.model, self.phi, hands, [self.history[hand.side].previous for hand in hands], self.config.fit)
        if self.config.warm_restart_frames:
            warm: list[int] = [i for i, hand in enumerate(hands) if self.history[hand.side].previous is None and self.history[hand.side].lost_pose is not None
                               and self._frame - self.history[hand.side].lost_frame <= self.config.warm_restart_frames]
            if warm:
                starts: list[HandPose | None] = [self.history[hands[i].side].lost_pose for i in warm]
                refits: list[FitResult] = fit_pose(self.model, self.phi, [hands[i] for i in warm], starts, replace(self.config.fit, temporal_weight=0.0))
                for i, refit in zip(warm, refits, strict=True):
                    first: FitResult = results_all[i]
                    if refit.converged and _finite(refit.pose) and (not math.isfinite(first.energy) or refit.e_2d + refit.e_dist < first.e_2d + first.e_dist):
                        results_all[i] = refit
        spent["fit"] += time.perf_counter() - begin
        return results_all

    def _recrop_acquisitions(self, frame: int, images: UInt8[Tensor, "c 480 640"], world_from_rig: Float32[Tensor, "4 4"], views: list["_View"],
                             spent: dict[str, float], native_images: tuple[UInt8[Tensor, "h w"], ...]) -> list["_View"]:
        """One ``acquire_recrop`` pass: acquisition views (no planning pose) whose KeyNet presence passes get a new circle, the enclosing
        circle of KeyNet's keypoints in the net frame; tracked views are unchanged."""
        fresh: list[int] = [index for index, view in enumerate(views) if view.pose is None]
        if not fresh:
            return views
        scratch: _FrameOutput = _FrameOutput.empty(len(self.cameras))
        self._observe(frame, images, world_from_rig, [views[index] for index in fresh], scratch, spent, native_images)
        updated: list[_View] = list(views)
        for index in fresh:
            view: _View = views[index]
            presence: float = float(scratch.presence[view.camera, view.side])
            points: Float32[Tensor, "21 2"] = scratch.keypoints[view.camera, view.side]
            if not math.isfinite(presence) or presence < self.config.presence_threshold or not bool(torch.isfinite(points).all()):
                continue
            circle: Float32[Tensor, "3"] = torch.from_numpy(enclosing_circles(points.numpy()[None], np.ones((1, 21), dtype=bool)))[0]
            if bool(torch.isfinite(circle).all()):
                updated[index] = _View(view.side, view.camera, circle, _crop_map(circle, view.side), torch.zeros(63))
        return updated

    def _refine_acquisitions(self, frame: int, images: UInt8[Tensor, "c 480 640"], world_from_rig: Float32[Tensor, "4 4"], hands: list[HandObservation],
                             results: list[FitResult], out: "_FrameOutput", spent: dict[str, float],
                             native_images: tuple[UInt8[Tensor, "h w"], ...]) -> tuple[list[HandObservation], list[FitResult]]:
        """One ``acquire_refine`` pass: re-observe each freshly acquired, converged hand in the best cameras of its fitted pose and refit."""
        again: list[_View] = []
        for hand, result in zip(hands, results, strict=True):
            if self.history[hand.side].previous is not None or not result.converged or not _finite(result.pose):
                continue
            projection: _Projection = self._project(result.pose, hand.side, world_from_rig)
            order: list[int] = sorted((c for c in range(len(self.cameras)) if int(projection.inside[c]) > 0), key=lambda c: -int(projection.inside[c]))
            again.extend(self._view(hand.side, camera, projection, result.pose) for camera in order[: self.config.max_views])
        if not again:
            return hands, results
        scratch: _FrameOutput = _FrameOutput.empty(len(self.cameras))
        observed: tuple[list[HandObservation], list[Side]] = self._observe(frame, images, world_from_rig, again, scratch, spent, native_images)
        if not observed[0]:
            return hands, results
        by_side: dict[Side, FitResult] = {hand.side: result for hand, result in zip(hands, results, strict=True)}
        begin: float = time.perf_counter()
        refits: list[FitResult] = fit_pose(self.model, self.phi, observed[0], [by_side[hand.side].pose for hand in observed[0]],
                                           replace(self.config.fit, temporal_weight=0.0))
        spent["fit"] += time.perf_counter() - begin
        replaced: dict[Side, tuple[HandObservation, FitResult]] = {}
        for hand, refit in zip(observed[0], refits, strict=True):
            if refit.converged and _finite(refit.pose):
                replaced[hand.side] = (hand, refit)
                ran: Bool[Tensor, "c"] = torch.isfinite(scratch.presence[:, hand.side])
                out.keypoints[ran, hand.side] = scratch.keypoints[ran, hand.side]
                out.presence[ran, hand.side] = scratch.presence[ran, hand.side]
        pairs: list[tuple[HandObservation, FitResult]] = [replaced.get(hand.side, (hand, result)) for hand, result in zip(hands, results, strict=True)]
        return [hand for hand, _ in pairs], [result for _, result in pairs]

    def _refine(
        self,
        frame: int,
        images: UInt8[Tensor, "c 480 640"],
        world_from_rig: Float32[Tensor, "4 4"],
        views: list[_View],
        hands: list[HandObservation],
        results: list[FitResult],
        out: "_FrameOutput",
        spent: dict[str, float],
        native_images: tuple[UInt8[Tensor, "h w"], ...],
    ) -> dict[Side, tuple[HandObservation, FitResult]]:
        """Re-cut the crops around the fitted pose where it moved off its crop, run KeyNet and the fit once more (our addition).

        A hand is refined when the fitted pose's circle centre moved by more than ``refine_shift`` crop radii in one of its
        views. A refinement that KeyNet rejects (presence below the threshold in every view) keeps the first fit and its
        KeyNet outputs in the record; an accepted one replaces them.
        """
        assert self.config.refine_shift is not None
        again: list[_View] = []
        for hand, result in zip(hands, results, strict=True):
            if not _finite(result.pose):
                continue
            projection: _Projection = self._project(result.pose, hand.side, world_from_rig)
            mine: list[_View] = [view for view in views if view.side == hand.side and bool(projection.inside[view.camera] > 0)]
            moved: list[float] = [
                float((projection.circles[view.camera, :2] - view.circle[:2]).norm() / view.circle[2].clamp_min(MIN_BOX_RADIUS)) for view in mine
            ]
            if moved and max(moved) > self.config.refine_shift:
                again.extend(self._view(hand.side, view.camera, projection, result.pose) for view in mine)
        if not again:
            return {}
        scratch: _FrameOutput = _FrameOutput.empty(len(self.cameras))
        observed: tuple[list[HandObservation], list[Side]] = self._observe(frame, images, world_from_rig, again, scratch, spent, native_images)
        if not observed[0]:
            return {}
        for hand in observed[0]:
            ran: Bool[Tensor, "c"] = torch.isfinite(scratch.presence[:, hand.side])
            out.keypoints[ran, hand.side] = scratch.keypoints[ran, hand.side]
            out.presence[ran, hand.side] = scratch.presence[ran, hand.side]
        return {hand.side: (hand, result) for hand, result in zip(observed[0], self._fit(observed[0], spent), strict=True)}


@dataclass(slots=True)
class _FrameOutput:
    """``FrameResult`` while a frame is being filled in."""

    circle: Float32[Tensor, "c 2 3"]
    box_source: Int8[Tensor, "c 2"]
    keypoints: Float32[Tensor, "c 2 21 2"]
    presence: Float32[Tensor, "c 2"]
    detnet_presence: Float32[Tensor, "2"]
    fit_energy: Float32[Tensor, "2"]
    landmarks: Float32[Tensor, "2 21 3"]
    poses: list[HandPose | None]
    observations: list[HandObservation | None]
    detnet_camera: int = -1
    visibility: Float32[Tensor, "c 2 21"] | None = None
    pinch: Float32[Tensor, "c 2"] | None = None

    @staticmethod
    def empty(cameras: int) -> "_FrameOutput":
        return _FrameOutput(
            circle=torch.full((cameras, 2, 3), torch.nan),
            box_source=torch.zeros((cameras, 2), dtype=torch.int8),
            keypoints=torch.full((cameras, 2, 21, 2), torch.nan),
            presence=torch.full((cameras, 2), torch.nan),
            detnet_presence=torch.full((2,), torch.nan),
            fit_energy=torch.full((2,), torch.nan),
            landmarks=torch.full((2, 21, 3), torch.nan),
            poses=[None, None],
            observations=[None, None],
        )

    def result(self) -> FrameResult:
        return FrameResult(
            tracked=torch.tensor([pose is not None for pose in self.poses]),
            poses=(self.poses[0], self.poses[1]),
            landmarks=self.landmarks,
            circle=self.circle,
            box_source=self.box_source,
            keypoints=self.keypoints,
            presence=self.presence,
            detnet_camera=self.detnet_camera,
            detnet_presence=self.detnet_presence,
            fit_energy=self.fit_energy,
            observations=(self.observations[0], self.observations[1]),
            visibility=self.visibility,
            pinch=self.pinch,
        )
