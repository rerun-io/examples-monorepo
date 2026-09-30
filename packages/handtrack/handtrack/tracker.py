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

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch
from jaxtyping import Bool, Float32, Int8, Int64, UInt8
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import MAX_VIEWS, HandObservation, ViewObservation
from handtrack.fit.pose_fit import FitConfig, FitResult, fit_pose
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox
from handtrack.hand.pose import HandPose, Side, extrapolate, landmarks
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import apply_affine, crop_boxes, crop_from_net, cut_crops
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.labels.keypoint_input import keypoint_input, relative_distances
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


@dataclass(slots=True)
class _History:
    previous: HandPose | None = None
    """θ(t−1); None while the hand is untracked."""
    before: HandPose | None = None
    """θ(t−2); None after the first tracked frame."""
    detnet_misses: int = 0
    """Consecutive UmeTrack frames without detector support."""


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
        self.timings_s: dict[str, float] = {"detnet": 0.0, "keynet": 0.0, "fit": 0.0, "tracker": 0.0}
        """Seconds per stage, summed over frames (``tracker`` is the rest: projection, circles, bookkeeping)."""

    def _drop(self, side: Side) -> None:
        self.history[side].previous = None
        self.history[side].before = None
        self.history[side].detnet_misses = 0

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
        """DetNet on this frame's round-robin camera; an untracked hand it reports gets an acquisition view (zero keypoint input)."""
        camera: int = self.next_detnet_camera
        self.next_detnet_camera = (camera + 1) % len(self.cameras)
        detections: Detections = self.detector(images[camera : camera + 1], frame, torch.tensor([camera]))
        out.detnet_camera = camera
        out.detnet_presence = detections.probability[0].clone()
        views: list[_View] = []
        for side in untracked:
            if float(detections.probability[0, side]) > self.config.detnet_threshold:
                out.circle[camera, side] = detections.circle[0, side]
                out.box_source[camera, side] = int(BoxSource.DETNET)
                views.append(_View(side, camera, detections.circle[0, side], _crop_map(detections.circle[0, side], side), torch.zeros(63)))
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
        observed: tuple[list[HandObservation], list[Side]] = self._observe(frame, images, world_from_rig, views, out, spent, native_images)
        for side in observed[1]:
            self._drop(side)
        hands: list[HandObservation] = observed[0]
        if not hands:
            return
        results: list[FitResult] = self._fit(hands, spent)
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
            if (acquiring and not result.converged) or not _finite(result.pose) or reach > self.config.max_reach_m:
                self._drop(hand.side)
                continue
            history: _History = self.history[hand.side]
            history.before = history.previous
            history.previous = result.pose
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
            good: list[int] = [index for index in mine if float(estimate.presence[index]) >= self.config.presence_threshold]
            if self.config.min_visible_keypoints and estimate.visibility is not None:
                good = [index for index in good if int((estimate.visibility[index] >= 0.5).sum()) >= self.config.min_visible_keypoints]
            if self.config.end_on_view_rejection and self.history[side].previous is not None and len(mine) >= 2 and len(good) == 1:
                good = []  # the pose is slipping off the hand: a one-view fit from here is poor; DetNet re-acquires next frame
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
                        weights=self._weights(estimate, index),
                        d_rel_mm=estimate.d_rel_mm[index],
                    )
                    for index in good
                )
                hands.append(HandObservation(side=side, views=views_seen))
        return hands, rejected

    def _weights(self, estimate: KeypointEstimate, index: int) -> Float32[Tensor, "21"]:
        """A view's keypoint weights: 0 for an empty heatmap, times the visibility weighting when it is on."""
        weights: Float32[Tensor, "21"] = (estimate.confidence[index] >= self.config.min_keypoint_confidence).to(torch.float32)
        if self.config.visibility_weights == "off" or estimate.visibility is None:
            return weights
        probability: Float32[Tensor, "21"] = estimate.visibility[index].to(torch.float32)
        if self.config.visibility_weights == "hard":
            return weights * (probability >= 0.5).to(torch.float32)
        return weights * probability.clamp_min(self.config.visibility_floor)

    def _fit(self, hands: list[HandObservation], spent: dict[str, float]) -> list[FitResult]:
        begin: float = time.perf_counter()
        results: list[FitResult] = fit_pose(self.model, self.phi, hands, [self.history[hand.side].previous for hand in hands], self.config.fit)
        spent["fit"] += time.perf_counter() - begin
        return results

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
        )
