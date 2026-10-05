"""Golden fixtures for robocap-live's Rust hand tracker, from handtrack's Python ``Tracker``.

Writes, under ``crates/robocap-live``:

- ``assets/generic_hand_model.json``: the generic UmeTrack hand model's arrays exactly as ``handtrack.fit.native`` hands them
  to handfit (float32 values, joints 0-19), plus the topology.
- ``tests/data/tracker/``: a synthetic RoboCap-like scene and the Python tracker's answers on it.
  - ``golden.json``: the rig (six 1920x1080 KB4 fisheye cameras), the headset pose per frame, and two Python runs with
    ``ROBUST_TRACKER_CONFIG`` and the native (handfit) fit, driven by the same recorded fake estimator outputs:
    ``calibration`` (generic model, phi 1, the first ``calibration_frames`` frames, as ``calibrate_unknown_hand`` tracks them),
    the scale calibration of its stereo observations (``calibrate_scale``), and ``tracking`` (every frame at the calibrated
    phi). Each frame records the KeyNet requests, the DetNet camera, the internal and reported state and the poses.
  - ``estimates.bin``: the fake KeyNet's output per (frame, camera, hand), float32 little-endian, 127 values each: points_net
    (21 x 2), points_px (21 x 2, = BarLetterbox.from_net(points_net) as the tracker computes it), d_rel_mm (21), confidence (21),
    presence (1).
  - ``detections.bin``: the fake DetNet's output per (frame, camera), float32: circle left (3), circle right (3), probability
    left, probability right (net frame).

The fake estimator's answer depends only on (frame, camera, hand), never on the crop, as in handtrack's tracker tests: the ground
truth projections of a hand 8 % smaller than the generic one, plus seeded noise and scripted events (a hand leaving, a noisy
re-acquisition, one rejected view, a lost headset pose).

Run from the repo root in the root pixi.toml's ``handtrack`` env, on the CPU (the native fit needs handfit's extension, built once
with ``pixi run -e handtrack --frozen handfit-build``):
    pixi run -e handtrack --frozen python packages/robocap-live/tools/golden_tracker.py
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from handtrack.fit.observations import HandObservation
from handtrack.fit.scale import ScaleCalibration, calibrate_scale, scaled_hand_model
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH, Letterbox
from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
from handtrack.labels.circles import enclosing_circles, square_boxes
from handtrack.labels.keypoint_input import relative_distances
from handtrack.models.detnet import Detections
from handtrack.tracker import ROBUST_TRACKER_CONFIG, CropRequest, FrameResult, KeypointEstimate, Tracker, TrackerConfig
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json
from torch import Tensor

CAMERA_NAMES: tuple[str, ...] = ("left_front", "right_front", "left_eye", "right_eye", "left", "right")
PHI_RANGE: tuple[float, float] = (0.75, 1.35)
"""``run_pipeline.PHI_RANGE``."""
ESTIMATE_VALUES: int = 21 * 2 + 21 * 2 + 21 + 21 + 1
DETECTION_VALUES: int = 8


@dataclass(frozen=True, slots=True)
class BarLetterbox(Letterbox):
    """robocap_track.py's 16:9 letterbox: full-width resize by 1/3, black bars of ``pad_y`` above and below."""

    pad_y: float = 0.0

    def __post_init__(self) -> None:
        pass

    def to_net(self, uv: Tensor) -> Tensor:
        scaled: Tensor = (uv + 0.5) * self.scale - 0.5
        return torch.stack((scaled[..., 0] + self.pad_x, scaled[..., 1] + self.pad_y), dim=-1)

    def from_net(self, uv: Tensor) -> Tensor:
        return (torch.stack((uv[..., 0] - self.pad_x, uv[..., 1] - self.pad_y), dim=-1) + 0.5) / self.scale - 0.5

    def apply(self, image: Tensor) -> Tensor:
        height: int = round(self.source_height * self.scale)
        width: int = round(self.source_width * self.scale)
        resized: Tensor = (
            F.interpolate(image.reshape(-1, 1, *image.shape[-2:]).float(), size=(height, width), mode="bilinear", antialias=True, align_corners=False)
            .round().to(torch.uint8).reshape(*image.shape[:-2], height, width)
        )
        return F.pad(resized, (int(self.pad_x), NET_WIDTH - width - int(self.pad_x), int(self.pad_y), NET_HEIGHT - height - int(self.pad_y)))


@dataclass(frozen=True, slots=True)
class Config:
    """Where to write and the scene's size."""

    crate: Path = Path(__file__).resolve().parents[1] / "crates" / "robocap-live"
    """The robocap-live crate directory."""
    frames: int = 100
    calibration_frames: int = 60
    true_phi: float = 0.92
    """The scene's hand relative to the generic hand."""
    seed: int = 7


@serde
@dataclass(frozen=True, slots=True)
class HandModelArrays:
    """``native._fitter``'s arrays (float32 values written as exact decimals)."""

    source: str
    joint_rotation_axes: list[list[float]]
    """20 x 3."""
    joint_rest_positions: list[list[float]]
    """20 x 3, millimetres."""
    landmark_rest_positions: list[list[float]]
    """21 x 3, millimetres."""
    landmark_rest_bone_weights: list[list[float]]
    """21 x 3."""
    landmark_rest_bone_indices: list[list[int]]
    """21 x 3."""
    joint_limits: list[list[float]]
    """20 x 2, radians."""
    topology: list[list[int]]
    """4 x 22: joint_parent, joint_frame_index, joint_first_child, joint_next_sibling."""


@serde
@dataclass(frozen=True, slots=True)
class RigCameraRecord:
    name: str
    width: int
    height: int
    cam_from_rig: list[list[float]]
    focal: list[float]
    principal: list[float]
    fisheye62: list[float] | None


@serde
@dataclass(frozen=True, slots=True)
class RigRecord:
    cameras: list[RigCameraRecord]
    source: str
    device: str


@serde
@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One KeyNet view a tracker asked for."""

    camera: int
    side: int
    acquisition: bool
    """No planning pose: the crop comes from ``circle``."""
    circle: list[float]
    """The circle in the net frame (DetNet's, the re-crop's, or the planning pose's)."""


@serde
@dataclass(frozen=True, slots=True)
class PoseRecord:
    rotation: list[float]
    """Row-major 3 x 3."""
    translation: list[float]
    joint_angles: list[float]
    """22."""


@serde
@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame: int
    detnet_camera: int
    """-1 when DetNet did not run; 6 (the camera count) when it ran on every camera (``detnet_all_cameras``)."""
    keynet_calls: list[list[RequestRecord]]
    """Every KeyNet call of the frame, in order (an acquisition's re-crop pass comes first)."""
    tracked: list[bool]
    """Per hand: tracked internally after the frame (θ(t) exists, reported or tentative)."""
    reported: list[bool]
    """Per hand: reported (past ``confirm_frames``)."""
    poses: list[PoseRecord | None]
    """θ(t) per hand when tracked."""
    landmarks: list[list[list[float]] | None]
    """World metres per reported hand."""
    view_cameras: list[list[int]]
    """The cameras of each reported hand's fit."""


@serde
@dataclass(frozen=True, slots=True)
class RunRecord:
    phi: float
    frames: list[FrameRecord]


@serde
@dataclass(frozen=True, slots=True)
class BlockView:
    camera: int
    weights: list[float]


@serde
@dataclass(frozen=True, slots=True)
class BlockRecord:
    """One calibration observation: the keypoints are the table's at (frame, camera, side)."""

    frame: int
    side: int
    initial: PoseRecord
    views: list[BlockView]


@serde
@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    phi_raw: float
    phi: float
    blocks: int
    iterations: int
    e_2d: float
    converged: bool
    termination: str
    hands: list[BlockRecord]
    """Every observation passed to ``calibrate_scale`` (stereo or not), in order."""


@serde
@dataclass(frozen=True, slots=True)
class GoldenRecord:
    format: str
    frames: int
    calibration_frames: int
    true_phi: float
    rig: RigRecord
    world_from_rig: list[list[float] | None]
    """Row-major 4 x 4 float32 values per frame; None = no headset pose."""
    calibration_run: RunRecord
    calibration: CalibrationRecord
    tracking_run: RunRecord
    all_cameras_run: RunRecord
    """``tracking_run`` with ``detnet_all_cameras=True`` (DetNet on every camera while a hand is untracked)."""
    events: list[str] = field(default_factory=list)


def _rotation(axis: int, angle: float) -> Float64[ndarray, "3 3"]:
    c, s = math.cos(angle), math.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix: Float64[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


CAMERAS: tuple[tuple[float, float, tuple[float, float, float]], ...] = (
    (-0.6, 0.45, (-0.06, 0.0, 0.03)),  # left_front: yaw, pitch down, position in the rig
    (0.6, 0.45, (0.06, 0.0, 0.03)),
    (-0.1, 0.6, (-0.03, 0.02, 0.05)),  # left_eye, right_eye: looking down
    (0.1, 0.6, (0.03, 0.02, 0.05)),
    (-1.5, 0.1, (-0.09, 0.0, -0.02)),  # left, right: sideways (the hands rarely appear there)
    (1.5, 0.1, (0.09, 0.0, -0.02)),
)


def _rig() -> tuple[CameraRig, RigRecord]:
    cam_from_rig_64: Float64[ndarray, "6 4 4"] = np.tile(np.eye(4), (6, 1, 1))
    for camera, (yaw, pitch, position) in enumerate(CAMERAS):
        rig_from_cam: Float64[ndarray, "3 3"] = _rotation(1, yaw) @ _rotation(0, -pitch)
        cam_from_rig_64[camera, :3, :3] = rig_from_cam.T
        cam_from_rig_64[camera, :3, 3] = -rig_from_cam.T @ np.asarray(position)
    cam_from_rig: Float32[ndarray, "6 4 4"] = cam_from_rig_64.astype(np.float32)
    focal: Float32[ndarray, "6 2"] = np.array([[1000.0, 1000.0]] * 6, dtype=np.float32) + np.arange(6, dtype=np.float32)[:, None] * 3.0
    principal: Float32[ndarray, "6 2"] = np.array([[959.5, 539.5]] * 6, dtype=np.float32) + np.array([[4.0, -3.0]], dtype=np.float32) * np.arange(6, dtype=np.float32)[:, None]
    fisheye: Float32[ndarray, "6 8"] = np.zeros((6, 8), dtype=np.float32)
    fisheye[:, :4] = np.array([0.031, -0.0072, 0.0011, -0.00009], dtype=np.float32) * (1.0 + 0.05 * np.arange(6, dtype=np.float32))[:, None]
    rig: CameraRig = CameraRig(
        names=tuple(f"/world/rig_00/cam_0{camera}" for camera in range(6)),
        image_size=torch.tensor([[1920.0, 1080.0]] * 6),
        cam_from_rig=torch.from_numpy(cam_from_rig),
        focal=torch.from_numpy(focal),
        principal=torch.from_numpy(principal),
        fisheye62=torch.from_numpy(fisheye),
    )
    record: RigRecord = RigRecord(
        cameras=[
            RigCameraRecord(
                name=CAMERA_NAMES[camera], width=1920, height=1080, cam_from_rig=[[float(x) for x in row] for row in cam_from_rig[camera]],
                focal=[float(x) for x in focal[camera]], principal=[float(x) for x in principal[camera]], fisheye62=[float(x) for x in fisheye[camera]],
            )
            for camera in range(6)
        ],
        source="synthetic golden_tracker.py scene",
        device="cap_a",
    )
    return rig, record


def _hand_rotation(side: Side) -> Float64[ndarray, "3 3"]:
    """Fingers forward and slightly down, palm down, in the rig frame (x right, y down, z forward)."""
    return _rotation(1, 0.35 if side == Side.LEFT else -0.35) @ _rotation(0, 1.2)


def _scene_poses(frames: int, limits: Float32[Tensor, "20 2"]) -> tuple[list[tuple[HandPose, HandPose]], list[Float32[ndarray, "4 4"] | None], list[str]]:
    """Per frame: both hands' θ in the world and the headset pose (None on the lost frame)."""
    poses: list[tuple[HandPose, HandPose]] = []
    headset: list[Float32[ndarray, "4 4"] | None] = []
    lower: Float64[ndarray, "20"] = limits[:, 0].numpy().astype(np.float64)
    upper: Float64[ndarray, "20"] = limits[:, 1].numpy().astype(np.float64)
    for frame in range(frames):
        t: float = frame / 30.0
        world_from_rig: Float64[ndarray, "4 4"] = np.eye(4)
        world_from_rig[:3, :3] = _rotation(1, 0.12 * math.sin(2 * math.pi * 0.15 * t)) @ _rotation(0, 0.05 * math.sin(2 * math.pi * 0.2 * t))
        world_from_rig[:3, 3] = (0.3 + 0.03 * math.sin(2 * math.pi * 0.2 * t), -1.5 + 0.01 * math.sin(2 * math.pi * 0.3 * t), 0.2 + 0.05 * t)
        pair: list[HandPose] = []
        for side in Side:
            sign: float = -1.0 if side == Side.LEFT else 1.0
            phase: float = 0.0 if side == Side.LEFT else 1.3
            wrist_rig: Float64[ndarray, "3"] = np.array([
                sign * 0.11 + 0.12 * math.sin(2 * math.pi * 0.4 * t + phase),
                0.24 + 0.03 * math.sin(2 * math.pi * 0.3 * t + phase),
                0.30 + 0.04 * math.sin(2 * math.pi * 0.25 * t + phase),
            ])
            rotation_rig: Float64[ndarray, "3 3"] = (
                _hand_rotation(side) @ _rotation(0, 0.2 * math.sin(2 * math.pi * 0.3 * t + phase)) @ _rotation(1, 0.25 * math.sin(2 * math.pi * 0.2 * t))
            )
            angles: Float64[ndarray, "22"] = np.zeros(22)
            angles[:20] = lower + (upper - lower) * (0.3 + 0.15 * np.sin(2 * math.pi * 0.35 * t + 0.4 * np.arange(20) + phase))
            pair.append(HandPose(
                torch.from_numpy((world_from_rig[:3, :3] @ rotation_rig).astype(np.float32)),
                torch.from_numpy((world_from_rig[:3, :3] @ wrist_rig + world_from_rig[:3, 3]).astype(np.float32)),
                torch.from_numpy(angles.astype(np.float32)),
            ))
        poses.append((pair[0], pair[1]))
        headset.append(world_from_rig.astype(np.float32))
    events: list[str] = ["frame 70: no headset pose (both tracks reset)"]
    headset[70] = None
    return poses, headset, events


@dataclass(frozen=True, slots=True)
class Tables:
    estimates: Float32[ndarray, "f c 2 127"]
    detections: Float32[ndarray, "f c 8"]
    events: list[str]


def _tables(config: Config, rig: CameraRig, letterbox: BarLetterbox, poses: list[tuple[HandPose, HandPose]],
            headset: list[Float32[ndarray, "4 4"] | None]) -> Tables:
    generic = generic_hand_model()
    truth = scaled_hand_model(generic, config.true_phi)
    generator: np.random.Generator = np.random.default_rng(config.seed)
    estimates: Float32[ndarray, "f c 2 127"] = np.zeros((config.frames, 6, 2, ESTIMATE_VALUES), dtype=np.float32)
    detections: Float32[ndarray, "f c 8"] = np.zeros((config.frames, 6, DETECTION_VALUES), dtype=np.float32)
    events: list[str] = [
        "frames 40-47: the right hand leaves (presence 0.05 everywhere, DetNet 0.1)",
        "frames 48-50: the right hand comes back with 30 px keypoint noise (relative-RMS gate)",
        "frame 25: the left hand's camera-3 view has presence 0.3 (end_on_view_rejection)",
    ]
    for frame in range(config.frames):
        world_from_rig: Tensor = torch.from_numpy(headset[frame] if headset[frame] is not None else headset[frame - 1])
        for side in Side:
            points_world: Float32[Tensor, "21 3"] = landmarks(truth, poses[frame][side], side)
            points_cam: Float32[Tensor, "6 21 3"] = world_to_cameras(rig, world_from_rig, points_world)
            pixels: Float32[Tensor, "6 21 2"] = project(rig, points_cam)
            front: Bool[Tensor, "6 21"] = in_front(points_cam)
            visible: Bool[Tensor, "6 21"] = front & inside_image(rig, pixels)
            count: Int64[Tensor, "6"] = visible.sum(-1)
            noise_px: float = 30.0 if side == Side.RIGHT and 48 <= frame <= 50 else 1.2
            noisy: Float32[Tensor, "6 21 2"] = pixels + torch.from_numpy(generator.normal(0.0, noise_px, (6, 21, 2)).astype(np.float32))
            noisy = torch.where(front[..., None], noisy, torch.zeros_like(noisy))
            net: Float32[Tensor, "6 21 2"] = letterbox.to_net(noisy)
            back: Float32[Tensor, "6 21 2"] = letterbox.from_net(net)
            d_rel: Float32[Tensor, "6 21"] = relative_distances(points_cam, torch.full((6,), config.true_phi)) + torch.from_numpy(
                generator.normal(0.0, 2.0, (6, 21)).astype(np.float32))
            confidence: Float32[Tensor, "6 21"] = visible.to(torch.float32)
            presence: Float32[Tensor, "6"] = torch.where(count >= 8, 0.95, 0.05)
            probability: Float32[Tensor, "6"] = torch.where(count >= 12, 0.9, 0.1)
            if side == Side.RIGHT and 40 <= frame <= 47:
                presence = torch.full((6,), 0.05)
                probability = torch.full((6,), 0.1)
            if side == Side.LEFT and frame == 25:
                presence[3] = min(float(presence[3]), 0.3)
            circles: Float32[ndarray, "6 3"] = enclosing_circles(letterbox.to_net(pixels).numpy(), front.numpy())
            circles[:, :2] += np.array([6.0, -4.0], dtype=np.float32)
            circles[:, 2] *= 1.25
            estimates[frame, :, side] = np.concatenate(
                [net.reshape(6, 42).numpy(), back.reshape(6, 42).numpy(), d_rel.numpy(), confidence.numpy(), presence[:, None].numpy()], axis=1)
            detections[frame, :, 3 * side : 3 * side + 3] = circles
            detections[frame, :, 6 + side] = probability.numpy()
    return Tables(estimates, detections, events)


@dataclass(slots=True)
class TableDetector:
    tables: Tables
    calls: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self, images: UInt8[Tensor, "b 480 640"], frame: int, cameras: Int64[Tensor, "b"]) -> Detections:
        rows: Float32[Tensor, "b 8"] = torch.from_numpy(self.tables.detections[frame, cameras.numpy()])
        circle: Float32[Tensor, "b 2 3"] = rows[:, :6].reshape(-1, 2, 3)
        probability: Float32[Tensor, "b 2"] = rows[:, 6:8]
        self.calls.extend((frame, int(camera)) for camera in cameras)
        return Detections(circle=circle, probability=probability, present=probability > 0.5, box=square_boxes(circle))


@dataclass(slots=True)
class TableKeyNet:
    tables: Tables
    calls: list[tuple[int, list[RequestRecord]]] = field(default_factory=list)

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        assert request.circles is not None
        cameras: list[int] = [int(c) for c in request.camera]
        sides: list[int] = [int(s) for s in request.side]
        self.calls.append((frame, [
            RequestRecord(camera=c, side=s, acquisition=request.poses[s] is None, circle=[float(x) for x in request.circles[i]])
            for i, (c, s) in enumerate(zip(cameras, sides, strict=True))
        ]))
        rows: Float32[Tensor, "n 127"] = torch.from_numpy(self.tables.estimates[frame, cameras, sides])
        return KeypointEstimate(
            points_net=rows[:, :42].reshape(-1, 21, 2),
            d_rel_mm=rows[:, 84:105],
            presence=rows[:, 126],
            confidence=rows[:, 105:126],
        )


def _pose_record(pose: HandPose) -> PoseRecord:
    return PoseRecord(rotation=[float(x) for x in pose.rotation.reshape(9)], translation=[float(x) for x in pose.translation],
                      joint_angles=[float(x) for x in pose.joint_angles])


def _run(rig: CameraRig, letterboxes: tuple[Letterbox, ...], phi: float, frames: int, tables: Tables,
         headset: list[Float32[ndarray, "4 4"] | None], config: TrackerConfig) -> tuple[RunRecord, list[FrameResult]]:
    generic = generic_hand_model()
    model = generic if phi == 1.0 else scaled_hand_model(generic, phi)
    detector: TableDetector = TableDetector(tables)
    keynet: TableKeyNet = TableKeyNet(tables)
    tracker: Tracker = Tracker(rig, letterboxes, model, phi, detector, keynet, config)
    images: UInt8[Tensor, "6 480 640"] = torch.zeros((6, NET_HEIGHT, NET_WIDTH), dtype=torch.uint8)
    records: list[FrameRecord] = []
    results: list[FrameResult] = []
    for frame in range(frames):
        world_from_rig: Tensor = torch.from_numpy(headset[frame]) if headset[frame] is not None else torch.full((4, 4), torch.nan)
        before: int = len(keynet.calls)
        result: FrameResult = tracker.step(frame, images, world_from_rig)
        results.append(result)
        previous: list[HandPose | None] = [tracker.history[side].previous for side in Side]
        records.append(FrameRecord(
            frame=frame,
            detnet_camera=result.detnet_camera,
            keynet_calls=[requests for _, requests in keynet.calls[before:]],
            tracked=[pose is not None for pose in previous],
            reported=[bool(x) for x in result.tracked],
            poses=[None if pose is None else _pose_record(pose) for pose in previous],
            landmarks=[None if not bool(result.tracked[side]) else [[float(x) for x in row] for row in result.landmarks[side]] for side in Side],
            view_cameras=[[] if observation is None else [int(rig.names.index(view.camera.names[0])) for view in observation.views]
                          for observation in result.observations],
        ))
    return RunRecord(phi=phi, frames=records), results


def _calibrate(results: list[FrameResult], rig: CameraRig) -> CalibrationRecord:
    """``run_pipeline.calibrate_unknown_hand`` on a finished φ = 1 run."""
    hands: list[HandObservation] = []
    initial: list[HandPose | None] = []
    blocks: list[BlockRecord] = []
    for frame, result in enumerate(results):
        for side, (observation, pose) in enumerate(zip(result.observations, result.poses, strict=True)):
            if observation is not None and pose is not None:
                hands.append(observation)
                initial.append(pose)
                blocks.append(BlockRecord(frame=frame, side=side, initial=_pose_record(pose), views=[
                    BlockView(camera=rig.names.index(view.camera.names[0]), weights=[float(x) for x in view.weights]) for view in observation.views
                ]))
    calibration: ScaleCalibration = calibrate_scale(generic_hand_model(), hands, initial)
    phi: float = min(max(calibration.phi, PHI_RANGE[0]), PHI_RANGE[1])
    return CalibrationRecord(phi_raw=calibration.phi, phi=phi, blocks=calibration.blocks, iterations=calibration.iterations, e_2d=calibration.e_2d,
                             converged=calibration.converged, termination=calibration.termination, hands=blocks)


def _export_hand_model(path: Path) -> None:
    model = generic_hand_model()
    arrays: HandModelArrays = HandModelArrays(
        source="handtrack generic_hand_model() (UmeTrack dataset/generic_hand_model.json) as handtrack.fit.native passes it to handfit",
        joint_rotation_axes=model.joint_rotation_axes[:20].tolist(),
        joint_rest_positions=model.joint_rest_positions[:20].tolist(),
        landmark_rest_positions=model.landmark_rest_positions.tolist(),
        landmark_rest_bone_weights=model.landmark_rest_bone_weights.tolist(),
        landmark_rest_bone_indices=model.landmark_rest_bone_indices.tolist(),
        joint_limits=model.joint_limits[:20].tolist(),
        topology=[model.joint_parent.tolist(), model.joint_frame_index.tolist(), model.joint_first_child.tolist(), model.joint_next_sibling.tolist()],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(arrays) + "\n")
    print(f"wrote {path}")


def main(config: Config) -> None:
    torch.set_num_threads(1)
    _export_hand_model(config.crate / "assets" / "generic_hand_model.json")
    rig, rig_record = _rig()
    letterbox: BarLetterbox = BarLetterbox(1920, 1080, False, 1.0 / 3.0, 0.0, 60.0)
    letterboxes: tuple[Letterbox, ...] = (letterbox,) * 6
    generic = generic_hand_model()
    poses, headset, events = _scene_poses(config.frames, generic.joint_limits[:20])
    tables: Tables = _tables(config, rig, letterbox, poses, headset)
    tracker_config: TrackerConfig = replace(ROBUST_TRACKER_CONFIG, fit_backend="native")
    calibration_run, calibration_results = _run(rig, letterboxes, 1.0, config.calibration_frames, tables, headset, tracker_config)
    calibration: CalibrationRecord = _calibrate(calibration_results, rig)
    print(f"calibration: phi {calibration.phi_raw:.5f} from {calibration.blocks} stereo blocks of {len(calibration.hands)} observations, "
          f"{calibration.iterations} iterations, {calibration.termination}")
    tracking_run, _ = _run(rig, letterboxes, calibration.phi, config.frames, tables, headset, tracker_config)
    all_cameras_run, _ = _run(rig, letterboxes, calibration.phi, config.frames, tables, headset, replace(tracker_config, detnet_all_cameras=True))
    for name, run in (("calibration", calibration_run), ("tracking", tracking_run), ("all cameras", all_cameras_run)):
        tracked: Bool[ndarray, "f 2"] = np.array([frame.tracked for frame in run.frames])
        reported: Bool[ndarray, "f 2"] = np.array([frame.reported for frame in run.frames])
        print(f"{name}: tracked {tracked.mean(axis=0)}, reported {reported.mean(axis=0)}, KeyNet calls {sum(len(f.keynet_calls) for f in run.frames)}, "
              f"DetNet runs {sum(f.detnet_camera >= 0 for f in run.frames)}")
        print("  reported per frame L:", "".join("1" if f.reported[0] else ("t" if f.tracked[0] else ".") for f in run.frames))
        print("  reported per frame R:", "".join("1" if f.reported[1] else ("t" if f.tracked[1] else ".") for f in run.frames))
    record: GoldenRecord = GoldenRecord(
        format="robocap-live-tracker-golden/1", frames=config.frames, calibration_frames=config.calibration_frames, true_phi=config.true_phi,
        rig=rig_record, world_from_rig=[None if h is None else [float(x) for x in h.reshape(16)] for h in headset],
        calibration_run=calibration_run, calibration=calibration, tracking_run=tracking_run, all_cameras_run=all_cameras_run,
        events=events + tables.events,
    )
    directory: Path = config.crate / "tests" / "data" / "tracker"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "golden.json").write_text(to_json(record) + "\n")
    tables.estimates.astype("<f4").tofile(directory / "estimates.bin")
    tables.detections.astype("<f4").tofile(directory / "detections.bin")
    print(f"wrote {directory}: golden.json {(directory / 'golden.json').stat().st_size / 1e3:.0f} kB, "
          f"estimates.bin {(directory / 'estimates.bin').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(tyro.cli(Config))
