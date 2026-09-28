"""The pose fit and the scale calibration on a real UmeTrack testing recording, from ground-truth keypoints plus noise.

Per frame and hand, the ground-truth landmarks (the recording's profile, ``pose.landmarks``) are projected into the four
cameras; the two cameras with the most keypoints inside (at least 17) are the views, with 1.5 px of noise on the 2D keypoints
and the exact d_rel. The known hand is tracked with the profile; the unknown hand with the generic model scaled by the ϕ
that ``calibrate_scale`` finds on the first 100 frames, tracked first with the generic model (the §5.1 protocol). MKPE is
the mean 3D landmark error in millimetres.
"""

import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import pyarrow as pa
import pytest
import torch
from jaxtyping import Bool, Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import MAX_VIEWS, HandObservation, ViewObservation
from handtrack.fit.pose_fit import FitConfig, FitResult, fit_pose
from handtrack.fit.scale import ScaleCalibration, calibrate_scale, scaled_hand_model
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.hand.pose import HandPose, Side, generic_hand_model, hand_model_from_profile, landmarks
from handtrack.labels.keypoint_input import hand_scale, relative_distances

pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")

import rerun as rr  # noqa: E402

CATALOG_URL: str = os.environ.get("HANDTRACK_CATALOG_URL", "rerun+http://127.0.0.1:51235")
DATASET: str = "dataforge-umetrack"
SEGMENT: str = "umetrack__real__hand_hand__testing__user_05__recording_00"
CAMERAS: tuple[str, ...] = tuple(f"/world/rig_00/cam_0{i}" for i in range(4))
MIN_INSIDE: int = 17
NOISE_PX: float = 1.5
TRACK_FRAMES: int = 60
CALIBRATION_FRAMES: int = 100


@dataclass(frozen=True, slots=True)
class _Recording:
    """What the test needs from one UmeTrack segment."""

    rig: CameraRig
    """The four cameras."""
    world_from_rig: Float32[Tensor, "t 4 4"]
    """NaN where the headset is untracked."""
    poses: dict[Side, HandPose]
    """Ground-truth θ per hand over all frames."""
    confidence: Float32[Tensor, "t 2"]
    """Per-hand confidence (0 = absent), left then right."""
    profile: str
    """The subject's hand profile JSON."""


def _require_catalog() -> None:
    url = urlparse(CATALOG_URL.replace("rerun+http://", "http://", 1))
    try:
        with socket.create_connection((url.hostname or "127.0.0.1", url.port or 80), timeout=2.0):
            pass
    except OSError as error:
        pytest.skip(f"catalog server {CATALOG_URL} is not reachable ({error}); it serves {DATASET}")


def _rotation_from_quaternion(xyzw: Float32[Tensor, "t 4"]) -> Float32[Tensor, "t 3 3"]:
    q: Float32[Tensor, "t 4"] = xyzw / xyzw.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    rows: list[Float32[Tensor, "t 3"]] = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def _column(table: pa.Table, name: str, width: int) -> Float32[Tensor, "t k"]:
    """One instance per row of a list<fixed_size_list<float>> (or list<double>) column; NaN where the row has none."""
    values: list[list[float]] = [
        [float("nan")] * width if not row else [float(v) for v in (row[0] if isinstance(row[0], list) else row)] for row in table[name].to_pylist()
    ]
    return torch.tensor(values, dtype=torch.float32)


def _read_recording() -> _Recording:
    dataset = rr.catalog.CatalogClient(CATALOG_URL).get_dataset(DATASET)
    view = dataset.filter_segments(SEGMENT)
    static = view.filter_contents([*CAMERAS, *(f"{c}/pinhole" for c in CAMERAS), "/world/gt/hands/profile"]).reader(index=None).to_arrow_table()

    def first(name: str) -> list[float]:
        return static[name][0].as_py()[0]

    cam_from_rig: Float32[Tensor, "4 4 4"] = torch.eye(4).repeat(4, 1, 1)
    focal: Float32[Tensor, "4 2"] = torch.zeros(4, 2)
    principal: Float32[Tensor, "4 2"] = torch.zeros(4, 2)
    fisheye62: Float32[Tensor, "4 8"] = torch.zeros(4, 8)
    image_size: Float32[Tensor, "4 2"] = torch.zeros(4, 2)
    for i, camera in enumerate(CAMERAS):
        assert static[f"{camera}:Transform3D:relation"][0].as_py()[0] == 2  # ChildFromParent: the value is cam_from_rig
        cam_from_rig[i, :3, :3] = torch.tensor(first(f"{camera}:Transform3D:mat3x3")).reshape(3, 3).T  # column-major
        cam_from_rig[i, :3, 3] = torch.tensor(first(f"{camera}:Transform3D:translation"))
        k: Float32[Tensor, "3 3"] = torch.tensor(first(f"{camera}/pinhole:Pinhole:image_from_camera")).reshape(3, 3).T
        focal[i] = torch.stack([k[0, 0], k[1, 1]])
        principal[i] = torch.stack([k[0, 2], k[1, 2]])
        fisheye62[i] = torch.tensor(first(f"{camera}/pinhole:simplecv.components.DistortionCoefficients"))
        image_size[i] = torch.tensor(first(f"{camera}/pinhole:Pinhole:resolution"))
    rig: CameraRig = CameraRig(names=CAMERAS, image_size=image_size, cam_from_rig=cam_from_rig, focal=focal, principal=principal, fisheye62=fisheye62)
    columns: list[str] = ["/world/rig_00:Transform3D:quaternion", "/world/rig_00:Transform3D:translation"]
    for side in ("left", "right"):
        columns += [
            f"/world/gt/hands/{side}/wrist:Transform3D:quaternion",
            f"/world/gt/hands/{side}/wrist:Transform3D:translation",
            f"/world/gt/hands/{side}/joint_angles:joint_angles",
            f"/world/gt/hands/{side}/confidence:Scalars:scalars",
        ]
    table = (
        view.filter_contents(["/world/rig_00", "/world/gt/hands/left/**", "/world/gt/hands/right/**"])
        .reader(index="video_time")
        .select("video_time", *columns)
        .to_arrow_table()
        .sort_by("video_time")
    )
    world_from_rig: Float32[Tensor, "t 4 4"] = torch.eye(4).repeat(table.num_rows, 1, 1)
    world_from_rig[:, :3, :3] = _rotation_from_quaternion(_column(table, columns[0], 4))
    world_from_rig[:, :3, 3] = _column(table, columns[1], 3)
    poses: dict[Side, HandPose] = {}
    confidence: list[Float32[Tensor, "t"]] = []
    for side, name in ((Side.LEFT, "left"), (Side.RIGHT, "right")):
        prefix: str = f"/world/gt/hands/{name}"
        poses[side] = HandPose(
            rotation=_rotation_from_quaternion(_column(table, f"{prefix}/wrist:Transform3D:quaternion", 4)),
            translation=_column(table, f"{prefix}/wrist:Transform3D:translation", 3),
            joint_angles=_column(table, f"{prefix}/joint_angles:joint_angles", 22),
        )
        confidence.append(torch.nan_to_num(_column(table, f"{prefix}/confidence:Scalars:scalars", 1)[:, 0]))
    return _Recording(
        rig=rig,
        world_from_rig=world_from_rig,
        poses=poses,
        confidence=torch.stack(confidence, dim=-1),
        profile=first("/world/gt/hands/profile:TextDocument:text"),
    )


def _one_camera(rig: CameraRig, index: int) -> CameraRig:
    return CameraRig(
        names=(rig.names[index],),
        image_size=rig.image_size[index : index + 1],
        cam_from_rig=rig.cam_from_rig[index : index + 1],
        focal=rig.focal[index : index + 1],
        principal=rig.principal[index : index + 1],
        fisheye62=None if rig.fisheye62 is None else rig.fisheye62[index : index + 1],
    )


def _pose_at(poses: HandPose, frame: int) -> HandPose:
    return HandPose(poses.rotation[frame], poses.translation[frame], poses.joint_angles[frame])


def observation(
    recording: _Recording,
    truth: Float32[Tensor, "21 3"],
    frame: int,
    side: Side,
    phi: float,
    generator: torch.Generator,
    noise_px: float = NOISE_PX,
    noise_mm: float = 0.0,
) -> HandObservation | None:
    """The hand as a perfect KeyNet on the (up to) two best views would see it, plus Gaussian noise; None if no view has 17 keypoints inside."""
    world_from_rig: Float32[Tensor, "4 4"] = recording.world_from_rig[frame]
    points_cam: Float32[Tensor, "4 21 3"] = world_to_cameras(recording.rig, world_from_rig, truth)
    pixels: Float32[Tensor, "4 21 2"] = project(recording.rig, points_cam)
    seen: Bool[Tensor, "4 21"] = in_front(points_cam) & inside_image(recording.rig, pixels)
    counts: list[int] = seen.sum(-1).tolist()
    ranked: list[int] = sorted((c for c in range(len(counts)) if counts[c] >= MIN_INSIDE), key=lambda c: -counts[c])[:MAX_VIEWS]
    if not ranked:
        return None
    views: list[ViewObservation] = []
    for c in ranked:
        views.append(
            ViewObservation(
                camera=_one_camera(recording.rig, c),
                world_from_rig=world_from_rig,
                keypoints_px=pixels[c] + noise_px * torch.randn(21, 2, generator=generator),
                weights=seen[c].to(torch.float32),
                d_rel_mm=relative_distances(points_cam[c], torch.tensor(phi)) + noise_mm * torch.randn(21, generator=generator),
            )
        )
    return HandObservation(side=side, views=tuple(views))


def track(
    model: HandModelTorch,
    phi: float,
    frames: list[dict[Side, HandObservation]],
    config: FitConfig = FitConfig(),  # noqa: B008 - frozen
) -> list[dict[Side, FitResult]]:
    """Fit every frame from the previous frame's fit; a hand with no observation loses its history."""
    previous: dict[Side, HandPose | None] = {Side.LEFT: None, Side.RIGHT: None}
    fitted: list[dict[Side, FitResult]] = []
    for observed in frames:
        sides: list[Side] = list(observed)
        results: list[FitResult] = fit_pose(model, phi, [observed[s] for s in sides], [previous[s] for s in sides], config) if sides else []
        previous = {side: None for side in Side}
        previous.update({side: result.pose for side, result in zip(sides, results, strict=True)})
        fitted.append(dict(zip(sides, results, strict=True)))
    return fitted


def mkpe_mm(model: HandModelTorch, fitted: list[dict[Side, FitResult]], truths: list[dict[Side, Float32[Tensor, "21 3"]]]) -> float:
    errors: list[float] = [
        float((landmarks(model, result.pose, side) - truths[t][side]).norm(dim=-1).mean()) * 1000.0
        for t, frame in enumerate(fitted)
        for side, result in frame.items()
    ]
    return sum(errors) / len(errors)


def scenario(
    noise_px: float = NOISE_PX, noise_mm: float = 0.0, seed: int = 0
) -> tuple[_Recording, list[dict[Side, Float32[Tensor, "21 3"]]], list[dict[Side, HandObservation]], float]:
    """The recording, its ground-truth landmarks per frame and hand, the noisy observations, and the profile's ϕ."""
    recording: _Recording = _read_recording()
    profile: HandModelTorch = hand_model_from_profile(recording.profile)
    phi: float = hand_scale(profile, generic_hand_model())
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    truths: list[dict[Side, Float32[Tensor, "21 3"]]] = []
    frames: list[dict[Side, HandObservation]] = []
    for frame in range(recording.world_from_rig.shape[0]):
        truth: dict[Side, Float32[Tensor, "21 3"]] = {}
        observed: dict[Side, HandObservation] = {}
        if bool(torch.isfinite(recording.world_from_rig[frame]).all()):
            for side in Side:
                if float(recording.confidence[frame, side]) <= 0.0:
                    continue
                truth[side] = landmarks(profile, _pose_at(recording.poses[side], frame), side)
                seen: HandObservation | None = observation(recording, truth[side], frame, side, phi, generator, noise_px, noise_mm)
                if seen is not None:
                    observed[side] = seen
        truths.append(truth)
        frames.append(observed)
    return recording, truths, frames, phi


@pytest.mark.integration
def test_tracks_a_real_recording_with_known_and_calibrated_hands() -> None:
    """Measured (1.5 px): known hand 1.6 mm, generic x calibrated ϕ 3.1 mm, generic x 1 8.7 mm over the first 60 frames.

    The calibrated ϕ (1.096) differs from the profile's least-squares ϕ (1.065) because the subject's hand is not a scaled
    generic hand; the generic model tracks better at the calibrated ϕ than at the profile's ϕ (4.2 mm).
    """
    _require_catalog()
    recording, truths, frames, phi = scenario()
    profile: HandModelTorch = hand_model_from_profile(recording.profile)
    generic: HandModelTorch = generic_hand_model()
    known: float = mkpe_mm(profile, track(profile, phi, frames[:TRACK_FRAMES]), truths)
    # §5.1: track the first 100 frames with the generic model, calibrate ϕ from those poses, then track again at generic x ϕ.
    first_pass: list[dict[Side, FitResult]] = track(generic, 1.0, frames[:CALIBRATION_FRAMES])
    uncalibrated: float = mkpe_mm(generic, first_pass[:TRACK_FRAMES], truths)
    blocks: list[tuple[HandObservation, HandPose]] = [
        (frame[side], first_pass[t][side].pose) for t, frame in enumerate(frames[:CALIBRATION_FRAMES]) for side in frame
    ]
    calibration: ScaleCalibration = calibrate_scale(generic, [block for block, _ in blocks], [pose for _, pose in blocks])
    scaled: HandModelTorch = scaled_hand_model(generic, calibration.phi)
    unknown: float = mkpe_mm(scaled, track(scaled, calibration.phi, frames[:TRACK_FRAMES]), truths)
    fitted_frames: int = sum(len(frame) for frame in frames[:TRACK_FRAMES])
    print(
        f"\n{SEGMENT}: {fitted_frames} hand-frames, profile phi {phi:.4f}, calibrated phi {calibration.phi:.4f} from {calibration.blocks} stereo blocks"
    )
    print(f"MKPE known hand {known:.2f} mm, unknown hand {unknown:.2f} mm (generic x calibrated phi), {uncalibrated:.2f} mm (generic x 1)")
    assert fitted_frames >= TRACK_FRAMES
    assert abs(calibration.phi - phi) < 0.05
    assert known < 3.0
    assert unknown < 5.0
    assert unknown < 0.6 * uncalibrated
