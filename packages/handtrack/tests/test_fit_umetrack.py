"""The pose fit and the scale calibration on a real UmeTrack testing recording, from ground-truth keypoints plus noise.

Per frame and hand, the ground-truth landmarks (the recording's profile, ``pose.landmarks``) are projected into the four
cameras; the two cameras with the most keypoints inside (at least 17) are the views, with 1.5 px of noise on the 2D keypoints
and the exact d_rel. The known hand is tracked with the profile; the unknown hand with the generic model scaled by the ϕ
that ``calibrate_scale`` finds on the first 100 frames, tracked first with the generic model (the §5.1 protocol). MKPE is
the mean 3D landmark error in millimetres.
"""

import pyarrow as pa
import pytest
import torch
from beartype.roar import BeartypeException
from jaxtyping import Bool, Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import MAX_VIEWS, HandObservation, ViewObservation
from handtrack.fit.pose_fit import DEFAULT_CONFIG, FitConfig, FitResult, fit_pose
from handtrack.fit.scale import ScaleCalibration, calibrate_scale, scaled_hand_model
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
from handtrack.labels.keypoint_input import relative_distances
from handtrack.labels.validity import MIN_VISIBLE_KEYPOINTS

rr = pytest.importorskip("rerun", reason="needs rerun-sdk with the catalog extra")

from handtrack.data.catalog import (  # noqa: E402
    CATALOG_URL,
    UMETRACK,
    HandTimeline,
    SegmentInfo,
    list_segments,
    read_hand_timeline,
    read_rig,
    read_statics,
)

SEGMENT: str = "umetrack__real__hand_hand__testing__user_05__recording_00"
NOISE_PX: float = 1.5
TRACK_FRAMES: int = 60
CALIBRATION_FRAMES: int = 100


def _read_recording() -> tuple[CameraRig, HandTimeline]:
    """The segment's four cameras and its ground-truth timeline, through the package's catalog readers; skips without the catalog."""
    try:
        dataset = rr.catalog.CatalogClient(CATALOG_URL).get_dataset(UMETRACK)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} dataset {UMETRACK} unreachable: {error}")
    infos: dict[str, SegmentInfo] = {info.segment_id: info for info in list_segments(dataset, UMETRACK)}
    if SEGMENT not in infos:
        pytest.skip(f"{SEGMENT} is not registered in {UMETRACK}")
    statics: pa.Table = read_statics(dataset, infos[SEGMENT])
    return read_rig(statics, infos[SEGMENT])[0], read_hand_timeline(dataset, infos[SEGMENT], statics)


def _pose_at(poses: HandPose, frame: int) -> HandPose:
    return HandPose(poses.rotation[frame], poses.translation[frame], poses.joint_angles[frame])


def observation(
    rig: CameraRig,
    world_from_rig: Float32[Tensor, "4 4"],
    truth: Float32[Tensor, "21 3"],
    side: Side,
    phi: float,
    generator: torch.Generator,
    noise_px: float = NOISE_PX,
    noise_mm: float = 0.0,
) -> HandObservation | None:
    """The hand as a perfect KeyNet on the (up to) two best views would see it, plus Gaussian noise; None if no view has 17 keypoints inside."""
    points_cam: Float32[Tensor, "4 21 3"] = world_to_cameras(rig, world_from_rig, truth)
    pixels: Float32[Tensor, "4 21 2"] = project(rig, points_cam)
    seen: Bool[Tensor, "4 21"] = in_front(points_cam) & inside_image(rig, pixels)
    counts: list[int] = seen.sum(-1).tolist()
    ranked: list[int] = sorted((c for c in range(len(counts)) if counts[c] >= MIN_VISIBLE_KEYPOINTS), key=lambda c: -counts[c])[:MAX_VIEWS]
    if not ranked:
        return None
    views: tuple[ViewObservation, ...] = tuple(
        ViewObservation(
            camera=rig.select([c]),
            world_from_rig=world_from_rig,
            keypoints_px=pixels[c] + noise_px * torch.randn(21, 2, generator=generator),
            weights=seen[c].to(torch.float32),
            d_rel_mm=relative_distances(points_cam[c], torch.tensor(phi)) + noise_mm * torch.randn(21, generator=generator),
        )
        for c in ranked
    )
    return HandObservation(side=side, views=views)


def track(model: HandModelTorch, phi: float, frames: list[dict[Side, HandObservation]], config: FitConfig = DEFAULT_CONFIG) -> list[dict[Side, FitResult]]:
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
    noise_px: float = NOISE_PX, noise_mm: float = 0.0, seed: int = 0, limit: int = CALIBRATION_FRAMES
) -> tuple[HandTimeline, list[dict[Side, Float32[Tensor, "21 3"]]], list[dict[Side, HandObservation]]]:
    """The recording's timeline (profile and ϕ included), its ground-truth landmarks per frame and hand, and the noisy observations.

    Only the first ``limit`` frames are observed (the noise is drawn in frame order, so a longer limit keeps the same frames).
    """
    rig, timeline = _read_recording()
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    truths: list[dict[Side, Float32[Tensor, "21 3"]]] = []
    frames: list[dict[Side, HandObservation]] = []
    for frame in range(min(limit, timeline.world_from_rig.shape[0])):
        truth: dict[Side, Float32[Tensor, "21 3"]] = {}
        observed: dict[Side, HandObservation] = {}
        if bool(timeline.headset_valid[frame]):
            for side in Side:
                if float(timeline.confidence[frame, side]) <= 0.0:
                    continue
                truth[side] = landmarks(timeline.hand_model, _pose_at(timeline.poses[side], frame), side)
                seen: HandObservation | None = observation(
                    rig, timeline.world_from_rig[frame], truth[side], side, timeline.hand_scale, generator, noise_px, noise_mm
                )
                if seen is not None:
                    observed[side] = seen
        truths.append(truth)
        frames.append(observed)
    return timeline, truths, frames


@pytest.mark.integration
def test_tracks_a_real_recording_with_known_and_calibrated_hands() -> None:
    """Measured (1.5 px): known hand 1.6 mm, generic x calibrated ϕ 3.1 mm, generic x 1 8.7 mm over the first 60 frames.

    The calibrated ϕ (1.096) differs from the profile's least-squares ϕ (1.065) because the subject's hand is not a scaled
    generic hand; the generic model tracks better at the calibrated ϕ than at the profile's ϕ (4.2 mm).
    """
    timeline, truths, frames = scenario()
    profile: HandModelTorch = timeline.hand_model
    phi: float = timeline.hand_scale
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
