"""The LM pose fit and the scale calibration on synthetic scenes: a real UmeTrack stereo pair and poses skinned from the generic model."""

import math

import torch
from jaxtyping import Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit.observations import HandObservation, ViewObservation, observe
from handtrack.fit.pose_fit import FitConfig, FitResult, fit_pose
from handtrack.fit.scale import ScaleCalibration, calibrate_scale, scaled_hand_model
from handtrack.geometry.camera import CameraRig
from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks

# cam_01 and cam_02 of umetrack__real__hand_hand__testing__user_05__recording_00 (the headset's stereo pair), read once from
# the catalog: cam_from_rig, focal, principal point and Fisheye62 [k1..k6, p1, p2]; images are 636 x 480.
CAM_FROM_RIG: Float32[Tensor, "2 4 4"] = torch.tensor(
    [
        [
            [0.85990775, 0.23873949, 0.45117828, 0.04033583],
            [0.39771041, 0.24070908, -0.88537300, -0.04256802],
            [-0.31997615, 0.94077766, 0.11203847, -0.04375378],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.58039391, 0.18884633, 0.79213631, 0.07179161],
            [0.80814075, -0.01381412, -0.58882707, -0.09532028],
            [-0.10025504, 0.98190951, -0.16063198, -0.07004973],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ]
)
FOCAL: Float32[Tensor, "2 2"] = torch.tensor([[239.01797485, 238.59420776], [239.02806091, 238.89305115]])
PRINCIPAL: Float32[Tensor, "2 2"] = torch.tensor([[317.98364258, 238.97972107], [317.16986084, 241.11093140]])
FISHEYE62: Float32[Tensor, "2 8"] = torch.tensor(
    [
        [-0.01455796, 0.07030687, -0.02130805, -0.02876545, 0.01763760, -0.00291994, 0.00298817, -0.00106629],
        [-0.00967496, 0.06433038, -0.02089349, -0.02452153, 0.01465230, -0.00233214, 0.00238000, -0.00047555],
    ]
)
RIG_QUATERNION_XYZW: tuple[float, float, float, float] = (0.80783969, 0.45242691, 0.36953306, -0.07842361)
RIG_TRANSLATION: tuple[float, float, float] = (-0.05733913, 0.40517554, -0.04792304)
MIN_INSIDE: int = 17


def _rotation_from_quaternion(q: Float32[Tensor, "4"]) -> Float32[Tensor, "3 3"]:
    x, y, z, w = (q / q.norm()).tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _world_from_rig() -> Float32[Tensor, "4 4"]:
    transform: Float32[Tensor, "4 4"] = torch.eye(4)
    transform[:3, :3] = _rotation_from_quaternion(torch.tensor(RIG_QUATERNION_XYZW))
    transform[:3, 3] = torch.tensor(RIG_TRANSLATION)
    return transform


def _camera(index: int) -> CameraRig:
    return CameraRig(
        names=(f"/world/rig_00/cam_0{index + 1}",),
        image_size=torch.tensor([[636.0, 480.0]]),
        cam_from_rig=CAM_FROM_RIG[index : index + 1],
        focal=FOCAL[index : index + 1],
        principal=PRINCIPAL[index : index + 1],
        fisheye62=FISHEYE62[index : index + 1],
    )


def _views(cameras: tuple[int, ...]) -> tuple[tuple[CameraRig, Float32[Tensor, "4 4"]], ...]:
    return tuple((_camera(index), _world_from_rig()) for index in cameras)


def _stereo_region() -> tuple[Float32[Tensor, "3"], Float32[Tensor, "3"]]:
    """The midpoint of the two camera centres and the mean optical axis, in the world frame."""
    world_from_rig: Float32[Tensor, "4 4"] = _world_from_rig()
    centres_rig: Float32[Tensor, "2 3"] = -torch.einsum("cji,cj->ci", CAM_FROM_RIG[:, :3, :3], CAM_FROM_RIG[:, :3, 3])
    axes_rig: Float32[Tensor, "2 3"] = CAM_FROM_RIG[:, 2, :3]
    origin: Float32[Tensor, "3"] = world_from_rig[:3, :3] @ centres_rig.mean(0) + world_from_rig[:3, 3]
    axis: Float32[Tensor, "3"] = world_from_rig[:3, :3] @ axes_rig.mean(0)
    return origin, axis / axis.norm()


def _random_pose(model: HandModelTorch, side: Side, generator: torch.Generator, distance_m: tuple[float, float] = (0.3, 0.6)) -> HandPose:
    """A pose with uniform random wrist rotation, joint angles inside the limits, and the palm centre 30-60 cm from the stereo pair."""
    rotation: Float32[Tensor, "3 3"] = _rotation_from_quaternion(torch.randn(4, generator=generator))
    limits: Float32[Tensor, "22 2"] = model.joint_limits
    fraction: Float32[Tensor, "22"] = 0.1 + 0.8 * torch.rand(22, generator=generator)
    joint_angles: Float32[Tensor, "22"] = limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])
    joint_angles[20:] = 0.0
    origin, axis = _stereo_region()
    jitter: Float32[Tensor, "3"] = 0.25 * torch.randn(3, generator=generator)
    direction: Float32[Tensor, "3"] = axis + jitter - (jitter @ axis) * axis
    distance: float = distance_m[0] + (distance_m[1] - distance_m[0]) * float(torch.rand(1, generator=generator))
    palm_centre_world: Float32[Tensor, "3"] = origin + distance * direction / direction.norm()
    at_origin: HandPose = HandPose(rotation, torch.zeros(3), joint_angles)
    palm_centre: Float32[Tensor, "3"] = landmarks(model, at_origin, side)[20]
    return HandPose(rotation, palm_centre_world - palm_centre, joint_angles)


def _visible_scene(model: HandModelTorch, side: Side, generator: torch.Generator, cameras: tuple[int, ...]) -> tuple[HandPose, HandObservation]:
    """Draw poses until one has at least 17 keypoints inside every camera in ``cameras``."""
    for _ in range(1000):
        pose: HandPose = _random_pose(model, side, generator)
        observation: HandObservation = observe(model, pose, side, 1.0, _views(cameras))
        if all(int(view.weights.sum()) >= MIN_INSIDE for view in observation.views):
            return pose, observation
    raise AssertionError("no visible pose in 1000 draws")


def _perturbed(pose: HandPose, generator: torch.Generator, angle_rad: float, translation_m: float, joint_rad: float) -> HandPose:
    axis: Float32[Tensor, "3"] = torch.randn(3, generator=generator)
    axis = axis / axis.norm() * angle_rad
    k: Float32[Tensor, "3 3"] = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation: Float32[Tensor, "3 3"] = torch.linalg.matrix_exp(k) @ pose.rotation
    shift: Float32[Tensor, "3"] = torch.randn(3, generator=generator)
    joint_angles: Float32[Tensor, "22"] = pose.joint_angles + joint_rad * (2 * torch.rand(22, generator=generator) - 1)
    joint_angles[20:] = pose.joint_angles[20:]
    return HandPose(rotation, pose.translation + translation_m * shift / shift.norm(), joint_angles)


def _landmark_error_mm(model: HandModelTorch, fitted: HandPose, truth: HandPose, side: Side) -> float:
    return float((landmarks(model, fitted, side) - landmarks(model, truth, side)).norm(dim=-1).mean()) * 1000.0


NO_TEMPORAL: FitConfig = FitConfig(temporal_weight=0.0)


def test_exact_stereo_keypoints_recover_the_pose_from_a_perturbed_start() -> None:
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(1)
    for side in Side:
        truth, observation = _visible_scene(model, side, generator, (0, 1))
        start: HandPose = _perturbed(truth, generator, angle_rad=math.radians(15.0), translation_m=0.02, joint_rad=0.2)
        result: FitResult = fit_pose(model, 1.0, [observation], [start], NO_TEMPORAL)[0]
        assert result.converged
        assert _landmark_error_mm(model, result.pose, truth, side) < 0.05
        assert result.e_2d < 1e-2


def _noisy(observation: HandObservation, generator: torch.Generator, sigma_px: float, sigma_mm: float = 0.0) -> HandObservation:
    views: tuple[ViewObservation, ...] = tuple(
        ViewObservation(
            camera=view.camera,
            world_from_rig=view.world_from_rig,
            keypoints_px=view.keypoints_px + sigma_px * torch.randn(21, 2, generator=generator),
            weights=view.weights,
            d_rel_mm=view.d_rel_mm + sigma_mm * torch.randn(21, generator=generator),
        )
        for view in observation.views
    )
    return HandObservation(side=observation.side, views=views)


def test_noisy_stereo_keypoints_fit_to_the_noise_floor() -> None:
    """1.5 px of noise: the fit ends below the energy of the true pose, and the mean landmark error stays under 6 mm (measured: 4.3 mm)."""
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(2)
    errors_mm: list[float] = []
    for trial in range(20):
        side: Side = Side(trial % 2)
        truth, observation = _visible_scene(model, side, generator, (0, 1))
        start: HandPose = _perturbed(truth, generator, angle_rad=math.radians(15.0), translation_m=0.02, joint_rad=0.2)
        noisy: HandObservation = _noisy(observation, generator, sigma_px=1.5)
        result: FitResult = fit_pose(model, 1.0, [noisy], [start], NO_TEMPORAL)[0]
        at_truth: FitResult = fit_pose(model, 1.0, [noisy], [truth], FitConfig(temporal_weight=0.0, max_iterations=0))[0]
        assert result.energy <= at_truth.energy * 1.0001
        errors_mm.append(_landmark_error_mm(model, result.pose, truth, side))
    assert sum(errors_mm) / len(errors_mm) < 6.0


def test_initial_pose_acquires_hands_with_random_wrist_rotation() -> None:
    """No previous pose: the neutral-pose initialiser finds hands 30-60 cm away in any wrist rotation (exact stereo keypoints)."""
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(3)
    trials: int = 40
    scenes: list[tuple[HandPose, HandObservation]] = [_visible_scene(model, Side(trial % 2), generator, (0, 1)) for trial in range(trials)]
    results: list[FitResult] = fit_pose(model, 1.0, [observation for _, observation in scenes], [None] * trials, NO_TEMPORAL)
    errors_mm: list[float] = [
        _landmark_error_mm(model, result.pose, truth, observation.side) for (truth, observation), result in zip(scenes, results, strict=True)
    ]
    assert sum(error < 10.0 for error in errors_mm) >= 0.95 * trials


def _pose_distance(a: HandPose, b: HandPose) -> float:
    """A scalar distance between two poses: translation in cm, rotation and joint angles in radians."""
    rotation: float = float(torch.linalg.matrix_norm(a.rotation - b.rotation))
    return float((a.translation - b.translation).norm()) * 100.0 + rotation + float((a.joint_angles[:20] - b.joint_angles[:20]).norm())


def test_temporal_term_pulls_toward_the_previous_pose() -> None:
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(4)
    truth, observation = _visible_scene(model, Side.RIGHT, generator, (0,))
    previous: HandPose = _perturbed(truth, generator, angle_rad=math.radians(4.0), translation_m=0.008, joint_rad=0.1)
    distances: list[float] = []
    for weight in (0.0, 20.0, 1e6):
        result: FitResult = fit_pose(model, 1.0, [observation], [previous], FitConfig(temporal_weight=weight))[0]
        distances.append(_pose_distance(result.pose, previous))
        assert (result.e_temporal > 0.0) == (weight > 0.0)
    assert distances[0] > distances[1] > distances[2]
    assert distances[2] < 0.1 * distances[0]


def test_joint_angles_stay_inside_the_limits() -> None:
    """Keypoints of a hand bent past its limits: the fit clamps at the limits (widened by the margin) instead of following them."""
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(5)
    margin: float = NO_TEMPORAL.joint_limit_margin_rad
    limits: Float32[Tensor, "22 2"] = model.joint_limits + torch.tensor([-margin, margin])
    for side in Side:
        truth, _ = _visible_scene(model, side, generator, (0, 1))
        overbent: Float32[Tensor, "22"] = truth.joint_angles.clone()
        overbent[:20] = torch.where(torch.arange(20) % 2 == 0, limits[:20, 1] + 0.3, limits[:20, 0] - 0.3)
        beyond: HandPose = HandPose(truth.rotation, truth.translation, overbent)
        observation: HandObservation = observe(model, beyond, side, 1.0, _views((0, 1)))
        for previous in (truth, None):
            angles: Float32[Tensor, "22"] = fit_pose(model, 1.0, [observation], [previous], NO_TEMPORAL)[0].pose.joint_angles
            assert bool(((angles[:20] >= limits[:20, 0] - 1e-6) & (angles[:20] <= limits[:20, 1] + 1e-6)).all())


def test_scaled_model_is_the_hand_enlarged_about_its_wrist() -> None:
    model: HandModelTorch = generic_hand_model()
    generator: torch.Generator = torch.Generator().manual_seed(6)
    pose: HandPose = _random_pose(model, Side.LEFT, generator)
    scaled: HandModelTorch = scaled_hand_model(model, 1.08)
    for side in Side:
        expected: Float32[Tensor, "21 3"] = pose.translation + 1.08 * (landmarks(model, pose, side) - pose.translation)
        torch.testing.assert_close(landmarks(scaled, pose, side), expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(scaled.mesh_vertices, 1.08 * model.mesh_vertices)


def test_calibration_recovers_the_hand_scale_from_noisy_stereo_keypoints() -> None:
    """A subject 8% larger than the generic hand, seen in stereo with 1.5 px of noise: ϕ = 1.08 ± 0.01 (measured 1.0887).

    The starting poses are the truth perturbed, as the tracker's first pass at ϕ = 1 would give them.
    """
    generic: HandModelTorch = generic_hand_model()
    subject: HandModelTorch = scaled_hand_model(generic, 1.08)
    generator: torch.Generator = torch.Generator().manual_seed(7)
    frames: list[HandObservation] = []
    starts: list[HandPose | None] = []
    for frame in range(30):
        side: Side = Side(frame % 2)
        pose: HandPose = _visible_scene(subject, side, generator, (0, 1))[0]
        exact: HandObservation = observe(subject, pose, side, 1.08, _views((0, 1)))
        frames.append(_noisy(exact, generator, sigma_px=1.5))
        starts.append(_perturbed(pose, generator, angle_rad=math.radians(5.0), translation_m=0.01, joint_rad=0.1))
    calibration: ScaleCalibration = calibrate_scale(generic, frames, starts)
    assert calibration.blocks == 30
    assert abs(calibration.phi - 1.08) < 0.01


def test_calibration_without_starting_poses_fits_them_first() -> None:
    """No poses given: each stereo observation is fitted from the neutral initialiser; mono observations are left out."""
    generic: HandModelTorch = generic_hand_model()
    subject: HandModelTorch = scaled_hand_model(generic, 0.92)
    generator: torch.Generator = torch.Generator().manual_seed(8)
    frames: list[HandObservation] = []
    for frame in range(6):
        side: Side = Side(frame % 2)
        pose: HandPose = _visible_scene(subject, side, generator, (0, 1))[0]
        frames.append(observe(subject, pose, side, 0.92, _views((0, 1) if frame < 4 else (0,))))
    calibration: ScaleCalibration = calibrate_scale(generic, frames)
    assert calibration.used == (0, 1, 2, 3)
    assert abs(calibration.phi - 0.92) < 1e-3
