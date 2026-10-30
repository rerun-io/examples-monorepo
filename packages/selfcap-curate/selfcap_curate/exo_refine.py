"""Deterministic exo-only refinement; Quest poses, hands and optical origins are fixed truth."""

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray
from scipy.optimize import OptimizeResult, least_squares
from serde import serde

from selfcap_curate.calibration_objective import ObjectiveShares, block_weights, robust_loss
from selfcap_curate.exo_calibration import Camera
from selfcap_curate.exo_landmarks import Landmarks
from selfcap_curate.quest_fit import Correspondences

FocalSource: TypeAlias = Literal["table", "fit", "fixed"]
VERSION: str = "selfcap-exo-refine-v2"
METHOD: str = "exo poses from Quest hands + fixed optical landmarks; Huber then Cauchy (6 px)"


@dataclass(frozen=True, slots=True)
class RefinementProblem:
    cameras: tuple[Camera, ...]
    """Current cameras in exo rig order."""
    hands: tuple[Correspondences, ...]
    """Mapped Quest/Sapiens hand pairs per exo rig."""
    landmarks: Landmarks
    """Catalog label record; ordering is preserved through fold selection."""
    quest_world: Float64[ndarray, "n 3"]
    """Factory optical centre per label before the fitted head translation."""
    head_rotations: Float64[ndarray, "n 3 3"]
    """World-from-head rotation per label."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Refinement:
    cameras: tuple[Camera, ...]
    """Refined camera models in exo rig order."""
    head_translation_m: Float64[ndarray, "3"]
    """Legacy compatibility field, always zero for exo-only fits. Never changes wearables."""
    label_layer_hash: str
    """SHA256 of the complete typed landmark record."""
    focal_source: FocalSource
    """Whether focal came from the device table, the joint fit, or a fixed record."""
    focal_record: str | None = None
    """SHA256 of the supplied fixed-focal record, when used."""
    method: str = METHOD
    """Exact estimator family."""
    version: str = VERSION
    """Solver and split-policy version."""
    objective: ObjectiveShares | None = None
    """Explicit count-invariant objective shares; null for legacy summed residuals."""
    converged: bool = True
    """Both robust optimization stages terminated before their step limit."""


def predict_labels(problem: RefinementProblem, cameras: tuple[Camera, ...], offset: Float64[ndarray, "3"]) -> Float64[ndarray, "n 2"]:
    """Project labelled centres with the shared optical lever arm."""
    world: Float64[ndarray, "n 3"] = problem.quest_world + np.einsum("nij,j->ni", problem.head_rotations, offset)
    for i, label in enumerate(problem.landmarks.labels):
        if label.target_id.startswith("phone"):
            world[i] = cameras[int(label.target_id[5:])].world_T_cam[:3, 3]
    result: Float64[ndarray, "n 2"] = np.empty((len(world), 2))
    for view, camera in enumerate(cameras):
        select: Bool[ndarray, "n"] = np.array([label.view == view for label in problem.landmarks.labels])
        result[select] = camera.project(world[select])[:, :2]
    return result


def select_labels(problem: RefinementProblem, fold: int, *, static_phones: bool = False) -> RefinementProblem:
    """Alternate sampled frame indices per view, retaining all targets in a frame."""
    frames: dict[int, list[float]] = {view: sorted({label.video_time for label in problem.landmarks.labels if label.view == view})
              for view in range(len(problem.cameras))}
    select: Bool[ndarray, "n"] = np.array([(frames[label.view].index(label.video_time) % 2 == fold or (static_phones and label.target_id.startswith("phone"))) for label in problem.landmarks.labels])
    return replace(problem, landmarks=replace(problem.landmarks, labels=[label for label, keep in zip(problem.landmarks.labels, select, strict=True) if keep]),
                   quest_world=problem.quest_world[select], head_rotations=problem.head_rotations[select])


def solve(problem: RefinementProblem, *, focal_source: FocalSource = "fit", fixed_focals: tuple[float, ...] | None = None,
          focal_record: str | None = None, landmark_target_count: int | None = None, objective: ObjectiveShares | None = None) -> Refinement:
    """Fit poses/focals/offset with the offline estimator's exact loss and bounds.

    Inputs are Float64 camera/point arrays in metres and source pixels. This
    solver uses no random draws or wall clock. Fixed focal values are explicit
    per-camera values; table mode uses the supplied table-calibrated seed.
    """
    count: int = len(problem.cameras)
    if not problem.landmarks.labels or count != len(problem.hands):
        raise ValueError("refinement needs landmarks and one hand correspondence set per camera")
    if problem.quest_world.shape != (len(problem.landmarks.labels), 3) or problem.head_rotations.shape != (len(problem.landmarks.labels), 3, 3):
        raise ValueError("one Quest point and head rotation is required per label")
    for label in problem.landmarks.labels:
        if label.view >= count or (label.target_id.startswith("phone") and int(label.target_id[5:]) >= count):
            raise ValueError("label references a camera outside this rig")
    free: bool = focal_source == "fit"
    if focal_source == "fixed" and (fixed_focals is None or len(fixed_focals) != count):
        raise ValueError("fixed focal source requires one focal for each camera")
    initial: Float64[ndarray, "c 11"] = np.stack([camera.parameters for camera in problem.cameras])
    if fixed_focals is not None:
        if focal_source != "fixed" or not all(np.isfinite(f) and f > 0.0 for f in fixed_focals):
            raise ValueError("positive fixed focal values require focal_source='fixed'")
        initial[:, 6] = fixed_focals
    width: int = 7 if free else 6
    nc: int = count * width
    parameters: Float64[ndarray, "p"] = np.concatenate([np.r_[p[:6], np.log(p[6])] if free else p[:6] for p in initial])
    targets: Float64[ndarray, "n 2"] = np.array([[label.u, label.v] for label in problem.landmarks.labels])
    repetitions = np.ones(len(problem.landmarks.labels), dtype=int)
    if landmark_target_count is not None:
        keys = [(label.view, label.target_id) for label in problem.landmarks.labels]
        repetitions = np.array([max(1, round(landmark_target_count / keys.count(key))) for key in keys])
    weights: Float64[ndarray, "n 1"] = np.sqrt([label.confidence for label in problem.landmarks.labels])[:, None] * 2.0
    loss_weights = None
    if objective is not None:
        repetitions[:] = 1
        weights[:] = 1.0
        hand_views = np.concatenate([np.full(len(data.world), i, dtype=np.int64) for i, data in enumerate(problem.hands)])
        hand_confidence = np.ones(len(hand_views))
        hand_weights = block_weights(hand_views, hand_confidence, objective.hands)
        label_views = np.array([label.view for label in problem.landmarks.labels], dtype=np.int64)
        confidence = np.array([label.confidence for label in problem.landmarks.labels])
        head = np.array([label.target_id.startswith('quest') for label in problem.landmarks.labels])
        label_weights = block_weights(label_views, confidence * head, objective.headset)
        label_weights += block_weights(label_views, confidence * ~head, objective.phones)
        loss_weights = np.repeat(np.r_[hand_weights, label_weights], 2) / 2.0


    def unpack(values: Float64[ndarray, "p"]) -> tuple[Camera, ...]:
        matrix: Float64[ndarray, "c 11"] = initial.copy()
        fitted: Float64[ndarray, "c w"] = values[:nc].reshape(count, width)
        matrix[:, :6] = fitted[:, :6]
        if free:
            matrix[:, 6] = np.exp(fitted[:, 6])
        return tuple(Camera(p) for p in matrix)

    def residual(values: Float64[ndarray, "p"]) -> Float64[ndarray, "r"]:
        cameras: tuple[Camera, ...] = unpack(values)
        errors: list[Float64[ndarray, "r"]] = [(camera.project(data.world)[:, :2] - data.image).ravel() for camera, data in zip(cameras, problem.hands, strict=True)]
        errors.append(np.repeat((predict_labels(problem, cameras, np.zeros(3)) - targets) * weights, repetitions, axis=0).ravel())
        return np.concatenate(errors)

    lower: Float64[ndarray, "p"] = np.full_like(parameters, -np.inf)
    upper: Float64[ndarray, "p"] = -lower
    if free:
        lower[6:nc:7], upper[6:nc:7] = np.log(400.0), np.log(1200.0)
    warm: OptimizeResult = least_squares(residual, parameters, bounds=(lower, upper), loss="huber" if loss_weights is None else robust_loss(loss_weights, "huber"), f_scale=6.0, x_scale="jac", max_nfev=150)
    solution: OptimizeResult = least_squares(residual, warm.x, bounds=(lower, upper), loss="cauchy" if loss_weights is None else robust_loss(loss_weights), f_scale=6.0, x_scale="jac", max_nfev=150)
    return Refinement(unpack(solution.x), np.zeros(3), problem.landmarks.digest, focal_source, focal_record,
                      objective=objective, converged=bool(warm.success and solution.success))
