"""Native warm and cold fitting.

The fitter owns copies of the model and camera calibration. Models and configs
are immutable while cached; clear ``_CACHE`` after deliberately changing one.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from handfit import FitOutput, HandFitter
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.fit.observations import HandObservation
from handtrack.fit.pose_fit import DEFAULT_CONFIG, FitConfig, FitResult, Termination
from handtrack.geometry.camera import CameraRig
from handtrack.hand.pose import HandPose


@dataclass(slots=True)
class _CachedFitter:
    model: HandModelTorch
    """Retain the model so its identity cannot be reused while cached."""
    fitter: HandFitter
    """Native immutable model and solver configuration."""
    cameras: dict[bytes, int] = field(default_factory=dict)
    """Content keys let temporary CameraRig selections share native calibration."""

    def camera_index(self, camera: CameraRig) -> int:
        transform: Float32[ndarray, "4 4"] = camera.cam_from_rig[0].numpy()
        focal: Float32[ndarray, "2"] = camera.focal[0].numpy()
        principal: Float32[ndarray, "2"] = camera.principal[0].numpy()
        distortion: Float32[ndarray, "8"] | None = None if camera.fisheye62 is None else camera.fisheye62[0].numpy()
        key: bytes = transform.tobytes() + focal.tobytes() + principal.tobytes() + (b"" if distortion is None else distortion.tobytes())
        index: int | None = self.cameras.get(key)
        if index is None:
            index = self.fitter.add_camera(transform, focal, principal, distortion)
            self.cameras[key] = index
        return index


_CACHE: OrderedDict[tuple[int, float, FitConfig], _CachedFitter] = OrderedDict()
"""Bounded model cache: a live tracker normally needs one entry."""


def _fitter(model: HandModelTorch, phi: float, config: FitConfig) -> _CachedFitter:
    key: tuple[int, float, FitConfig] = (id(model), phi, config)
    cached: _CachedFitter | None = _CACHE.get(key)
    if cached is None:
        topology: Int64[ndarray, "4 22"] = np.stack(
            [
                model.joint_parent.numpy(),
                model.joint_frame_index.numpy(),
                model.joint_first_child.numpy(),
                model.joint_next_sibling.numpy(),
            ]
        )
        cached = _CachedFitter(
            model,
            HandFitter(
                model.joint_rotation_axes[:20].numpy(),
                model.joint_rest_positions[:20].numpy(),
                model.landmark_rest_positions.numpy(),
                model.landmark_rest_bone_weights.numpy(),
                model.joint_limits[:20].numpy(),
                model.landmark_rest_bone_indices.numpy(),
                topology,
                np.array(
                    [
                        phi,
                        config.dist_weight,
                        config.temporal_weight,
                        config.temporal_translation_unit_m,
                        config.joint_limit_margin_rad,
                        config.max_iterations,
                        config.relative_tolerance,
                        config.absolute_tolerance,
                        config.initial_damping,
                        config.init_iterations,
                        config.init_relative_tolerance,
                        config.rotation_hypotheses,
                        config.full_fit_hypotheses,
                        config.finger_starts,
                    ],
                    dtype=np.float64,
                ),
            ),
        )
        _CACHE[key] = cached
        if len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    return cached


def fit_pose(
    model: HandModelTorch, phi: float, hands: Sequence[HandObservation], previous: Sequence[HandPose | None], config: FitConfig = DEFAULT_CONFIG
) -> list[FitResult]:
    """Fit one frame, preserving the torch signature and input order."""
    return _fit_pose(model, phi, hands, previous, config, central_difference=False)


def fit_pose_central_difference(
    model: HandModelTorch, phi: float, hands: Sequence[HandObservation], previous: Sequence[HandPose | None], config: FitConfig = DEFAULT_CONFIG
) -> list[FitResult]:
    """Diagnostic native replay using the torch finite-difference step sizes."""
    return _fit_pose(model, phi, hands, previous, config, central_difference=True)


def _fit_outputs(
    model: HandModelTorch,
    phi: float,
    hands: Sequence[HandObservation],
    previous: Sequence[HandPose | None],
    config: FitConfig,
    *,
    central_difference: bool,
) -> list[FitOutput]:
    if len(hands) != len(previous):
        raise ValueError(f"{len(hands)} hands but {len(previous)} previous poses")
    if not hands:
        return []
    cached: _CachedFitter = _fitter(model, phi, config)
    poses: list[HandPose] = [pose if pose is not None else HandPose(torch.eye(3), torch.zeros(3), torch.zeros(22)) for pose in previous]
    n: int = len(hands)
    cameras: Int64[ndarray, "n 2"] = np.full((n, 2), -1, dtype=np.int64)
    world_from_rig: Float32[ndarray, "n 2 4 4"] = np.zeros((n, 2, 4, 4), dtype=np.float32)
    pixels: Float32[ndarray, "n 2 21 2"] = np.zeros((n, 2, 21, 2), dtype=np.float32)
    weights: Float32[ndarray, "n 2 21"] = np.zeros((n, 2, 21), dtype=np.float32)
    distances: Float32[ndarray, "n 2 21"] = np.zeros((n, 2, 21), dtype=np.float32)
    for row, hand in enumerate(hands):
        for slot, view in enumerate(hand.views):
            cameras[row, slot] = cached.camera_index(view.camera)
            world_from_rig[row, slot] = view.world_from_rig.numpy()
            pixels[row, slot] = view.keypoints_px.numpy()
            weights[row, slot] = view.weights.numpy()
            distances[row, slot] = view.d_rel_mm.numpy()
    outputs: list[FitOutput] = cached.fitter.fit(
        np.array([int(hand.side) for hand in hands], dtype=np.int64),
        np.stack([pose.rotation.numpy() for pose in poses]),
        np.stack([pose.translation.numpy() for pose in poses]),
        np.stack([pose.joint_angles.numpy() for pose in poses]),
        cameras,
        world_from_rig,
        pixels,
        weights,
        distances,
        has_previous=np.array([pose is not None for pose in previous], dtype=np.bool_),
        central_difference=central_difference,
    )
    return outputs


def _fit_pose(
    model: HandModelTorch,
    phi: float,
    hands: Sequence[HandObservation],
    previous: Sequence[HandPose | None],
    config: FitConfig,
    *,
    central_difference: bool,
) -> list[FitResult]:
    return [
        FitResult(
            pose=HandPose(torch.from_numpy(output.rotation), torch.from_numpy(output.translation), torch.from_numpy(output.joint_angles)),
            e_2d=output.e_2d,
            e_dist=output.e_dist,
            e_temporal=output.e_temporal,
            energy=output.energy,
            iterations=output.iterations,
            converged=output.converged,
            termination=cast(Termination, output.termination),
        )
        for output in _fit_outputs(model, phi, hands, previous, config, central_difference=central_difference)
    ]
