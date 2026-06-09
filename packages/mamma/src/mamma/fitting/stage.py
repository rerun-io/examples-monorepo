"""Per-tick fitting stage: triangulate landmarks, feed per-person window fitters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32

from mamma.calibration.npz_contract import CameraCalibration
from mamma.fitting.triangulation import triangulate_points
from mamma.fitting.window_fitter import FitResult, FitterConfig, SlidingWindowFitter
from mamma.landmarks.estimator import CameraLandmarks


@dataclass(slots=True)
class TickFitOutput:
    """Fitting outputs for one tick."""

    fits: dict[int, FitResult]
    """Per-person newest-frame SMPL-X fits (absent while a fitter warms up)."""
    triangulated: dict[int, tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]]
    """Per-person triangulated landmark clouds + validity."""


class FittingStage:
    """Owns triangulation plus one :class:`SlidingWindowFitter` per person."""

    def __init__(self, cameras: list[CameraCalibration], config: FitterConfig) -> None:
        self.config: FitterConfig = config
        self.k_per_cam: list[Float32[torch.Tensor, "3 3"]] = [
            torch.as_tensor(cam.k_matrix, dtype=torch.float32, device=config.device) for cam in cameras
        ]
        self.world_to_cam_per_cam: list[Float32[torch.Tensor, "4 4"]] = [
            torch.as_tensor(cam.world_to_cam, dtype=torch.float32, device=config.device) for cam in cameras
        ]
        self._fitters: dict[int, SlidingWindowFitter] = {}
        self.faces = None
        """Triangle faces of the SMPL-X mesh (set after the first fitter exists)."""

    def step(self, frame_idx: int, landmarks: list[CameraLandmarks]) -> TickFitOutput:
        """Triangulate + fit every person visible in >= 2 cameras this tick."""
        obj_ids: set[int] = set()
        for cam_landmarks in landmarks:
            obj_ids.update(cam_landmarks.keys())

        fits: dict[int, FitResult] = {}
        triangulated: dict[int, tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]] = {}
        for obj_id in sorted(obj_ids):
            cam_indices: list[int] = [c for c, cl in enumerate(landmarks) if obj_id in cl]
            if len(cam_indices) < 2:
                continue
            device: torch.device = self.k_per_cam[0].device
            n_landmarks: int = landmarks[cam_indices[0]][obj_id].joints2d.shape[0]
            # Build full per-camera tensors; cameras without this person get zeros
            # (excluded by the validity masks inside triangulation/reprojection).
            pts2d_all: list[Float32[torch.Tensor, "n 3"]] = []
            vis_all: list[Float32[torch.Tensor, "n"]] = []
            for cam_landmarks in landmarks:
                if obj_id in cam_landmarks:
                    pts2d_all.append(cam_landmarks[obj_id].joints2d)
                    vis_all.append(cam_landmarks[obj_id].visibility)
                else:
                    pts2d_all.append(torch.zeros(n_landmarks, 3, device=device))
                    vis_all.append(torch.zeros(n_landmarks, device=device))

            points3d, valid = triangulate_points(pts2d_all, self.k_per_cam, self.world_to_cam_per_cam, vis_all)
            triangulated[obj_id] = (points3d, valid)

            fitter: SlidingWindowFitter | None = self._fitters.get(obj_id)
            if fitter is None:
                fitter = SlidingWindowFitter(self.k_per_cam, self.world_to_cam_per_cam, self.config)
                self._fitters[obj_id] = fitter
                if self.faces is None:
                    self.faces = fitter.model.faces
            optimize: bool = self.config.fit_stride <= 1 or frame_idx % self.config.fit_stride == 0
            result: FitResult | None = fitter.push(frame_idx, pts2d_all, vis_all, points3d, valid, optimize=optimize)
            if result is not None:
                fits[obj_id] = result
        return TickFitOutput(fits=fits, triangulated=triangulated)
