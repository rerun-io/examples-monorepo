"""Per-tick fitting stage: triangulate landmarks, feed per-person window fitters."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Float32

from mamma.calibration.npz_contract import CameraCalibration
from mamma.fitting.triangulation import triangulate_gated, triangulate_points
from mamma.fitting.window_fitter import FitResult, FitterConfig, SlidingWindowFitter
from mamma.landmarks.estimator import CameraLandmarks


@dataclass(slots=True)
class TickFitOutput:
    """Fitting outputs for one tick."""

    fits: dict[int, FitResult]
    """Per-person newest-frame SMPL-X fits (absent while a fitter warms up)."""
    triangulated: dict[int, tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]]
    """Per-person triangulated landmark clouds + validity."""
    metrics: dict[str, float] = field(default_factory=dict)
    """Per-tick fit health scalars (valid anchor count, contact landmark count)."""


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
        self._stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=config.device) if config.fit_stream and torch.device(config.device).type == "cuda" else None
        )
        if config.compile_iter:
            self._prewarm_compile()

    def _prewarm_compile(self) -> None:
        """Compile the fitter's graphed iterations now, before streaming starts.

        Fitters are created lazily when a person first appears, so without this
        the inductor compile (~27 s cold cache, ~1.7 s warm) stalls the fit
        worker mid-stream. A throwaway fitter fed one window of zero
        observations runs both captures (bootstrap + steady state); later
        fitters reuse the compiled code (~0.15 s to re-capture).
        """
        dummy: SlidingWindowFitter = SlidingWindowFitter(self.k_per_cam, self.world_to_cam_per_cam, self.config)
        n: int = dummy.verts_to_landmarks.shape[0]
        device: torch.device = self.k_per_cam[0].device
        n_cams: int = len(self.k_per_cam)
        for i in range(self.config.window_size + 1):
            dummy.push(
                i,
                [torch.zeros(n, 3, device=device) for _ in range(n_cams)],
                [torch.zeros(n, device=device) for _ in range(n_cams)],
                torch.zeros(n, 3, device=device),
                torch.zeros(n, dtype=torch.bool, device=device),
            )
        torch.cuda.synchronize()

    def step(self, frame_idx: int, landmarks: list[CameraLandmarks]) -> TickFitOutput:
        """Triangulate + fit every person visible in >= 2 cameras this tick.

        With ``fit_stream`` the work runs on a private non-blocking stream: the
        pipeline's main thread (track/landmarks) syncs the legacy default
        stream many times per tick, and each such sync otherwise also waits
        for every fit kernel queued from the worker thread. The step ends with
        a sync of its own stream only, so every tensor it read or produced is
        settled before references are dropped or handed to the logger.
        """
        if self._stream is None:
            return self._step(frame_idx, landmarks)
        self._stream.wait_stream(torch.cuda.current_stream())  # landmarks were produced on the caller's stream
        with torch.cuda.stream(self._stream):
            out: TickFitOutput = self._step(frame_idx, landmarks)
        self._stream.synchronize()
        return out

    def _step(self, frame_idx: int, landmarks: list[CameraLandmarks]) -> TickFitOutput:
        """:meth:`step` body on the current stream."""
        obj_ids: set[int] = set()
        for cam_landmarks in landmarks:
            obj_ids.update(cam_landmarks.keys())

        fits: dict[int, FitResult] = {}
        triangulated: dict[int, tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]] = {}
        metrics: dict[str, float] = {}
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
            fc_all: list[Float32[torch.Tensor, "n"]] = []
            for cam_landmarks in landmarks:
                if obj_id in cam_landmarks:
                    pts2d_all.append(cam_landmarks[obj_id].joints2d)
                    vis_all.append(cam_landmarks[obj_id].visibility)
                    fc_all.append(cam_landmarks[obj_id].floor_contact)
                else:
                    pts2d_all.append(torch.zeros(n_landmarks, 3, device=device))
                    vis_all.append(torch.zeros(n_landmarks, device=device))
            # Fused per-landmark floor-contact probability (mean over observing
            # cameras, like the original's contact fusion), low-prob zeroed.
            floor_contact: Float32[torch.Tensor, "n"] = torch.stack(fc_all, dim=0).mean(dim=0)
            floor_contact = torch.where(floor_contact < 0.25, torch.zeros_like(floor_contact), floor_contact)

            # Multi-view consistency gate: drop any camera whose mask swapped to
            # the other person this tick (its landmarks reproject far from the
            # 3D estimate) so it neither triangulates nor pulls the fit. Returns
            # gated visibility used for BOTH the anchor and the fit reprojection.
            vis_gated: list[Float32[torch.Tensor, "n"]]
            vis_gated, points3d, valid = triangulate_gated(
                pts2d_all, self.k_per_cam, self.world_to_cam_per_cam, vis_all, reproj_thresh_px=self.config.reproj_gate_px
            )
            # Dense cloud for visualization: triangulate every marker seen by
            # >= 2 cameras (sigma/log-variance-weighted, like the original DAG
            # which writes all 512), not just the confident vis>0.5 ones. The
            # FIT still anchors only to points3d/valid (strict) below, so fit
            # quality is unchanged; this only enriches the logged cloud.
            dense3d, dense_valid = triangulate_points(
                pts2d_all, self.k_per_cam, self.world_to_cam_per_cam, vis_all, vis_thresh=-1.0
            )
            triangulated[obj_id] = (dense3d, dense_valid)
            # Contact prediction alone fires mid-jump (feet visually near the
            # grass) — gate the pull by the triangulated height: only landmarks
            # the geometry also places near the ground get pulled to it.
            floor_contact = floor_contact * valid.float() * torch.exp(-(points3d[:, 2] / 0.05) ** 2)

            fitter: SlidingWindowFitter | None = self._fitters.get(obj_id)
            if fitter is None:
                fitter = SlidingWindowFitter(self.k_per_cam, self.world_to_cam_per_cam, self.config)
                self._fitters[obj_id] = fitter
                if self.faces is None:
                    self.faces = fitter.model.faces
            optimize: bool = self.config.fit_stride <= 1 or frame_idx % self.config.fit_stride == 0
            result: FitResult | None = fitter.push(frame_idx, pts2d_all, vis_gated, points3d, valid, floor_contact=floor_contact, optimize=optimize)
            if result is not None:
                fits[obj_id] = result
            if obj_id == 0:
                metrics["valid_anchors"] = float(valid.sum().item())
                metrics["floor_contacts"] = float((floor_contact > 0.25).sum().item())
        return TickFitOutput(fits=fits, triangulated=triangulated, metrics=metrics)

    def drain_head(self) -> list[TickFitOutput]:
        """Flush the bootstrap-window head frames of every fitter (frames solved
        before the first causal emit point). Mirror of :meth:`drain` for the
        clip head — one output per frame, oldest first."""
        outputs: list[TickFitOutput] = []
        for obj_id, fitter in self._fitters.items():
            for result in fitter.drain_head():
                outputs.append(TickFitOutput(fits={obj_id: result}, triangulated={}))
        return outputs

    def drain(self) -> list[TickFitOutput]:
        """Flush fixed-lag tails of every fitter at end of stream (one output per frame)."""
        outputs: list[TickFitOutput] = []
        for obj_id, fitter in self._fitters.items():
            for result in fitter.drain():
                outputs.append(TickFitOutput(fits={obj_id: result}, triangulated={}))
        return outputs
