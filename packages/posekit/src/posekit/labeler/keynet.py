"""MEgATrack's KeyNet-F as the MEgATrack propagator's 2D keypoint network (``posekit.labeler.megatrack.KeypointEstimator``).

The network, the perspective crops and the heatmap decoding come from ``packages/handtrack`` (imported, not vendored): it is not
a dependency of the posekit environments, so the server runs with ``PYTHONPATH=packages/handtrack`` (it needs only torch,
einops, jaxtyping and simplecv, which posekit-dev and posekit have). Weights: the KeyNet-F that RoboCap's ``hands`` layer runs
(robocap-live ``models/MODELS.md``: ``keynet-strong-pinch/best.weights.pt``, sha256 ``067cf7f6...``).

A crop is UmeTrack's perspective crop (handtrack ``labels/perspective.py``): a pinhole crop camera at the source camera's
centre, aimed at the hand, its focal fitting the hand with KeyNet's training margin (1.2); every crop pixel's ray goes back
through the source camera's KB4 lens (handtrack's Fisheye62 with k5 = k6 = p1 = p2 = 0 is KB4) and samples the native grey
frame. Right hands are mirrored, so KeyNet sees left hands only; RoboCap needs no crop roll (robocap-live ``estimator.rs``).
A crop aims either at a box (the reviewer's: its centre and edge midpoints play the hand's extent, prior zero, as for a
DetNet acquisition) or at the previous frame's 3D keypoints in that camera (MEgATrack's tracked crop; with ``use_prior`` the
same keypoints are KeyNet's 63-value prior). Decoded keypoints go back through the crop camera to rays and through KB4 to
native pixels. Frames are converted to grey with OpenCV's BT.601 luma.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

try:
    from handtrack.geometry.camera import CameraRig
    from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
    from handtrack.labels.keypoint_input import keypoint_input, relative_distances
    from handtrack.labels.perspective import CropCameras, crop_cameras, from_crop, sample_crops, to_crop
    from handtrack.models.keynet import KeyNetF, KeyNetOutput, keynet_for_state
except ImportError as error:  # pragma: no cover - environment check
    raise ImportError("posekit.labeler.keynet imports handtrack: run with PYTHONPATH=packages/handtrack (from the repo root)") from error

from posekit.labeler.megatrack import CropRequest
from posekit.labeler.rig import NUM_KEYPOINTS, Kb4Rig, ViewKeypoints

DEFAULT_WEIGHTS: Path = Path.home() / "hand-labeler-data" / "keynet" / "keynet-strong-pinch.weights.pt"
"""RoboCap's deployed KeyNet-F (with the pinch head), copied from pablo-dl-server ``handtrack-data/runs/keynet-strong-pinch``."""
PHI: float = 1.0
"""Hand scale for KeyNet's relative distances (robocap-live's default before its live calibration)."""


def handtrack_rig(rig: Kb4Rig, device: torch.device) -> CameraRig:
    """The rig as handtrack's ``CameraRig`` (float32, Fisheye62 coefficients ``[k1..k4, 0, 0, 0, 0]``)."""
    fisheye62: Float64[ndarray, "c 8"] = np.concatenate([rig.kb4, np.zeros((len(rig.names), 4))], axis=1)
    return CameraRig(
        names=rig.names,
        image_size=torch.as_tensor(rig.size_wh, dtype=torch.float32, device=device),
        cam_from_rig=torch.as_tensor(rig.cam_from_rig, dtype=torch.float32, device=device),
        focal=torch.as_tensor(rig.focal, dtype=torch.float32, device=device),
        principal=torch.as_tensor(rig.principal, dtype=torch.float32, device=device),
        fisheye62=torch.as_tensor(fisheye62, dtype=torch.float32, device=device),
    )


class KeyNetEstimator:
    """KeyNet-F on perspective crops of one segment's native frames; see the module docstring. Thread-safe (one call at a time)."""

    def __init__(self, rig: Kb4Rig, weights: Path = DEFAULT_WEIGHTS, *, device: str = "cuda", use_prior: bool = False) -> None:
        """Load the network.

        Args:
            rig: The segment's cameras.
            weights: KeyNet-F ``state_dict`` file.
            device: Torch device.
            use_prior: Feed tracked crops the previous keypoints as KeyNet's prior (else zeros, as on acquisition).
        """
        self.rig: Kb4Rig = rig
        self.device: torch.device = torch.device(device)
        state: dict[str, Tensor] = torch.load(weights, map_location="cpu", weights_only=True)
        model: KeyNetF = keynet_for_state(state)
        model.load_state_dict(state)
        self.model: KeyNetF = model.eval().to(self.device)
        self.cameras: CameraRig = handtrack_rig(rig, self.device)
        self.use_prior: bool = use_prior
        self._lock = threading.Lock()

    def _plan(self, request: CropRequest) -> tuple[CropCameras, Float32[Tensor, "63"]] | None:
        """The crop camera (on the CPU) and keypoint prior of one request; None when no usable crop camera exists."""
        mirror: Bool[Tensor, "1"] = torch.tensor([request.side == "right"])
        roll: Float32[Tensor, "1"] = torch.zeros(1)
        prior: Float32[Tensor, "63"] = torch.zeros(63)
        if request.points_cam is not None:
            points: Float32[Tensor, "1 21 3"] = torch.as_tensor(request.points_cam, dtype=torch.float32)[None]
            valid: Bool[Tensor, "1 21"] = torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
            cameras: CropCameras = crop_cameras(torch.nan_to_num(points), valid, roll, mirror)
            if self.use_prior and bool(valid.all()) and bool(torch.isfinite(cameras.focal).all()):
                uv, _ = to_crop(cameras, points)
                prior = keypoint_input(uv, relative_distances(points, torch.tensor([PHI])))[0]
        elif request.box is not None:
            box = request.box
            cx, cy = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
            anchors: Float64[ndarray, "5 2"] = np.asarray([[cx, cy], [box.x0, cy], [box.x1, cy], [cx, box.y0], [cx, box.y1]], dtype=np.float64)
            rays: Float32[Tensor, "1 5 3"] = torch.as_tensor(self.rig.unproject(request.cam, anchors), dtype=torch.float32)[None]
            cameras = crop_cameras(rays, torch.ones((1, 5), dtype=torch.bool), roll, mirror)
        else:
            raise ValueError("a crop request needs a box or keypoints")
        if not (bool(torch.isfinite(cameras.focal).all()) and bool(torch.isfinite(cameras.rotation).all())):
            return None
        return cameras, prior

    def estimate(self, frames_rgb: Mapping[int, UInt8[ndarray, "h w 3"]], requests: Sequence[CropRequest]) -> list[ViewKeypoints | None]:
        """KeyNet keypoints for each request, in order; None where the crop is unusable or its camera has no frame.

        Args:
            frames_rgb: Native RGB frame per camera index (the cameras the requests name).
            requests: Crops to run, any cameras and sides.
        """
        planned: list[tuple[int, CropCameras, Float32[Tensor, "63"]]] = []
        for index, request in enumerate(requests):
            plan: tuple[CropCameras, Float32[Tensor, "63"]] | None = self._plan(request) if request.cam in frames_rgb else None
            if plan is not None:
                planned.append((index, plan[0], plan[1]))
        results: list[ViewKeypoints | None] = [None] * len(requests)
        if not planned:
            return results
        with self._lock, torch.inference_mode():
            crops: Float32[Tensor, "n 1 96 96"] = torch.zeros((len(planned), 1, 96, 96), device=self.device)
            for cam in sorted({requests[index].cam for index, _, _ in planned}):
                members: list[int] = [k for k, (index, _, _) in enumerate(planned) if requests[index].cam == cam]
                grey: UInt8[ndarray, "h w"] = cv2.cvtColor(frames_rgb[cam], cv2.COLOR_RGB2GRAY)
                frame: UInt8[Tensor, "1 h w"] = torch.from_numpy(grey).to(self.device)[None]
                cameras: CropCameras = CropCameras(
                    torch.cat([planned[k][1].rotation for k in members]).to(self.device),
                    torch.cat([planned[k][1].focal for k in members]).to(self.device),
                    torch.cat([planned[k][1].mirror for k in members]).to(self.device),
                )
                image: Tensor = torch.zeros(len(members), dtype=torch.int64, device=self.device)
                crops[members] = sample_crops(frame, image, cameras, self.cameras.select([cam]))
            features: Float32[Tensor, "n 63"] = torch.stack([prior for _, _, prior in planned]).to(self.device)
            output: KeyNetOutput = self.model(crops, features)
            points_crop, peaks = decode_heatmaps(output.heatmaps.float())
            d_rel: Float32[Tensor, "n 21"] = decode_distance(output.distance.float())
            presence: Float32[Tensor, "n"] = output.presence_logit.float().sigmoid()
            points_cpu: Float32[Tensor, "n 21 2"] = points_crop.cpu()
            peaks_cpu: Float32[Tensor, "n 21"] = peaks.cpu()
            d_rel_cpu: Float32[Tensor, "n 21"] = d_rel.cpu()
            presence_cpu: Float32[Tensor, "n"] = presence.cpu()
        for k, (index, camera, _) in enumerate(planned):
            rays: Float64[ndarray, "21 3"] = from_crop(camera, points_cpu[k : k + 1])[0].double().numpy()
            uv: Float64[ndarray, "21 2"] = self.rig.project(requests[index].cam, rays)
            if not np.isfinite(uv).all() or uv.shape != (NUM_KEYPOINTS, 2):
                continue
            results[index] = ViewKeypoints(
                cam=requests[index].cam,
                uv=uv,
                confidence=peaks_cpu[k].double().numpy(),
                d_rel_mm=d_rel_cpu[k].double().numpy(),
                presence=float(presence_cpu[k]),
            )
        return results
