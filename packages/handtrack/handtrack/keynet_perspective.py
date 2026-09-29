"""KeyNet-F on perspective crops as the tracker's keypoint stage (the crops ``keynet_crop='perspective'`` trains on).

A tracked view's crop camera fits the planning pose's landmarks in that camera (``labels.perspective.crop_cameras``, the
training crops' margin, no jitter); an acquisition's is aimed through its DetNet circle's centre, with a focal that gives the
circle's angular radius the same margin. The crop samples the native frame through the camera's lens; the keypoint input is the
pose's landmarks in that crop (zeros on acquisition); decoded keypoints go back through the crop camera and the lens to the net
frame, so the fit and the tracker are unchanged.
"""

import math

import torch
from jaxtyping import Bool, Float32, UInt8
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.geometry.camera import CameraRig, in_front, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox
from handtrack.hand.pose import HandPose, Side, landmarks
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.labels.keypoint_input import keypoint_input, relative_distances
from handtrack.labels.perspective import CROP_CENTRE, CROP_MARGIN, CropCameras, crop_cameras, from_crop, look_at, sample_crops, to_crop, unproject
from handtrack.models.keynet import KeyNetF, KeyNetOutput
from handtrack.tracker import CropRequest, KeypointEstimate


class PerspectiveKeyNetEstimator:
    """A ``KeypointEstimator``: KeyNet-F on perspective crops of the native frames."""

    def __init__(self, model: KeyNetF, rig: CameraRig, letterboxes: tuple[Letterbox, ...], camera_angles_deg: tuple[float, ...],
                 hand_model: HandModelTorch, phi: float, detnet_confirmation: bool = False) -> None:
        self.model: KeyNetF = model.eval()
        self.detnet_confirmation: bool = detnet_confirmation
        """Also ask the tracker to confirm views with DetNet and end drifted tracks (the UmeTrack stage's rule, TrackerConfig.umetrack_*)."""
        self.rig: CameraRig = rig
        self.letterboxes: tuple[Letterbox, ...] = letterboxes
        self.roll: tuple[float, ...] = tuple(math.radians(angle) for angle in (camera_angles_deg or (0.0,) * len(letterboxes)))
        self.hand_model: HandModelTorch = hand_model
        self.phi: float = phi

    def _cameras(self, request: CropRequest, camera: int, side: int, index: int) -> tuple[CropCameras, Float32[Tensor, "63"]]:
        """One view's crop camera and keypoint input, on the CPU."""
        one: CameraRig = self.rig.select([camera])
        mirror: Bool[Tensor, "1"] = torch.tensor([side == int(Side.RIGHT)])
        roll: Float32[Tensor, "1"] = torch.tensor([self.roll[camera]])
        pose: HandPose | None = request.poses[side]
        if pose is not None and request.world_from_rig is not None:
            points_world: Float32[Tensor, "21 3"] = landmarks(self.hand_model, pose, Side(side))
            points_cam: Float32[Tensor, "21 3"] = world_to_cameras(self.rig, request.world_from_rig, points_world)[camera]
            cameras: CropCameras = crop_cameras(points_cam[None], in_front(points_cam)[None], roll, mirror)
            if bool(torch.isfinite(cameras.focal).all()):
                uv, _ = to_crop(cameras, points_cam[None])
                features: Float32[Tensor, "63"] = keypoint_input(uv, relative_distances(points_cam[None], torch.tensor([self.phi])))[0]
                return cameras, features
        if request.circles is None:
            raise ValueError("an untracked view needs its DetNet circle")
        letterbox: Letterbox = self.letterboxes[camera]
        centre: Float32[Tensor, "1 2"] = letterbox.from_net(request.circles[index, :2][None])
        radius: float = float(request.circles[index, 2]) / letterbox.scale
        rays: Float32[Tensor, "2 3"] = unproject(one, torch.cat([centre, centre + torch.tensor([[radius, 0.0]])]))
        angle: float = float(torch.arccos((rays[0] * rays[1]).sum().clamp(-1.0, 1.0)))
        focal: Float32[Tensor, "1"] = torch.tensor([CROP_CENTRE / (math.tan(max(angle, 1e-3)) * CROP_MARGIN)])
        return CropCameras(look_at(rays[:1], roll), focal, mirror), torch.zeros(63)

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        del frame
        if not request.native_images:
            raise ValueError("perspective crops need the native camera frames (CropRequest.native_images)")
        device: torch.device = images.device
        count: int = len(request.camera)
        planned: list[tuple[CropCameras, Float32[Tensor, "63"]]] = [self._cameras(request, int(request.camera[i]), int(request.side[i]), i) for i in range(count)]
        crops: Float32[Tensor, "n 1 96 96"] = torch.zeros((count, 1, 96, 96), device=device)
        for camera in sorted({int(c) for c in request.camera.tolist()}):
            rows: list[int] = [i for i in range(count) if int(request.camera[i]) == camera]
            # An unusable crop camera samples nothing (identity rotation, a finite focal): its view is zeroed after the network.
            cameras: CropCameras = CropCameras(torch.cat([planned[i][0].rotation if self._usable(planned[i][0]) else torch.eye(3)[None] for i in rows]).to(device),
                                               torch.cat([planned[i][0].focal if self._usable(planned[i][0]) else torch.ones(1) for i in rows]).to(device),
                                               torch.cat([planned[i][0].mirror for i in rows]).to(device))
            frames: UInt8[Tensor, "1 h w"] = request.native_images[camera].to(device)[None]
            crops[rows] = sample_crops(frames, torch.zeros(len(rows), dtype=torch.int64, device=device), cameras, self.rig.select([camera]).to(device))
        features: Float32[Tensor, "n 63"] = torch.nan_to_num(torch.stack([feature for _, feature in planned])).to(device)
        with torch.inference_mode():
            output: KeyNetOutput = self.model(crops, features)
            points_crop, confidence = decode_heatmaps(output.heatmaps.float())
            d_rel: Float32[Tensor, "n 21"] = decode_distance(output.distance.float())
            presence: Float32[Tensor, "n"] = output.presence_logit.float().sigmoid()
        points_net: Float32[Tensor, "n 21 2"] = torch.zeros((count, 21, 2))
        crop_uv: Float32[Tensor, "n 21 2"] = points_crop.cpu()
        for i, (cameras, _) in enumerate(planned):
            camera = int(request.camera[i])
            rays: Float32[Tensor, "1 21 3"] = from_crop(cameras, crop_uv[i : i + 1])
            native: Float32[Tensor, "21 2"] = project(self.rig.select([camera]), rays[:, None])[0, 0]
            points_net[i] = self.letterboxes[camera].to_net(native)
        # A view without a usable crop camera (the hand at or past the lens edge, or no finite circle) reports nothing:
        # presence and confidence 0 keep it out of the fit (the tracker compares presence to a threshold; NaN never fails it).
        usable: Bool[Tensor, "n"] = torch.tensor([self._usable(planned[i][0]) for i in range(count)], dtype=torch.bool)
        usable = usable & torch.isfinite(points_net).all(dim=(-1, -2))
        presence_out: Float32[Tensor, "n"] = torch.where(usable, presence.cpu(), torch.zeros(count))
        confidence_out: Float32[Tensor, "n 21"] = torch.where(usable[:, None], confidence.cpu(), torch.zeros(count, 21))
        return KeypointEstimate(points_net=torch.nan_to_num(points_net), d_rel_mm=torch.nan_to_num(d_rel.cpu()), presence=presence_out,
                                confidence=confidence_out, uses_detnet_presence=self.detnet_confirmation)

    @staticmethod
    def _usable(cameras: CropCameras) -> bool:
        return bool(torch.isfinite(cameras.focal).all() & torch.isfinite(cameras.rotation).all())
