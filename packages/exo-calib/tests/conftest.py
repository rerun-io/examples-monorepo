"""Synthetic multi-view scenes shared by the geometry tests."""

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ops.correspondences import ObservationSet


def ring_rig(n_views: int, radius_m: float) -> Float64[ndarray, "v 4 4"]:
    """Cameras on a horizontal ring of ``radius_m`` around the origin, each looking at the origin (RDF).

    Returns:
        Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
    """
    angles: Float64[ndarray, "v"] = np.arange(n_views, dtype=np.float64) * (2.0 * np.pi / float(n_views))
    centers: Float64[ndarray, "v 3"] = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(n_views, dtype=np.float64)))
    cam_T_world: Float64[ndarray, "v 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    world_up: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for view_idx in range(n_views):
        forward: Float64[ndarray, "3"] = -centers[view_idx] / np.linalg.norm(centers[view_idx])
        right: Float64[ndarray, "3"] = np.cross(forward, world_up)
        down: Float64[ndarray, "3"] = np.cross(forward, right)
        rotation: Float64[ndarray, "3 3"] = np.stack((right, down, forward))
        cam_T_world[view_idx, :3, :3] = rotation
        cam_T_world[view_idx, :3, 3] = -rotation @ centers[view_idx]
    return cam_T_world


def observe_points(
    points_xyz: Float64[ndarray, "n 3"], intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
) -> ObservationSet:
    """Project every point into every view with unit confidence, one point per frame and joint 0 (point-major, view-minor order)."""
    n_views: int = cam_T_world.shape[0]
    n_points: int = points_xyz.shape[0]
    points_homo: Float64[ndarray, "n 4"] = np.column_stack((points_xyz, np.ones(n_points, dtype=np.float64)))
    points_cam: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3], points_homo)
    projected: Float64[ndarray, "v n 3"] = np.einsum("vij,vnj->vni", intrinsics, points_cam)
    projected_xy: Float64[ndarray, "v n 2"] = projected[:, :, :2] / projected[:, :, 2:3]
    return ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=np.repeat(np.arange(n_points, dtype=np.int64), n_views),
        obs_view_idx=np.tile(np.arange(n_views, dtype=np.int64), n_points),
        obs_xy=projected_xy.transpose(1, 0, 2).reshape(-1, 2),
        obs_conf=np.ones(n_points * n_views, dtype=np.float64),
    )


class FakeScene:
    """A 4-camera ring (320x240 images) watching a 133-joint skeleton that drifts along x; fake models read the frame
    index from pixel (0, 0, 0) of every view, so they stay stateless."""

    def __init__(self, n_views: int = 4) -> None:
        import torch
        from posekit.ops.crops import CropSpec
        from simplecv.types.rig import RigCameras

        K: Float64[ndarray, "3 3"] = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
        names: tuple[str, ...] = tuple(f"rig_{i:02d}/cam_00" for i in range(n_views))
        self.rig = RigCameras(names, np.repeat(K[None], n_views, axis=0), ring_rig(n_views, 4.0), np.tile(np.array([320, 240]), (n_views, 1)))
        self.skeleton: Float64[ndarray, "133 3"] = np.random.default_rng(3).uniform(-0.3, 0.3, (133, 3))
        self.crop_spec = CropSpec(input_size=(192, 256))
        self.torch = torch
        from dagkit.sources import SegmentRef

        self.segment = SegmentRef("synthetic", "ring")

    def project(self, frame_index: int) -> Float64[ndarray, "v 133 2"]:
        points = self.skeleton + np.array([0.002 * frame_index, 0.0, 0.0])
        cam = np.einsum("vij,nj->vni", self.rig.cam_T_world[:, :3], np.column_stack((points, np.ones(133))))
        uv = np.einsum("vij,vnj->vni", self.rig.intrinsics, cam)
        return uv[..., :2] / uv[..., 2:]

    def frames(self, frame_index: int):  # MultiViewFrames
        from simplecv.types.frames import MultiViewFrames

        rgb = self.torch.zeros((len(self.rig.names), 3, 240, 320), dtype=self.torch.uint8)
        rgb[:, 0, 0, 0] = frame_index
        rgb[:, 1, 0, 0] = self.torch.arange(len(self.rig.names))  # the view, for the detector (it sees a subset)
        return MultiViewFrames(1_000_000 * frame_index, frame_index, self.rig.names, rgb, self.torch.full((len(self.rig.names),), 1_000_000 * frame_index))

    def detector(self, frames_hwc):  # BoxDetections: the padded box of the projected skeleton
        from posekit.predictions import BoxDetections

        xy = self.project(int(frames_hwc[0, 0, 0, 0]))
        lo, hi = xy.min(axis=1), xy.max(axis=1)
        center, size = (lo + hi) / 2.0, (hi - lo) * 1.5
        boxes = self.torch.as_tensor(np.concatenate((center - size / 2, center + size / 2), axis=1), dtype=self.torch.float32)
        views = frames_hwc[:, 0, 0, 1].long()
        return BoxDetections(xyxy=boxes[views], scores=self.torch.ones(len(views)), frame_indices=self.torch.arange(len(views)))

    def pose(self, frames_hwc, boxes):  # exact projections, confidence 0.95
        from types import SimpleNamespace

        xy = self.project(int(frames_hwc[0, 0, 0, 0]))[boxes.frame_indices.numpy()]
        scores = self.torch.full((len(xy), 133), 0.95)
        return SimpleNamespace(frame_indices=boxes.frame_indices, xy=self.torch.as_tensor(xy, dtype=self.torch.float32), scores=scores)

    def depth(self, frame_rgb, K_33):  # MoGe stand-in: every pixel at 4 m
        from types import SimpleNamespace

        return SimpleNamespace(depth_meters=np.full(frame_rgb.shape[:2], 4.0, dtype=np.float32))

    def frames_at(self, view: int, indices):
        return self.torch.zeros((len(indices), 3, 240, 320), dtype=self.torch.uint8)

    def resources(self) -> dict:
        """Fake graph resources, keyed like the App's model factories."""
        self.pose_model = _Callable(self.pose, crop_spec=self.crop_spec)
        return {"detector": _Callable(self.detector), "pose": self.pose_model, "moge": _Callable(self.depth), "frames": self, "g3t": _Callable(self.detector)}


class _Callable:
    def __init__(self, fn, **attributes) -> None:
        self.fn = fn
        self.__dict__.update(attributes)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
