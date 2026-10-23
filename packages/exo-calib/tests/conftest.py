"""Synthetic multi-view scenes shared by the geometry tests."""

from typing import NamedTuple

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ops.correspondences import ObservationSet
from exo_calib.types import RigLayout


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


def project_points(points_xyz: Float64[ndarray, "n 3"], intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]) -> Float64[ndarray, "v n 2"]:
    """Pinhole pixels of every point in every view."""
    uvw: Float64[ndarray, "v n 3"] = np.einsum("vij,vnj->vni", intrinsics, np.einsum("vij,nj->vni", cam_T_world[:, :3], np.column_stack((points_xyz, np.ones(len(points_xyz))))))
    return uvw[..., :2] / uvw[..., 2:]


def observe_points(
    points_xyz: Float64[ndarray, "n 3"], intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
) -> ObservationSet:
    """Project every point into every view with unit confidence, one point per frame and joint 0 (point-major, view-minor order)."""
    n_views: int = cam_T_world.shape[0]
    n_points: int = points_xyz.shape[0]
    projected_xy: Float64[ndarray, "v n 2"] = project_points(points_xyz, intrinsics, cam_T_world)
    return ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=np.repeat(np.arange(n_points, dtype=np.int64), n_views),
        obs_view_idx=np.tile(np.arange(n_views, dtype=np.int64), n_points),
        obs_xy=projected_xy.transpose(1, 0, 2).reshape(-1, 2),
        obs_conf=np.ones(n_points * n_views, dtype=np.float64),
    )


class Person(NamedTuple):
    """Someone in a ``FakeScene``: the subject's skeleton moved by ``offset_m``, found by the detector in ``views`` (None: all)."""

    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    views: tuple[int, ...] | None = None
    score: float = 1.0


class FakeScene:
    """A ring of cameras (320x240 images) watching people with a 133-joint skeleton that drifts along x; ``people[0]`` is the
    subject. Fake models read the frame index from pixel (0, 0, 0) of every view, so they stay stateless."""

    def __init__(self, n_views: int = 4, people: tuple[Person, ...] = (Person(),)) -> None:
        import torch
        from posekit.ops.crops import CropSpec
        from simplecv.types.rig import RigCameras

        K: Float64[ndarray, "3 3"] = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
        names: tuple[str, ...] = tuple(f"rig_{i:02d}/cam_00" for i in range(n_views))
        self.rig = RigCameras(names, np.repeat(K[None], n_views, axis=0), ring_rig(n_views, 4.0), np.tile(np.array([320, 240]), (n_views, 1)))
        self.skeleton: Float64[ndarray, "133 3"] = np.random.default_rng(3).uniform(-0.3, 0.3, (133, 3))
        for shoulder, elbow, wrist, side in ((5, 7, 9, 1.0), (6, 8, 10, -1.0)):  # an average adult's arms, hanging down
            self.skeleton[elbow] = self.skeleton[shoulder] + np.array([0.0, 0.05 * side, -0.255])
            self.skeleton[wrist] = self.skeleton[elbow] + np.array([0.05, 0.0, -0.2317])
        self.people = people
        self.crop_spec = CropSpec(input_size=(192, 256))
        self.torch = torch
        from dagkit.sources import SegmentRef

        self.segment = SegmentRef("synthetic", "ring")
        self.layout = RigLayout(names, (), names)  # every camera exo, calibrated and a pinhole

    def project(self, frame_index: int, person: int = 0) -> Float64[ndarray, "v 133 2"]:
        """A person's exact keypoints in every view."""
        return project_points(self.skeleton + np.array([0.002 * frame_index, 0.0, 0.0]) + np.array(self.people[person].offset_m), self.rig.intrinsics, self.rig.cam_T_world)

    def person_box(self, frame_index: int, person: int, view: int) -> Float64[ndarray, "4"]:
        """The padded box of a person's keypoints in one view."""
        xy = self.project(frame_index, person)[view]
        center, size = (xy.min(axis=0) + xy.max(axis=0)) / 2.0, (xy.max(axis=0) - xy.min(axis=0)) * 1.5
        return np.concatenate((center - size / 2, center + size / 2))

    def seen(self, person: int, view: int) -> bool:
        views = self.people[person].views
        return views is None or view in views

    def frames(self, frame_index: int):  # MultiViewFrames
        from simplecv.types.frames import MultiViewFrames

        rgb = self.torch.zeros((len(self.rig.names), 3, 240, 320), dtype=self.torch.uint8)
        rgb[:, 0, 0, 0] = frame_index
        rgb[:, 1, 0, 0] = self.torch.arange(len(self.rig.names))  # the view, for the detector (it sees a subset)
        return MultiViewFrames(1_000_000 * frame_index, frame_index, self.rig.names, rgb, self.torch.full((len(self.rig.names),), 1_000_000 * frame_index))

    def detector(self, frames_hwc):  # BoxDetections: every person's box in the views that see them
        from posekit.predictions import BoxDetections

        frame_index = int(frames_hwc[0, 0, 0, 0])
        found = [(row, person) for row, view in enumerate(frames_hwc[:, 0, 0, 1].long().tolist()) for person in range(len(self.people)) if self.seen(person, view)]
        views = frames_hwc[:, 0, 0, 1].long().tolist()
        return BoxDetections(
            xyxy=self.torch.as_tensor(np.array([self.person_box(frame_index, person, views[row]) for row, person in found]).reshape(-1, 4), dtype=self.torch.float32),
            scores=self.torch.as_tensor([self.people[person].score for _, person in found]),
            frame_indices=self.torch.as_tensor([row for row, _ in found], dtype=self.torch.long),
        )

    def pose(self, frames_hwc, boxes):  # the keypoints of the first person seen whose box holds the crop centre (0.95); else 0.05
        from types import SimpleNamespace

        frame_index = int(frames_hwc[0, 0, 0, 0])
        xy = np.empty((len(boxes.frame_indices), 133, 2))
        conf = np.full(len(boxes.frame_indices), 0.05)
        for row, (view, box) in enumerate(zip(boxes.frame_indices.tolist(), boxes.xyxy.numpy(), strict=True)):
            center = (box[:2] + box[2:]) / 2.0
            holders = [p for p in range(len(self.people)) if self.seen(p, view) and (self.person_box(frame_index, p, view)[:2] <= center).all() and (center <= self.person_box(frame_index, p, view)[2:]).all()]
            xy[row] = self.project(frame_index, holders[0] if holders else 0)[view]
            conf[row] = 0.95 if holders else 0.05
        scores = self.torch.as_tensor(conf[:, None] * np.ones((1, 133)), dtype=self.torch.float32)
        return SimpleNamespace(frame_indices=boxes.frame_indices, xy=self.torch.as_tensor(xy, dtype=self.torch.float32), scores=scores)

    def fit_body(self, kp_xy, conf, intrinsics, cam_T_world):  # BodyFit whose "mesh" is the triangulated shoulders and left hip
        from simplecv.ops.triangulate import triangulate_weighted

        from exo_calib.ops.body import BodyFit

        n_views, n_frames = kp_xy.shape[:2]
        xyz, _ = triangulate_weighted(
            self.torch.from_numpy(kp_xy.reshape(n_views, -1, 2)), self.torch.from_numpy(conf.reshape(n_views, -1)), self.torch.from_numpy(intrinsics @ cam_T_world[:, :3])
        )
        joints = xyz.numpy().reshape(n_frames, 133, 3)
        zeros = np.zeros((n_frames, 165), np.float32)
        return BodyFit(np.zeros(16, np.float32), zeros, zeros[:, :3], joints[:, [5, 6, 11]].astype(np.float32))

    def depth(self, frame_rgb, K_33):  # MoGe stand-in: every pixel at 4 m
        from types import SimpleNamespace

        return SimpleNamespace(depth_meters=np.full(frame_rgb.shape[:2], 4.0, dtype=np.float32))

    def frames_at(self, view: int, indices):
        return self.torch.zeros((len(indices), 3, 240, 320), dtype=self.torch.uint8)

    def times_ns(self, view: int) -> np.ndarray:
        """Every video frame's time on every camera (``frames`` samples some of them)."""
        return 1_000_000 * np.arange(64, dtype=np.int64)

    def close(self) -> None:  # a FrameProvider: the App's reader of the segment's videos
        pass

    def resources(self) -> dict:
        """Fake graph resources, keyed like the App's model factories."""
        self.pose_model = FakeModel(self.pose, crop_spec=self.crop_spec)
        return {"detector": FakeModel(self.detector), "pose": self.pose_model, "moge": FakeModel(self.depth), "frames": self, "g3t": FakeModel(self.detector), "smplx": FakeModel(self.fit_body, faces=np.array([[0, 1, 2]]))}


class FakeModel:
    def __init__(self, fn, **attributes) -> None:
        self.fn = fn
        self.__dict__.update(attributes)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
