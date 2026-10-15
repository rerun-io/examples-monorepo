"""Unit-lane fixtures: a ring of pinhole cameras looking at the origin (world z up) and fake models for the node and graph tests."""

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

K_ENGINE: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 376.0], [0.0, 0.0, 1.0]])
"""Intrinsics of a 1024 x 752 engine image."""


def look_at(camera_position: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """cam_T_world (RDF camera) of a camera at ``camera_position`` looking at the origin."""
    forward: Float64[ndarray, "3"] = -camera_position / np.linalg.norm(camera_position)
    right: Float64[ndarray, "3"] = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    rotation: Float64[ndarray, "3 3"] = np.stack([right, np.cross(forward, right), forward], axis=0)
    cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = rotation
    cam_T_world[:3, 3] = -rotation @ camera_position
    return cam_T_world


def camera_ring(views: int, radius_m: float = 4.0, height_m: float = 1.4) -> Float64[ndarray, "v 4 4"]:
    """cam_T_world of ``views`` cameras evenly spaced on a horizontal ring."""
    angles: Float64[ndarray, " v"] = np.linspace(0.0, 2.0 * np.pi, views, endpoint=False)
    return np.stack([look_at(np.array([radius_m * np.cos(a), radius_m * np.sin(a), height_m])) for a in angles])


def project(points: Float64[ndarray, "n 3"], cam_T_world: Float64[ndarray, "v 4 4"]) -> Float64[ndarray, "v n 2"]:
    """Pixels of ``points`` in every camera of the ring (``K_ENGINE``)."""
    camera: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3, :3], points) + cam_T_world[:, None, :3, 3]
    pixels: Float64[ndarray, "v n 3"] = camera @ K_ENGINE.T
    return pixels[..., :2] / pixels[..., 2:]


# ---- fakes for the node and graph tests (CPU tensors, no models) -------------------------------------------------------

import dataclasses  # noqa: E402
from collections.abc import Iterator  # noqa: E402

import torch  # noqa: E402
from dagkit.node import Event, Fenced  # noqa: E402
from dagkit.time import STATIC  # noqa: E402
from posekit.models import IdentityEncoder, PersonDetector  # noqa: E402
from posekit.predictions import BoxDetections  # noqa: E402
from simplecv.types.frames import MultiViewFrames  # noqa: E402
from simplecv.types.rig import RigCameras  # noqa: E402

from mamma.models.landmarks.estimator import DenseLandmarks  # noqa: E402
from mamma.ops.window_fit import FitResult  # noqa: E402

ENGINE_HW: tuple[int, int] = (48, 64)
K_SMALL: Float64[ndarray, "3 3"] = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
BODY: Float64[ndarray, "6 3"] = np.array([[0.0, 0.0, 0.1], [0.1, 0.0, 0.9], [-0.1, 0.0, 0.9], [0.0, 0.1, 1.5], [0.0, -0.1, 1.2], [0.05, 0.05, 0.5]])
"""Six landmarks of a person standing at the origin (metres, z up)."""


class FakeScene:
    """Three cameras on a ring, stored at twice the engine size, one person at the origin, 30 frames at 30 fps."""

    def __init__(self, views: int = 3) -> None:
        names = tuple(f"rig_{i:02d}/cam_00" for i in range(views))
        stored = K_SMALL.copy()
        stored[:2] *= 2.0
        self.rig = RigCameras(names, np.stack([stored] * views), camera_ring(views), np.tile(np.array([128, 96], dtype=np.int64), (views, 1)))

    def frames(self, index: int) -> MultiViewFrames:
        v: int = len(self.rig.names)
        return MultiViewFrames(index * 33_333_333, index, self.rig.names, torch.zeros((v, 3, *ENGINE_HW), dtype=torch.uint8), torch.full((v,), index * 33_333_333))

    def events(self, count: int = 30) -> Iterator[Event]:
        yield Event(STATIC, "rig", self.rig)
        for index in range(count):
            frames = self.frames(index)
            yield Event(frames.t_ns, "frames", Fenced(frames, None))

    def pixels(self, view: int) -> Float64[ndarray, "6 2"]:
        camera: Float64[ndarray, "6 3"] = BODY @ self.rig.cam_T_world[view, :3, :3].T + self.rig.cam_T_world[view, :3, 3]
        image: Float64[ndarray, "6 3"] = camera @ K_SMALL.T
        return image[:, :2] / image[:, 2:]


class FakeDetector(PersonDetector):
    def __call__(self, frames_rgb: torch.Tensor) -> BoxDetections:
        b: int = frames_rgb.shape[0]
        return BoxDetections(torch.tensor([[20.0, 8.0, 44.0, 44.0]] * b), torch.full((b,), 0.9), torch.arange(b))


class FakeEncoder(IdentityEncoder):
    embed_dim = 512

    def __call__(self, frames_rgb: torch.Tensor, detections: BoxDetections) -> torch.Tensor:
        return torch.ones((detections.num_detections, 512))


class FakeMasks:
    """Every prompted object keeps a box mask at its last prompt."""

    def new_states(self, views: int, video_hw: tuple[int, int]) -> list[dict]:
        return [{} for _ in range(views)]

    def __call__(self, states: list[dict], frame_index: int, frames: torch.Tensor, prompts: list[dict]) -> list[dict]:
        out: list[dict] = []
        for state, prompt in zip(states, prompts, strict=True):
            state.update({obj: np.asarray(box, dtype=np.int64) for obj, box in prompt.items()})
            masks: dict = {}
            for obj, (x1, y1, x2, y2) in state.items():
                mask = torch.zeros(frames.shape[2:], dtype=torch.bool)
                mask[y1:y2, x1:x2] = True
                masks[obj] = (mask, torch.tensor([0.9]))
            out.append(masks)
        return out


class FakeLandmarker:
    """The scene's six landmarks, exactly, for every crop."""

    def __init__(self, scene: FakeScene) -> None:
        self.scene = scene

    def __call__(self, frames: torch.Tensor, view: np.ndarray, masks: torch.Tensor, boxes_xyxy: np.ndarray, rgb_scale: float = 1.0) -> DenseLandmarks:
        xy = torch.as_tensor(np.stack([self.scene.pixels(v) for v in view.tolist()]), dtype=torch.float32).reshape(-1, 6, 2)
        k: int = xy.shape[0]
        return DenseLandmarks(xy, torch.full((k, 6), -9.0), torch.ones((k, 6)), torch.zeros((k, 6)))


class FakeFitter:
    """``SlidingWindowFitter``'s emission schedule (window 8, lag 4, heads at bootstrap) with the anchors' centroid as the body."""

    def __init__(self, window: int = 8, lag: int = 4) -> None:
        self.window, self.lag, self.frames, self.heads = window, lag, [], []

    def _result(self, frame: tuple[int, np.ndarray]) -> FitResult:
        z: np.ndarray = np.zeros(3, dtype=np.float32)
        return FitResult(frame[0], np.tile(frame[1], (4, 1)), np.zeros((5, 3), np.float32), np.zeros(165, np.float32), np.zeros(16, np.float32), z, np.zeros((55, 3), np.float32))

    def push(self, frame_idx, pts2d_per_cam, vis_per_cam, anchors3d, anchors_valid, floor_contact=None, optimize=True) -> FitResult | None:  # type: ignore[no-untyped-def]
        self.frames = [*self.frames, (frame_idx, anchors3d[anchors_valid].mean(dim=0).numpy().astype(np.float32))][-self.window :]
        if len(self.frames) < self.window:
            return None
        if not hasattr(self, "booted"):
            self.booted, self.heads = True, [self._result(f) for f in self.frames[: self.window - 1 - self.lag]]
        return self._result(self.frames[-1 - self.lag])

    def drain_head(self) -> list[FitResult]:
        heads, self.heads = self.heads, []
        return heads

    def drain(self) -> list[FitResult]:
        return [self._result(f) for f in self.frames[-self.lag :]] if hasattr(self, "booted") else []


def assert_same(a: object, b: object) -> None:
    for f in dataclasses.fields(a):  # type: ignore[arg-type]
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, torch.Tensor | np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype and x.shape == y.shape, f.name
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-4, err_msg=f.name)
        else:
            assert x == y, f.name
