"""The map's surface: keyframe depth fused into a sparse colour TSDF (``packages/tsdf-rs``), logged block by block.

:class:`LiveSurface` runs on its own thread (:class:`LatestWinsWorker`, fed by the async logger on every new
keyframe): each new keyframe is integrated into a live volume and only the changed blocks are logged; every few
keyframes and at the end, everything is re-fused at the backend's latest poses. ``tsdf-rs`` releases the GIL while it
integrates and extracts.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import lietorch_mojo as lietorch
import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, UInt8, UInt32
from numpy import ndarray
from tsdf_rs import TsdfVolume

from mast3r_slam.lietorch_utils import as_SE3


@dataclass(frozen=True, slots=True)
class KeyframeDepth:
    """What a keyframe contributes to the surface; its pose comes separately, since the backend refines it."""

    depth_hw: Float32[ndarray, "h w"]
    """Z of the keyframe's canonical pointmap in its own camera frame, before the Sim3 scale; 0 where the
    confidence is at or below the logger's threshold."""
    k_33: Float32[ndarray, "3 3"]
    """Pinhole intrinsics on the depth raster."""
    rgb_hw3: UInt8[ndarray, "h w 3"]
    """The model's cropped image, on the depth raster."""


@dataclass(frozen=True, slots=True)
class SurfaceSettings:
    """TSDF fusion parameters, in map units (MASt3R's pointmaps are roughly metric)."""

    voxel_size: float = 0.04
    """Voxel edge."""
    truncation_voxels: float = 3.0
    """Signed-distance truncation in voxels (Open3D's ``3 * voxel`` convention)."""


def new_volume(settings: SurfaceSettings) -> TsdfVolume:
    """An empty colour-fusing TSDF volume."""
    return TsdfVolume(voxel_size=settings.voxel_size, sdf_trunc=settings.truncation_voxels * settings.voxel_size, fuse_color=True)


def integrate_keyframe(volume: TsdfVolume, keyframe: KeyframeDepth, world_sim3_cam_data: Float32[ndarray, "8"]) -> None:
    """Integrate one keyframe at its world-from-camera Sim3 (lietorch ``[t, q, s]``).

    A Sim3 maps a camera point ``X`` to ``s R X + t``: the rigid pose ``(R, t)`` with the depth scaled by ``s``.
    """
    sim3: lietorch.Sim3 = lietorch.Sim3(torch.from_numpy(world_sim3_cam_data).unsqueeze(0))
    world_T_cam: Float32[ndarray, "4 4"] = as_SE3(sim3).matrix()[0].numpy().astype(np.float32)
    scale: float = float(world_sim3_cam_data[7])
    volume.integrate(
        np.ascontiguousarray(keyframe.depth_hw * scale, dtype=np.float32),
        keyframe.k_33,
        np.linalg.inv(world_T_cam.astype(np.float64)).astype(np.float32),
        np.ascontiguousarray(keyframe.rgb_hw3),
    )


def front_faced_mesh(
    vertices: Float32[ndarray, "v 3"], triangles: UInt32[ndarray, "t 3"], colors: UInt8[ndarray, "v 3"] | None, normals: Float32[ndarray, "v 3"]
) -> rr.Mesh3D:
    """A ``tsdf-rs`` mesh as a Mesh3D that renders only front faces (the side the cameras saw), so walls seen from
    outside a room do not hide it."""
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=colors,
        vertex_normals=normals,
        face_rendering=rr.components.MeshFaceRendering.Front,
    )


T = TypeVar("T")


class LatestWinsWorker(Generic[T]):
    """Runs ``handle`` on its own thread for the newest request; a request that arrives while one is pending
    replaces it (a size-1 slot, never a growing queue). :meth:`close` handles the pending request, then joins."""

    def __init__(self, handle: Callable[[T], None], name: str) -> None:
        self._handle: Callable[[T], None] = handle
        self._condition: threading.Condition = threading.Condition()
        self._pending: T | None = None
        self._closing: bool = False
        self._thread: threading.Thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, request: T) -> None:
        """Replace the pending request, if any, and wake the worker; returns at once."""
        with self._condition:
            self._pending = request
            self._condition.notify()

    def close(self, timeout: float | None = None) -> None:
        """Handle the pending request, then stop the thread and join it."""
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                request: T | None = self._pending
                self._pending = None
                if request is None:
                    return
            # An exception ends the thread; threading.excepthook prints it (no bare except around beartype checks).
            self._handle(request)


@dataclass(frozen=True, slots=True)
class SurfaceRequest:
    """Every keyframe so far (index, depth, latest pose) and the time point the update is logged at."""

    keyframes: list[tuple[int, KeyframeDepth, Float32[ndarray, "8"]]]
    frame_idx: int
    timestamp_ns: int | None
    final: bool = False
    """The run's end: re-fuse everything at the final poses."""


class LiveSurface:
    """The map surface, grown on the worker thread and logged block by block under ``mesh_path``.

    Each request integrates only the keyframes not yet in the live volume, at their current pose, and logs the
    blocks that changed (``mesh_path/<x>_<y>_<z>``), so an update is small. The backend keeps refining poses,
    which does not move what was integrated: every ``refuse_every`` keyframes, and for the final request, a fresh
    volume is fused from every keyframe at its latest pose, the logged blocks are cleared, and all of its blocks are
    logged; it becomes the live volume. Blocks are logged at the request's time point, so scrubbing shows growth.
    (MetricTrail's ``live_surface`` pattern; see ``packages/metrictrail``.)
    """

    def __init__(
        self,
        mesh_path: str,
        recording: rr.RecordingStream | None,
        timelines: tuple[str, str],
        refuse_every: int = 10,
        settings: SurfaceSettings | None = None,
    ) -> None:
        self._mesh_path: str = mesh_path
        self._recording: rr.RecordingStream | None = recording
        self._timelines: tuple[str, str] = timelines
        self._refuse_every: int = refuse_every
        self._settings: SurfaceSettings = settings if settings is not None else SurfaceSettings()
        self._volume: TsdfVolume = new_volume(self._settings)
        self._integrated: set[int] = set()
        self._n_kf_at_refuse: int = 0

    def handle(self, request: SurfaceRequest) -> None:
        """Bring the surface up to ``request`` and log what changed (run on the worker thread)."""
        refuse: bool = request.final or len(request.keyframes) - self._n_kf_at_refuse >= self._refuse_every
        if refuse:
            self._volume = new_volume(self._settings)
            self._integrated = set()
            self._n_kf_at_refuse = len(request.keyframes)
        for kf_idx, keyframe, world_sim3_cam_data in request.keyframes:
            if kf_idx not in self._integrated:
                integrate_keyframe(self._volume, keyframe, world_sim3_cam_data)
                self._integrated.add(kf_idx)

        frame_timeline, video_timeline = self._timelines
        rr.set_time(frame_timeline, sequence=request.frame_idx, recording=self._recording)
        if request.timestamp_ns is not None:
            rr.set_time(video_timeline, duration=1e-9 * float(request.timestamp_ns), recording=self._recording)
        if refuse:
            rr.log(self._mesh_path, rr.Clear(recursive=True), recording=self._recording)
        for (x, y, z), (vertices, triangles, colors, normals) in self._volume.extract_dirty_block_meshes():
            block_path: str = f"{self._mesh_path}/{x}_{y}_{z}"
            if triangles.shape[0] == 0:
                rr.log(block_path, rr.Clear(recursive=False), recording=self._recording)
            else:
                rr.log(block_path, front_faced_mesh(vertices, triangles, colors, normals), recording=self._recording)
