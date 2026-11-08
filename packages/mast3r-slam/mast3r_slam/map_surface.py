"""The map's surface: every keyframe's depth fused into a sparse colour TSDF (``packages/tsdf-rs``), one mesh.

The backend keeps refining keyframe poses, so a volume that only grows goes stale; every keyframe is re-fused at its
latest pose from scratch instead (:func:`fuse_keyframes`). That runs on its own thread (:class:`LatestWinsWorker`,
fed by the async logger on every new keyframe), so the logger never waits on it; ``tsdf-rs`` releases the GIL while
it integrates and extracts.
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


def fuse_keyframes(
    keyframes: list[tuple[KeyframeDepth, Float32[ndarray, "8"]]],
    settings: SurfaceSettings | None = None,
) -> rr.Mesh3D | None:
    """Fuse keyframes, each at its world-from-camera Sim3 (lietorch ``[t, q, s]``), into one front-faced mesh.

    A Sim3 maps a camera point ``X`` to ``s R X + t``: the rigid pose ``(R, t)`` with the depth scaled by ``s``.

    Returns:
        The mesh, rendering only front faces (the side the cameras saw), so walls seen from outside a room do not
        hide it; None if no surface came out.
    """
    if settings is None:
        settings = SurfaceSettings()
    volume: TsdfVolume = TsdfVolume(
        voxel_size=settings.voxel_size, sdf_trunc=settings.truncation_voxels * settings.voxel_size, fuse_color=True
    )
    for keyframe, world_sim3_cam_data in keyframes:
        sim3: lietorch.Sim3 = lietorch.Sim3(torch.from_numpy(world_sim3_cam_data).unsqueeze(0))
        world_T_cam: Float32[ndarray, "4 4"] = as_SE3(sim3).matrix()[0].numpy().astype(np.float32)
        scale: float = float(world_sim3_cam_data[7])
        volume.integrate(
            np.ascontiguousarray(keyframe.depth_hw * scale, dtype=np.float32),
            keyframe.k_33,
            np.linalg.inv(world_T_cam.astype(np.float64)).astype(np.float32),
            np.ascontiguousarray(keyframe.rgb_hw3),
        )
    vertices: Float32[ndarray, "v 3"]
    triangles: UInt32[ndarray, "t 3"]
    colors: UInt8[ndarray, "v 3"] | None
    normals: Float32[ndarray, "v 3"]
    vertices, triangles, colors, normals = volume.extract_mesh()
    if triangles.shape[0] == 0:
        return None
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
    """Every keyframe at its latest pose, and the time point the mesh is logged at."""

    keyframes: list[tuple[KeyframeDepth, Float32[ndarray, "8"]]]
    frame_idx: int
    timestamp_ns: int | None


def log_surface_request(request: SurfaceRequest, mesh_path: str, recording: rr.RecordingStream | None, timelines: tuple[str, str]) -> None:
    """Fuse ``request`` and log the mesh at its time point on ``recording`` (pass it explicitly: a worker thread
    does not see the caller's thread- or context-local recording)."""
    mesh: rr.Mesh3D | None = fuse_keyframes(request.keyframes)
    if mesh is None:
        return
    frame_timeline, video_timeline = timelines
    rr.set_time(frame_timeline, sequence=request.frame_idx, recording=recording)
    if request.timestamp_ns is not None:
        rr.set_time(video_timeline, duration=1e-9 * float(request.timestamp_ns), recording=recording)
    rr.log(mesh_path, mesh, recording=recording)
