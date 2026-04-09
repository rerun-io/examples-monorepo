"""Async Rerun logger thread.

Consumes ``LogEvent`` objects from a queue and performs all Rerun logging
off the tracking hot-path.  The thread binds to the caller's
``RecordingStream`` (for Gradio) or uses the global recording (for CLI).

Handles JPEG compression, focal estimation, pointmap/depth conversion,
camera path accumulation, and blueprint refreshes — all work that was
previously synchronous on the tracking thread.

Usage as a context manager::

    event_queue: queue.Queue[LogEvent] = queue.Queue(maxsize=32)
    with AsyncRerunLogger(event_queue, parent_log_path, timeline, recording=rec) as logger:
        for frame in frames:
            event_queue.put(snapshot_current_frame(frame, ...))
        # LogTerminate is sent automatically on __exit__
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path

import lietorch
import numpy as np
import rerun as rr
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8, UInt16
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import log_pinhole
from torch import Tensor

from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.log_events import (
    KeyframeSnapshot,
    LogCurrentFrame,
    LogEvent,
    LogMapUpdate,
    LogTerminate,
    LogText,
)
from mast3r_slam.mast3r_utils import frame_to_extrinsics, frame_to_pinhole
from mast3r_slam.rerun_log_utils import FRAME_TIMELINE, VIDEO_TIMELINE, create_blueprints

# ── Snapshot helpers ──────────────────────────────────────────────────────
# These create CPU-only event objects from live Frame / SharedKeyframes /
# SharedStates data.  Called on the pipeline thread; consumed by the
# logger thread.


def snapshot_current_frame(
    frame: Frame,
    frame_idx: int,
    timestamp_ns: int | None,
    keyframes: SharedKeyframes | None = None,
) -> LogCurrentFrame:
    """Create a CPU snapshot of the current frame for async logging.

    Args:
        frame: The current tracked frame.
        frame_idx: Frame index in the sequence.
        timestamp_ns: Video timestamp in nanoseconds, or None.
        keyframes: Shared keyframe buffer (to snapshot the last keyframe's
            evolving pointmap/confidence for the 2D panel). None skips it.

    Returns:
        A ``LogCurrentFrame`` event with CPU-only numpy arrays.
    """
    h: int = int(frame.img_shape.squeeze()[0].item())
    w: int = int(frame.img_shape.squeeze()[1].item())
    world_sim3_cam_data: Float32[ndarray, "8"] = frame.world_sim3_cam.data.squeeze().cpu().numpy().astype(np.float32)
    rgb_cpu: Float32[ndarray, "H W 3"] = frame.rgb.numpy().astype(np.float32)
    X_canon_cpu: Float32[ndarray, "hw 3"] | None = (
        frame.X_canon.detach().cpu().numpy().astype(np.float32) if frame.X_canon is not None else None
    )
    C_cpu: Float32[ndarray, "hw 1"] | None = (
        frame.C.detach().cpu().numpy().astype(np.float32) if frame.C is not None else None
    )

    # Snapshot the last keyframe's current state for the 2D panel.
    # The tracker continuously updates the active keyframe's pointmap/
    # confidence, so we re-read it each frame.
    last_kf_rgb: Float32[ndarray, "H W 3"] | None = None
    last_kf_X_canon: Float32[ndarray, "hw 3"] | None = None
    last_kf_C: Float32[ndarray, "hw 1"] | None = None
    if keyframes is not None:
        last_kf: Frame | None = keyframes.last_keyframe()
        if last_kf is not None:
            last_kf_rgb = last_kf.rgb.numpy().astype(np.float32)
            if last_kf.X_canon is not None:
                last_kf_X_canon = last_kf.X_canon.detach().cpu().numpy().astype(np.float32)
            if last_kf.C is not None:
                last_kf_C = last_kf.C.detach().cpu().numpy().astype(np.float32)

    return LogCurrentFrame(
        frame_idx=frame_idx,
        timestamp_ns=timestamp_ns,
        world_sim3_cam_data=world_sim3_cam_data,
        rgb=rgb_cpu,
        X_canon=X_canon_cpu,
        C=C_cpu,
        img_shape=(h, w),
        last_kf_rgb=last_kf_rgb,
        last_kf_X_canon=last_kf_X_canon,
        last_kf_C=last_kf_C,
    )


def snapshot_keyframe(frame: Frame, kf_idx: int) -> KeyframeSnapshot:
    """Create a CPU snapshot of a new keyframe for async logging.

    Args:
        frame: The keyframe frame.
        kf_idx: Keyframe buffer index.

    Returns:
        A ``KeyframeSnapshot`` with CPU-only numpy arrays.
    """
    h: int = int(frame.img_shape.squeeze()[0].item())
    w: int = int(frame.img_shape.squeeze()[1].item())
    assert frame.X_canon is not None
    assert frame.C is not None
    return KeyframeSnapshot(
        kf_idx=kf_idx,
        world_sim3_cam_data=frame.world_sim3_cam.data.squeeze().cpu().numpy().astype(np.float32),
        rgb=frame.rgb.numpy().astype(np.float32),
        X_canon=frame.X_canon.detach().cpu().numpy().astype(np.float32),
        C=frame.C.detach().cpu().numpy().astype(np.float32),
        img_shape=(h, w),
    )


def snapshot_edges(
    states: SharedStates,
    keyframes: SharedKeyframes,
) -> tuple[Float32[ndarray, "n 3"], Float32[ndarray, "n 3"]] | None:
    """Snapshot factor graph edge positions from shared state.

    Args:
        states: Shared system state (for edge index lists).
        keyframes: Shared keyframe buffer (for poses).

    Returns:
        (positions_i, positions_j) tuple or None if no edges.
    """
    with states.lock:
        ii: Int[Tensor, "num_edges"] = torch.tensor(states.edges_ii, dtype=torch.long)
        jj: Int[Tensor, "num_edges"] = torch.tensor(states.edges_jj, dtype=torch.long)
        if ii.numel() == 0 or jj.numel() == 0:
            return None
        world_sim3_cami: lietorch.Sim3 = lietorch.Sim3(keyframes.world_sim3_cam[ii, 0])
        world_sim3_camj: lietorch.Sim3 = lietorch.Sim3(keyframes.world_sim3_cam[jj, 0])
    positions_i: Float32[ndarray, "n 3"] = world_sim3_cami.matrix()[:, :3, 3].cpu().numpy().astype(np.float32)
    positions_j: Float32[ndarray, "n 3"] = world_sim3_camj.matrix()[:, :3, 3].cpu().numpy().astype(np.float32)
    return (positions_i, positions_j)


def compute_orient(
    keyframes: SharedKeyframes,
    n_kf: int,
) -> tuple[Float32[ndarray, "3 3"], Float32[ndarray, "3"]]:
    """Compute gravity-alignment transform from all keyframe poses.

    Args:
        keyframes: Shared keyframe buffer.
        n_kf: Number of keyframes to use.

    Returns:
        (orient_R, orient_t) rotation and translation arrays.
    """
    world_T_cam_list: list[Float32[ndarray, "4 4"]] = []
    for i in range(n_kf):
        kf: Frame = keyframes[i]
        kf_ext = frame_to_extrinsics(kf, camera_conventions="RUB")
        assert kf_ext.world_T_cam is not None
        world_T_cam_list.append(kf_ext.world_T_cam.astype(np.float32))

    world_T_cam: Float32[ndarray, "n_kf 4 4"] = np.stack(world_T_cam_list)
    orient_34: Float[ndarray, "3 4"] = auto_orient_and_center_poses(
        world_T_cam, method="up", center_method="poses"
    ).transform
    orient_R: Float32[ndarray, "3 3"] = orient_34[:, :3].astype(np.float32)
    orient_t: Float32[ndarray, "3"] = orient_34[:, 3].astype(np.float32)
    return (orient_R, orient_t)


# ── Internal helpers ──────────────────────────────────────────────────────


def snapshot_to_frame(
    world_sim3_cam_data: Float32[ndarray, "8"],
    rgb: Float32[ndarray, "H W 3"],
    X_canon: Float32[ndarray, "hw 3"] | None,
    C: Float32[ndarray, "hw 1"] | None,
    img_shape: tuple[int, int],
    frame_id: int = 0,
) -> Frame:
    """Reconstruct a CPU-only Frame from snapshot arrays.

    The returned Frame has all tensors on CPU and is suitable for
    ``frame_to_pinhole()`` / ``frame_to_extrinsics()`` which already
    call ``.cpu()`` internally.

    Args:
        world_sim3_cam_data: Raw lietorch Sim3 data (8 floats).
        rgb: Normalized RGB image [0, 1], HWC layout.
        X_canon: Canonical 3D point map, or None.
        C: Per-point confidence, or None.
        img_shape: (height, width).
        frame_id: Frame index for the reconstructed Frame.

    Returns:
        A Frame with CPU tensors suitable for ``frame_to_pinhole()``.
    """
    h: int = img_shape[0]
    w: int = img_shape[1]
    sim3: lietorch.Sim3 = lietorch.Sim3(torch.from_numpy(world_sim3_cam_data).unsqueeze(0))
    rgb_tensor: torch.Tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    img_shape_tensor: torch.Tensor = torch.tensor([[h, w]])
    frame: Frame = Frame(
        frame_id=frame_id,
        rgb_tensor=rgb_tensor,
        img_shape=img_shape_tensor,
        img_true_shape=img_shape_tensor.clone(),
        rgb=torch.from_numpy(rgb),
        world_sim3_cam=sim3,
    )
    if X_canon is not None:
        frame.X_canon = torch.from_numpy(X_canon)
    if C is not None:
        frame.C = torch.from_numpy(C)
    return frame


# ── AsyncRerunLogger ──────────────────────────────────────────────────────


class AsyncRerunLogger:
    """Consumes log events from a queue on a dedicated thread.

    All expensive work (JPEG compression, focal estimation, pointmap
    conversion, ``rr.log()`` calls) happens in this thread, keeping the
    tracking pipeline unblocked.

    Use as a context manager to guarantee the logger thread is started
    and cleanly shut down::

        with AsyncRerunLogger(event_queue, path, timeline) as logger:
            event_queue.put(snapshot_current_frame(...))

    Args:
        event_queue: Queue of ``LogEvent`` objects.
        parent_log_path: Root Rerun entity path.
        timeline: Active timeline name.
        recording: Rerun recording to bind in this thread (Gradio).
            ``None`` uses the global recording (CLI).
    """

    def __init__(
        self,
        event_queue: queue.Queue[LogEvent],
        parent_log_path: Path,
        timeline: str,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self._queue: queue.Queue[LogEvent] = event_queue
        self._recording: rr.RecordingStream | None = recording
        self._parent_log_path: Path = parent_log_path
        self._timeline: str = timeline
        self._thread: threading.Thread | None = None

        # Internal visualization state
        self._path_list: list[list[float]] = []
        self._logged_keyframes: set[int] = set()
        self._keyframe_pinhole_static: dict[int, PinholeParameters] = {}
        self._last_kf_idx: int = -1
        self._last_blueprint_n_kf: int = -1
        self._n_keyframes: int = 0
        self._conf_thresh: int = 7
        self._image_plane_distance: float = 0.2
        self._error: Exception | None = None

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> AsyncRerunLogger:
        """Start the logger thread."""
        self.start()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Send LogTerminate, join the thread, and warn if it outlives the timeout."""
        # Use a bounded put so we don't deadlock if the logger thread
        # crashed and the queue is full.
        try:
            self._queue.put(LogTerminate(), timeout=5.0)
        except queue.Full:
            # Queue is stuck — the logger thread is likely dead.
            if self._error is not None:
                warnings.warn(
                    f"Async logger thread crashed: {self._error}. {self._queue.qsize()} events were not processed.",
                    stacklevel=2,
                )
            return

        self.join(timeout=30.0)
        if self.is_alive():
            warnings.warn(
                f"Async logger thread did not finish within 30s ({self._queue.qsize()} events remaining). "
                "Some trailing log entries may be lost.",
                stacklevel=2,
            )

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the logger thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="async-rerun-logger",
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = 30.0) -> None:
        """Wait for the logger thread to finish.

        Args:
            timeout: Maximum seconds to wait (None = wait forever).
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Return True if the logger thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    # ── Thread main loop ──────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: bind recording, then process events until LogTerminate."""
        if self._recording is not None:
            rr.set_thread_local_data_recording(self._recording)

        try:
            while True:
                try:
                    event: LogEvent = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(event, LogTerminate):
                    self._drain_queue()
                    break

                self._handle_event(event)
        except Exception as exc:
            self._error = exc
            traceback.print_exc(file=sys.stderr)

    def _drain_queue(self) -> None:
        """Process all remaining non-terminate events."""
        while True:
            try:
                event: LogEvent = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, LogTerminate):
                continue
            self._handle_event(event)

    def _handle_event(self, event: LogEvent) -> None:
        """Dispatch a single event to its handler."""
        match event:
            case LogMapUpdate():
                self._handle_map_update(event)
            case LogCurrentFrame():
                self._handle_current_frame(event)
            case LogText():
                rr.log(event.path, rr.TextLog(event.message, level=event.level))

    # ── Timeline management ───────────────────────────────────────────

    def _set_time(self, frame_idx: int, timestamp_ns: int | None) -> None:
        """Set Rerun timeline state before logging."""
        rr.set_time(FRAME_TIMELINE, sequence=frame_idx)
        if timestamp_ns is not None:
            rr.set_time(VIDEO_TIMELINE, duration=1e-9 * float(timestamp_ns))

    # ── Image helpers ─────────────────────────────────────────────────

    def _log_rgb_image(self, image_path: str, rgb: Float32[ndarray, "H W 3"]) -> UInt8[ndarray, "H W 3"]:
        """Convert [0,1] float RGB to uint8, JPEG-compress, and log."""
        rgb_uint8: UInt8[ndarray, "H W 3"] = (rgb * 255).astype(np.uint8)
        rr.log(
            image_path,
            rr.Image(image=rgb_uint8, color_model=rr.ColorModel.RGB).compress(jpeg_quality=75),
        )
        return rgb_uint8

    def _normalize_to_uint8(self, values: Float32[ndarray, "..."]) -> UInt8[ndarray, "..."]:
        """Normalize float array values to 0-255 uint8 range."""
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        finite_mask: ndarray = np.isfinite(values)
        if not np.any(finite_mask):
            return np.zeros(values.shape, dtype=np.uint8)
        finite_values: Float32[ndarray, "n"] = values[finite_mask]
        min_value: float = float(finite_values.min())
        max_value: float = float(finite_values.max())
        if max_value <= min_value:
            return np.zeros(values.shape, dtype=np.uint8)
        normalized: Float32[ndarray, "..."] = (values - min_value) / (max_value - min_value)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _depth_meters_to_uint16_mm(self, depth_meters: Float32[ndarray, "H W"]) -> UInt16[ndarray, "H W"]:
        """Convert depth in meters to uint16 millimeters."""
        depth_mm: Float32[ndarray, "H W"] = np.clip(depth_meters * 1000.0, 0.0, float(np.iinfo(np.uint16).max))
        return depth_mm.astype(np.uint16)

    def _log_pointmap_and_confidence(
        self,
        X_canon: Float32[ndarray, "hw 3"],
        C: Float32[ndarray, "hw 1"],
        log_path: str,
        height: int,
        width: int,
    ) -> None:
        """Log pointmap depth, pointmap image, and confidence to Rerun."""
        pointmap_hw3: Float32[ndarray, "H W 3"] = X_canon.reshape(height, width, 3).astype(np.float32)
        pointmap_hw3 = np.nan_to_num(pointmap_hw3, nan=0.0, posinf=0.0, neginf=0.0)
        depth_hw: Float32[ndarray, "H W"] = np.maximum(pointmap_hw3[..., 2], 0.0)
        confidence_mask: Bool[ndarray, "H W"] = C.reshape(height, width) > self._conf_thresh
        filtered_depth_hw: Float32[ndarray, "H W"] = np.where(confidence_mask, depth_hw, 0.0)

        confidence_hw: Float32[ndarray, "H W"] = C.reshape(height, width).astype(np.float32)
        confidence_hw = np.nan_to_num(confidence_hw, nan=0.0, posinf=0.0, neginf=0.0)
        confidence_uint8: UInt8[ndarray, "H W"] = self._normalize_to_uint8(confidence_hw)
        pointmap_uint8: UInt8[ndarray, "H W 3"] = self._normalize_to_uint8(pointmap_hw3)
        depth_uint16_mm: UInt16[ndarray, "H W"] = self._depth_meters_to_uint16_mm(filtered_depth_hw)

        rr.log(f"{log_path}/pointmap", rr.Image(pointmap_uint8, color_model=rr.ColorModel.RGB).compress())
        rr.log(f"{log_path}/pointmap_depth", rr.DepthImage(depth_uint16_mm, meter=1000.0).compress())
        rr.log(f"{log_path}/confidence", rr.Image(confidence_uint8, color_model=rr.ColorModel.L).compress())

    # ── Event handlers ────────────────────────────────────────────────

    def _handle_current_frame(self, event: LogCurrentFrame) -> None:
        """Log current camera image, pointmap, transform, and camera path."""
        self._set_time(event.frame_idx, event.timestamp_ns)

        # Reconstruct Frame for frame_to_pinhole()
        frame: Frame = snapshot_to_frame(
            event.world_sim3_cam_data,
            event.rgb,
            event.X_canon,
            event.C,
            event.img_shape,
            frame_id=event.frame_idx,
        )

        cam_log_path: str = f"{self._parent_log_path}/current_camera"
        pinhole = frame_to_pinhole(frame)
        log_pinhole(
            camera=pinhole,
            cam_log_path=Path(cam_log_path),
            image_plane_distance=self._image_plane_distance * 2,
        )

        # RGB image
        rgb_uint8: UInt8[ndarray, "H W 3"] = self._log_rgb_image(
            f"{cam_log_path}/pinhole/image",
            event.rgb,
        )

        # Pointmap, depth, confidence
        if event.X_canon is not None and event.C is not None:
            self._log_pointmap_and_confidence(
                event.X_canon,
                event.C,
                f"{cam_log_path}/pinhole",
                rgb_uint8.shape[0],
                rgb_uint8.shape[1],
            )

        # Camera path
        assert pinhole.extrinsics.world_t_cam is not None
        translation: Float32[ndarray, "3"] = pinhole.extrinsics.world_t_cam
        self._path_list.append(translation.tolist())
        rr.log(
            f"{self._parent_log_path}/path",
            rr.LineStrips3D(
                strips=self._path_list,
                colors=(255, 0, 0),
                labels="Camera Path",
            ),
        )

        # Last keyframe 2D panel — the tracker continuously updates the
        # active keyframe's pointmap/confidence, so relog it every frame.
        if event.last_kf_rgb is not None:
            lk_path: str = f"{self._parent_log_path}/last_keyframe"
            self._log_rgb_image(lk_path, event.last_kf_rgb)
            if event.last_kf_X_canon is not None and event.last_kf_C is not None:
                h_lk: int = event.last_kf_rgb.shape[0]
                w_lk: int = event.last_kf_rgb.shape[1]
                self._log_pointmap_and_confidence(
                    event.last_kf_X_canon,
                    event.last_kf_C,
                    lk_path,
                    h_lk,
                    w_lk,
                )

    def _handle_map_update(self, event: LogMapUpdate) -> None:
        """Handle batched structural changes: new keyframes, poses, edges, orient."""
        self._set_time(event.frame_idx, event.timestamp_ns)

        # New keyframes
        for kf_snapshot in event.new_keyframes:
            self._log_new_keyframe(kf_snapshot)

        # Dirty keyframe pose updates from backend
        for kf_idx, world_sim3_cam_data in event.pose_updates:
            self._relog_keyframe_camera(kf_idx, world_sim3_cam_data)

        # Factor graph edges
        if event.edge_positions is not None:
            positions_i: Float32[ndarray, "n 3"] = event.edge_positions[0]
            positions_j: Float32[ndarray, "n 3"] = event.edge_positions[1]
            self._log_edges(positions_i, positions_j)

        # Orient transform
        if event.orient is not None:
            orient_R: Float32[ndarray, "3 3"] = event.orient[0]
            orient_t: Float32[ndarray, "3"] = event.orient[1]
            rr.log(
                f"{self._parent_log_path}",
                rr.Transform3D(mat3x3=orient_R, translation=orient_t),
            )

        # Blueprint refresh if keyframe count changed
        if self._n_keyframes != self._last_blueprint_n_kf:
            rr.send_blueprint(
                create_blueprints(self._parent_log_path, timeline=self._timeline, n_keyframes=self._n_keyframes)
            )
            self._last_blueprint_n_kf = self._n_keyframes

    def _log_new_keyframe(self, kf: KeyframeSnapshot) -> None:
        """Log a keyframe's static data (image, pointcloud, transform) once."""
        if kf.kf_idx in self._logged_keyframes:
            return

        kf_cam_log_path: str = f"{self._parent_log_path}/keyframes/keyframe-{kf.kf_idx}"

        # Reconstruct Frame for frame_to_pinhole()
        frame: Frame = snapshot_to_frame(
            kf.world_sim3_cam_data,
            kf.rgb,
            kf.X_canon,
            kf.C,
            kf.img_shape,
            frame_id=kf.kf_idx,
        )

        # Camera transform
        pinhole = frame_to_pinhole(frame)
        log_pinhole(
            camera=pinhole,
            cam_log_path=Path(kf_cam_log_path),
            image_plane_distance=self._image_plane_distance,
        )
        self._keyframe_pinhole_static[kf.kf_idx] = PinholeParameters(
            name=pinhole.name,
            extrinsics=pinhole.extrinsics,
            intrinsics=pinhole.intrinsics,
        )

        # RGB image
        kf_rgb_uint8: UInt8[ndarray, "H W 3"] = self._log_rgb_image(
            f"{kf_cam_log_path}/pinhole/image",
            kf.rgb,
        )

        # Pointmap + confidence
        self._log_pointmap_and_confidence(
            kf.X_canon,
            kf.C,
            f"{kf_cam_log_path}/pinhole",
            kf.img_shape[0],
            kf.img_shape[1],
        )

        # Confidence-filtered point cloud
        mask: Bool[ndarray, "hw"] = kf.C.squeeze() > self._conf_thresh
        positions: Float32[ndarray, "num_points 3"] = kf.X_canon
        colors: UInt8[ndarray, "num_points 3"] = kf_rgb_uint8.reshape(-1, 3)
        rr.log(
            f"{kf_cam_log_path}/pointcloud",
            rr.Points3D(positions=positions[mask], colors=colors[mask]),
        )

        self._logged_keyframes.add(kf.kf_idx)
        self._n_keyframes = max(self._n_keyframes, kf.kf_idx + 1)

        # Update last-keyframe 2D view
        self._log_last_keyframe(kf)
        self._last_kf_idx = kf.kf_idx

    def _log_last_keyframe(self, kf: KeyframeSnapshot) -> None:
        """Log the last keyframe's image and pointmap to the 2D view."""
        lk_path: str = f"{self._parent_log_path}/last_keyframe"
        self._log_rgb_image(lk_path, kf.rgb)
        self._log_pointmap_and_confidence(
            kf.X_canon,
            kf.C,
            lk_path,
            kf.img_shape[0],
            kf.img_shape[1],
        )

    def _relog_keyframe_camera(self, kf_idx: int, world_sim3_cam_data: Float32[ndarray, "8"]) -> None:
        """Replay a keyframe camera update after backend refinement using cached intrinsics."""
        kf_cam_log_path: str = f"{self._parent_log_path}/keyframes/keyframe-{kf_idx}"
        cached_pinhole: PinholeParameters | None = self._keyframe_pinhole_static.get(kf_idx)
        if cached_pinhole is None:
            return
        frame: Frame = snapshot_to_frame(
            world_sim3_cam_data,
            rgb=np.zeros((1, 1, 3), dtype=np.float32),
            X_canon=None,
            C=None,
            img_shape=(1, 1),
            frame_id=kf_idx,
        )
        extrinsics = frame_to_extrinsics(frame)
        pinhole = PinholeParameters(
            name=cached_pinhole.name,
            extrinsics=extrinsics,
            intrinsics=cached_pinhole.intrinsics,
        )
        log_pinhole(
            camera=pinhole,
            cam_log_path=Path(kf_cam_log_path),
            image_plane_distance=self._image_plane_distance,
        )

    def _log_edges(
        self,
        positions_i: Float32[ndarray, "n 3"],
        positions_j: Float32[ndarray, "n 3"],
    ) -> None:
        """Log factor graph edges as line strips."""
        line_strips: list[list[float]] = []
        for t_i, t_j in zip(positions_i, positions_j, strict=False):
            line_strips.append(t_i.tolist())
            line_strips.append(t_j.tolist())
        rr.log(
            f"{self._parent_log_path}/edges",
            rr.LineStrips3D(strips=line_strips, colors=(0, 255, 0), labels="Factor Graph"),
        )
