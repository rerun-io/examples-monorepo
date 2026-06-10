"""Streaming Rerun logger: per-camera H.264 VideoStreams + shared 3D scene.

The hot-loop contract is one ``log_tick_video`` call per synchronized frame
set. Frames arrive as GPU-resident RGB CHW uint8 tensors (already resized);
each camera's frames are NVENC-encoded to H.264 packets which Rerun's viewer
decodes — the RRD stays small (video packets, not images) and the logging
path never re-decodes source video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
import torch
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import log_pinhole
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder

from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.types import CameraTracks
from mamma.fitting.window_fitter import FitResult
from mamma.landmarks.estimator import CameraLandmarks
from mamma.viz.blueprint import WORLD_TAG, camera_entity, default_blueprint, pinhole_entity


def _vertex_normals(vertices: ndarray, faces: ndarray) -> ndarray:
    """Area-weighted vertex normals (port of the original ``_vertex_normals``)."""
    v0: ndarray = vertices[faces[:, 0]]
    v1: ndarray = vertices[faces[:, 1]]
    v2: ndarray = vertices[faces[:, 2]]
    face_normals: ndarray = np.cross(v1 - v0, v2 - v0)
    normals: ndarray = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths: ndarray = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(lengths, 1e-8, None)

_ID_PALETTE: list[tuple[int, int, int]] = [
    (230, 110, 80),
    (90, 160, 230),
    (120, 200, 120),
    (220, 180, 80),
    (180, 120, 220),
    (90, 200, 200),
    (230, 140, 180),
    (160, 160, 160),
]
"""Stable per-person colors; person ``obj_id`` indexes modulo this palette."""

# SMPL-X kinematic parents (kintree_table row 0 of SMPLX_NEUTRAL.npz; 55 tree
# joints: 0-21 body, 22 jaw, 23-24 eyes, 25-39 left hand, 40-54 right hand).
# Joints 55+ in the 127-joint output are regressed landmarks with no bones.
_SMPLX_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]
_SMPLX_BONES: list[tuple[int, int]] = [(p, i) for i, p in enumerate(_SMPLX_PARENTS) if p >= 0]
def _smplx_joint_name(j: int) -> str:
    """Human name for tree joint ``j`` (generic for the hand chain tail)."""
    return _SMPLX_JOINT_NAMES[j] if j < len(_SMPLX_JOINT_NAMES) else f"j{j}"


_SMPLX_JOINT_NAMES: list[str] = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye", "right_eye",
]


TIMELINE: str = "time"
"""Shared timeline name; ticks are elapsed seconds (``frame_idx / fps``)."""


class StreamLogger:
    """Owns all Rerun logging for one streaming run."""

    def __init__(
        self,
        sequence: MultiViewSequence,
        resize_hw: tuple[int, int],
        video_codec: VideoCodecChoice = VideoCodecChoice.H264,
    ) -> None:
        """Args:
        sequence: The capture being processed (native-resolution calibration).
        resize_hw: ``(height, width)`` of the frames the engine works on;
            pinholes are logged rescaled to this grid so video, 2D overlays,
            and reprojections all share one pixel space.
        video_codec: Codec for the per-camera VideoStreams.
        """
        self.sequence: MultiViewSequence = sequence
        self.resize_hw: tuple[int, int] = resize_hw
        self.fps: float = sequence.fps
        self._video_codec: VideoCodecChoice = video_codec
        self._encoders: dict[str, VideoEncoder] = {
            name: VideoEncoder(codec=video_codec, fps=sequence.fps) for name in sequence.camera_names
        }
        self._pts_offset: dict[str, int] = dict.fromkeys(sequence.camera_names, 0)
        # Video encode+log runs on a worker thread: the D2H copy + swscale +
        # NVENC submission cost ~13 ms/tick of CPU that otherwise serializes
        # with model enqueue. Rerun's clock is per-thread, so the worker sets
        # its own timeline position. Bounded queue gives backpressure.
        from concurrent.futures import ThreadPoolExecutor

        self._video_worker: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-log")
        self._video_pending: list = []
        rerun_codec_by_choice: dict[VideoCodecChoice, rr.VideoCodec] = {
            VideoCodecChoice.H264: rr.VideoCodec.H264,
            VideoCodecChoice.H265: rr.VideoCodec.H265,
            VideoCodecChoice.AV1: rr.VideoCodec.AV1,
        }
        self._rerun_codec: rr.VideoCodec = rerun_codec_by_choice[video_codec]
        self._skeleton_frames_logged: set[int] = set()

    def setup(self) -> None:
        """Send blueprint and static scene structure (run once, before the loop)."""
        rr.send_blueprint(default_blueprint(self.sequence.camera_names))
        rr.log(WORLD_TAG, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # Calibrated ground plane (world z=0) — the same plane the fitter's
        # floor-penetration loss references; matches the original scene.rrd.
        half: float = 6.0
        rr.log(
            f"{WORLD_TAG}/ground",
            rr.Mesh3D(
                vertex_positions=[[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
                triangle_indices=[[0, 1, 2], [0, 2, 3]],
                vertex_colors=[[70, 80, 90, 110]] * 4,
            ),
            static=True,
        )
        keypoint_names: list[rr.AnnotationInfo] = [
            rr.AnnotationInfo(id=j, label=_SMPLX_JOINT_NAMES[j] if j < len(_SMPLX_JOINT_NAMES) else f"j{j}")
            for j in range(127)
        ]
        rr.log(
            WORLD_TAG,
            rr.AnnotationContext(
                [
                    # Explicit transparent background: without an id=0 entry the
                    # viewer auto-colors class 0, tinting the whole frame.
                    rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0))),
                ]
                + [
                    # keypoint_connections turn the bare 127-joint Points3D into a
                    # labeled wireframe skeleton (bones exist for tree joints 0-54).
                    rr.ClassDescription(
                        info=rr.AnnotationInfo(
                            id=obj_id + 1, label=f"person_{obj_id}", color=_ID_PALETTE[obj_id % len(_ID_PALETTE)]
                        ),
                        keypoint_annotations=keypoint_names,
                        keypoint_connections=_SMPLX_BONES,
                    )
                    for obj_id in range(len(_ID_PALETTE))
                ]
            ),
            static=True,
        )
        # Time-series styling for the metrics panel (names once, static).
        for stage_name in ("track", "landmarks", "fit_wait", "log_video", "log_tracks", "log_landmarks"):
            rr.log(f"metrics/timing/{stage_name}", rr.SeriesLines(names=stage_name, widths=1.5), static=True)
        rr.log("metrics/fit/valid_anchors", rr.SeriesLines(names="valid anchors (of 512)", widths=1.5), static=True)
        rr.log("metrics/fit/floor_contacts", rr.SeriesLines(names="floor-contact landmarks", widths=1.5), static=True)
        height: int
        width: int
        height, width = self.resize_hw
        for cam in self.sequence.cameras:
            pinhole: PinholeParameters = cam.scaled_to(height=height, width=width).to_pinhole_parameters()
            log_pinhole(pinhole, cam_log_path=Path(camera_entity(cam.name)), image_plane_distance=0.4, static=True)
            rr.log(f"{pinhole_entity(cam.name)}/video", rr.VideoStream(codec=self._rerun_codec), static=True)

    def _rgb_to_yuv420_gpu(
        self, rgb_chw: UInt8[torch.Tensor, "3 h w"]
    ) -> tuple[UInt8[ndarray, "h w"], UInt8[ndarray, "h2 w2"], UInt8[ndarray, "h2 w2"]]:
        """BT.601 limited-range RGB->YUV420 on GPU (replaces CPU swscale).

        The CPU rgb24->yuv420p reformat was the dominant logging cost
        (~12 ms/tick of GIL-held work across 4 cameras); on GPU it is ~0.2 ms
        and the D2H copy shrinks from 2.7 MB to 1.4 MB per frame.
        """
        import torch.nn.functional as F

        rgb: torch.Tensor = rgb_chw.float()
        r, g, b = rgb[0], rgb[1], rgb[2]
        y = (16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b).round_().clamp_(16, 235).to(torch.uint8)
        u_full = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
        v_full = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
        uv = torch.stack([u_full, v_full]).unsqueeze(0)
        uv_small = F.avg_pool2d(uv, kernel_size=2)[0].round_().clamp_(16, 240).to(torch.uint8)
        return (
            y.contiguous().cpu().numpy(),
            uv_small[0].contiguous().cpu().numpy(),
            uv_small[1].contiguous().cpu().numpy(),
        )

    def log_tick_video(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> None:
        """Encode and log one synchronized frame per camera (async worker)."""

        def encode_and_log(frames_gpu: list[UInt8[torch.Tensor, "3 h w"]]) -> None:
            for cam_name, frame_chw in zip(self.sequence.camera_names, frames_gpu, strict=True):
                y_plane, u_plane, v_plane = self._rgb_to_yuv420_gpu(frame_chw)
                packets: list[tuple[int, bytes]] = self._encoders[cam_name].encode_yuv_planes(y_plane, u_plane, v_plane)
                self._log_packets(cam_name, packets)

        # Backpressure: never queue more than 4 ticks of frames.
        if len(self._video_pending) >= 4:
            self._video_pending.pop(0).result()
        self._video_pending.append(self._video_worker.submit(encode_and_log, list(frames)))

    def log_tick_tracks(self, frame_idx: int, tracks: list[CameraTracks], seg_stride: int = 5) -> None:
        """Log per-camera person boxes every tick + segmentation ids at a stride.

        Full-resolution segmentation images every tick would dominate the RRD
        (~1 MB/cam/tick raw), so masks are logged only every ``seg_stride``
        ticks (full resolution — they must share the pinhole pixel space);
        boxes are logged every tick.
        """
        set_tick_time(frame_idx, self.fps)
        for cam_name, cam_tracks in zip(self.sequence.camera_names, tracks, strict=True):
            boxes: list[ndarray] = []
            class_ids: list[int] = []
            for obj_id, track in sorted(cam_tracks.items()):
                if track.bbox_xyxy is not None:
                    boxes.append(track.bbox_xyxy)
                    class_ids.append(obj_id + 1)
            entity: str = pinhole_entity(cam_name)
            if boxes:
                boxes_arr: ndarray = np.stack(boxes, axis=0)
                rr.log(
                    f"{entity}/persons",
                    rr.Boxes2D(
                        array=boxes_arr,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=class_ids,
                    ),
                )
            if frame_idx % seg_stride == 0 and cam_tracks:
                seg: torch.Tensor = torch.zeros(
                    self.resize_hw,
                    dtype=torch.uint8,
                    device=next(iter(cam_tracks.values())).mask.device,
                )
                for obj_id, track in sorted(cam_tracks.items()):
                    seg[track.mask] = obj_id + 1
                rr.log(f"{entity}/mask", rr.SegmentationImage(seg.cpu().numpy()))

    def log_tick_landmarks(self, frame_idx: int, landmarks: list[CameraLandmarks]) -> None:
        """Log dense 2D landmarks per camera/person, colored by visibility."""
        set_tick_time(frame_idx, self.fps)
        for cam_name, cam_landmarks in zip(self.sequence.camera_names, landmarks, strict=True):
            entity_base: str = pinhole_entity(cam_name)
            for obj_id, result in sorted(cam_landmarks.items()):
                positions: ndarray = result.joints2d[:, :2].cpu().numpy()
                vis: ndarray = result.visibility.cpu().numpy()
                colors: ndarray = np.zeros((positions.shape[0], 4), dtype=np.uint8)
                colors[:, 1] = (vis * 255).astype(np.uint8)
                colors[:, 0] = ((1.0 - vis) * 255).astype(np.uint8)
                colors[:, 3] = 255
                rr.log(
                    f"{entity_base}/landmarks/person_{obj_id}",
                    rr.Points2D(positions=positions, colors=colors, radii=1.5),
                )

    def log_tick_fit(
        self,
        frame_idx: int,
        fits: dict[int, FitResult],
        triangulated: dict[int, tuple[torch.Tensor, torch.Tensor]],
        faces: ndarray | None,
    ) -> None:
        """Log SMPL-X meshes + triangulated landmark clouds for one tick."""
        set_tick_time(frame_idx, self.fps)
        for obj_id, (points3d, valid) in triangulated.items():
            cloud: ndarray = points3d[valid].cpu().numpy()
            rr.log(f"{WORLD_TAG}/triangulated/person_{obj_id}", rr.Points3D(positions=cloud, radii=0.008))
        for obj_id, fit in fits.items():
            color: tuple[int, int, int] = _ID_PALETTE[obj_id % len(_ID_PALETTE)]
            normals: ndarray | None = _vertex_normals(fit.vertices, faces) if faces is not None else None
            rr.log(
                f"{WORLD_TAG}/meshes/person_{obj_id}",
                rr.Mesh3D(
                    vertex_positions=fit.vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    albedo_factor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0],
                ),
            )
            rr.log(
                f"{WORLD_TAG}/joints/person_{obj_id}",
                rr.Points3D(
                    positions=fit.joints,
                    radii=0.012,
                    class_ids=obj_id + 1,
                    keypoint_ids=list(range(fit.joints.shape[0])),
                ),
            )
            # SMPL-X parameters logged as what they ARE: a kinematic tree of
            # relative transforms, expressed as 0.33 NAMED TRANSFORM FRAMES
            # (tf2-style): flat entity paths, each joint bound to a named frame
            # via CoordinateFrame, and the frame graph carrying the per-joint
            # axis-angle rotation relative to its parent frame with the
            # constant (betas-determined) bone offset. Root frame parents to
            # the world's implicit frame.
            root: str = f"{WORLD_TAG}/skeleton/person_{obj_id}"
            frames_: list[str] = [f"person{obj_id}/{_smplx_joint_name(j)}" for j in range(len(_SMPLX_PARENTS))]
            if obj_id not in self._skeleton_frames_logged:
                for j in range(len(_SMPLX_PARENTS)):
                    rr.log(f"{root}/{_smplx_joint_name(j)}", rr.CoordinateFrame(frames_[j]), static=True)
                self._skeleton_frames_logged.add(obj_id)
            pose_aa: ndarray = fit.pose.reshape(55, 3)
            rest: ndarray = fit.rest_joints
            for j, parent in enumerate(_SMPLX_PARENTS):
                offset: ndarray = rest[j] + fit.trans if parent < 0 else rest[j] - rest[parent]
                angle: float = float(np.linalg.norm(pose_aa[j]))
                axis: ndarray = pose_aa[j] / angle if angle > 1e-8 else np.array([1.0, 0.0, 0.0])
                rr.log(
                    f"{root}/{_smplx_joint_name(j)}",
                    rr.Transform3D(
                        translation=offset,
                        rotation=rr.RotationAxisAngle(axis=axis, angle=angle),
                        child_frame=frames_[j],
                        parent_frame=f"tf#{WORLD_TAG}" if parent < 0 else frames_[parent],
                    ),
                )
            rr.log(f"metrics/params/person_{obj_id}/trans", rr.Scalars(fit.trans))
            rr.log(f"metrics/params/person_{obj_id}/betas", rr.BarChart(fit.betas))

    def log_tick_metrics(self, frame_idx: int, timings_ms: dict[str, float], fit_metrics: dict[str, float]) -> None:
        """Per-tick scalars: stage timings + fit health (valid anchors, contacts)."""
        set_tick_time(frame_idx, self.fps)
        for stage_name, ms in timings_ms.items():
            rr.log(f"metrics/timing/{stage_name}", rr.Scalars(ms))
        for name, value in fit_metrics.items():
            rr.log(f"metrics/fit/{name}", rr.Scalars(value))

    def flush(self) -> None:
        """Drain buffered encoder packets and reset encoders for reuse.

        A flushed libav context is in EOF state and cannot encode again, so
        fresh encoders replace the drained ones; the per-camera PTS offset
        keeps the shared timeline monotonic across multiple ``run()`` calls.
        """
        for pending in self._video_pending:
            pending.result()
        self._video_pending.clear()
        for cam_name, encoder in self._encoders.items():
            self._log_packets(cam_name, encoder.flush())
            self._pts_offset[cam_name] += encoder.next_pts
            self._encoders[cam_name] = VideoEncoder(codec=self._video_codec, fps=self.fps)

    def _log_packets(self, cam_name: str, packets: list[tuple[int, bytes]]) -> None:
        entity: str = f"{pinhole_entity(cam_name)}/video"
        offset: int = self._pts_offset[cam_name]
        for pts, data in packets:
            rr.set_time(TIMELINE, duration=(offset + pts) / self.fps)
            rr.log(entity, rr.VideoStream.from_fields(sample=data))

    @property
    def encoder_stats(self) -> dict[str, dict[str, object]]:
        """Per-camera encoder performance metrics."""
        return {name: enc.stats for name, enc in self._encoders.items()}


def set_tick_time(frame_idx: int, fps: float) -> None:
    """Position the shared timeline at ``frame_idx`` for non-video entities."""
    rr.set_time(TIMELINE, duration=frame_idx / fps)
