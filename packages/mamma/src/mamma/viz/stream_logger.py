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
        rerun_codec_by_choice: dict[VideoCodecChoice, rr.VideoCodec] = {
            VideoCodecChoice.H264: rr.VideoCodec.H264,
            VideoCodecChoice.H265: rr.VideoCodec.H265,
            VideoCodecChoice.AV1: rr.VideoCodec.AV1,
        }
        self._rerun_codec: rr.VideoCodec = rerun_codec_by_choice[video_codec]

    def setup(self) -> None:
        """Send blueprint and static scene structure (run once, before the loop)."""
        rr.send_blueprint(default_blueprint(self.sequence.camera_names))
        rr.log(WORLD_TAG, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            WORLD_TAG,
            rr.AnnotationContext(
                [
                    # Explicit transparent background: without an id=0 entry the
                    # viewer auto-colors class 0, tinting the whole frame.
                    rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)),
                ]
                + [
                    rr.AnnotationInfo(id=obj_id + 1, label=f"person_{obj_id}", color=_ID_PALETTE[obj_id % len(_ID_PALETTE)])
                    for obj_id in range(len(_ID_PALETTE))
                ]
            ),
            static=True,
        )
        height: int
        width: int
        height, width = self.resize_hw
        for cam in self.sequence.cameras:
            pinhole: PinholeParameters = cam.scaled_to(height=height, width=width).to_pinhole_parameters()
            log_pinhole(pinhole, cam_log_path=Path(camera_entity(cam.name)), image_plane_distance=0.4, static=True)
            rr.log(f"{pinhole_entity(cam.name)}/video", rr.VideoStream(codec=self._rerun_codec), static=True)

    def log_tick_video(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> None:
        """Encode and log one synchronized frame per camera at this tick."""
        for cam_name, frame_chw in zip(self.sequence.camera_names, frames, strict=True):
            rgb_hwc: UInt8[ndarray, "h w 3"] = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
            packets: list[tuple[int, bytes]] = self._encoders[cam_name].encode_frame(rgb_hwc)
            self._log_packets(cam_name, packets)

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
        fits: "dict[int, FitResult]",
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
            normals: ndarray = _vertex_normals(fit.vertices, faces) if faces is not None else None
            rr.log(
                f"{WORLD_TAG}/meshes/person_{obj_id}",
                rr.Mesh3D(
                    vertex_positions=fit.vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    albedo_factor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0],
                ),
            )
            rr.log(f"{WORLD_TAG}/joints/person_{obj_id}", rr.Points3D(positions=fit.joints, radii=0.012, colors=color))

    def flush(self) -> None:
        """Drain buffered encoder packets (run once, after the loop)."""
        for cam_name, encoder in self._encoders.items():
            self._log_packets(cam_name, encoder.flush())

    def _log_packets(self, cam_name: str, packets: list[tuple[int, bytes]]) -> None:
        entity: str = f"{pinhole_entity(cam_name)}/video"
        for pts, data in packets:
            rr.set_time(TIMELINE, duration=pts / self.fps)
            rr.log(entity, rr.VideoStream.from_fields(sample=data))

    @property
    def encoder_stats(self) -> dict[str, dict[str, object]]:
        """Per-camera encoder performance metrics."""
        return {name: enc.stats for name, enc in self._encoders.items()}


def set_tick_time(frame_idx: int, fps: float) -> None:
    """Position the shared timeline at ``frame_idx`` for non-video entities."""
    rr.set_time(TIMELINE, duration=frame_idx / fps)
