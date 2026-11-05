"""Video 3D hands demo: CUDA decode -> hand detector -> top-down 3D hand model -> Rerun.

Frames stay on the GPU from TorchCodec's decoder through detection and the
hand model; boxes, joints and meshes cross to the CPU for logging. Every hand
is logged in its side slot (``left_0``, ``right_0``, ...): boxes and 2D joints
on the image, 3D joints and the mesh in the camera frame beside it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, Int64, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm.auto import tqdm

from posekit.cameras import CalibratedCamera, load_calibrated_camera
from posekit.models import AnnotatedHandDetectorConfig, AnnotatedHandPose3dConfig, HandPose3dPipeline, WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.models.wilor import CROP_SIZE, FOCAL_LENGTH, WilorHandPose3d
from posekit.predictions import HandDetections, HandPoses3d, HandSide
from posekit.rerun_logging import log_person_points2d
from posekit.skeletons import KeypointSkeleton

EXAMPLES: Path = Path(__file__).resolve().parents[3] / "assets" / "examples"
CAMERA_PATH: str = "world/camera"
"""Frame of the source camera: the 3D hands live below it, beside the image."""
IMAGE_PATH: str = f"{CAMERA_PATH}/image"
"""Pinhole and video of the source camera; boxes and 2D joints live below it."""
SIDE_CLASS: dict[HandSide, int] = {"left": 0, "right": 1}
"""Annotation class of each side (skeleton links and colour)."""
SIDE_COLOR: dict[HandSide, tuple[int, int, int]] = {"left": (66, 135, 245), "right": (245, 135, 66)}
"""Display colour of each side."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoHandsConfig:
    """Configuration for the GPU video 3D hands demo."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""
    video_path: Path = EXAMPLES / "selfcap_oak_left.mp4"
    """Input video path."""
    camera_path: Path | None = EXAMPLES / "selfcap_oak_left.json"
    """CalibratedCamera JSON of the video (EgoHand needs it); None for uncalibrated video with WiLoR."""
    detector: AnnotatedHandDetectorConfig = field(default_factory=WilorHandDetectorConfig)
    """Hand detector selected by subcommand (``wilor`` or ``egohand``)."""
    pose: AnnotatedHandPose3dConfig = field(default_factory=WilorHandPoseConfig)
    """Top-down 3D hand model selected by subcommand (``wilor`` or ``egohand``)."""
    frame_batch_size: int = 16
    """Frames decoded and processed per step."""
    max_frames: int | None = None
    """Maximum number of frames to process; ``None`` runs the full video."""
    keypoint_threshold: float = 0.3
    """Minimum keypoint confidence for 2D visibility."""


def log_hand_annotation_context(skeleton: KeypointSkeleton) -> None:
    """One annotation class per side, both with the model's skeleton links."""
    rr.log(
        "world",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=class_id, label=f"{side} hand", color=SIDE_COLOR[side]),
                    keypoint_annotations=[rr.AnnotationInfo(id=index, label=name) for index, name in enumerate(skeleton.keypoint_names)],
                    keypoint_connections=list(skeleton.links),
                )
                for side, class_id in SIDE_CLASS.items()
            ]
        ),
        static=True,
    )


def _slots(sides: Sequence[HandSide]) -> list[str]:
    """``left_0``, ``left_1``, ``right_0``...: the k-th hand of a side in one frame."""
    counts: dict[HandSide, int] = {"left": 0, "right": 0}
    slots: list[str] = []
    for side in sides:
        slots.append(f"{side}_{counts[side]}")
        counts[side] += 1
    return slots


def log_chunk_hands(
    *,
    frame_indices: Sequence[int],
    timestamps_ns: Sequence[int],
    detections: HandDetections,
    hands: HandPoses3d,
    keypoint_threshold: float,
    live_slots: set[str],
) -> None:
    """Log one processed chunk onto the ``frame`` and ``video_time`` timelines.

    Args:
        frame_indices: Global frame index of each chunk frame.
        timestamps_ns: Video timestamp of each chunk frame.
        detections: Hand boxes of the chunk (batch-relative frame indices).
        hands: Reconstructed hands of the chunk (batch-relative frame indices).
        keypoint_threshold: Minimum keypoint confidence for 2D visibility.
        live_slots: Slots currently holding data, updated in place; a slot is
            cleared only on its populated -> empty transition.
    """
    box_frames: Int64[ndarray, "n"] = detections.boxes.frame_indices.cpu().numpy()
    boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
    hand_frames: Int64[ndarray, "m"] = hands.keypoints.frame_indices.cpu().numpy()
    xy: Float32[ndarray, "m k 2"] = hands.keypoints.xy.detach().cpu().numpy()
    scores: Float32[ndarray, "m k"] = hands.keypoints.scores.detach().cpu().numpy()
    xyz: Float32[ndarray, "m k 3"] | None = None if hands.keypoints.xyz_camera is None else hands.keypoints.xyz_camera.cpu().numpy()
    vertices: Float32[ndarray, "m v 3"] | None = None if hands.vertices is None else hands.vertices.cpu().numpy()
    faces: dict[HandSide, ndarray] = {} if hands.faces is None else {side: value.cpu().numpy() for side, value in hands.faces.items()}
    keypoint_ids: UInt16[ndarray, "k"] = np.arange(xy.shape[1], dtype=np.uint16)
    for local, (frame_index, timestamp_ns) in enumerate(zip(frame_indices, timestamps_ns, strict=True)):
        rr.set_time("frame", sequence=int(frame_index))
        rr.set_time("video_time", duration=1e-9 * float(timestamp_ns))
        box_rows: Int64[ndarray, "a"] = np.flatnonzero(box_frames == local)
        hand_rows: Int64[ndarray, "b"] = np.flatnonzero(hand_frames == local)
        present: set[str] = set()
        for row, slot in zip(box_rows, _slots([detections.sides[r] for r in box_rows]), strict=True):
            side: HandSide = detections.sides[row]
            present.add(slot)
            rr.log(
                f"{IMAGE_PATH}/hands/{slot}/box",
                rr.Boxes2D(array=boxes[row][None], array_format=rr.Box2DFormat.XYXY, class_ids=SIDE_CLASS[side], labels=f"{side} hand"),
            )
        for row, slot in zip(hand_rows, _slots([hands.sides[r] for r in hand_rows]), strict=True):
            side = hands.sides[row]
            present.add(slot)
            log_person_points2d(
                f"{IMAGE_PATH}/hands/{slot}/keypoints", xy[row], scores[row], keypoint_threshold,
                keypoint_ids=keypoint_ids, class_ids=SIDE_CLASS[side],
            )
            if xyz is not None:
                rr.log(
                    f"{CAMERA_PATH}/hands/{slot}/joints",
                    rr.Points3D(xyz[row], keypoint_ids=keypoint_ids, class_ids=SIDE_CLASS[side], radii=0.004, show_labels=False),
                )
            if vertices is not None and side in faces:
                rr.log(
                    f"{CAMERA_PATH}/hands/{slot}/mesh",
                    rr.Mesh3D(vertex_positions=vertices[row], triangle_indices=faces[side], albedo_factor=(*SIDE_COLOR[side], 160)),
                )
        for slot in sorted(live_slots - present):
            for path in (f"{IMAGE_PATH}/hands/{slot}/box", f"{IMAGE_PATH}/hands/{slot}/keypoints", f"{CAMERA_PATH}/hands/{slot}/joints", f"{CAMERA_PATH}/hands/{slot}/mesh"):
                rr.log(path, rr.Clear(recursive=False))
        live_slots.clear()
        live_slots.update(present)


def send_blueprint(target: Float32[ndarray, "3"]) -> None:
    """Video with 2D hands beside a 3D view that looks at the hands from behind the camera.

    The eye sits ~0.75 m from the first hands, above and to the right of the camera ray:
    EgoHand's metric hands are at arm's length, WiLoR's virtual camera puts them metres away.
    """
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=IMAGE_PATH, name="video + 2D hands"),
            rrb.Spatial3DView(
                origin="world",
                name="3D hands",
                contents=["+ world/**", f"- {IMAGE_PATH}/hands/**"],
                eye_controls=rrb.EyeControls3D(
                    position=(target + np.asarray([0.25, -0.35, -0.6])).tolist(), look_target=target.tolist(), eye_up=(0.0, -1.0, 0.0)
                ),
            ),
        )
    )


def pinhole_matrix(pose: object, camera: CalibratedCamera | None, width: int, height: int) -> Float32[ndarray, "3 3"]:
    """The camera the 3D hands are expressed in: WiLoR's virtual pinhole, else the calibration's ``K``."""
    if isinstance(pose, WilorHandPose3d) or camera is None:
        focal: float = FOCAL_LENGTH / CROP_SIZE * max(width, height)
        return np.asarray([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    return camera.intrinsics.astype(np.float32)


def main(config: VideoHandsConfig) -> None:
    """Run the GPU video 3D hands demo and log results to Rerun.

    Args:
        config: Demo configuration.

    Raises:
        ValueError: If the video metadata cannot be read.
    """
    from torchcodec.decoders import VideoDecoder

    camera: CalibratedCamera | None = None if config.camera_path is None else load_calibrated_camera(config.camera_path)
    pipeline = HandPose3dPipeline(config.detector.setup(), config.pose.setup())
    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    num_frames_raw = decoder.metadata.num_frames
    fps_raw = decoder.metadata.average_fps
    if num_frames_raw is None or fps_raw is None or float(fps_raw) <= 0.0:
        raise ValueError(f"Could not read valid video metadata from {config.video_path}.")
    num_frames: int = int(num_frames_raw) if config.max_frames is None else min(int(num_frames_raw), config.max_frames)
    assert decoder.metadata.width is not None and decoder.metadata.height is not None
    width: int = int(decoder.metadata.width)
    height: int = int(decoder.metadata.height)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(IMAGE_PATH, rr.Pinhole(image_from_camera=pinhole_matrix(pipeline.pose, camera, width, height), resolution=[width, height], image_plane_distance=0.05), static=True)
    video_timestamps_ns: Int64[ndarray, "frames"] = log_video(config.video_path, Path(IMAGE_PATH), timeline="video_time")
    log_hand_annotation_context(pipeline.pose.skeleton)
    total_hands: int = 0
    live_slots: set[str] = set()
    for start in tqdm(range(0, num_frames, config.frame_batch_size), desc="posekit video hands"):
        stop: int = min(start + config.frame_batch_size, num_frames)
        frames_rgb: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(start, stop).data.contiguous()
        with torch.inference_mode():
            detections, hands = pipeline(frames_rgb, camera)
        if total_hands == 0 and hands.num_hands and hands.keypoints.xyz_camera is not None:
            send_blueprint(hands.keypoints.xyz_camera.reshape(-1, 3).median(dim=0).values.cpu().numpy())
        total_hands += hands.num_hands
        chunk: Int64[ndarray, "chunk"] = np.arange(start, stop, dtype=np.int64)
        timestamps: Int64[ndarray, "chunk"] = (
            video_timestamps_ns[chunk] if len(video_timestamps_ns) >= stop else (chunk / float(fps_raw) * 1e9).astype(np.int64)
        )
        log_chunk_hands(
            frame_indices=chunk.tolist(), timestamps_ns=timestamps.tolist(), detections=detections, hands=hands,
            keypoint_threshold=config.keypoint_threshold, live_slots=live_slots,
        )
    torch.cuda.synchronize()
    print(f"[posekit] processed {num_frames} frames, {total_hands} hands")
