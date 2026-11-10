"""Video 3D hands demo: CUDA decode -> hand detector -> top-down 3D hand model -> Rerun.

Frames stay on the GPU from TorchCodec's decoder through detection and the
hand model; boxes, joints and meshes cross to the CPU for logging. The layout is
dataforge's: the camera node ``world/camera`` holds the camera transform, its
``pinhole`` child the Pinhole with the video below it and the detector boxes under
``pinhole/boxes``. Every hand is logged in its side slot (``left_0``, ``right_0``, ...):
its 3D joints and mesh once, in the camera frame (``world/camera/hands``, a sibling of
the pinhole), and the video view, whose origin is the pinhole, projects them onto the
video. The 3D view looks over the wearer's head at the camera frustum and the hands.

The camera is a pinhole: ``fx``/``fy``/``cx``/``cy`` when given, else an assumed horizontal field
of view (``hfov_deg``) with the principal point at the image centre. It does not have to be
exact; it fixes where the metric 3D hands sit in front of the camera. Every stack's hands end up
in it: WiLoR predicts in its own virtual camera, so its hands keep their wrist-relative shape and
get the translation that reprojects their joints in this pinhole (``WilorHands.in_pinhole``).

Any container FFmpeg reads is accepted. The models decode the file as it is; the viewer's video reader
needs MP4, so ``as_mp4`` remuxes other containers without re-encoding. Both readers apply the display
rotation (an iPhone portrait .MOV), so the frames, their size, the pinhole and the hands are all upright.
"""

import math
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, Int64, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_custom_types import Points3DWithConfidence
from simplecv.rerun_eye_controls import follow_eye_controls
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm.auto import tqdm

from posekit.cameras import CalibratedCamera
from posekit.models import AnnotatedHandDetectorConfig, AnnotatedHandPose3dConfig, HandPose3dPipeline, WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.predictions import HandDetections, HandPoses3d, HandSide
from posekit.skeletons import KeypointSkeleton

EXAMPLES: Path = Path(__file__).resolve().parents[3] / "assets" / "examples"
CAMERA_PATH: str = "world/camera"
"""Camera node (dataforge's ``cam_path``): the camera transform; the 3D hands live below it, beside the pinhole."""
PINHOLE_PATH: str = f"{CAMERA_PATH}/pinhole"
"""Pinhole of the source camera (dataforge's ``pinhole_path``): the video and the detector boxes live below it."""
VIDEO_PATH: str = f"{PINHOLE_PATH}/video"
"""Video stream of the source camera (dataforge's ``video_path``)."""
SIDE_CLASS: dict[HandSide, int] = {"left": 0, "right": 1}
"""Annotation class of each side (box and label colour)."""
SIDE_COLOR: dict[HandSide, tuple[int, int, int]] = {"left": (66, 135, 245), "right": (245, 135, 66)}
"""Display colour of each side."""
MESH_ALPHA: int = 60
"""Mesh opacity (0-255): low enough that the hand in the video and the green joints show through."""
JOINT_CLASS: int = 2
"""Annotation class of the joints of either side: it carries the skeleton links in ``LINK_COLOR``."""
LINK_COLOR: tuple[int, int, int] = (255, 255, 255)
"""Skeleton link colour: white stands out on both side colours and on skin (red would read as low confidence)."""
MP4_CODECS: frozenset[str] = frozenset({"h264", "hevc", "av1", "vp9"})
"""Video codecs (FFmpeg names) that go into MP4 as they are and that the Rerun viewer decodes."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoHandsSettings:
    """What the 3D hands demo runs: the video, its pinhole camera and one hand stack. Shared by the CLI and the app."""

    video_path: Path = EXAMPLES / "gsi_sewing_left_front.mp4"
    """Input video path."""
    fx: float | None = None
    """Focal length along x in pixels; None assumes ``hfov_deg``."""
    fy: float | None = None
    """Focal length along y in pixels; None assumes ``hfov_deg``."""
    cx: float | None = None
    """Principal point x in pixels; None takes the image centre."""
    cy: float | None = None
    """Principal point y in pixels; None takes the image centre."""
    hfov_deg: float = 65.0
    """Horizontal field of view assumed for the intrinsics left unset."""
    detector: AnnotatedHandDetectorConfig = field(default_factory=WilorHandDetectorConfig)
    """Hand detector selected by stack name (``wilor``, ``egohand``, ``egohand-fast`` or ``egohand-fast-bf16``)."""
    pose: AnnotatedHandPose3dConfig = field(default_factory=WilorHandPoseConfig)
    """Top-down 3D hand model selected by the same stack name as ``detector``."""
    frame_batch_size: int = 16
    """Frames decoded and processed per step."""
    max_frames: int | None = None
    """Maximum number of frames to process; ``None`` runs the full video."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoHandsConfig(VideoHandsSettings):
    """Configuration for the GPU video 3D hands demo on the command line."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""


def log_hand_annotation_context(skeleton: KeypointSkeleton) -> None:
    """One annotation class per side (boxes, labels) and one for the joints with the model's skeleton links in ``LINK_COLOR``."""
    rr.log(
        "world",
        rr.AnnotationContext(
            [
                *(rr.ClassDescription(info=rr.AnnotationInfo(id=class_id, label=f"{side} hand", color=SIDE_COLOR[side])) for side, class_id in SIDE_CLASS.items()),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=JOINT_CLASS, label="hand joints", color=LINK_COLOR),
                    keypoint_annotations=[rr.AnnotationInfo(id=index, label=name) for index, name in enumerate(skeleton.keypoint_names)],
                    keypoint_connections=list(skeleton.links),
                ),
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
    live_slots: set[str],
) -> None:
    """Log one processed chunk onto the ``frame`` and ``video_time`` timelines.

    Each hand's joints and mesh go under ``world/camera/hands/<slot>`` in the camera frame
    (the video view projects them through the pinhole); its detector box goes under ``world/camera/pinhole/boxes/<slot>``.
    Joints are simplecv's ``Points3DWithConfidence`` at confidence 1.0 (no model here gives one per joint), so they show green;
    their class draws the skeleton links in ``LINK_COLOR``.

    Args:
        frame_indices: Global frame index of each chunk frame.
        timestamps_ns: Video timestamp of each chunk frame.
        detections: Hand boxes of the chunk (batch-relative frame indices).
        hands: Reconstructed hands of the chunk (batch-relative frame indices).
        live_slots: Slots currently holding data, updated in place; a slot is
            cleared only on its populated -> empty transition.
    """
    box_frames: Int64[ndarray, "n"] = detections.boxes.frame_indices.cpu().numpy()
    boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
    hand_frames: Int64[ndarray, "m"] = hands.keypoints.frame_indices.cpu().numpy()
    xyz: Float32[ndarray, "m k 3"] | None = None if hands.keypoints.xyz_camera is None else hands.keypoints.xyz_camera.cpu().numpy()
    vertices: Float32[ndarray, "m v 3"] | None = None if hands.vertices is None else hands.vertices.cpu().numpy()
    faces: dict[HandSide, ndarray] = {} if hands.faces is None else {side: value.cpu().numpy() for side, value in hands.faces.items()}
    keypoint_ids: UInt16[ndarray, "k"] = np.arange(hands.keypoints.skeleton.num_keypoints, dtype=np.uint16)
    # The hand models give no per-joint confidence (WiLoR's score is its box's), so every joint is logged at 1.0: green.
    full_confidence: Float32[ndarray, "k"] = np.ones(len(keypoint_ids), dtype=np.float32)
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
                f"{PINHOLE_PATH}/boxes/{slot}",
                rr.Boxes2D(array=boxes[row][None], array_format=rr.Box2DFormat.XYXY, class_ids=SIDE_CLASS[side], labels=f"{side} hand"),
            )
        for row, slot in zip(hand_rows, _slots([hands.sides[r] for r in hand_rows]), strict=True):
            side = hands.sides[row]
            present.add(slot)
            if xyz is not None:
                rr.log(
                    f"{CAMERA_PATH}/hands/{slot}/joints",
                    Points3DWithConfidence(xyz[row], full_confidence, class_ids=JOINT_CLASS, keypoint_ids=keypoint_ids, radii=0.004),
                )
            if vertices is not None and side in faces:
                rr.log(
                    f"{CAMERA_PATH}/hands/{slot}/mesh",
                    rr.Mesh3D(vertex_positions=vertices[row], triangle_indices=faces[side], albedo_factor=(*SIDE_COLOR[side], MESH_ALPHA)),
                )
        for slot in sorted(live_slots - present):
            for path in (f"{PINHOLE_PATH}/boxes/{slot}", f"{CAMERA_PATH}/hands/{slot}/joints", f"{CAMERA_PATH}/hands/{slot}/mesh"):
                rr.log(path, rr.Clear(recursive=False))
        live_slots.clear()
        live_slots.update(present)


def send_blueprint() -> None:
    """The video with the hands projected onto it, beside a 3D view of the camera frustum and the same hands.

    The video view's origin is the pinhole, so the 3D joints and meshes under ``world/camera/hands``
    project onto the video, as dataforge's camera panes show its hand meshes.

    The 3D eye is dataforge's headset chase view (``blueprints.headset_eye_controls``): behind and
    above the head, looking down past the camera at the hands.

    The time panel plays ``video_time`` from the start: a streamed viewer would otherwise follow the
    newest data and sit at the end. The whole video is logged before the hands, so playback starts at
    once and the hands appear as their chunks arrive.
    """
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=PINHOLE_PATH, name="video + projected hands", contents=[f"+ {PINHOLE_PATH}/**", f"+ {CAMERA_PATH}/hands/**"]),
                rrb.Spatial3DView(
                    origin="world",
                    name="3D hands",
                    contents=["+ world/**", f"- {PINHOLE_PATH}/boxes/**"],
                    eye_controls=follow_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), back_m=0.4, up_m=0.45, ahead_m=0.5, aim_down_m=0.4, side_m=0.15),
                ),
            ),
            rrb.TimePanel(timeline="video_time", play_state="playing"),
        )
    )


def assumed_pinhole(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """``fx, fy, cx, cy`` of a square-pixel pinhole with horizontal field of view ``hfov_deg``, centred on the image."""
    focal: float = width / 2.0 / math.tan(math.radians(hfov_deg) / 2.0)
    return focal, focal, width / 2.0, height / 2.0


def video_size(path: Path) -> tuple[int, int]:
    """``width, height`` of a video, from its metadata (no decoding)."""
    from torchcodec.decoders import VideoDecoder

    metadata = VideoDecoder(path, seek_mode="approximate").metadata
    if metadata.width is None or metadata.height is None:
        raise ValueError(f"Could not read the frame size of {path}.")
    return int(metadata.width), int(metadata.height)


def as_mp4(video: Path, out_dir: Path) -> Path:
    """The video in an MP4 (ISO BMFF) file, the container the Rerun video reader takes.

    MP4 and QuickTime files (.mp4, .mov, .m4v) come back as they are. Other containers (.mkv, .webm, ...)
    are remuxed into ``out_dir`` with stream copy, which keeps the display rotation; only a codec outside
    ``MP4_CODECS`` is re-encoded, to H.264, and that is printed.
    """
    import av

    with av.open(str(video)) as source:
        if "mp4" in source.format.name.split(","):
            return video
        codec: str = source.streams.video[0].codec_context.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target: Path = out_dir / f"{video.stem}.mp4"
    video_args: list[str] = ["-c:v", "copy"]
    if codec not in MP4_CODECS:
        print(f"[posekit] re-encoding {video.name} to H.264: its {codec} video cannot go into MP4 for the viewer")
        video_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-map", "0:v:0", *video_args, str(target)], check=True
    )
    return target


def pinhole_camera(settings: VideoHandsSettings, width: int, height: int) -> CalibratedCamera:
    """The settings' pinhole; each unset intrinsic comes from ``assumed_pinhole``."""
    given: tuple[float | None, ...] = (settings.fx, settings.fy, settings.cx, settings.cy)
    fx, fy, cx, cy = (value if value is not None else assumed for value, assumed in zip(given, assumed_pinhole(width, height, settings.hfov_deg), strict=True))
    return CalibratedCamera(np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), np.zeros(0), width, height, lens="PINHOLE")


def describe_camera(settings: VideoHandsSettings, camera: CalibratedCamera) -> str:
    """``assumed pinhole, f = ...`` when the camera is the assumed one, else its four intrinsics.

    Within half a pixel counts as assumed: the app fills its fields with the assumed values rounded to 0.1 px
    (``hands_ui.fill_intrinsics``), so an untouched app run reports the assumed camera like the CLI does.
    """
    fx, fy, cx, cy = camera.intrinsics[0, 0], camera.intrinsics[1, 1], camera.intrinsics[0, 2], camera.intrinsics[1, 2]
    if np.allclose((fx, fy, cx, cy), assumed_pinhole(camera.width, camera.height, settings.hfov_deg), atol=0.5):
        return f"assumed pinhole, f = {fx:.0f} px ({settings.hfov_deg:g}° horizontal FOV)"
    return f"pinhole fx = {fx:.0f}, fy = {fy:.0f}, cx = {cx:.0f}, cy = {cy:.0f} px"


def build_pipeline(settings: VideoHandsSettings) -> HandPose3dPipeline:
    """Set up the detector and the 3D hand model the settings name (exports and engine builds happen here)."""
    return HandPose3dPipeline(settings.detector.setup(), settings.pose.setup())


def run_video_hands(settings: VideoHandsSettings, pipeline: HandPose3dPipeline | None = None) -> Iterator[str]:
    """Decode the video on the GPU, run the hand stack chunk by chunk and log everything to the active recording.

    The CLI drains this generator into its global recording; the app streams a thread-local
    recording to its viewer after every yield.

    Args:
        settings: Video, pinhole intrinsics and hand stack.
        pipeline: A pipeline already set up for ``settings``; None builds one.

    Yields:
        A progress line after every chunk, then the summary; both name the camera used.

    Raises:
        ValueError: If the video metadata cannot be read.
    """
    from torchcodec.decoders import VideoDecoder

    if pipeline is None:
        pipeline = build_pipeline(settings)
    decoder = VideoDecoder(settings.video_path, dimension_order="NHWC", device="cuda")
    num_frames_raw = decoder.metadata.num_frames
    fps_raw = decoder.metadata.average_fps
    if num_frames_raw is None or fps_raw is None or float(fps_raw) <= 0.0:
        raise ValueError(f"Could not read valid video metadata from {settings.video_path}.")
    num_frames: int = int(num_frames_raw) if settings.max_frames is None else min(int(num_frames_raw), settings.max_frames)
    assert decoder.metadata.width is not None and decoder.metadata.height is not None
    width: int = int(decoder.metadata.width)
    height: int = int(decoder.metadata.height)
    camera: CalibratedCamera = pinhole_camera(settings, width, height)
    camera_note: str = describe_camera(settings, camera)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(CAMERA_PATH, rr.Transform3D(), static=True)
    rr.log(
        PINHOLE_PATH,
        rr.Pinhole(image_from_camera=camera.intrinsics, resolution=[width, height], image_plane_distance=0.05),
        static=True,
    )
    with tempfile.TemporaryDirectory(prefix="posekit-video-") as scratch:
        video_timestamps_ns: Int64[ndarray, "frames"] = log_video(
            as_mp4(settings.video_path, Path(scratch)), Path(VIDEO_PATH), timeline="video_time"
        )
    send_blueprint()
    log_hand_annotation_context(pipeline.pose.skeleton)
    total_hands: int = 0
    live_slots: set[str] = set()
    for start in tqdm(range(0, num_frames, settings.frame_batch_size), desc="posekit video hands"):
        stop: int = min(start + settings.frame_batch_size, num_frames)
        frames_rgb: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(start, stop).data.contiguous()
        with torch.inference_mode():
            detections, hands = pipeline(frames_rgb, camera)
            hands = hands.in_pinhole(camera.intrinsics)
        total_hands += hands.num_hands
        chunk: Int64[ndarray, "chunk"] = np.arange(start, stop, dtype=np.int64)
        timestamps: Int64[ndarray, "chunk"] = (
            video_timestamps_ns[chunk] if len(video_timestamps_ns) >= stop else (chunk / float(fps_raw) * 1e9).astype(np.int64)
        )
        log_chunk_hands(
            frame_indices=chunk.tolist(), timestamps_ns=timestamps.tolist(), detections=detections, hands=hands,
            live_slots=live_slots,
        )
        yield f"Frames {stop}/{num_frames}, {total_hands} hands; {camera_note}"
    torch.cuda.synchronize()
    yield f"processed {num_frames} frames, {total_hands} hands; {camera_note}"


def main(config: VideoHandsConfig) -> None:
    """Run the GPU video 3D hands demo and log results to the recording ``config.rr_config`` set up.

    Args:
        config: Demo configuration.
    """
    statuses: list[str] = list(run_video_hands(config))
    print(f"[posekit] {statuses[-1]}")
