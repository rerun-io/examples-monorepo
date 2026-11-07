"""Whole-body 3D people in images or a video: SAM3 person masks -> SAM 3D Body -> Rerun.

Moved from ``packages/sam3d-body`` (its ``sam3d-body-cli`` demo), now over posekit roles: the SAM3
``PromptableSegmenter`` finds every person with a mask, MoGe-2 estimates the camera unless a calibration is
given, and the ``TopDownBodyPose3d`` role reconstructs each person. Images also log MoGe-2 depth clouds.
A video is logged once as a VideoStream on the ``video_time`` timeline; each frame's people go on the same
timeline at that frame's timestamp, and its frames stay on the GPU.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm.auto import tqdm

from posekit.body_rerun import create_view, log_people, set_annotation_context
from posekit.cameras import CalibratedCamera, load_calibrated_camera
from posekit.models import AnnotatedBodyPose3dConfig, BodyPose3dConfig, SegmentationPrompts
from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig
from posekit.models.sam3d_body import Sam3dBodyConfig, Sam3dBodyPredictor, Sam3dBodyReconstruction
from posekit.predictions import BoxDetections

VIDEO_TIMELINE: str = "video_time"
"""Timeline of the source VideoStream and of the per-frame people over it."""
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
EXAMPLES: Path = Path(__file__).resolve().parents[3] / "data" / "body3d-examples"
"""Example images (``pixi run -e posekit-body posekit-body-download-examples``)."""


@dataclass(frozen=True, slots=True)
class Body3dFrame:
    """One frame's people, the detections they came from, and the camera and depth used."""

    people: Sam3dBodyReconstruction
    """Reconstructed people, row-aligned with ``detections``."""
    detections: BoxDetections
    """SAM3 person boxes and masks."""
    camera: CalibratedCamera | None
    """The camera given to the body model; None means its default field of view."""
    depth_meters: Float32[ndarray, "h w"] | None
    """MoGe-2 metric depth, when it estimated the camera."""
    depth_confidence: Float32[ndarray, "h w"] | None
    """MoGe-2 depth confidence."""


def pinhole_camera(intrinsics: Float32[ndarray, "3 3"], width: int, height: int) -> CalibratedCamera:
    """An undistorted camera from estimated intrinsics."""
    return CalibratedCamera(intrinsics.astype(np.float64), np.zeros(0, dtype=np.float64), width, height, lens="PINHOLE")


class Body3dPipeline:
    """SAM3 person masks + optional MoGe-2 camera + SAM 3D Body, one RGB image at a time."""

    def __init__(self, body: BodyPose3dConfig, segmenter: Sam3SegmenterConfig, *, prompt: str = "person", estimate_fov: bool = True) -> None:
        self.segmenter: Sam3Segmenter = segmenter.setup()
        self.body: Sam3dBodyPredictor = body.setup()
        self.prompt: str = prompt
        self.fov = None
        if estimate_fov:
            from monopriors.models.metric_depth import MoGeV2MetricPredictor

            self.fov = MoGeV2MetricPredictor(device="cuda")

    def estimate_camera(self, rgb: UInt8[ndarray, "h w 3"]) -> tuple[CalibratedCamera, Float32[ndarray, "h w"], Float32[ndarray, "h w"]]:
        """MoGe-2 intrinsics, metric depth and confidence of one image."""
        assert self.fov is not None, "estimate_fov is off"
        prediction = self.fov(rgb=rgb, K_33=None)
        camera = pinhole_camera(prediction.K_33.astype(np.float32), int(rgb.shape[1]), int(rgb.shape[0]))
        return camera, prediction.depth_meters.astype(np.float32), prediction.confidence.astype(np.float32)

    def frames(self, frames_rgb: UInt8[Tensor, "b h w 3"], camera: CalibratedCamera | None) -> tuple[BoxDetections, Sam3dBodyReconstruction]:
        """Segment and reconstruct every person in a CUDA frame batch seen by one camera."""
        detections: BoxDetections = self.segmenter(frames_rgb, SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text=self.prompt))
        return detections, self.body(frames_rgb, detections, camera)

    def __call__(self, rgb: UInt8[ndarray, "h w 3"], camera: CalibratedCamera | None = None) -> Body3dFrame:
        """Reconstruct the people in one image; without ``camera``, MoGe-2 estimates it (if enabled)."""
        depth: Float32[ndarray, "h w"] | None = None
        confidence: Float32[ndarray, "h w"] | None = None
        if camera is None and self.fov is not None:
            camera, depth, confidence = self.estimate_camera(rgb)
        with torch.inference_mode():
            detections, people = self.frames(torch.from_numpy(np.ascontiguousarray(rgb)).cuda()[None], camera)
        return Body3dFrame(people, detections, camera, depth, confidence)


def log_body3d_frame(frame: Body3dFrame, rgb: UInt8[ndarray, "h w 3"], *, log_depth: bool) -> None:
    """Log one reconstructed image under ``/world`` (image, overlays, joints, meshes, optional depth clouds)."""
    masks: Bool[ndarray, "n h w"] | None = None if frame.detections.masks is None else frame.detections.masks.cpu().numpy()
    with_depth: bool = log_depth and frame.depth_meters is not None and frame.camera is not None
    log_people(
        frame.people,
        rgb,
        Path("/world"),
        masks=masks,
        depth_meters=frame.depth_meters if with_depth else None,
        depth_confidence=frame.depth_confidence if with_depth else None,
        depth_intrinsics=frame.camera.intrinsics.astype(np.float32) if with_depth and frame.camera is not None else None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Body3dConfig:
    """Whole-body 3D people in one image, a folder of images, or a video."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""
    image_path: Path | None = None
    """One input image."""
    image_folder: Path | None = None
    """A folder of input images."""
    video_path: Path | None = None
    """An input video (one camera for the whole clip)."""
    body: AnnotatedBodyPose3dConfig = field(default_factory=Sam3dBodyConfig)
    """SAM 3D Body weights and mode (``full`` adds hands)."""
    segmenter: Sam3SegmenterConfig = field(default_factory=Sam3SegmenterConfig)
    """SAM3 person segmentation."""
    prompt: str = "person"
    """Text concept SAM3 segments."""
    camera_path: Path | None = None
    """CalibratedCamera JSON of the input; None estimates intrinsics with MoGe-2 (per image, or once per video)."""
    estimate_fov: bool = True
    """Without ``camera_path``, estimate the camera with MoGe-2; off, SAM 3D Body assumes its default field of view."""
    log_depth: bool = True
    """Log MoGe-2 depth point clouds for images (full, background, people)."""
    frame_batch_size: int = 4
    """Video frames segmented and reconstructed per step."""
    max_frames: int | None = None
    """Maximum number of images or video frames; None processes all."""


def list_images(config: Body3dConfig) -> list[Path]:
    """The input images, sorted.

    Raises:
        ValueError: If no image input is set or none is found.
    """
    if config.image_path is not None:
        return [config.image_path]
    if config.image_folder is None:
        raise ValueError("Set one of image_path, image_folder or video_path.")
    images: list[Path] = sorted(path for path in config.image_folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images in {config.image_folder}.")
    return images


def read_rgb(path: Path) -> UInt8[ndarray, "h w 3"]:
    """An image file as RGB.

    Raises:
        ValueError: If OpenCV cannot read it.
    """
    bgr: ndarray | None = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def video_chunks(path: Path, batch_size: int, max_frames: int | None) -> Iterator[tuple[int, UInt8[Tensor, "b h w 3"]]]:
    """CUDA-decoded frame batches with the index of their first frame."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path, dimension_order="NHWC", device="cuda")
    count: int = int(decoder.metadata.num_frames or 0)
    if max_frames is not None:
        count = min(count, max_frames)
    for start in tqdm(range(0, count, batch_size), desc="posekit body3d"):
        yield start, decoder.get_frames_in_range(start, min(start + batch_size, count)).data.contiguous()


def main(config: Body3dConfig) -> None:
    """Reconstruct people in the configured images or video and log them to Rerun."""
    camera: CalibratedCamera | None = None if config.camera_path is None else load_calibrated_camera(config.camera_path)
    is_video: bool = config.video_path is not None
    images: list[Path] = [] if is_video else list_images(config)[: config.max_frames]
    log_depth: bool = config.log_depth and not is_video and camera is None and config.estimate_fov
    set_annotation_context()
    rr.send_blueprint(rrb.Blueprint(create_view(log_depth=log_depth), collapse_panels=True))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    pipeline = Body3dPipeline(config.body, config.segmenter, prompt=config.prompt, estimate_fov=config.estimate_fov and camera is None)
    people: int = 0
    if config.video_path is not None:
        video_times_ns: Int64[ndarray, "num_frames"] = log_video(config.video_path, Path("/world/cam/pinhole/image"), timeline=VIDEO_TIMELINE)
        for start, frames_rgb in video_chunks(config.video_path, config.frame_batch_size, config.max_frames):
            if start + len(frames_rgb) > len(video_times_ns):
                raise ValueError(f"{config.video_path}: the logged VideoStream has {len(video_times_ns)} frames, fewer than torchcodec decodes.")
            if camera is None and pipeline.fov is not None:
                camera = pipeline.estimate_camera(frames_rgb[0].cpu().numpy())[0]
            with torch.inference_mode():
                detections, result = pipeline.frames(frames_rgb, camera)
            frame_hw: tuple[int, int] = (int(frames_rgb.shape[1]), int(frames_rgb.shape[2]))
            for local in range(len(frames_rgb)):
                rows: Tensor = torch.nonzero(result.keypoints.frame_indices == local).flatten()
                masks: ndarray | None = None if detections.masks is None else detections.masks[detections.frame_indices == local].cpu().numpy()
                rr.set_time(VIDEO_TIMELINE, duration=np.timedelta64(int(video_times_ns[start + local]), "ns"))
                log_people(result.take(rows), frame_hw, Path("/world"), masks=masks)
            people += result.num_people
    else:
        for index, path in enumerate(tqdm(images, desc="posekit body3d")):
            rr.set_time("image_sequence", sequence=index)
            rgb: UInt8[ndarray, "h w 3"] = read_rgb(path)
            frame = pipeline(rgb, camera)
            if frame.people.num_people == 0:
                print(f"[warn] No people in {path}; skipping.")
                continue
            log_body3d_frame(frame, rgb, log_depth=log_depth)
            people += frame.people.num_people
    print(f"[posekit] reconstructed {people} people")
