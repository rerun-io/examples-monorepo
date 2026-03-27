"""Video-to-image frame extraction node.

Extracts evenly-spaced frames from a single video and saves them as JPEG
images to an output directory. Reuses the same ``np.linspace`` spacing
strategy as :func:`pysfm.apis.pycolmap_rig_recon.extract_synchronized_frames`
but simplified for a single video.

Uses a class-based node pattern (like ``MultiViewCalibrator``) so that
intermediate Rerun logging lives inside the node, gated behind
``config.verbose``. Callers only need to set the Rerun recording context.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import VideoReader

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------
@dataclass
class VideoToImageConfig:
    """Configuration for video-to-image frame extraction."""

    num_frames: int = 20
    """Number of evenly-spaced frames to extract from the video."""
    verbose: bool = False
    """Log each extracted frame to Rerun as it is written (requires active recording context)."""


@dataclass
class VideoToImageResult:
    """Result of video-to-image frame extraction."""

    output_dir: Path
    """Directory where extracted JPEG images were saved."""
    num_frames_extracted: int
    """Actual number of frames extracted (may be less than requested if video is short)."""
    frame_indices: list[int]
    """Original video frame indices that were extracted."""
    image_paths: list[Path]
    """Paths to the extracted JPEG images, in order."""


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------
TIMELINE: str = "video_time"
"""Timeline name shared between video and extracted images."""


class VideoToImageNode:
    """Extract evenly-spaced frames from a single video.

    Follows the ``MultiViewCalibrator`` pattern: takes ``parent_log_path`` and
    ``config`` in the constructor, optionally logs intermediate results to Rerun
    when ``config.verbose`` is ``True``. The caller is responsible for setting
    the Rerun recording context (``with recording:`` or global init).

    Args:
        config: Extraction configuration.
        parent_log_path: Root Rerun entity path for this node's data.
    """

    def __init__(self, config: VideoToImageConfig, parent_log_path: Path) -> None:
        self.config: VideoToImageConfig = config
        self.parent_log_path: Path = parent_log_path

    def __call__(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        frame_timestamps_ns: Int[ndarray, "num_frames"] | None = None,
    ) -> VideoToImageResult:
        """Extract evenly-spaced frames from a video and save as JPEG images.

        When ``self.config.verbose`` is True and a Rerun recording context is
        active, each extracted frame is logged to
        ``{parent_log_path}/extracted/image`` on the ``video_time`` timeline
        (synchronized with the video asset if ``frame_timestamps_ns`` is given).

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to write extracted images into.
            frame_timestamps_ns: Optional per-frame video timestamps in nanoseconds
                (as returned by ``log_video``). When provided and verbose is True,
                extracted frames are logged at the correct video time.

        Returns:
            VideoToImageResult with paths and metadata about extracted frames.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video has zero frames.
        """
        if not video_path.exists():
            msg: str = f"Video not found: {video_path}"
            raise FileNotFoundError(msg)

        reader: VideoReader = VideoReader(filename=video_path)
        total_frames: int = reader.frame_cnt

        if total_frames <= 0:
            msg = f"Video has zero frames: {video_path}"
            raise RuntimeError(msg)

        actual_num_frames: int = min(self.config.num_frames, total_frames)
        frame_indices: list[int] = np.linspace(0, total_frames - 1, actual_num_frames, dtype=int).tolist()
        pad_width: int = len(str(actual_num_frames))

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Extracting %d evenly-spaced frames from %s (%d total frames)",
            actual_num_frames,
            video_path.name,
            total_frames,
        )

        image_paths: list[Path] = [Path()] * actual_num_frames

        # Use random access (get_frame) to read only the needed frames.
        for out_idx, frame_idx in enumerate(frame_indices):
            bgr_frame: UInt8[ndarray, "H W 3"] = reader.get_frame(frame_idx)
            filename: str = f"image{out_idx + 1:0{pad_width}d}.jpg"
            out_path: Path = output_dir / filename
            cv2.imwrite(str(out_path), bgr_frame)
            image_paths[out_idx] = out_path

            # Verbose: log each frame as it's extracted
            if self.config.verbose:
                if frame_timestamps_ns is not None:
                    rr.set_time(TIMELINE, duration=1e-9 * float(frame_timestamps_ns[frame_idx]))
                rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                rr.log(
                    f"{self.parent_log_path}/extracted/image",
                    rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
                )

        logger.info("Wrote %d frames to %s", actual_num_frames, output_dir)

        return VideoToImageResult(
            output_dir=output_dir,
            num_frames_extracted=actual_num_frames,
            frame_indices=frame_indices,
            image_paths=image_paths,
        )


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
def create_video_to_image_blueprint(parent_log_path: Path) -> rrb.ContainerLike:
    """Create a Rerun blueprint for video-to-image visualization.

    Args:
        parent_log_path: Root log path for this node's data.

    Returns:
        Rerun container layout.
    """
    return rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/video",
                name="Input Video",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/extracted",
                name="Extracted Frames",
            ),
        ],
        column_shares=[1, 1],
    )


# ---------------------------------------------------------------------------
# CLI config & main
# ---------------------------------------------------------------------------
@dataclass
class VideoToImageCLIConfig:
    """CLI configuration for video-to-image frame extraction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    video_path: Path = Path("data/examples/unknown-rig/cam1.mp4")
    """Path to input video file."""
    output_dir: Path = Path("/tmp/video_to_image_output")
    """Directory to write extracted JPEG images."""
    config: VideoToImageConfig = field(default_factory=lambda: VideoToImageConfig(verbose=True))
    """Frame extraction configuration."""


def main(config: VideoToImageCLIConfig) -> None:
    """CLI entry point for video-to-image extraction with Rerun visualization."""
    from simplecv.rerun_log_utils import log_video

    parent_log_path: Path = Path("world")

    # Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_video_to_image_blueprint(parent_log_path),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    # Log the video asset — returns timestamps for all frames
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=config.video_path,
        video_log_path=parent_log_path / "video",
        timeline=TIMELINE,
    )

    # Run extraction — node handles intermediate logging when verbose
    node: VideoToImageNode = VideoToImageNode(config=config.config, parent_log_path=parent_log_path)
    node(
        video_path=config.video_path,
        output_dir=config.output_dir,
        frame_timestamps_ns=frame_timestamps_ns,
    )
