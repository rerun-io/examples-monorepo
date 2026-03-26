"""Monocular video reconstruction.

End-to-end pipeline: extracts frames from a single video, runs
ALIKED+LightGlue feature extraction and sequential matching, then
produces a 3D reconstruction via either incremental or global mapping.

Uses the pycolmap Python API throughout — no subprocess calls to COLMAP CLI.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig

from pysfm.apis.sfm_reconstruction import _extract_intrinsics, _extract_visible_keypoints
from pysfm.apis.video_to_image import VideoToImageConfig, VideoToImageNode

logger: logging.Logger = logging.getLogger(__name__)

TIMELINE: str = "frame"
"""Timeline name used for sequential camera logging."""

SfMCameraModel: TypeAlias = Literal["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]
"""Supported COLMAP camera model names."""


# ---------------------------------------------------------------------------
# TimingLogger
# ---------------------------------------------------------------------------
class TimingLogger:
    """Builds a markdown timing table and logs it to Rerun as a TextDocument.

    Each pipeline step is wrapped in ``with timer.log_time("section"):``.
    The markdown table grows incrementally — each exit appends a row and
    updates the Rerun log so the viewer shows progress live.
    """

    def __init__(self, header: str, log_path: str = "logs") -> None:
        self.start_time: float = time.perf_counter()
        self.log_path: str = log_path
        self.markdown_table: str = f"# {header}\n" "| Section | Time |\n" "|---------|------|\n"
        rr.log(self.log_path, rr.TextDocument(self.markdown_table, media_type="text/markdown"), static=True)

    @contextmanager
    def log_time(self, section_name: str) -> Generator[None, None, None]:
        """Time a pipeline section and append it to the markdown table."""
        t_start: float = time.perf_counter()
        logger.info(f"{section_name} ...")
        try:
            yield
        finally:
            elapsed: float = time.perf_counter() - t_start
            minutes: int
            seconds: float
            minutes, seconds = divmod(elapsed, 60)
            time_str: str = f"{int(minutes)}m {seconds:.1f}s"
            self.markdown_table += f"| {section_name} | {time_str} |\n"
            rr.log(self.log_path, rr.TextDocument(self.markdown_table, media_type="text/markdown"), static=True)
            logger.info(f"{section_name}: {time_str}")

    def log_total(self) -> None:
        """Append a total row to the timing table."""
        total: float = time.perf_counter() - self.start_time
        minutes: int
        seconds: float
        minutes, seconds = divmod(total, 60)
        time_str: str = f"{int(minutes)}m {seconds:.1f}s"
        self.markdown_table += f"| **Total** | **{time_str}** |\n"
        rr.log(self.log_path, rr.TextDocument(self.markdown_table, media_type="text/markdown"), static=True)
        logger.info(f"Total pipeline time: {time_str}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class VidReconConfig:
    """Configuration for monocular video reconstruction."""

    video_path: Path = Path("data/examples/unknown-rig/cam1.mp4")
    """Path to the input video file."""
    output_dir: Path | None = None
    """Output directory. Defaults to ``video_path.parent / 'output'``."""
    num_frames: int = 100
    """Number of evenly-spaced frames to extract from the video."""
    camera_model: SfMCameraModel = "SIMPLE_RADIAL"
    """COLMAP camera model for intrinsics estimation."""
    overlap: int = 10
    """Sequential matching overlap (number of neighboring frames to match)."""
    mapping: Literal["incremental", "global"] = "incremental"
    """Mapping method: incremental (robust, slower) or global (faster)."""
    use_gpu: bool = True
    """Use GPU for feature extraction and matching."""
    verbose: bool = False
    """Emit detailed COLMAP logging."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class VidReconResult:
    """Output of monocular video reconstruction pipeline."""

    output_dir: Path
    """Root output directory."""
    images_dir: Path
    """Directory containing extracted frames."""
    database_path: Path
    """Path to the COLMAP database."""
    model_dir: Path
    """Path to the sparse model directory."""
    num_frames_extracted: int
    """Number of frames extracted from the video."""
    mapping_method: str
    """Which mapping method was used ('incremental' or 'global')."""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_vid_recon(*, config: VidReconConfig, timer: TimingLogger | None = None) -> VidReconResult:
    """Run the monocular video reconstruction pipeline.

    Pipeline:
        1. Extract frames from video
        2. Feature extraction (ALIKED_N16ROT, GPU)
        3. Sequential matching (ALIKED_LIGHTGLUE, GPU)
        4. Incremental or global mapping (user choice)

    Args:
        config: Pipeline configuration.
        timer: Optional timing logger. When provided, each step is timed
            and the results are logged to Rerun as a markdown table.

    Returns:
        Result containing paths to all outputs.
    """
    # -- Resolve output paths -------------------------------------------------
    output_dir: Path = config.output_dir if config.output_dir is not None else config.video_path.parent / "output"
    images_dir: Path = output_dir / "images"
    database_path: Path = output_dir / "database.db"
    sparse_dir: Path = output_dir / "sparse"

    for d in (images_dir, sparse_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- 1. Extract frames ----------------------------------------------------
    with timer.log_time("Frame extraction") if timer else nullcontext():
        node: VideoToImageNode = VideoToImageNode(
            config=VideoToImageConfig(num_frames=config.num_frames),
            parent_log_path=Path("world"),
        )
        extraction_result = node(
            video_path=config.video_path,
            output_dir=images_dir,
        )
    num_frames_extracted: int = extraction_result.num_frames_extracted

    # -- 2. Feature extraction (ALIKED_N16ROT, GPU) ---------------------------
    pycolmap.set_random_seed(0)

    if config.verbose:
        pycolmap.logging.minloglevel = pycolmap.logging.INFO

    reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()
    reader_options.camera_model = config.camera_model

    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    extraction_options.use_gpu = config.use_gpu
    extraction_options.gpu_index = "0"

    with timer.log_time(f"Feature extraction (ALIKED_N16ROT, {config.camera_model})") if timer else nullcontext():
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

    # -- 3. Sequential matching (ALIKED_LIGHTGLUE) ----------------------------
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    matching_options.use_gpu = config.use_gpu
    matching_options.gpu_index = "0"

    pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = config.overlap
    pairing_options.quadratic_overlap = False

    with timer.log_time(f"Sequential matching (overlap={config.overlap})") if timer else nullcontext():
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
        )

    # -- 4. Mapping -----------------------------------------------------------
    with timer.log_time(f"{config.mapping.capitalize()} mapping") if timer else nullcontext():
        if config.mapping == "incremental":
            incremental_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions()
            incremental_options.multiple_models = False
            recs: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=sparse_dir,
                options=incremental_options,
            )
        else:
            global_options: pycolmap.GlobalPipelineOptions = pycolmap.GlobalPipelineOptions()
            recs = pycolmap.global_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=sparse_dir,
                options=global_options,
            )

    if not recs:
        msg: str = f"{config.mapping.capitalize()} mapping failed: COLMAP produced no reconstruction"
        raise RuntimeError(msg)

    # Use the largest reconstruction.
    rec_id: int = max(recs, key=lambda k: recs[k].num_reg_images())
    rec: pycolmap.Reconstruction = recs[rec_id]
    logger.info(f"Reconstruction: {rec.num_reg_images()} images registered (model {rec_id})")

    if timer:
        timer.log_total()

    return VidReconResult(
        output_dir=output_dir,
        images_dir=images_dir,
        database_path=database_path,
        model_dir=sparse_dir / str(rec_id),
        num_frames_extracted=num_frames_extracted,
        mapping_method=config.mapping,
    )


# ---------------------------------------------------------------------------
# Blueprint builder
# ---------------------------------------------------------------------------
def create_vid_blueprint(parent_log_path: Path) -> rrb.ContainerLike:
    """Create a Rerun blueprint for monocular video reconstruction visualization.

    Layout: 3D view (left) | 2D camera view + timing table (right, stacked).

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).

    Returns:
        Rerun blueprint layout.
    """
    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        origin="/",
        contents=[f"+ {parent_log_path}/**"],
    )

    view_2d: rrb.Spatial2DView = rrb.Spatial2DView(
        origin=f"{parent_log_path}/camera/pinhole",
    )

    view_timing: rrb.TextDocumentView = rrb.TextDocumentView(
        origin="logs",
        name="Timing",
    )

    return rrb.Horizontal(
        contents=[view_3d, rrb.Vertical(contents=[view_2d, view_timing], row_shares=[3, 1])],
        column_shares=[3, 2],
    )


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------
def log_vid_reconstruction(result: VidReconResult, parent_log_path: Path) -> None:
    """Load the reconstruction and log it to Rerun.

    Computes a gravity-alignment transform from the camera poses using
    ``auto_orient_and_center_poses`` (method="up") and logs it as a static
    ``Transform3D`` on ``parent_log_path``.

    Logs 3D point cloud (static), then camera frustums, images, and
    keypoints over a timeline.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    rec: pycolmap.Reconstruction = pycolmap.Reconstruction(str(result.model_dir))

    # -- Compute gravity-alignment transform from camera poses ----------------
    world_T_cam_all: list[Float64[ndarray, "4 4"]] = []
    for image in rec.images.values():
        world_from_cam: pycolmap.Rigid3d = image.cam_from_world().inverse()
        world_T_cam_cv: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        world_T_cam_cv[:3, :3] = world_from_cam.rotation.matrix()
        world_T_cam_cv[:3, 3] = world_from_cam.translation
        world_T_cam_all.append(world_T_cam_cv)

    if world_T_cam_all:
        world_T_cam_batch: Float64[ndarray, "N 4 4"] = np.stack(world_T_cam_all)
        world_T_cam_gl: Float64[ndarray, "N 4 4"] = convert_pose(
            world_T_cam_batch, CameraConventions.CV, CameraConventions.GL
        )
        orient_34: Float64[ndarray, "3 4"] = auto_orient_and_center_poses(
            world_T_cam_gl, method="up", center_method="poses"
        ).transform
        orient_R: Float64[ndarray, "3 3"] = orient_34[:, :3]
        orient_t: Float64[ndarray, "3"] = orient_34[:, 3]
        rr.log(
            f"{parent_log_path}",
            rr.Transform3D(mat3x3=orient_R, translation=orient_t),
            static=True,
        )
        logger.info(f"Gravity-alignment transform logged on '{parent_log_path}' (det={np.linalg.det(orient_R):.4f})")

    # -- Static 3D point cloud ------------------------------------------------
    xyz_list: list[Float64[ndarray, "3"]] = []
    rgb_list: list[UInt8[ndarray, "3"]] = []
    for point in rec.points3D.values():
        xyz_list.append(point.xyz)
        rgb_list.append(point.color)

    if xyz_list:
        points3d_xyz: Float32[ndarray, "N 3"] = np.array(xyz_list, dtype=np.float32)
        points3d_rgb: UInt8[ndarray, "N 3"] = np.array(rgb_list, dtype=np.uint8)
        rr.log(
            f"{parent_log_path}/point_cloud",
            rr.Points3D(
                positions=points3d_xyz,
                colors=points3d_rgb,
            ),
            static=True,
        )

    # -- Log cameras over timeline --------------------------------------------
    sorted_images: list[tuple[int, pycolmap.Image]] = sorted(rec.images.items(), key=lambda x: x[1].name)

    for frame_idx, (_image_id, image) in enumerate(sorted_images):
        rr.set_time(TIMELINE, sequence=frame_idx)

        camera_path: str = f"{parent_log_path}/camera"

        # Transform (world_T_cam)
        world_from_cam: pycolmap.Rigid3d = image.cam_from_world().inverse()
        world_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        world_T_cam[:3, :3] = world_from_cam.rotation.matrix()
        world_T_cam[:3, 3] = world_from_cam.translation

        rr.log(camera_path, rr.Transform3D(mat3x3=world_T_cam[:3, :3], translation=world_T_cam[:3, 3]))

        # Pinhole camera
        cam: pycolmap.Camera = image.camera
        K: Float64[ndarray, "3 3"] = _extract_intrinsics(cam)
        rr.log(
            f"{camera_path}/pinhole",
            rr.Pinhole(
                image_from_camera=K,
                width=cam.width,
                height=cam.height,
                image_plane_distance=1.0,
            ),
        )

        # Image
        image_path: Path = result.images_dir / image.name
        if image_path.exists():
            bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rr.log(f"{camera_path}/pinhole/image", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress())

        # 2D keypoints
        keypoints_xy: Float32[ndarray, "M 2"] = _extract_visible_keypoints(image, rec.points3D)
        if keypoints_xy.shape[0] > 0:
            rr.log(
                f"{camera_path}/pinhole/image/keypoints",
                rr.Points2D(
                    positions=keypoints_xy,
                    radii=np.full(keypoints_xy.shape[0], 2.0, dtype=np.float32),
                    colors=np.full((keypoints_xy.shape[0], 3), [34, 138, 167], dtype=np.uint8),
                ),
            )


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------
@dataclass
class VidReconCLIConfig:
    """CLI configuration for monocular video reconstruction with Rerun visualization."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration (--save, --connect, --spawn)."""
    config: VidReconConfig = field(default_factory=VidReconConfig)
    """Reconstruction pipeline configuration."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(cli_config: VidReconCLIConfig) -> None:
    """Run the monocular video pipeline and visualize results in Rerun."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parent_log_path: Path = Path("world")

    # 1. Run the full pipeline with timing
    timer: TimingLogger = TimingLogger(header="Monocular Video Reconstruction")
    result: VidReconResult = run_vid_recon(config=cli_config.config, timer=timer)

    # 2. Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_vid_blueprint(parent_log_path=parent_log_path),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    # Gravity-aligned data has Z-up; tell the viewer accordingly.
    rr.log("/", rr.ViewCoordinates.RFU, static=True)

    # 3. Log the reconstruction
    log_vid_reconstruction(result, parent_log_path)

    print(f"Output: {result.output_dir}")
    print(f"Mapping: {result.mapping_method}")
    print(f"Model: {result.model_dir}")
