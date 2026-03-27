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

MATCH_TIMELINE: str = "match_pair"
"""Timeline name used for match-pair visualization."""

SfMCameraModel: TypeAlias = Literal[
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"
]
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
        self.markdown_table: str = f"# {header}\n| Section | Time |\n|---------|------|\n"
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
    overlap: int
    """Sequential matching overlap used for pairing."""


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
        overlap=config.overlap,
    )


# ---------------------------------------------------------------------------
# Blueprint builder
# ---------------------------------------------------------------------------
def create_vid_blueprint_tabs(
    parent_log_path: Path,
    *,
    active_tab: int = 0,
    timeline: str = TIMELINE,
) -> rrb.Blueprint:
    """Create a tabbed Rerun blueprint for video reconstruction visualization.

    Three tabs:
        0 — **Features**: per-frame 2D keypoints + timing
        1 — **Matches**: side-by-side match pairs + timing
        2 — **Reconstruction**: 3D + 2D views of the final model + timing

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).
        active_tab: Index of the tab to display initially.
        timeline: Which timeline the time panel should display.

    Returns:
        Blueprint with tabbed layout.
    """
    # -- Tab 0: Features -------------------------------------------------------
    features_tab: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/features",
                name="Features",
            ),
            rrb.TextDocumentView(origin="logs", name="Timing"),
        ],
        column_shares=[3, 1],
        name="Features",
    )

    # -- Tab 1: Matches --------------------------------------------------------
    matches_tab: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/matches",
                name="Match Pairs",
            ),
            rrb.TextDocumentView(origin="logs", name="Timing"),
        ],
        column_shares=[3, 1],
        name="Matches",
    )

    # -- Tab 2: Reconstruction -------------------------------------------------
    recon_tab: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial3DView(
                origin="/",
                contents=[f"+ {parent_log_path}/**"],
            ),
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole"),
                    rrb.TextDocumentView(origin="logs", name="Timing"),
                ],
                row_shares=[5, 2],
            ),
        ],
        column_shares=[3, 2],
        name="Reconstruction",
    )

    return rrb.Blueprint(
        rrb.Tabs(
            features_tab,
            matches_tab,
            recon_tab,
            active_tab=active_tab,
        ),
        rrb.TimePanel(timeline=timeline),
        collapse_panels=True,
    )


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------
def log_vid_reconstruction(result: VidReconResult, parent_log_path: Path) -> None:
    """Load the reconstruction and log it to Rerun.

    Computes a gravity-alignment transform from the camera poses using
    ``auto_orient_and_center_poses`` (method="up") and logs it as a static
    ``Transform3D`` on ``parent_log_path``.

    The 3D point cloud is logged **incrementally**: each frame adds the
    points visible from that camera so the cloud grows as you scrub through
    the timeline.  Camera frustums, images, and 2D keypoints are logged
    per-frame as before.

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

    # -- Log cameras + incremental point cloud over timeline -------------------
    sorted_images: list[tuple[int, pycolmap.Image]] = sorted(rec.images.items(), key=lambda x: x[1].name)

    seen_point_ids: set[int] = set()
    accumulated_xyz: list[Float64[ndarray, "3"]] = []
    accumulated_rgb: list[UInt8[ndarray, "3"]] = []

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
                    colors=[0, 255, 0],
                ),
            )

        # Accumulate 3D points visible from this camera
        for p2d in image.points2D:
            if p2d.has_point3D() and p2d.point3D_id in rec.points3D and p2d.point3D_id not in seen_point_ids:
                seen_point_ids.add(p2d.point3D_id)
                point: pycolmap.Point3D = rec.points3D[p2d.point3D_id]
                accumulated_xyz.append(point.xyz)
                accumulated_rgb.append(point.color)

        # Log the growing point cloud
        if accumulated_xyz:
            rr.log(
                f"{parent_log_path}/point_cloud",
                rr.Points3D(
                    positions=np.array(accumulated_xyz, dtype=np.float32),
                    colors=np.array(accumulated_rgb, dtype=np.uint8),
                ),
            )


# ---------------------------------------------------------------------------
# Feature visualization
# ---------------------------------------------------------------------------
def log_features(result: VidReconResult, parent_log_path: Path) -> None:
    """Log per-frame images with **all** detected keypoints from the database.

    Unlike ``log_vid_reconstruction`` which shows only triangulated keypoints,
    this shows every keypoint found by ALIKED — giving a complete picture
    of feature coverage before matching and reconstruction.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    with pycolmap.Database.open(result.database_path) as db:
        all_images: list[pycolmap.Image] = db.read_all_images()
        sorted_images: list[pycolmap.Image] = sorted(all_images, key=lambda img: img.name)

        for frame_idx, db_image in enumerate(sorted_images):
            rr.set_time(TIMELINE, sequence=frame_idx)

            image_path: Path = result.images_dir / db_image.name
            if not image_path.exists():
                continue
            bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log(
                f"{parent_log_path}/features/image",
                rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
            )

            # All keypoints from DB (Nx6: x, y, a11, a12, a21, a22)
            kps: Float32[ndarray, "N 6"] = db.read_keypoints(db_image.image_id)
            if kps.shape[0] > 0:
                kps_xy: Float32[ndarray, "N 2"] = kps[:, :2]
                rr.log(
                    f"{parent_log_path}/features/image/keypoints",
                    rr.Points2D(positions=kps_xy, colors=[255, 165, 0]),
                )

    logger.info(f"Logged features for {len(sorted_images)} frames")


# ---------------------------------------------------------------------------
# Match visualization
# ---------------------------------------------------------------------------
def log_matches(
    result: VidReconResult,
    parent_log_path: Path,
    *,
    max_dim: int = 640,
) -> None:
    """Log feature matches as side-by-side image pairs with connecting lines.

    Creates horizontally stacked image pairs, draws line strips between
    matched keypoint positions, and sends everything via ``rr.send_columns``
    on a separate ``"match_pair"`` timeline.

    Uses geometrically verified inlier matches from two-view geometry
    (post-RANSAC) for cleaner visualization.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
        max_dim: Maximum dimension for resized stacked images.
    """
    with pycolmap.Database.open(result.database_path) as db:
        all_images: list[pycolmap.Image] = db.read_all_images()
        sorted_images: list[pycolmap.Image] = sorted(all_images, key=lambda img: img.name)
        num_images: int = len(sorted_images)

        # -- Collect all match data in a single pass ----------------------------
        pair_indices: list[int] = []
        jpeg_blobs: list[bytes] = []
        kp_left_list: list[Float32[ndarray, "M 2"]] = []
        kp_right_list: list[Float32[ndarray, "M 2"]] = []
        strip_list: list[list[Float32[ndarray, "2 2"]]] = []
        pair_idx: int = 0

        for i in range(num_images):
            img_a: pycolmap.Image = sorted_images[i]
            kps_a: Float32[ndarray, "Na 6"] = db.read_keypoints(img_a.image_id)

            bgr_a: UInt8[ndarray, "H W 3"] | None = cv2.imread(
                str(result.images_dir / img_a.name), cv2.IMREAD_COLOR
            )
            if bgr_a is None:
                continue

            for j in range(i + 1, min(i + result.overlap + 1, num_images)):
                img_b: pycolmap.Image = sorted_images[j]

                # Read geometrically verified matches
                tvg: pycolmap.TwoViewGeometry = db.read_two_view_geometry(
                    img_a.image_id, img_b.image_id
                )
                inlier_matches: np.ndarray = tvg.inlier_matches
                if inlier_matches.shape[0] == 0:
                    continue

                kps_b: Float32[ndarray, "Nb 6"] = db.read_keypoints(img_b.image_id)
                bgr_b: UInt8[ndarray, "H W 3"] | None = cv2.imread(
                    str(result.images_dir / img_b.name), cv2.IMREAD_COLOR
                )
                if bgr_b is None:
                    continue

                # Stack images side-by-side
                h_a: int = bgr_a.shape[0]
                w_a: int = bgr_a.shape[1]
                h_b: int = bgr_b.shape[0]
                w_b: int = bgr_b.shape[1]

                # Resize both to same height for hconcat
                max_h: int = max(h_a, h_b)
                if h_a != max_h:
                    scale_a: float = max_h / h_a
                    bgr_a_resized: UInt8[ndarray, "Hm Wa 3"] = cv2.resize(bgr_a, (int(w_a * scale_a), max_h))
                else:
                    bgr_a_resized = bgr_a
                    scale_a = 1.0
                if h_b != max_h:
                    scale_b: float = max_h / h_b
                    bgr_b_resized: UInt8[ndarray, "Hm Wb 3"] = cv2.resize(bgr_b, (int(w_b * scale_b), max_h))
                else:
                    bgr_b_resized = bgr_b
                    scale_b = 1.0

                stacked: UInt8[ndarray, "Hm Ws 3"] = cv2.hconcat([bgr_a_resized, bgr_b_resized])
                w_left: int = bgr_a_resized.shape[1]

                # Resize stacked image to max_dim
                sh: int = stacked.shape[0]
                sw: int = stacked.shape[1]
                resize_scale: float = max_dim / max(sh, sw) if max(sh, sw) > max_dim else 1.0
                if resize_scale < 1.0:
                    stacked = cv2.resize(stacked, (int(sw * resize_scale), int(sh * resize_scale)))

                # JPEG compress
                ok: bool
                buf: np.ndarray
                ok, buf = cv2.imencode(".jpg", stacked)
                if not ok:
                    continue
                jpeg_blobs.append(buf.tobytes())

                # Map match indices to keypoint coordinates
                idx_a: np.ndarray = inlier_matches[:, 0]
                idx_b: np.ndarray = inlier_matches[:, 1]

                pts_a: Float32[ndarray, "M 2"] = kps_a[idx_a, :2] * scale_a * resize_scale
                pts_b: Float32[ndarray, "M 2"] = kps_b[idx_b, :2] * scale_b * resize_scale
                pts_b_offset: Float32[ndarray, "M 2"] = pts_b.copy()
                pts_b_offset[:, 0] += w_left * resize_scale

                kp_left_list.append(pts_a.astype(np.float32))
                kp_right_list.append(pts_b_offset.astype(np.float32))

                # Line strips: each match is a 2-point strip
                num_matches: int = pts_a.shape[0]
                strips: list[Float32[ndarray, "2 2"]] = [
                    np.stack([pts_a[k], pts_b_offset[k]]).astype(np.float32)
                    for k in range(num_matches)
                ]
                strip_list.append(strips)

                pair_indices.append(pair_idx)
                pair_idx += 1

    if not pair_indices:
        logger.warning("No match pairs found to visualize")
        return

    num_pairs: int = len(pair_indices)
    logger.info(f"Sending {num_pairs} match pairs via send_columns ...")

    # -- Batch send images via send_columns ------------------------------------
    pair_indices_arr: np.ndarray = np.array(pair_indices, dtype=np.int64)

    image_cols: rr.ComponentColumnList = rr.EncodedImage.columns(
        blob=jpeg_blobs,
        media_type=["image/jpeg"] * num_pairs,
    )
    rr.send_columns(
        f"{parent_log_path}/matches/image",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=image_cols,
    )

    # -- Batch send left keypoints ---------------------------------------------
    all_kp_left: Float32[ndarray, "T 2"] = np.concatenate(kp_left_list, axis=0)
    kp_left_lengths: list[int] = [kp.shape[0] for kp in kp_left_list]
    kp_left_cols: rr.ComponentColumnList = rr.Points2D.columns(
        positions=all_kp_left,
        colors=np.full((all_kp_left.shape[0], 3), [0, 255, 0], dtype=np.uint8),
    ).partition(lengths=kp_left_lengths)
    rr.send_columns(
        f"{parent_log_path}/matches/image/kp_left",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=kp_left_cols,
    )

    # -- Batch send right keypoints --------------------------------------------
    all_kp_right: Float32[ndarray, "T 2"] = np.concatenate(kp_right_list, axis=0)
    kp_right_lengths: list[int] = [kp.shape[0] for kp in kp_right_list]
    kp_right_cols: rr.ComponentColumnList = rr.Points2D.columns(
        positions=all_kp_right,
        colors=np.full((all_kp_right.shape[0], 3), [0, 255, 0], dtype=np.uint8),
    ).partition(lengths=kp_right_lengths)
    rr.send_columns(
        f"{parent_log_path}/matches/image/kp_right",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=kp_right_cols,
    )

    # -- Batch send line strips ------------------------------------------------
    all_strips: list[Float32[ndarray, "2 2"]] = []
    strip_counts: list[int] = []
    for strips in strip_list:
        all_strips.extend(strips)
        strip_counts.append(len(strips))

    line_cols: rr.ComponentColumnList = rr.LineStrips2D.columns(
        strips=all_strips,
        colors=np.full((len(all_strips), 3), [0, 255, 0], dtype=np.uint8),
    ).partition(lengths=strip_counts)
    rr.send_columns(
        f"{parent_log_path}/matches/image/lines",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=line_cols,
    )

    logger.info(f"Logged {num_pairs} match pairs")


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
    """Run the monocular video pipeline and visualize results in Rerun.

    Sends the blueprint multiple times to switch the active tab as
    features → matches → reconstruction are progressively logged.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parent_log_path: Path = Path("world")

    # 1. Initial blueprint — Features tab shows timing updates live
    rr.send_blueprint(create_vid_blueprint_tabs(parent_log_path, active_tab=0))
    rr.log("/", rr.ViewCoordinates.RFU, static=True)

    # 2. Run the full pipeline (computation only, timed)
    timer: TimingLogger = TimingLogger(header="Monocular Video Reconstruction")
    result: VidReconResult = run_vid_recon(config=cli_config.config, timer=timer)

    # 3. Log features → stay on Features tab
    log_features(result, parent_log_path)

    # 4. Log matches → switch to Matches tab + match_pair timeline
    rr.send_blueprint(create_vid_blueprint_tabs(parent_log_path, active_tab=1, timeline=MATCH_TIMELINE))
    log_matches(result, parent_log_path)

    # 5. Log reconstruction → switch to Reconstruction tab + frame timeline
    rr.send_blueprint(create_vid_blueprint_tabs(parent_log_path, active_tab=2, timeline=TIMELINE))
    log_vid_reconstruction(result, parent_log_path)

    print(f"Output: {result.output_dir}")
    print(f"Mapping: {result.mapping_method}")
    print(f"Model: {result.model_dir}")
