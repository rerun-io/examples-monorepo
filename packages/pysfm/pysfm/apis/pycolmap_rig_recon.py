"""Unknown-rig multi-camera reconstruction from videos.

End-to-end pipeline: discovers videos in a directory, extracts synchronized
frames, runs ALIKED+LightGlue feature extraction and matching, bootstraps a
no-rig incremental reconstruction, estimates rig calibration from that
bootstrap, then produces a final rig-aware global reconstruction.

Uses the pycolmap Python API throughout — no subprocess calls to COLMAP CLI.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from serde import field, serde
from serde.json import to_json
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import MultiVideoReader

from pysfm.apis.pycolmap_vid_recon import MATCH_TIMELINE, SfMCameraModel, TimingLogger
from pysfm.streamed_pipeline import compare_databases, extract_features_streamed

logger: logging.Logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv"})
"""Recognized video file extensions."""

TIMELINE: str = "frame"
"""Timeline name used for sequential camera logging."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RigReconConfig:
    """Configuration for unknown-rig video reconstruction."""

    videos_dir: Path = Path("data/examples/unknown-rig")
    """Directory containing video files (cam1.mp4, cam2.mp4, ...)."""
    output_dir: Path | None = None
    """Output directory. Defaults to ``videos_dir / 'output'``."""
    num_frames: int = 50
    """Number of frames to extract per camera."""
    rig_name: str = "rig1"
    """Name for the camera rig (used in image path structure)."""
    ref_camera: str | None = None
    """Reference camera name. Defaults to first camera alphabetically."""
    camera_model: SfMCameraModel = "SIMPLE_RADIAL"
    """COLMAP camera model for intrinsics estimation (e.g. OPENCV_FISHEYE for fisheye lenses)."""
    overlap: int = 5
    """Sequential matching overlap (number of neighboring frames to match)."""
    use_gpu: bool = True
    """Use GPU for feature extraction and matching."""
    image_plane_distance: float | None = None
    """Rerun frustum size. Controls how large camera frustums appear in the 3D view.
    When None, Rerun picks a default. Try 0.2 for a compact view."""
    verbose: bool = False
    """Emit detailed COLMAP logging."""
    validate: bool = False
    """When True, run streamed implementations of pipeline stages alongside
    the black-box pycolmap calls and assert equivalence.  Increases runtime
    proportionally — intended for development and testing."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class RigReconResult:
    """Output of unknown-rig reconstruction pipeline."""

    output_dir: Path
    """Root output directory."""
    images_dir: Path
    """Directory containing extracted frames (COLMAP image_path root)."""
    database_path: Path
    """Path to the COLMAP database."""
    no_rig_model_dir: Path
    """Path to no-rig bootstrap sparse model directory."""
    rig_model_dir: Path
    """Path to final rig-aware sparse model directory."""
    num_frames_extracted: int
    """Number of frames extracted per camera."""
    num_cameras: int
    """Number of cameras in the rig."""
    camera_names: list[str]
    """Names of discovered cameras (sorted alphabetically)."""
    ref_camera: str
    """Name of the reference camera."""
    overlap: int
    """Sequential matching overlap used for pairing."""
    image_plane_distance: float | None
    """Rerun frustum size for camera visualization."""


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------
def discover_videos(videos_dir: Path) -> list[tuple[str, Path]]:
    """Find video files in a directory and derive camera names from filenames.

    Args:
        videos_dir: Directory to scan for video files.

    Returns:
        Sorted list of ``(camera_name, video_path)`` tuples.

    Raises:
        FileNotFoundError: If ``videos_dir`` does not exist.
        RuntimeError: If fewer than 2 videos are found.
    """
    if not videos_dir.is_dir():
        msg: str = f"Videos directory not found: {videos_dir}"
        raise FileNotFoundError(msg)

    videos: list[tuple[str, Path]] = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            camera_name: str = p.stem
            videos.append((camera_name, p))

    if len(videos) < 2:
        msg = f"Expected at least 2 videos in {videos_dir}, found {len(videos)}"
        raise RuntimeError(msg)

    return videos


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_synchronized_frames(
    videos: list[tuple[str, Path]],
    output_images_dir: Path,
    rig_name: str,
    num_frames: int,
) -> int:
    """Extract evenly-spaced frames from each video and save as JPEG.

    All cameras use the same frame indices so that identical filenames across
    camera folders represent the same timestamp (synchronized rig frames).

    Images are saved to ``output_images_dir/<rig_name>/<camera_name>/imageXXXX.jpg``
    with one-based, zero-padded filenames.

    Args:
        videos: List of ``(camera_name, video_path)`` from :func:`discover_videos`.
        output_images_dir: Root directory for extracted images.
        rig_name: Rig subdirectory name.
        num_frames: Target number of frames to extract.

    Returns:
        Actual number of frames extracted (may be less than ``num_frames``
        if a video is shorter).

    Raises:
        RuntimeError: If a video cannot be opened or has zero frames.
    """
    video_paths: list[Path] = [path for _, path in videos]
    camera_names: list[str] = [name for name, _ in videos]
    reader: MultiVideoReader = MultiVideoReader(video_paths)

    total_frames: int = len(reader)
    if total_frames <= 0:
        msg: str = "No synchronized frames available — videos may have zero frames"
        raise RuntimeError(msg)
    actual_num_frames: int = min(num_frames, total_frames)
    frame_indices: list[int] = np.linspace(0, total_frames - 1, actual_num_frames, dtype=int).tolist()
    pad_width: int = len(str(actual_num_frames))

    logger.info(
        f"Extracting {actual_num_frames} frames from {len(videos)} videos ({total_frames} synchronized frames available)"
    )

    # Create output directories for each camera.
    cam_dirs: list[Path] = []
    for cam_name in camera_names:
        cam_dir: Path = output_images_dir / rig_name / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        cam_dirs.append(cam_dir)

    # Read sequentially through the synchronized videos, saving only frames
    # at target indices.  Sequential decoding avoids the costly random seeks
    # that __getitem__/get_frame performs on compressed video.
    target_set: set[int] = set(frame_indices)
    index_to_out: dict[int, int] = {idx: out_idx for out_idx, idx in enumerate(frame_indices)}
    frames_saved: int = 0
    last_target: int = frame_indices[-1]

    for video_idx, bgr_list in enumerate(reader):
        if bgr_list is None:
            break
        if video_idx in target_set:
            out_idx: int = index_to_out[video_idx]
            filename: str = f"image{out_idx + 1:0{pad_width}d}.jpg"
            for cam_idx, bgr in enumerate(bgr_list):
                cv2.imwrite(str(cam_dirs[cam_idx] / filename), bgr)
            frames_saved += 1

            if frames_saved == actual_num_frames:
                break

        if video_idx >= last_target:
            break

    logger.info(f"Wrote frames to {output_images_dir / rig_name}")
    return actual_num_frames


# ---------------------------------------------------------------------------
# Rig config dataclasses (pyserde — matches COLMAP rig_config.json format)
# ---------------------------------------------------------------------------
@serde
class ColmapRigCameraConfig:
    """Single camera entry in a COLMAP rig_config.json."""

    image_prefix: str
    """Image path prefix for this camera (e.g. ``rig1/cam1/``)."""
    ref_sensor: bool = field(default=False, skip_if=lambda v: not v)
    """Whether this is the reference sensor. Omitted from JSON when False."""


@serde
class ColmapRigConfig:
    """One rig entry in a COLMAP rig_config.json (the file is a list of these)."""

    cameras: list[ColmapRigCameraConfig]
    """Cameras belonging to this rig."""


# ---------------------------------------------------------------------------
# Rig config generation
# ---------------------------------------------------------------------------
def generate_rig_config_json(
    camera_names: list[str],
    rig_name: str,
    ref_camera: str,
    output_path: Path,
) -> None:
    """Write a COLMAP rig_config.json for the discovered cameras.

    The config only declares sensor grouping and the reference sensor; it does
    NOT include ``cam_from_rig`` poses.  Those are derived later by
    :func:`pycolmap.apply_rig_config` from the no-rig bootstrap reconstruction.

    Args:
        camera_names: Sorted list of camera names.
        rig_name: Rig subdirectory name (used in ``image_prefix``).
        ref_camera: Which camera is the reference sensor.
        output_path: Where to write the JSON file.
    """
    cameras: list[ColmapRigCameraConfig] = [
        ColmapRigCameraConfig(image_prefix=f"{rig_name}/{name}/", ref_sensor=(name == ref_camera))
        for name in camera_names
    ]
    rig: ColmapRigConfig = ColmapRigConfig(cameras=cameras)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_json([rig]))
    logger.info(f"Wrote rig config to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_rig_recon(*, config: RigReconConfig, timer: TimingLogger | None = None) -> RigReconResult:
    """Run the full unknown-rig reconstruction pipeline.

    This function is the main orchestrator.  It does NOT call ``rr.log()`` —
    visualization is handled by the CLI / Gradio layer.

    Pipeline:
        1. Discover videos
        2. Extract synchronized frames
        3. Generate rig_config.json
        4. Feature extraction (ALIKED_N16ROT, GPU)
        5. Sequential matching (ALIKED_LIGHTGLUE, GPU) — no rig info
        6. Incremental mapping (no-rig bootstrap)
        7. Apply rig config (auto-derives cam_from_rig from bootstrap)
        8. Rig-aware sequential matching (expand_rig_images)
        9. Global mapping (refine_sensor_from_rig)

    Args:
        config: Pipeline configuration.
        timer: Optional timing logger. When provided, each step is timed
            and the results are logged to Rerun as a markdown table.

    Returns:
        Result containing paths to all outputs.
    """
    # -- Resolve output paths -------------------------------------------------
    output_dir: Path = config.output_dir if config.output_dir is not None else config.videos_dir / "output"
    images_dir: Path = output_dir / "images"
    database_path: Path = output_dir / "database.db"
    no_rig_sparse_dir: Path = output_dir / "sparse" / "no_rig"
    rig_sparse_dir: Path = output_dir / "sparse"

    # Clean stale data from previous runs to prevent mixed-naming collisions.
    if images_dir.exists() or database_path.exists() or rig_sparse_dir.exists():
        logger.info(f"Removing stale output from previous run: {output_dir}")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if database_path.exists():
        database_path.unlink()
    if rig_sparse_dir.exists():
        shutil.rmtree(rig_sparse_dir)

    for d in (images_dir, no_rig_sparse_dir, rig_sparse_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- 1. Discover videos ---------------------------------------------------
    videos: list[tuple[str, Path]] = discover_videos(config.videos_dir)
    camera_names: list[str] = [name for name, _ in videos]
    ref_camera: str = config.ref_camera if config.ref_camera is not None else camera_names[0]
    if ref_camera not in camera_names:
        msg: str = f"ref_camera '{ref_camera}' not among discovered cameras: {camera_names}"
        raise ValueError(msg)
    logger.info(f"Discovered {len(videos)} cameras: {camera_names} (ref={ref_camera})")

    # -- 2. Extract frames ----------------------------------------------------
    with timer.log_time("Frame extraction") if timer else nullcontext():
        num_frames_extracted: int = extract_synchronized_frames(
            videos=videos,
            output_images_dir=images_dir,
            rig_name=config.rig_name,
            num_frames=config.num_frames,
        )

    # -- 3. Generate rig config -----------------------------------------------
    rig_config_path: Path = images_dir / "rig_config.json"
    generate_rig_config_json(
        camera_names=camera_names,
        rig_name=config.rig_name,
        ref_camera=ref_camera,
        output_path=rig_config_path,
    )

    # -- 3b. Read rig config (needed for early apply and step 7) --------------
    rig_configs: list[pycolmap.RigConfig] = pycolmap.read_rig_config(rig_config_path)

    # -- 4. Feature extraction (ALIKED_N16ROT, GPU) ---------------------------
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
        logger.info(
            f"Extracting features (ALIKED_N16ROT, camera_model={config.camera_model}, gpu={config.use_gpu}) ..."
        )
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

    # -- 4b. Streamed extraction validation -----------------------------------
    if config.validate:
        validation_db_path: Path = output_dir / "database_validation.db"
        if validation_db_path.exists():
            validation_db_path.unlink()

        logger.info("Running streamed feature extraction for validation ...")
        pycolmap.set_random_seed(0)  # Reset for reproducibility
        extract_features_streamed(
            database_path=validation_db_path,
            image_path=images_dir,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

        compare_databases(database_path, validation_db_path)
        logger.info("Extraction validation passed — databases are equivalent.")

    # -- 4c. Rig config (frame grouping for cross-camera matching) -----------
    # Populate frame_id/rig_id so expand_rig_images generates cross-camera
    # pairs in the bootstrap matching pass.
    logger.info("Applying rig config: populating frame_id/rig_id for cross-camera matching ...")
    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config(rig_configs, db)

    # -- 5. Sequential matching (ALIKED_LIGHTGLUE, rig-aware) -----------------
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE  # Neural matcher paired with ALIKED features.
    matching_options.use_gpu = config.use_gpu
    matching_options.gpu_index = "0"

    pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = config.overlap  # Match each image against this many neighbors in sequence.
    pairing_options.quadratic_overlap = False  # Skip quadratically-spaced far-apart matches (not needed here).

    # Allow same-timestamp cross-camera pairs and expand matching to all
    # cameras in neighboring rig frames (not just the same camera).
    matching_options.skip_image_pairs_in_same_frame = False
    pairing_options.expand_rig_images = True

    with timer.log_time(f"Sequential matching (overlap={config.overlap})") if timer else nullcontext():
        logger.info(
            f"Sequential matching (ALIKED_LIGHTGLUE, overlap={config.overlap}, expand_rig={pairing_options.expand_rig_images}) ..."
        )
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
        )

    # -- 5b. Clear early rig info before bootstrap mapper ---------------------
    # The incremental mapper rejects reconstructions with rig info but no
    # sensor_from_rig poses.  Clear rigs/frames so the bootstrap runs without
    # rig constraints; step 7 re-applies rig config with derived poses.
    logger.info("Clearing rig/frame info before no-rig bootstrap mapper ...")
    with pycolmap.Database.open(database_path) as db:
        db.clear_rigs()
        db.clear_frames()

    # -- 6. Incremental mapping (no-rig bootstrap) ----------------------------
    # NOTE: GPU BA via cuDSS is not working in this build — using CPU for BA.
    incremental_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions()
    incremental_options.multiple_models = False

    with timer.log_time("Incremental mapping (no-rig bootstrap)") if timer else nullcontext():
        logger.info("Incremental mapping (no-rig bootstrap) ...")
        no_rig_recs: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=no_rig_sparse_dir,
            options=incremental_options,
        )

    if not no_rig_recs:
        msg: str = "No-rig bootstrap failed: COLMAP produced no reconstruction"
        raise RuntimeError(msg)

    # Use the largest reconstruction.
    no_rig_rec_id: int = max(no_rig_recs, key=lambda k: no_rig_recs[k].num_reg_images())
    no_rig_rec: pycolmap.Reconstruction = no_rig_recs[no_rig_rec_id]
    logger.info(f"No-rig bootstrap: {no_rig_rec.num_reg_images()} images registered (model {no_rig_rec_id})")

    # -- 7. Apply rig config (auto-derive cam_from_rig from bootstrap) --------
    logger.info("Applying rig config and deriving cam_from_rig from bootstrap ...")
    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config(rig_configs, db, no_rig_rec)

    # -- 8. Rig-aware sequential matching (expand_rig_images) -----------------
    rig_matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    rig_matching_options.type = (
        pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    )  # Neural matcher paired with ALIKED features.
    rig_matching_options.use_gpu = config.use_gpu
    rig_matching_options.gpu_index = "0"
    # Allow matching images that share the same rig-frame timestamp (cross-camera
    # pairs like cam1/image0001 ↔ cam2/image0001).  Without this, same-frame
    # pairs are skipped and rig cameras would never be directly linked.
    rig_matching_options.skip_image_pairs_in_same_frame = False

    rig_pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    rig_pairing_options.overlap = config.overlap  # Match each image against this many neighbors in sequence.
    rig_pairing_options.quadratic_overlap = False  # Skip quadratically-spaced far-apart matches (not needed here).
    # When matching image N, also match against ALL cameras in neighboring rig
    # frames — not just the same camera.  e.g. cam1/image0005 gets matched
    # against cam2/image0004, cam3/image0006, etc.  This produces the cross-camera
    # pairs that let the global mapper enforce rig constraints.
    rig_pairing_options.expand_rig_images = True

    with timer.log_time("Rig-aware matching (expand_rig_images)") if timer else nullcontext():
        logger.info("Rig-aware sequential matching (expand_rig_images=True) ...")
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=rig_matching_options,
            pairing_options=rig_pairing_options,
        )

    # -- 9. Global mapping (refine_sensor_from_rig) ---------------------------
    global_options: pycolmap.GlobalPipelineOptions = pycolmap.GlobalPipelineOptions()
    global_options.mapper.bundle_adjustment.refine_sensor_from_rig = True

    with timer.log_time("Global mapping (refine_sensor_from_rig)") if timer else nullcontext():
        logger.info("Global mapping (refine_sensor_from_rig=True) ...")
        rig_recs: dict[int, pycolmap.Reconstruction] = pycolmap.global_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=rig_sparse_dir,
            options=global_options,
        )

    if not rig_recs:
        msg = "Rig-aware global mapping failed: COLMAP produced no reconstruction"
        raise RuntimeError(msg)

    rig_rec_id: int = max(rig_recs, key=lambda k: rig_recs[k].num_reg_images())
    rig_rec: pycolmap.Reconstruction = rig_recs[rig_rec_id]
    logger.info(f"Final rig reconstruction: {rig_rec.num_reg_images()} images registered (model {rig_rec_id})")

    if timer:
        timer.log_total()

    # -- Done -----------------------------------------------------------------
    return RigReconResult(
        output_dir=output_dir,
        images_dir=images_dir,
        database_path=database_path,
        no_rig_model_dir=no_rig_sparse_dir / str(no_rig_rec_id),
        rig_model_dir=rig_sparse_dir / str(rig_rec_id),
        num_frames_extracted=num_frames_extracted,
        num_cameras=len(camera_names),
        camera_names=camera_names,
        ref_camera=ref_camera,
        overlap=config.overlap,
        image_plane_distance=config.image_plane_distance,
    )


# ---------------------------------------------------------------------------
# Blueprint builder (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def create_rig_blueprint_tabs(
    parent_log_path: Path,
    camera_names: list[str],
    *,
    active_tab: int = 0,
    timeline: str = TIMELINE,
) -> rrb.Blueprint:
    """Create a tabbed Rerun blueprint for multi-camera rig visualization.

    Three tabs:
        0 — **Features**: per-frame 2D keypoints for the reference camera + timing
        1 — **Matches**: side-by-side match pairs + timing
        2 — **Reconstruction**: 3D + 2D views of the final model + timing

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).
        camera_names: List of camera names in the rig.
        active_tab: Index of the tab to display initially.
        timeline: Which timeline the time panel should display.

    Returns:
        Blueprint with tabbed layout.
    """
    ref_cam: str = camera_names[0] if camera_names else "cam1"

    # -- Tab 0: Features -------------------------------------------------------
    features_tab: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/features/{ref_cam}",
                name=f"Features ({ref_cam})",
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
                    rrb.Spatial2DView(origin=f"{parent_log_path}/{ref_cam}/pinhole"),
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
# Rerun visualization (used by CLI and Gradio)
# ---------------------------------------------------------------------------
def log_rig_reconstruction(result: RigReconResult, parent_log_path: Path) -> None:
    """Load the final rig reconstruction and log it to Rerun.

    Computes a gravity-alignment transform from the camera poses using
    ``auto_orient_and_center_poses`` (method="up") and logs it as a static
    ``Transform3D`` on ``parent_log_path``.

    The 3D point cloud is logged **incrementally**: each frame adds the
    points visible from all cameras at that frame so the cloud grows as
    you scrub through the timeline.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    from pysfm.apis.sfm_reconstruction import _extract_intrinsics, _extract_visible_keypoints

    rec: pycolmap.Reconstruction = pycolmap.Reconstruction(str(result.rig_model_dir))

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

    # -- Group images by camera -----------------------------------------------
    cam_images: dict[str, list[tuple[int, pycolmap.Image]]] = {}
    for image_id, image in sorted(rec.images.items()):
        parts: list[str] = image.name.split("/")
        cam_name: str = parts[-2] if len(parts) >= 2 else "unknown"
        if cam_name not in cam_images:
            cam_images[cam_name] = []
        cam_images[cam_name].append((image_id, image))

    for cam_name in cam_images:
        cam_images[cam_name].sort(key=lambda x: x[1].name)

    max_frames: int = max((len(imgs) for imgs in cam_images.values()), default=0)

    # -- Log cameras + incremental point cloud over timeline -------------------
    seen_point_ids: set[int] = set()
    accumulated_xyz: list[Float64[ndarray, "3"]] = []
    accumulated_rgb: list[UInt8[ndarray, "3"]] = []

    for frame_idx in range(max_frames):
        rr.set_time(TIMELINE, sequence=frame_idx)

        for cam_name in result.camera_names:
            if cam_name not in cam_images or frame_idx >= len(cam_images[cam_name]):
                continue

            _image_id: int
            image: pycolmap.Image
            _image_id, image = cam_images[cam_name][frame_idx]
            camera_path: str = f"{parent_log_path}/{cam_name}"

            # Transform (world_T_cam)
            world_from_cam: pycolmap.Rigid3d = image.cam_from_world().inverse()
            world_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
            world_T_cam[:3, :3] = world_from_cam.rotation.matrix()
            world_T_cam[:3, 3] = world_from_cam.translation

            rr.log(camera_path, rr.Transform3D(mat3x3=world_T_cam[:3, :3], translation=world_T_cam[:3, 3]))

            # Pinhole camera — ref camera is green, others are gray.
            cam: pycolmap.Camera = image.camera
            K: Float64[ndarray, "3 3"] = _extract_intrinsics(cam)
            frustum_color: list[int] = [0, 200, 0] if cam_name == result.ref_camera else [200, 200, 200]
            rr.log(
                f"{camera_path}/pinhole",
                rr.Pinhole(
                    image_from_camera=K,
                    width=cam.width,
                    height=cam.height,
                    image_plane_distance=result.image_plane_distance,
                    color=frustum_color,
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

        # Log the growing point cloud (once per frame, after all cameras)
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
def log_rig_features(result: RigReconResult, parent_log_path: Path) -> None:
    """Log per-frame per-camera images with all detected keypoints from the database.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    with pycolmap.Database.open(result.database_path) as db:
        all_images: list[pycolmap.Image] = db.read_all_images()

        # Group by camera (image names like "rig1/cam1/image0001.jpg")
        cam_db_images: dict[str, list[pycolmap.Image]] = {}
        for db_image in all_images:
            parts: list[str] = db_image.name.split("/")
            cam_name: str = parts[-2] if len(parts) >= 2 else "unknown"
            if cam_name not in cam_db_images:
                cam_db_images[cam_name] = []
            cam_db_images[cam_name].append(db_image)

        for cam_name in cam_db_images:
            cam_db_images[cam_name].sort(key=lambda img: img.name)

        max_frames: int = max((len(imgs) for imgs in cam_db_images.values()), default=0)

        for frame_idx in range(max_frames):
            rr.set_time(TIMELINE, sequence=frame_idx)

            for cam_name in result.camera_names:
                if cam_name not in cam_db_images or frame_idx >= len(cam_db_images[cam_name]):
                    continue

                db_image: pycolmap.Image = cam_db_images[cam_name][frame_idx]
                image_path: Path = result.images_dir / db_image.name
                if not image_path.exists():
                    continue
                bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb_img: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rr.log(
                    f"{parent_log_path}/features/{cam_name}/image",
                    rr.Image(rgb_img, color_model=rr.ColorModel.RGB).compress(),
                )

                kps: Float32[ndarray, "N 6"] = db.read_keypoints(db_image.image_id)
                if kps.shape[0] > 0:
                    kps_xy: Float32[ndarray, "N 2"] = kps[:, :2]
                    rr.log(
                        f"{parent_log_path}/features/{cam_name}/image/keypoints",
                        rr.Points2D(positions=kps_xy, colors=[255, 165, 0]),
                    )

    logger.info(f"Logged features for {max_frames} frames across {len(cam_db_images)} cameras")


# ---------------------------------------------------------------------------
# Match visualization
# ---------------------------------------------------------------------------
def log_rig_matches(
    result: RigReconResult,
    parent_log_path: Path,
    *,
    max_dim: int = 640,
) -> None:
    """Log feature matches as side-by-side image pairs with connecting lines.

    Uses ``rr.send_columns`` on a separate ``"match_pair"`` timeline.
    Includes both same-camera sequential pairs and cross-camera rig pairs.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
        max_dim: Maximum dimension for resized stacked images.
    """
    with pycolmap.Database.open(result.database_path) as db:
        all_images: list[pycolmap.Image] = db.read_all_images()
        sorted_images: list[pycolmap.Image] = sorted(all_images, key=lambda img: img.name)
        num_images: int = len(sorted_images)

        pair_indices: list[int] = []
        jpeg_blobs: list[bytes] = []
        kp_left_list: list[Float32[ndarray, "M 2"]] = []
        kp_right_list: list[Float32[ndarray, "M 2"]] = []
        strip_list: list[list[Float32[ndarray, "2 2"]]] = []
        pair_idx: int = 0

        for i in range(num_images):
            img_a: pycolmap.Image = sorted_images[i]
            kps_a: Float32[ndarray, "Na 6"] = db.read_keypoints(img_a.image_id)

            bgr_a: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(result.images_dir / img_a.name), cv2.IMREAD_COLOR)
            if bgr_a is None:
                continue

            for j in range(i + 1, min(i + result.overlap + 1, num_images)):
                img_b: pycolmap.Image = sorted_images[j]

                tvg: pycolmap.TwoViewGeometry = db.read_two_view_geometry(img_a.image_id, img_b.image_id)
                inlier_matches: np.ndarray = tvg.inlier_matches
                if inlier_matches.shape[0] == 0:
                    continue

                kps_b: Float32[ndarray, "Nb 6"] = db.read_keypoints(img_b.image_id)
                bgr_b: UInt8[ndarray, "H W 3"] | None = cv2.imread(
                    str(result.images_dir / img_b.name), cv2.IMREAD_COLOR
                )
                if bgr_b is None:
                    continue

                h_a: int = bgr_a.shape[0]
                w_a: int = bgr_a.shape[1]
                h_b: int = bgr_b.shape[0]
                w_b: int = bgr_b.shape[1]

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

                sh: int = stacked.shape[0]
                sw: int = stacked.shape[1]
                resize_scale: float = max_dim / max(sh, sw) if max(sh, sw) > max_dim else 1.0
                if resize_scale < 1.0:
                    stacked = cv2.resize(stacked, (int(sw * resize_scale), int(sh * resize_scale)))

                ok: bool
                buf: np.ndarray
                ok, buf = cv2.imencode(".jpg", stacked)
                if not ok:
                    continue
                jpeg_blobs.append(buf.tobytes())

                idx_a: np.ndarray = inlier_matches[:, 0]
                idx_b: np.ndarray = inlier_matches[:, 1]

                pts_a: Float32[ndarray, "M 2"] = kps_a[idx_a, :2] * scale_a * resize_scale
                pts_b: Float32[ndarray, "M 2"] = kps_b[idx_b, :2] * scale_b * resize_scale
                pts_b_offset: Float32[ndarray, "M 2"] = pts_b.copy()
                pts_b_offset[:, 0] += w_left * resize_scale

                kp_left_list.append(pts_a.astype(np.float32))
                kp_right_list.append(pts_b_offset.astype(np.float32))

                num_matches: int = pts_a.shape[0]
                strips: list[Float32[ndarray, "2 2"]] = [
                    np.stack([pts_a[k], pts_b_offset[k]]).astype(np.float32) for k in range(num_matches)
                ]
                strip_list.append(strips)

                pair_indices.append(pair_idx)
                pair_idx += 1

    if not pair_indices:
        logger.warning("No match pairs found to visualize")
        return

    num_pairs: int = len(pair_indices)
    logger.info(f"Sending {num_pairs} match pairs via send_columns ...")

    pair_indices_arr: np.ndarray = np.array(pair_indices, dtype=np.int64)

    rr.send_columns(
        f"{parent_log_path}/matches/image",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=rr.EncodedImage.columns(blob=jpeg_blobs, media_type=["image/jpeg"] * num_pairs),
    )

    all_kp_left: Float32[ndarray, "T 2"] = np.concatenate(kp_left_list, axis=0)
    rr.send_columns(
        f"{parent_log_path}/matches/image/kp_left",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=rr.Points2D.columns(
            positions=all_kp_left,
            colors=np.full((all_kp_left.shape[0], 3), [0, 255, 0], dtype=np.uint8),
        ).partition(lengths=[kp.shape[0] for kp in kp_left_list]),
    )

    all_kp_right: Float32[ndarray, "T 2"] = np.concatenate(kp_right_list, axis=0)
    rr.send_columns(
        f"{parent_log_path}/matches/image/kp_right",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=rr.Points2D.columns(
            positions=all_kp_right,
            colors=np.full((all_kp_right.shape[0], 3), [0, 255, 0], dtype=np.uint8),
        ).partition(lengths=[kp.shape[0] for kp in kp_right_list]),
    )

    all_strips: list[Float32[ndarray, "2 2"]] = []
    strip_counts: list[int] = []
    for strips in strip_list:
        all_strips.extend(strips)
        strip_counts.append(len(strips))

    rr.send_columns(
        f"{parent_log_path}/matches/image/lines",
        indexes=[rr.TimeColumn(MATCH_TIMELINE, sequence=pair_indices_arr)],
        columns=rr.LineStrips2D.columns(
            strips=all_strips,
            colors=np.full((len(all_strips), 3), [0, 255, 0], dtype=np.uint8),
        ).partition(lengths=strip_counts),
    )

    logger.info(f"Logged {num_pairs} match pairs")


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------
@dataclass
class RigReconCLIConfig:
    """CLI configuration for unknown-rig reconstruction with Rerun visualization."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration (--save, --connect, --spawn)."""
    config: RigReconConfig = field(default_factory=RigReconConfig)
    """Reconstruction pipeline configuration."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(cli_config: RigReconCLIConfig) -> None:
    """Run the unknown-rig pipeline and visualize results in Rerun.

    Sends the blueprint multiple times to switch the active tab as
    features → matches → reconstruction are progressively logged.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parent_log_path: Path = Path("world")

    # 1. Initial blueprint — Features tab shows timing updates live
    # (camera_names not known yet, use placeholder — will be updated after pipeline)
    rr.log("/", rr.ViewCoordinates.RFU, static=True)

    # 2. Run the full pipeline (computation only, timed)
    timer: TimingLogger = TimingLogger(header="Unknown-Rig Reconstruction")
    result: RigReconResult = run_rig_recon(config=cli_config.config, timer=timer)

    # Now we know camera names — send proper blueprint
    rr.send_blueprint(create_rig_blueprint_tabs(parent_log_path, result.camera_names, active_tab=0))

    # 3. Log features → stay on Features tab
    log_rig_features(result, parent_log_path)

    # 4. Log matches → switch to Matches tab + match_pair timeline
    rr.send_blueprint(
        create_rig_blueprint_tabs(parent_log_path, result.camera_names, active_tab=1, timeline=MATCH_TIMELINE)
    )
    log_rig_matches(result, parent_log_path)

    # 5. Log reconstruction → switch to Reconstruction tab + frame timeline
    rr.send_blueprint(create_rig_blueprint_tabs(parent_log_path, result.camera_names, active_tab=2, timeline=TIMELINE))
    log_rig_reconstruction(result, parent_log_path)

    print(f"Output: {result.output_dir}")
