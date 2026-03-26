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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import pycolmap
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from serde import field, serde
from serde.json import to_json
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import MultiVideoReader

from pysfm.streamed_pipeline import compare_databases, extract_features_streamed

logger: logging.Logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv"})
"""Recognized video file extensions."""

TIMELINE: str = "frame"
"""Timeline name used for sequential camera logging."""

SfMCameraModel: TypeAlias = Literal["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]
"""Supported COLMAP camera model names."""


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
    camera_model: SfMCameraModel = "OPENCV"
    """COLMAP camera model for intrinsics estimation (e.g. OPENCV_FISHEYE for fisheye lenses)."""
    overlap: int = 5
    """Sequential matching overlap (number of neighboring frames to match)."""
    use_gpu: bool = True
    """Use GPU for feature extraction and matching."""
    verbose: bool = False
    """Emit detailed COLMAP logging."""
    validate: bool = False
    """When True, run streamed implementations of pipeline stages alongside
    the black-box pycolmap calls and assert equivalence.  Increases runtime
    proportionally — intended for development and testing."""
    early_rig_config: bool = True
    """Apply rig frame grouping before the no-rig bootstrap matching.
    Populates frame_id/rig_id in the database so expand_rig_images generates
    cross-camera pairs in the bootstrap pass. Set to False for legacy behavior
    (boundary-only cross-camera pairs)."""


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
        "Extracting %d frames from %d videos (%d synchronized frames available)",
        actual_num_frames,
        len(videos),
        total_frames,
    )

    # Create output directories for each camera.
    cam_dirs: list[Path] = []
    for cam_name in camera_names:
        cam_dir: Path = output_images_dir / rig_name / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        cam_dirs.append(cam_dir)

    # Extract frames using random access (MultiVideoReader.__getitem__).
    for out_idx, frame_idx in enumerate(frame_indices):
        bgr_list: list[UInt8[ndarray, "H W 3"]] = reader[frame_idx]
        filename: str = f"image{out_idx + 1:0{pad_width}d}.jpg"
        for cam_idx, bgr in enumerate(bgr_list):
            cv2.imwrite(str(cam_dirs[cam_idx] / filename), bgr)

    logger.info("Wrote frames to %s", output_images_dir / rig_name)
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
    logger.info("Wrote rig config to %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_rig_recon(*, config: RigReconConfig) -> RigReconResult:
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

    Returns:
        Result containing paths to all outputs.
    """
    # -- Resolve output paths -------------------------------------------------
    output_dir: Path = config.output_dir if config.output_dir is not None else config.videos_dir / "output"
    images_dir: Path = output_dir / "images"
    database_path: Path = output_dir / "database.db"
    no_rig_sparse_dir: Path = output_dir / "sparse" / "no_rig"
    rig_sparse_dir: Path = output_dir / "sparse"

    for d in (images_dir, no_rig_sparse_dir, rig_sparse_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- 1. Discover videos ---------------------------------------------------
    videos: list[tuple[str, Path]] = discover_videos(config.videos_dir)
    camera_names: list[str] = [name for name, _ in videos]
    ref_camera: str = config.ref_camera if config.ref_camera is not None else camera_names[0]
    if ref_camera not in camera_names:
        msg: str = f"ref_camera '{ref_camera}' not among discovered cameras: {camera_names}"
        raise ValueError(msg)
    logger.info("Discovered %d cameras: %s (ref=%s)", len(videos), camera_names, ref_camera)

    # -- 2. Extract frames ----------------------------------------------------
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

    logger.info("Extracting features (ALIKED_N16ROT, camera_model=%s, gpu=%s) ...", config.camera_model, config.use_gpu)
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

    # -- 4c. Early rig config (frame grouping for cross-camera matching) ------
    if config.early_rig_config:
        logger.info("Early rig config: populating frame_id/rig_id for cross-camera matching ...")
        with pycolmap.Database.open(database_path) as db:
            pycolmap.apply_rig_config(rig_configs, db)

    # -- 5. Sequential matching (ALIKED_LIGHTGLUE, no rig) --------------------
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE  # Neural matcher paired with ALIKED features.
    matching_options.use_gpu = config.use_gpu
    matching_options.gpu_index = "0"

    pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = config.overlap  # Match each image against this many neighbors in sequence.
    pairing_options.quadratic_overlap = False  # Skip quadratically-spaced far-apart matches (not needed here).

    if config.early_rig_config:
        # Allow same-timestamp cross-camera pairs and expand matching to all
        # cameras in neighboring rig frames (not just the same camera).
        matching_options.skip_image_pairs_in_same_frame = False
        pairing_options.expand_rig_images = True

    logger.info(
        "Sequential matching (ALIKED_LIGHTGLUE, overlap=%d, expand_rig=%s) ...",
        config.overlap,
        pairing_options.expand_rig_images,
    )
    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=matching_options,
        pairing_options=pairing_options,
    )

    # -- 6. Incremental mapping (no-rig bootstrap) ----------------------------
    # NOTE: GPU BA via cuDSS is not working in this build — using CPU for BA.
    incremental_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions()
    incremental_options.multiple_models = False

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
    logger.info("No-rig bootstrap: %d images registered (model %d)", no_rig_rec.num_reg_images(), no_rig_rec_id)

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

    logger.info("Rig-aware sequential matching (expand_rig_images=True) ...")
    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=rig_matching_options,
        pairing_options=rig_pairing_options,
    )

    # -- 9. Global mapping (refine_sensor_from_rig) ---------------------------
    global_options: pycolmap.GlobalPipelineOptions = pycolmap.GlobalPipelineOptions()
    global_options.mapper.bundle_adjustment.refine_sensor_from_rig = True

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
    logger.info("Final rig reconstruction: %d images registered (model %d)", rig_rec.num_reg_images(), rig_rec_id)

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
    )


# ---------------------------------------------------------------------------
# Blueprint builder (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def create_rig_blueprint(parent_log_path: Path, camera_names: list[str]) -> rrb.ContainerLike:
    """Create a Rerun blueprint for multi-camera rig visualization.

    Shows a 3D spatial view with the full point cloud and per-camera frustums,
    plus a 2D view for scrubbing through frames.

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).
        camera_names: List of camera names in the rig.

    Returns:
        Rerun blueprint layout.
    """
    # Origin must be an ancestor of parent_log_path (not parent_log_path
    # itself) so that the gravity-alignment Transform3D logged on
    # parent_log_path is visible in the view.
    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        origin="/",
        contents=[f"+ {parent_log_path}/**"],
    )

    # 2D view for the first camera — user can switch via entity panel.
    first_cam: str = camera_names[0] if camera_names else "cam1"
    view_2d: rrb.Spatial2DView = rrb.Spatial2DView(
        origin=f"{parent_log_path}/{first_cam}/pinhole",
    )

    return rrb.Horizontal(contents=[view_3d, view_2d], column_shares=[3, 2])


# ---------------------------------------------------------------------------
# Rerun visualization (used by CLI and Gradio)
# ---------------------------------------------------------------------------
def log_rig_reconstruction(result: RigReconResult, parent_log_path: Path) -> None:
    """Load the final rig reconstruction and log it to Rerun.

    Computes a gravity-alignment transform from the camera poses using
    ``auto_orient_and_center_poses`` (method="up") and logs it as a static
    ``Transform3D`` on ``parent_log_path``.  Rerun's transform hierarchy then
    applies it to the entire scene (point cloud, camera frustums, images,
    keypoints).

    .. note::

        The Spatial3DView **must** have its ``origin`` set to an ancestor of
        ``parent_log_path`` (e.g. ``"/"``) for the gravity-alignment transform
        to be visible.  If ``origin`` equals ``parent_log_path``, the view
        renders from that entity's perspective and the transform is invisible.

    Logs 3D point cloud (static), then per-camera frustums, images, and
    keypoints over a timeline.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    import rerun as rr

    from pysfm.apis.sfm_reconstruction import _extract_intrinsics, _extract_visible_keypoints

    rec: pycolmap.Reconstruction = pycolmap.Reconstruction(str(result.rig_model_dir))

    # -- Compute gravity-alignment transform from camera poses ----------------
    # Extract all world_T_cam poses, convert CV→GL (required by
    # auto_orient_and_center_poses), then log the resulting rotation+centering
    # as a static Transform3D on parent_log_path so that Rerun's hierarchy
    # applies it to all children (point cloud, cameras, etc.).
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
        logger.info("Gravity-alignment transform logged on '%s' (det=%.4f)", parent_log_path, np.linalg.det(orient_R))

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

    # -- Group images by camera -----------------------------------------------
    #    Image names look like "rig1/cam1/image0001.jpg".
    cam_images: dict[str, list[tuple[int, pycolmap.Image]]] = {}
    for image_id, image in sorted(rec.images.items()):
        parts: list[str] = image.name.split("/")
        cam_name: str = parts[-2] if len(parts) >= 2 else "unknown"
        if cam_name not in cam_images:
            cam_images[cam_name] = []
        cam_images[cam_name].append((image_id, image))

    # Sort each camera's images by filename for consistent frame ordering.
    for cam_name in cam_images:
        cam_images[cam_name].sort(key=lambda x: x[1].name)

    max_frames: int = max((len(imgs) for imgs in cam_images.values()), default=0)

    # -- Log per-camera data over timeline ------------------------------------
    for frame_idx in range(max_frames):
        rr.set_time(TIMELINE, sequence=frame_idx)

        for cam_name in result.camera_names:
            if cam_name not in cam_images or frame_idx >= len(cam_images[cam_name]):
                continue

            _image_id: int
            image: pycolmap.Image
            _image_id, image = cam_images[cam_name][frame_idx]
            camera_path: str = f"{parent_log_path}/{cam_name}"

            # Transform (world_T_cam) — raw COLMAP pose; gravity-alignment
            # is handled by the parent Transform3D on parent_log_path.
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
                    image_plane_distance=1.0,
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
                        radii=np.full(keypoints_xy.shape[0], 2.0, dtype=np.float32),
                        colors=np.full((keypoints_xy.shape[0], 3), [34, 138, 167], dtype=np.uint8),
                    ),
                )


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
    """Run the unknown-rig pipeline and visualize results in Rerun."""
    import rerun as rr

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parent_log_path: Path = Path("world")

    # 1. Run the full pipeline
    result: RigReconResult = run_rig_recon(config=cli_config.config)

    # 2. Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_rig_blueprint(parent_log_path=parent_log_path, camera_names=result.camera_names),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    # Gravity-aligned data has Z-up; tell the viewer accordingly.
    rr.log("/", rr.ViewCoordinates.RFU, static=True)

    # 3. Log the reconstruction
    log_rig_reconstruction(result, parent_log_path)

    print(f"Output: {result.output_dir}")
