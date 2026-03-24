"""Unknown-rig multi-camera reconstruction from videos.

End-to-end pipeline: discovers videos in a directory, extracts synchronized
frames, runs ALIKED+LightGlue feature extraction and matching, bootstraps a
no-rig incremental reconstruction, estimates rig calibration from that
bootstrap, then produces a final rig-aware global reconstruction.

Uses the pycolmap Python API throughout — no subprocess calls to COLMAP CLI.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import cv2
import numpy as np
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

if TYPE_CHECKING:
    import pycolmap
    import rerun.blueprint as rrb

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
    """COLMAP camera model for intrinsics estimation."""
    overlap: int = 5
    """Sequential matching overlap (number of neighboring frames to match)."""
    use_gpu: bool = True
    """Use GPU for feature extraction and matching."""
    verbose: bool = False
    """Emit detailed COLMAP logging."""


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
    # Determine frame count from the shortest video.
    total_frames_per_video: list[int] = []
    for _cam_name, video_path in videos:
        cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            msg: str = f"Cannot open video: {video_path}"
            raise RuntimeError(msg)
        total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            msg = f"Video has zero frames: {video_path}"
            raise RuntimeError(msg)
        total_frames_per_video.append(total)
        cap.release()

    min_total: int = min(total_frames_per_video)
    actual_num_frames: int = min(num_frames, min_total)
    frame_indices: list[int] = np.linspace(0, min_total - 1, actual_num_frames, dtype=int).tolist()
    pad_width: int = len(str(actual_num_frames))

    logger.info(
        "Extracting %d frames from %d videos (shortest has %d frames)", actual_num_frames, len(videos), min_total
    )

    for cam_name, video_path in videos:
        cam_dir: Path = output_images_dir / rig_name / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        for out_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret: bool
            frame: UInt8[ndarray, "H W 3"]
            ret, frame = cap.read()
            if not ret:
                msg = f"Failed to read frame {frame_idx} from {video_path}"
                raise RuntimeError(msg)

            # One-based, zero-padded filename: image0001.jpg, image0002.jpg, ...
            filename: str = f"image{out_idx + 1:0{pad_width}d}.jpg"
            cv2.imwrite(str(cam_dir / filename), frame)
        cap.release()

    logger.info("Wrote frames to %s", output_images_dir / rig_name)
    return actual_num_frames


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
    cameras: list[dict[str, str | bool]] = []
    for name in camera_names:
        entry: dict[str, str | bool] = {"image_prefix": f"{rig_name}/{name}/"}
        if name == ref_camera:
            entry["ref_sensor"] = True
        cameras.append(entry)

    config: list[dict[str, list[dict[str, str | bool]]]] = [{"cameras": cameras}]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
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
    import pycolmap

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

    # -- 4. Feature extraction (ALIKED_N16ROT, GPU) ---------------------------
    pycolmap.set_random_seed(0)

    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    extraction_options.use_gpu = config.use_gpu
    extraction_options.gpu_index = "0"

    logger.info("Extracting features (ALIKED_N16ROT, gpu=%s) ...", config.use_gpu)
    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        extraction_options=extraction_options,
    )

    # -- 5. Sequential matching (ALIKED_LIGHTGLUE, no rig) --------------------
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    matching_options.use_gpu = config.use_gpu
    matching_options.gpu_index = "0"

    pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = config.overlap
    pairing_options.quadratic_overlap = False

    logger.info("Sequential matching (ALIKED_LIGHTGLUE, overlap=%d, no rig) ...", config.overlap)
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
    no_rig_rec: pycolmap.Reconstruction = max(no_rig_recs.values(), key=lambda r: r.num_reg_images())
    logger.info("No-rig bootstrap: %d images registered", no_rig_rec.num_reg_images())

    # -- 7. Apply rig config (auto-derive cam_from_rig from bootstrap) --------
    rig_configs: list[pycolmap.RigConfig] = pycolmap.read_rig_config(rig_config_path)

    logger.info("Applying rig config and deriving cam_from_rig from bootstrap ...")
    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config(rig_configs, db, no_rig_rec)

    # -- 8. Rig-aware sequential matching (expand_rig_images) -----------------
    rig_matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    rig_matching_options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    rig_matching_options.use_gpu = config.use_gpu
    rig_matching_options.gpu_index = "0"
    rig_matching_options.skip_image_pairs_in_same_frame = False

    rig_pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    rig_pairing_options.overlap = config.overlap
    rig_pairing_options.quadratic_overlap = False
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

    rig_rec: pycolmap.Reconstruction = max(rig_recs.values(), key=lambda r: r.num_reg_images())
    logger.info("Final rig reconstruction: %d images registered", rig_rec.num_reg_images())

    # -- Done -----------------------------------------------------------------
    return RigReconResult(
        output_dir=output_dir,
        images_dir=images_dir,
        database_path=database_path,
        no_rig_model_dir=no_rig_sparse_dir / "0",
        rig_model_dir=rig_sparse_dir / "0",
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
    import rerun.blueprint as rrb

    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=["+ $origin/**"],
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

    Logs 3D point cloud (static), then per-camera frustums, images, and
    keypoints over a timeline.

    Args:
        result: Pipeline result with paths to outputs.
        parent_log_path: Root Rerun log path.
    """
    import pycolmap
    import rerun as rr

    from pysfm.apis.sfm_reconstruction import _extract_intrinsics, _extract_visible_keypoints

    rec: pycolmap.Reconstruction = pycolmap.Reconstruction(str(result.rig_model_dir))

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
                # radii=np.full(points3d_xyz.shape[0], 0.005, dtype=np.float32),
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

            # Transform (world_T_cam)
            world_from_cam: pycolmap.Rigid3d = image.cam_from_world().inverse()
            R: Float64[ndarray, "3 3"] = world_from_cam.rotation.matrix()
            t: Float64[ndarray, "3"] = world_from_cam.translation

            world_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
            world_T_cam[:3, :3] = R
            world_T_cam[:3, 3] = t

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
    import rerun.blueprint as rrb

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
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # 3. Log the reconstruction
    log_rig_reconstruction(result, parent_log_path)

    print(f"Output: {result.output_dir}")
