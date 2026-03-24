"""SfM reconstruction node.

Wraps pycolmap's incremental Structure-from-Motion pipeline to produce a 3D
reconstruction (sparse point cloud + camera poses) from a directory of images.
Uses the high-level pycolmap API: feature extraction → exhaustive matching →
incremental mapping.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

if TYPE_CHECKING:
    import pycolmap
    import rerun.blueprint as rrb

TIMELINE: str = "frame"
"""Timeline name used for sequential camera logging."""

SfMCameraModel: TypeAlias = Literal["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]
"""Supported COLMAP camera model names."""


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class SfMConfig:
    """Configuration for COLMAP SfM reconstruction."""

    camera_model: SfMCameraModel = "OPENCV"
    """COLMAP camera model name (SIMPLE_PINHOLE, PINHOLE, OPENCV, etc.)."""
    random_seed: int = 0
    """Random seed for reproducibility."""
    verbose: bool = False
    """Emit detailed logging when True."""


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------
@dataclass
class SfMImageResult:
    """Per-image reconstruction output."""

    world_T_cam: Float64[ndarray, "4 4"]
    """world_T_cam transformation matrix."""
    intrinsics: Float64[ndarray, "3 3"]
    """Camera intrinsic matrix (K)."""
    image_size: tuple[int, int]
    """(width, height) of the image."""
    image_name: str
    """Filename of the registered image."""
    keypoints_xy: Float32[ndarray, "M 2"]
    """2D keypoint positions (x, y) in pixel coordinates for visible 3D points."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SfMResult:
    """Output of COLMAP SfM reconstruction."""

    points3d_xyz: Float32[ndarray, "N 3"]
    """Reconstructed 3D point positions."""
    points3d_rgb: UInt8[ndarray, "N 3"]
    """Per-point RGB colors."""
    images: list[SfMImageResult]
    """Per-image results (poses, intrinsics, keypoints), sorted by COLMAP image_id."""
    num_images_registered: int
    """Number of successfully registered images."""


def _extract_intrinsics(camera: pycolmap.Camera) -> Float64[ndarray, "3 3"]:
    """Extract a 3x3 intrinsic matrix from a pycolmap Camera.

    Handles SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL (fx=fy, cx, cy)
    and PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV (fx, fy, cx, cy).

    Args:
        camera: pycolmap Camera object.

    Returns:
        3x3 intrinsic matrix K.
    """
    import pycolmap

    K: Float64[ndarray, "3 3"] = np.eye(3, dtype=np.float64)

    if camera.model in (
        pycolmap.CameraModelId.SIMPLE_PINHOLE,
        pycolmap.CameraModelId.SIMPLE_RADIAL,
        pycolmap.CameraModelId.RADIAL,
    ):
        fx: float = camera.params[0]
        fy: float = camera.params[0]
        cx: float = camera.params[1]
        cy: float = camera.params[2]
    elif camera.model in (
        pycolmap.CameraModelId.PINHOLE,
        pycolmap.CameraModelId.OPENCV,
        pycolmap.CameraModelId.OPENCV_FISHEYE,
        pycolmap.CameraModelId.FULL_OPENCV,
    ):
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
    else:
        msg: str = f"Unsupported camera model: {camera.model}"
        raise ValueError(msg)

    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _extract_visible_keypoints(
    image: pycolmap.Image,
    points3d: pycolmap.Point3DMap,
) -> Float32[ndarray, "M 2"]:
    """Extract 2D keypoint positions for points that have a valid 3D match.

    Args:
        image: pycolmap Image with points2D.
        points3d: Reconstruction's points3D dict for existence checks.

    Returns:
        Array of (x, y) pixel coordinates for visible keypoints.
    """
    xys: list[Float32[ndarray, "2"]] = []
    for p2d in image.points2D:
        if p2d.has_point3D() and p2d.point3D_id in points3d:
            xys.append(np.array(p2d.xy, dtype=np.float32))

    if not xys:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(xys)


# ---------------------------------------------------------------------------
# Core pipeline function (NO Rerun, NO file I/O beyond what pycolmap needs)
# ---------------------------------------------------------------------------
def run_sfm(
    *,
    image_dir: Path,
    config: SfMConfig,
) -> SfMResult:
    """Run COLMAP incremental SfM reconstruction.

    This function is pure computation (from the node's perspective). It does NOT
    call ``rr.log()``. The caller (CLI or Gradio) handles visualization.

    Note: pycolmap's high-level API requires an image directory on disk. A future
    iteration will use lower-level APIs (``pycolmap.Sift``, ``pycolmap.Database``)
    to accept numpy arrays directly.

    Args:
        image_dir: Directory containing input images.
        config: SfM configuration.

    Returns:
        SfMResult with reconstruction outputs.
    """
    import pycolmap

    tmp_dir: str = tempfile.mkdtemp(prefix="pysfm_")
    database_path: Path = Path(tmp_dir) / "database.db"
    sfm_path: Path = Path(tmp_dir) / "sfm"
    sfm_path.mkdir(exist_ok=True)

    pycolmap.set_random_seed(config.random_seed)

    if config.verbose:
        pycolmap.logging.set_log_destination(pycolmap.logging.INFO, Path(tmp_dir) / "INFO.log.")

    # Feature extraction
    pycolmap.extract_features(database_path, image_dir)

    # Exhaustive matching
    pycolmap.match_exhaustive(database_path)

    # Incremental mapping
    reconstructions: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
        database_path, image_dir, sfm_path
    )

    if not reconstructions:
        msg: str = "COLMAP failed to produce any reconstruction"
        raise RuntimeError(msg)

    # Use the largest reconstruction (most registered images)
    rec: pycolmap.Reconstruction = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    # Extract 3D points
    xyz_list: list[Float64[ndarray, "3"]] = []
    rgb_list: list[UInt8[ndarray, "3"]] = []
    for point in rec.points3D.values():
        xyz_list.append(point.xyz)
        rgb_list.append(point.color)

    points3d_xyz: Float32[ndarray, "N 3"] = (
        np.array(xyz_list, dtype=np.float32) if xyz_list else np.zeros((0, 3), dtype=np.float32)
    )
    points3d_rgb: UInt8[ndarray, "N 3"] = (
        np.array(rgb_list, dtype=np.uint8) if rgb_list else np.zeros((0, 3), dtype=np.uint8)
    )

    # Extract per-image data (sorted by image_id for determinism)
    image_results: list[SfMImageResult] = []
    sorted_images: list[tuple[int, pycolmap.Image]] = sorted(rec.images.items())
    for _image_id, image in sorted_images:
        # world_T_cam = inverse of cam_from_world
        world_from_cam: pycolmap.Rigid3d = image.cam_from_world().inverse()
        R: Float64[ndarray, "3 3"] = world_from_cam.rotation.matrix()
        t: Float64[ndarray, "3"] = world_from_cam.translation

        world_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        world_T_cam[:3, :3] = R
        world_T_cam[:3, 3] = t

        cam: pycolmap.Camera = image.camera
        K: Float64[ndarray, "3 3"] = _extract_intrinsics(cam)
        keypoints_xy: Float32[ndarray, "M 2"] = _extract_visible_keypoints(image, rec.points3D)

        image_results.append(
            SfMImageResult(
                world_T_cam=world_T_cam,
                intrinsics=K,
                image_size=(cam.width, cam.height),
                image_name=image.name,
                keypoints_xy=keypoints_xy,
            )
        )

    # Clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return SfMResult(
        points3d_xyz=points3d_xyz,
        points3d_rgb=points3d_rgb,
        images=image_results,
        num_images_registered=len(image_results),
    )


# ---------------------------------------------------------------------------
# Blueprint builder (shared between CLI and Gradio)
# ---------------------------------------------------------------------------
def create_sfm_blueprint(
    parent_log_path: Path,
) -> rrb.ContainerLike:
    """Create a Rerun blueprint for SfM reconstruction visualization.

    Uses a single ``camera`` entity with a timeline to scrub through frames,
    plus a 3D view of the full point cloud and all camera frustums.

    Args:
        parent_log_path: Root log path (typically ``Path("world")``).

    Returns:
        Rerun blueprint layout.
    """
    import rerun.blueprint as rrb

    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=["+ $origin/**"],
    )

    view_2d: rrb.Spatial2DView = rrb.Spatial2DView(
        origin=f"{parent_log_path}/camera/pinhole",
    )

    return rrb.Horizontal(contents=[view_3d, view_2d], column_shares=[3, 2])


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------
@dataclass
class SfMCLIConfig:
    """CLI configuration for COLMAP SfM reconstruction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration (--save, --connect, --spawn)."""
    image_dir: Path = Path("data/examples/sfm_reconstruction/Fountain/images")
    """Directory containing input images."""
    sfm_config: SfMConfig = field(default_factory=lambda: SfMConfig(verbose=True))
    """SfM reconstruction configuration."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(config: SfMCLIConfig) -> None:
    """CLI entry point for COLMAP SfM reconstruction with Rerun visualization.

    Loads images from a directory, runs the SfM pipeline, sets up the Rerun
    blueprint, and logs cameras sequentially over a timeline with 2D keypoints.
    """
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb

    parent_log_path: Path = Path("world")

    # 1. Validate input
    if not config.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {config.image_dir}")

    # 2. Run SfM pipeline
    result: SfMResult = run_sfm(
        image_dir=config.image_dir,
        config=config.sfm_config,
    )

    # 3. Setup Rerun blueprint
    blueprint: rrb.Blueprint = rrb.Blueprint(
        create_sfm_blueprint(parent_log_path=parent_log_path),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    # 4. Log 3D point cloud (static — full cloud visible at all times)
    if result.points3d_xyz.shape[0] > 0:
        rr.log(
            f"{parent_log_path}/point_cloud",
            rr.Points3D(
                positions=result.points3d_xyz,
                colors=result.points3d_rgb,
                radii=np.full(result.points3d_xyz.shape[0], 0.005, dtype=np.float32),
            ),
            static=True,
        )

    # 5. Log cameras sequentially over the timeline
    for i, img_result in enumerate(result.images):
        rr.set_time(TIMELINE, sequence=i)

        camera_path: str = f"{parent_log_path}/camera"

        # Log transform (world_T_cam)
        rr.log(
            camera_path,
            rr.Transform3D(
                mat3x3=img_result.world_T_cam[:3, :3],
                translation=img_result.world_T_cam[:3, 3],
            ),
        )

        # Log pinhole camera
        width: int = img_result.image_size[0]
        height: int = img_result.image_size[1]
        rr.log(
            f"{camera_path}/pinhole",
            rr.Pinhole(
                image_from_camera=img_result.intrinsics,
                width=width,
                height=height,
            ),
        )

        # Log image
        image_path: Path = config.image_dir / img_result.image_name
        if image_path.exists():
            bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rr.log(
                    f"{camera_path}/pinhole/image",
                    rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
                )

        # Log 2D keypoints
        if img_result.keypoints_xy.shape[0] > 0:
            rr.log(
                f"{camera_path}/pinhole/image/keypoints",
                rr.Points2D(
                    positions=img_result.keypoints_xy,
                    radii=np.full(img_result.keypoints_xy.shape[0], 2.0, dtype=np.float32),
                    colors=np.full((img_result.keypoints_xy.shape[0], 3), [34, 138, 167], dtype=np.uint8),
                ),
            )
