"""Streamed pipeline stage implementations and validation helpers.

Provides manual per-image implementations of COLMAP pipeline stages that
produce identical results to the high-level ``pycolmap`` API calls.  Each
streamed function exposes the inner loop so that Rerun logging can be added
directly when needed.

Also provides :func:`compare_databases` for asserting database equivalence
between black-box and streamed runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray

if TYPE_CHECKING:
    import pycolmap

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PerImageExtractionResult:
    """Per-image feature extraction data for Rerun logging."""

    image_id: int
    """COLMAP database image_id."""
    image_name: str
    """Relative image path (e.g. ``'rig1/cam1/image0001.jpg'``)."""
    image_rgb: UInt8[ndarray, "H W 3"]
    """Original-resolution RGB image."""
    keypoints_xy: Float32[ndarray, "N 2"]
    """Extracted keypoint positions (x, y) in original image coordinates."""
    num_keypoints: int
    """Number of extracted keypoints."""
    image_index: int
    """0-based index in the extraction sequence."""
    total_images: int
    """Total number of images being processed."""


# ---------------------------------------------------------------------------
# Bitmap resize helper (uses pycolmap.Bitmap for exact COLMAP equivalence)
# ---------------------------------------------------------------------------
def _maybe_rescale_bitmap(
    bitmap: pycolmap.Bitmap,
    max_image_size: int,
) -> tuple[pycolmap.Bitmap, float, float]:
    """Downscale a Bitmap if its largest dimension exceeds *max_image_size*.

    Uses ``pycolmap.Bitmap.rescale`` for identical interpolation to COLMAP's
    internal ``MaybeRescaleBitmap``.

    Args:
        bitmap: Input bitmap.
        max_image_size: Maximum allowed dimension.  No resizing if ``<= 0``.

    Returns:
        Tuple of (bitmap, scale_x, scale_y).  The bitmap is rescaled in-place
        (a clone is made first).  Scale factors map from rescaled coordinates
        back to the original resolution (1.0 when no resizing is performed).
    """
    import pycolmap as _pycolmap  # noqa: F811 — runtime import

    if max_image_size <= 0:
        return bitmap, 1.0, 1.0

    orig_w: int = bitmap.width
    orig_h: int = bitmap.height
    max_dim: int = max(orig_w, orig_h)

    if max_dim <= max_image_size:
        return bitmap, 1.0, 1.0

    scale: float = max_image_size / max_dim
    # COLMAP uses static_cast<int> which truncates (floors toward zero).
    new_w: int = int(orig_w * scale)
    new_h: int = int(orig_h * scale)

    rescaled: _pycolmap.Bitmap = bitmap.clone()
    rescaled.rescale(new_w, new_h)

    scale_x: float = orig_w / new_w
    scale_y: float = orig_h / new_h
    return rescaled, scale_x, scale_y


# ---------------------------------------------------------------------------
# Streamed feature extraction
# ---------------------------------------------------------------------------
def extract_features_streamed(
    *,
    database_path: Path,
    image_path: Path,
    camera_mode: pycolmap.CameraMode,
    reader_options: pycolmap.ImageReaderOptions,
    extraction_options: pycolmap.FeatureExtractionOptions,
) -> None:
    """Feature extraction with per-image loop access.

    Produces identical database contents as ``pycolmap.extract_features()``
    but exposes the inner loop so that Rerun logging can be added directly.

    Three phases:
        1. ``pycolmap.import_images()`` — register images + camera models.
        2. Create a ``FeatureExtractor`` instance.
        3. Per-image loop: read → resize → extract → rescale keypoints →
           write to DB.

    Args:
        database_path: Path to the COLMAP SQLite database.
        image_path: Root directory containing images.
        camera_mode: How camera intrinsics are shared (AUTO, PER_FOLDER, …).
        reader_options: Image reader configuration.
        extraction_options: Feature extraction configuration.
    """
    import pycolmap

    # -- Phase 1: Import images (camera models, EXIF) without extraction -----
    # Create database if it does not exist (extract_features does this
    # automatically, but import_images requires an existing file).
    if not database_path.exists():
        with pycolmap.Database.open(database_path):
            pass  # creates empty DB with schema

    pycolmap.import_images(
        database_path=database_path,
        image_path=image_path,
        camera_mode=camera_mode,
        options=reader_options,
    )

    # -- Phase 2: Create extractor -------------------------------------------
    extractor: pycolmap.FeatureExtractor = pycolmap.FeatureExtractor.create(
        extraction_options
    )
    max_image_size: int = extraction_options.eff_max_image_size()
    needs_rgb: bool = extraction_options.requires_rgb()

    # -- Phase 3: Per-image extraction loop ----------------------------------
    with pycolmap.Database.open(database_path) as db:
        all_images: list[pycolmap.Image] = db.read_all_images()
        total_images: int = len(all_images)

        for idx, db_image in enumerate(all_images):
            image_id: int = db_image.image_id
            image_name: str = db_image.name
            full_path: Path = image_path / image_name

            # Read image using pycolmap.Bitmap (same loader as extract_features)
            bitmap: pycolmap.Bitmap | None = pycolmap.Bitmap.read(
                str(full_path), as_rgb=needs_rgb
            )
            if bitmap is None:
                logger.warning("Could not read image: %s — skipping", full_path)
                continue

            # Resize using pycolmap.Bitmap.rescale (exact COLMAP equivalence)
            rescaled: pycolmap.Bitmap
            scale_x: float
            scale_y: float
            rescaled, scale_x, scale_y = _maybe_rescale_bitmap(bitmap, max_image_size)

            # Extract features from bitmap
            kps: pycolmap.FeatureKeypoints
            descs: pycolmap.FeatureDescriptors
            kps, descs = extractor.extract(rescaled)

            # Convert keypoints to numpy Nx6 (x, y, a11, a12, a21, a22)
            if len(kps) > 0:
                kps_np: Float32[ndarray, "N 6"] = np.array(
                    [[kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22] for kp in kps],
                    dtype=np.float32,
                )
                # Rescale keypoints back to original resolution
                if scale_x != 1.0 or scale_y != 1.0:
                    kps_np[:, 0] *= scale_x  # x
                    kps_np[:, 1] *= scale_y  # y
                    kps_np[:, 2] *= scale_x  # a11
                    kps_np[:, 3] *= scale_x  # a12
                    kps_np[:, 4] *= scale_y  # a21
                    kps_np[:, 5] *= scale_y  # a22
            else:
                kps_np: Float32[ndarray, "N 6"] = np.zeros((0, 6), dtype=np.float32)  # type: ignore[no-redef]

            # Write to database
            with pycolmap.DatabaseTransaction(db):
                db.write_keypoints(image_id, kps_np)
                db.write_descriptors(image_id, descs)

            logger.info(
                "Extracted %d features from %s [%d/%d]",
                len(kps),
                image_name,
                idx + 1,
                total_images,
            )


# ---------------------------------------------------------------------------
# Database comparison
# ---------------------------------------------------------------------------
def compare_databases(
    db_path_a: Path,
    db_path_b: Path,
    *,
    keypoint_atol: float = 0.0,
) -> None:
    """Compare two COLMAP databases for feature extraction equivalence.

    Asserts that both databases contain the same images with identical
    keypoints and descriptors (within the given tolerance).

    Args:
        db_path_a: Path to reference database (e.g. from pycolmap.extract_features).
        db_path_b: Path to test database (e.g. from streamed extraction).
        keypoint_atol: Absolute tolerance for keypoint position comparison.

    Raises:
        AssertionError: If the databases differ.
    """
    import pycolmap

    with pycolmap.Database.open(db_path_a) as db_a, pycolmap.Database.open(db_path_b) as db_b:
        images_a: list[pycolmap.Image] = db_a.read_all_images()
        images_b: list[pycolmap.Image] = db_b.read_all_images()

        # Same number of images
        assert len(images_a) == len(images_b), f"Image count mismatch: {len(images_a)} vs {len(images_b)}"

        # Sort by name for stable comparison
        images_a.sort(key=lambda img: img.name)
        images_b.sort(key=lambda img: img.name)

        for img_a, img_b in zip(images_a, images_b, strict=True):
            assert img_a.name == img_b.name, f"Image name mismatch: {img_a.name} vs {img_b.name}"

            # Compare keypoints
            kps_a: Float32[ndarray, "N_a C_a"] = db_a.read_keypoints(img_a.image_id)
            kps_b: Float32[ndarray, "N_b C_b"] = db_b.read_keypoints(img_b.image_id)

            assert kps_a.shape == kps_b.shape, (
                f"Keypoint shape mismatch for {img_a.name}: {kps_a.shape} vs {kps_b.shape}"
            )

            if keypoint_atol == 0.0:
                np.testing.assert_array_equal(
                    kps_a,
                    kps_b,
                    err_msg=f"Keypoints differ for {img_a.name}",
                )
            else:
                np.testing.assert_allclose(
                    kps_a,
                    kps_b,
                    atol=keypoint_atol,
                    err_msg=f"Keypoints differ for {img_a.name}",
                )

            # Compare descriptors
            descs_a: pycolmap.FeatureDescriptors = db_a.read_descriptors(img_a.image_id)
            descs_b: pycolmap.FeatureDescriptors = db_b.read_descriptors(img_b.image_id)

            assert descs_a.data.shape == descs_b.data.shape, (
                f"Descriptor shape mismatch for {img_a.name}: {descs_a.data.shape} vs {descs_b.data.shape}"
            )
            np.testing.assert_array_equal(
                descs_a.data,
                descs_b.data,
                err_msg=f"Descriptors differ for {img_a.name}",
            )
