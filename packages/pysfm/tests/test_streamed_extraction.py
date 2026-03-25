"""Tests for streamed feature extraction equivalence.

Validates that ``extract_features_streamed`` produces identical database
contents to ``pycolmap.extract_features`` on both synthetic images and the
real Fountain dataset.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32, UInt8
from numpy import ndarray

from pysfm.apis.pycolmap_recon_streamed import (
    PerImageExtractionResult,
    _maybe_rescale_bitmap,
    extract_features_streamed,
)

FOUNTAIN_DIR: Path = Path(__file__).parents[1] / "data" / "examples" / "sfm_reconstruction" / "Fountain" / "images"
"""Path to Fountain dataset images (11 images, 3072×2048)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_synthetic_images(output_dir: Path, num_images: int = 5) -> Path:
    """Create a directory of small synthetic images with detectable features.

    Uses a checkerboard-like pattern that produces ALIKED keypoints.

    Args:
        output_dir: Parent directory.  Images are written to ``output_dir/images/``.
        num_images: Number of images to generate.

    Returns:
        Path to the images directory.
    """
    images_dir: Path = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng: np.random.Generator = np.random.default_rng(42)
    for i in range(num_images):
        # Base checkerboard
        img: UInt8[ndarray, "H W 3"] = np.zeros((200, 300, 3), dtype=np.uint8)
        for row in range(0, 200, 20):
            for col in range(0, 300, 20):
                if (row // 20 + col // 20) % 2 == 0:
                    img[row : row + 20, col : col + 20] = 255
        # Add some per-image variation so keypoints differ slightly
        noise: UInt8[ndarray, "H W 3"] = rng.integers(0, 30, size=img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        bgr: UInt8[ndarray, "H W 3"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / f"image{i:04d}.jpg"), bgr)

    return images_dir


def _run_blackbox_extraction(
    images_dir: Path,
    db_path: Path,
    extraction_options: pycolmap.FeatureExtractionOptions | None = None,
) -> None:
    """Run the original ``pycolmap.extract_features()`` for comparison."""
    if extraction_options is None:
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT

    reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()

    pycolmap.set_random_seed(0)
    pycolmap.extract_features(
        database_path=db_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )


def _run_streamed_extraction(
    images_dir: Path,
    db_path: Path,
    extraction_options: pycolmap.FeatureExtractionOptions | None = None,
    callback: Callable[[PerImageExtractionResult], None] | None = None,
) -> None:
    """Run ``extract_features_streamed`` for comparison."""
    if extraction_options is None:
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT

    reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()

    pycolmap.set_random_seed(0)
    extract_features_streamed(
        database_path=db_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        reader_options=reader_options,
        extraction_options=extraction_options,
        callback=callback,
    )


def _compare_databases(
    db_path_a: Path,
    db_path_b: Path,
    *,
    keypoint_atol: float = 0.0,
) -> None:
    """Compare two COLMAP databases for feature extraction equivalence.

    Args:
        db_path_a: Path to reference database (black-box extract_features).
        db_path_b: Path to test database (streamed extraction).
        keypoint_atol: Absolute tolerance for keypoint position comparison.
    """
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


# ---------------------------------------------------------------------------
# Test: API availability
# ---------------------------------------------------------------------------
def test_pycolmap_low_level_apis_exist() -> None:
    """Verify that the pycolmap low-level APIs we depend on are available."""
    assert hasattr(pycolmap, "import_images")
    assert hasattr(pycolmap, "FeatureExtractor")
    assert hasattr(pycolmap.FeatureExtractor, "create")
    assert hasattr(pycolmap, "DatabaseTransaction")

    opts: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    assert hasattr(opts, "eff_max_image_size")
    assert hasattr(opts, "requires_rgb")


# ---------------------------------------------------------------------------
# Test: Equivalence on synthetic images (no resize)
# ---------------------------------------------------------------------------
def test_streamed_extraction_equivalence_synthetic(tmp_path: Path) -> None:
    """Streamed extraction produces identical DB contents on small synthetic images."""
    images_dir: Path = _create_synthetic_images(tmp_path / "data")

    db_a: Path = tmp_path / "blackbox.db"
    db_b: Path = tmp_path / "streamed.db"

    _run_blackbox_extraction(images_dir, db_a)
    _run_streamed_extraction(images_dir, db_b)

    _compare_databases(db_a, db_b, keypoint_atol=0.0)


# ---------------------------------------------------------------------------
# Test: Equivalence on Fountain dataset (with resize)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not FOUNTAIN_DIR.is_dir(),
    reason="Fountain dataset not downloaded — run: pixi run -e pysfm _download-sfm-example",
)
def test_streamed_extraction_equivalence_fountain(tmp_path: Path) -> None:
    """Streamed extraction produces equivalent DB contents on Fountain (resize path)."""
    # Use first 5 images for speed
    subset_dir: Path = tmp_path / "fountain_subset"
    subset_dir.mkdir()
    image_files: list[Path] = sorted(FOUNTAIN_DIR.glob("*.jpg"))[:5]
    if not image_files:
        image_files = sorted(FOUNTAIN_DIR.glob("*.png"))[:5]
    assert len(image_files) >= 3, f"Expected >=3 images in {FOUNTAIN_DIR}, found {len(image_files)}"

    for f in image_files:
        shutil.copy2(f, subset_dir / f.name)

    db_a: Path = tmp_path / "blackbox.db"
    db_b: Path = tmp_path / "streamed.db"

    _run_blackbox_extraction(subset_dir, db_a)
    _run_streamed_extraction(subset_dir, db_b)

    # Using pycolmap.Bitmap for reading and resizing gives exact COLMAP equivalence.
    _compare_databases(db_a, db_b, keypoint_atol=0.0)


# ---------------------------------------------------------------------------
# Test: Callback invocation
# ---------------------------------------------------------------------------
def test_callback_invoked_per_image(tmp_path: Path) -> None:
    """Callback is invoked once per image with correct metadata."""
    images_dir: Path = _create_synthetic_images(tmp_path / "data", num_images=3)

    results: list[PerImageExtractionResult] = []

    def record_callback(result: PerImageExtractionResult) -> None:
        results.append(result)

    db_path: Path = tmp_path / "test.db"
    _run_streamed_extraction(images_dir, db_path, callback=record_callback)

    assert len(results) == 3, f"Expected 3 callbacks, got {len(results)}"

    for i, r in enumerate(results):
        assert r.image_index == i, f"Expected image_index={i}, got {r.image_index}"
        assert r.total_images == 3, f"Expected total_images=3, got {r.total_images}"
        assert r.image_rgb.ndim == 3, "Expected 3D RGB array"
        assert r.image_rgb.shape[2] == 3, "Expected 3 channels (RGB)"
        assert r.num_keypoints == r.keypoints_xy.shape[0], (
            f"num_keypoints ({r.num_keypoints}) != keypoints_xy rows ({r.keypoints_xy.shape[0]})"
        )
        assert r.num_keypoints > 0, f"Expected keypoints for image {r.image_name}"


# ---------------------------------------------------------------------------
# Test: _maybe_rescale_bitmap
# ---------------------------------------------------------------------------
def test_maybe_rescale_bitmap_noop() -> None:
    """No resizing when image is smaller than max_image_size."""
    arr: UInt8[ndarray, "H W 3"] = np.zeros((100, 200, 3), dtype=np.uint8)
    bm: pycolmap.Bitmap = pycolmap.Bitmap.from_array(arr)
    rescaled: pycolmap.Bitmap
    sx: float
    sy: float
    rescaled, sx, sy = _maybe_rescale_bitmap(bm, 300)
    assert rescaled.width == 200
    assert rescaled.height == 100
    assert sx == 1.0
    assert sy == 1.0


def test_maybe_rescale_bitmap_downscale() -> None:
    """Downscale when image exceeds max_image_size."""
    arr: UInt8[ndarray, "H W 3"] = np.zeros((2000, 3000, 3), dtype=np.uint8)
    bm: pycolmap.Bitmap = pycolmap.Bitmap.from_array(arr)
    rescaled: pycolmap.Bitmap
    sx: float
    sy: float
    rescaled, sx, sy = _maybe_rescale_bitmap(bm, 1600)
    assert max(rescaled.width, rescaled.height) <= 1600
    assert sx > 1.0
    assert sy > 1.0


def test_maybe_rescale_bitmap_disabled() -> None:
    """No resizing when max_image_size <= 0."""
    arr: UInt8[ndarray, "H W 3"] = np.zeros((500, 500, 3), dtype=np.uint8)
    bm: pycolmap.Bitmap = pycolmap.Bitmap.from_array(arr)
    rescaled: pycolmap.Bitmap
    sx: float
    sy: float
    rescaled, sx, sy = _maybe_rescale_bitmap(bm, -1)
    assert rescaled.width == 500
    assert rescaled.height == 500
    assert sx == 1.0
    assert sy == 1.0
