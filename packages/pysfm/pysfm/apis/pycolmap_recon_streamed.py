"""Unknown-rig multi-camera reconstruction with streamed feature extraction.

Identical to :mod:`pysfm.apis.pycolmap_recon` except that feature extraction
(step 4) is performed image-by-image rather than via the black-box
``pycolmap.extract_features()`` call.  This gives callers per-image access
(e.g. for Rerun logging) via an optional callback.

All other pipeline steps (matching, mapping, rig config) are unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray

if TYPE_CHECKING:
    import pycolmap

from pysfm.apis.pycolmap_recon import (
    RigReconConfig,
    RigReconResult,
    discover_videos,
    extract_synchronized_frames,
    generate_rig_config_json,
)

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PerImageExtractionResult:
    """Per-image feature extraction data passed to callbacks."""

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
    callback: Callable[[PerImageExtractionResult], None] | None = None,
) -> None:
    """Feature extraction with per-image callback access.

    Produces identical database contents as ``pycolmap.extract_features()``
    but exposes each image via *callback* after its features are committed.

    Three phases:
        1. ``pycolmap.import_images()`` — register images + camera models.
        2. Create a ``FeatureExtractor`` instance.
        3. Per-image loop: read → resize → extract → rescale keypoints →
           write to DB → invoke callback.

    Args:
        database_path: Path to the COLMAP SQLite database.
        image_path: Root directory containing images.
        camera_mode: How camera intrinsics are shared (AUTO, PER_FOLDER, …).
        reader_options: Image reader configuration.
        extraction_options: Feature extraction configuration.
        callback: Optional function invoked after each image's features are
            extracted and written to the database.
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

            # Keep original-resolution RGB array for the callback
            orig_rgb: UInt8[ndarray, "H W 3"] = bitmap.to_array() if needs_rgb else np.stack([bitmap.to_array()] * 3, axis=-1)

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

            # Invoke callback
            if callback is not None:
                keypoints_xy: Float32[ndarray, "N 2"] = kps_np[:, :2] if len(kps) > 0 else np.zeros((0, 2), dtype=np.float32)
                callback(
                    PerImageExtractionResult(
                        image_id=image_id,
                        image_name=image_name,
                        image_rgb=orig_rgb,
                        keypoints_xy=keypoints_xy,
                        num_keypoints=len(kps),
                        image_index=idx,
                        total_images=total_images,
                    )
                )


# ---------------------------------------------------------------------------
# Streamed rig reconstruction pipeline
# ---------------------------------------------------------------------------
def run_rig_recon_streamed(
    *,
    config: RigReconConfig,
    extraction_callback: Callable[[PerImageExtractionResult], None] | None = None,
) -> RigReconResult:
    """Run the full unknown-rig reconstruction pipeline with streamed extraction.

    Identical to :func:`pysfm.apis.pycolmap_recon.run_rig_recon` except step 4
    (feature extraction) uses :func:`extract_features_streamed` which invokes
    *extraction_callback* after each image.

    Pipeline:
        1. Discover videos
        2. Extract synchronized frames
        3. Generate rig_config.json
        4. **Streamed** feature extraction (ALIKED_N16ROT, GPU) with callback
        5. Sequential matching (ALIKED_LIGHTGLUE, GPU) — no rig info
        6. Incremental mapping (no-rig bootstrap)
        7. Apply rig config (auto-derives cam_from_rig from bootstrap)
        8. Rig-aware sequential matching (expand_rig_images)
        9. Global mapping (refine_sensor_from_rig)

    Args:
        config: Pipeline configuration.
        extraction_callback: Optional per-image callback for step 4.

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

    # -- 4. Streamed feature extraction (ALIKED_N16ROT, GPU) ------------------
    pycolmap.set_random_seed(0)

    if config.verbose:
        pycolmap.logging.minloglevel = pycolmap.logging.INFO

    reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()
    reader_options.camera_model = config.camera_model

    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    extraction_options.use_gpu = config.use_gpu
    extraction_options.gpu_index = "0"

    logger.info("Streamed feature extraction (ALIKED_N16ROT, camera_model=%s, gpu=%s) ...", config.camera_model, config.use_gpu)
    extract_features_streamed(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        reader_options=reader_options,
        extraction_options=extraction_options,
        callback=extraction_callback,
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

    no_rig_rec_id: int = max(no_rig_recs, key=lambda k: no_rig_recs[k].num_reg_images())
    no_rig_rec: pycolmap.Reconstruction = no_rig_recs[no_rig_rec_id]
    logger.info("No-rig bootstrap: %d images registered (model %d)", no_rig_rec.num_reg_images(), no_rig_rec_id)

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
