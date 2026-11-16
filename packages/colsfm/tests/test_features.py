"""ALIKED extraction through COLMAP's own ONNX runner, on real Galileo images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32
from numpy import ndarray

from colsfm.database import create_database, read_keypoints
from colsfm.features import (
    BLOB_MAX_KEYPOINTS,
    ExtractionReport,
    FeatureOptions,
    cudnn_is_available,
    extract_features,
    resolve_device,
)
from colsfm.frames_meta import FramesMeta
from colsfm.keyframe_selection import KeyframeSelection, apply_selection, select_keyframes

FAST_IMAGE_SIZE: int = 640
"""Longest side used where the test cares about wiring rather than keypoint counts.

COLMAP rescales the detections back to the original image, so the coordinates
stay comparable; only the detector's input resolution shrinks. On the CPU
provider this is the difference between 2.8 s and 0.4 s per 1920x1200 image.
"""





@pytest.fixture(scope="module")
def two_image_extraction(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, FramesMeta, ExtractionReport]:
    """Extract two full-resolution Galileo images once, at the blob's own settings."""
    subset: FramesMeta = galileo_input.filtered([keyframe.keyframe_id for keyframe in galileo_input.keyframes[:2]])
    database_path: Path = tmp_path_factory.mktemp("features") / "two.db"
    create_database(database_path, subset)
    report: ExtractionReport = extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in subset.keyframes],
        FeatureOptions(),
    )
    return database_path, subset, report


def test_resolve_device_never_returns_an_unusable_cuda(galileo_input: FramesMeta) -> None:
    """`auto` degrades to the CPU provider; explicit `cuda` raises instead of aborting.

    ONNX Runtime's CUDA provider dlopens `libcudnn.so` lazily and throws inside a
    COLMAP worker thread when it is missing, which terminates the process. The
    only safe check is the one made before the call.
    """
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "cpu")
    usable: bool = pycolmap.has_cuda and pycolmap.get_num_cuda_devices() > 0 and cudnn_is_available()
    if usable:
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("auto") == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
        assert resolve_device("auto") == "cpu"


def test_extraction_fills_the_pre_created_image_rows(
    two_image_extraction: tuple[Path, FramesMeta, ExtractionReport],
) -> None:
    """COLMAP reuses the image rows the pipeline wrote instead of inserting its own.

    That is what keeps cuSFM's `KeyframeMetaData.id` as the COLMAP `image_id`
    and its `camera_params_id` as the `camera_id`, and it is the single
    assumption the whole database design rests on.
    """
    database_path, subset, _ = two_image_extraction
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    assert database.num_images() == len(subset.keyframes)
    assert database.num_cameras() == len(subset.cameras)
    assert database.num_rigs() == 1
    for keyframe in subset.keyframes:
        image: pycolmap.Image = database.read_image(keyframe.keyframe_id)
        assert image.name == keyframe.image_name
        assert image.camera_id == keyframe.camera_params_id
        assert image.frame_id == keyframe.synced_sample_id
    database.close()


def test_keypoint_counts_land_on_the_blob_budget(
    two_image_extraction: tuple[Path, FramesMeta, ExtractionReport],
) -> None:
    """The blob's Galileo keyframes hold 2047-2048 points; COLMAP's ALIKED matches that.

    `max_key_points_for_tensorrt: 4000` never binds in the blob because its ONNX
    detector head is a fixed top-2048 (feature_extractor_main.md §6.3), so 2048
    is the number to reproduce, not 4000.
    """
    _, subset, report = two_image_extraction
    assert set(report.keypoint_counts) == {keyframe.keyframe_id for keyframe in subset.keyframes}
    for count in report.keypoint_counts.values():
        assert 1900 <= count <= BLOB_MAX_KEYPOINTS
    assert report.elapsed_seconds > 0.0
    assert "keypoints" in report.summary()


def test_keypoints_are_pixels_in_the_original_image(
    two_image_extraction: tuple[Path, FramesMeta, ExtractionReport],
) -> None:
    """Detections come back in the source image's pixel frame, not the network's."""
    database_path, subset, _ = two_image_extraction
    keyframe = subset.keyframes[0]
    camera = subset.cameras[keyframe.camera_params_id]
    keypoints_xy: Float32[ndarray, "num_keypoints 2"] = read_keypoints(database_path, keyframe.keyframe_id)
    assert keypoints_xy.shape[1] == 2
    assert keypoints_xy[:, 0].min() >= 0.0
    assert keypoints_xy[:, 1].min() >= 0.0
    assert keypoints_xy[:, 0].max() <= camera.image_width
    assert keypoints_xy[:, 1].max() <= camera.image_height
    assert keypoints_xy[:, 0].max() > 0.5 * camera.image_width


def test_descriptors_are_float32_aliked_blocks(
    two_image_extraction: tuple[Path, FramesMeta, ExtractionReport],
) -> None:
    """ALIKED descriptors are stored as a uint8 reinterpretation of float32x128.

    COLMAP tags them with the extractor type, which the LightGlue matcher then
    checks, so nothing has to carry the descriptors in a side array.
    """
    database_path, subset, _ = two_image_extraction
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    stored: pycolmap.FeatureDescriptors = database.read_descriptors(subset.keyframes[0].keyframe_id)
    database.close()
    assert stored.type == pycolmap.FeatureExtractorType.ALIKED_N16ROT
    assert stored.data.dtype == np.uint8
    assert stored.data.shape[1] == 128 * 4
    recovered = stored.to_float()
    norms = np.linalg.norm(recovered.data, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "ALIKED descriptors should be L2-normalised"


def test_only_selected_keyframes_reach_the_extractor(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path
) -> None:
    """The pipeline's stages 1 and 2 in sequence: select, create the database, extract.

    With gates far above anything the trajectory does, cuSFM keeps the first
    sample alone — six Galileo frames (feature_extractor_main.md §5) — and only
    those six may reach the database or the extraction report.
    """
    selection: KeyframeSelection = select_keyframes(
        galileo_input,
        min_inter_frame_distance_m=99.0,
        min_inter_frame_rotation_degrees=999.0,
        sample_sync_threshold_microseconds=100,
    )
    selected: FramesMeta = apply_selection(galileo_input, selection)
    assert len(selection.kept_keyframe_ids) == 6
    assert len(selected.keyframes) == 6
    # The filtered collection carries the dense 1-based rig numbering.
    assert {keyframe.synced_sample_id for keyframe in selected.keyframes} == {1}
    assert selection.kept_synced_sample_ids == (1,)

    database_path: Path = tmp_path / "selected.db"
    create_database(database_path, selected)
    report: ExtractionReport = extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in selected.keyframes],
        FeatureOptions(max_image_size=FAST_IMAGE_SIZE, max_num_features=256),
    )
    assert set(report.keypoint_counts) == set(selection.kept_keyframe_ids)

    with pycolmap.Database.open(database_path) as database:
        assert database.num_images() == 6
        assert database.num_frames() == 1


def test_extract_features_reports_a_missing_database(galileo_input_dir: Path, tmp_path: Path) -> None:
    """A missing database is an error here, not a silently created empty one."""
    with pytest.raises(FileNotFoundError):
        extract_features(tmp_path / "absent.db", galileo_input_dir, ["a.jpeg"])


def test_extract_features_reports_a_missing_image_root(galileo_input: FramesMeta, tmp_path: Path) -> None:
    """A missing image root is caught before COLMAP is handed the job."""
    database_path: Path = tmp_path / "empty.db"
    create_database(database_path, galileo_input.filtered([galileo_input.keyframes[0].keyframe_id]))
    with pytest.raises(FileNotFoundError):
        extract_features(database_path, tmp_path / "absent", [galileo_input.keyframes[0].image_name])


def test_report_summary_handles_an_empty_run() -> None:
    """The summary line is safe to print even when nothing was extracted."""
    assert "no images" in ExtractionReport().summary()
