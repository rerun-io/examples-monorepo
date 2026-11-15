"""The frame source: the catalog manifest, the backend an input directory selects, and the image-folder decode."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from colsfm.frames import (
    FRAME_SOURCE_FILE,
    FrameSourceManifest,
    ImageFolderFrames,
    StreamKeyframes,
    VideoStreamFrames,
    open_frame_source,
    read_manifest,
    write_manifest,
)


def _manifest() -> FrameSourceManifest:
    return FrameSourceManifest(
        catalog_url="rerun+http://127.0.0.1:51235",
        dataset="msd-g2",
        segment="segment",
        timeline="timestamp",
        streams=[StreamKeyframes(entity="world/rig_00/cam_00/pinhole/video", names=["cam0/10", "cam0/20"], t_ns=[10, 20])],
    )


def test_a_manifest_round_trips_through_its_file(tmp_path: Path) -> None:
    """What the catalog door writes is what the run reads back."""
    path = tmp_path / FRAME_SOURCE_FILE
    write_manifest(path, _manifest())
    assert read_manifest(path) == _manifest()


def test_a_manifest_with_an_unknown_field_is_refused_naming_the_file(tmp_path: Path) -> None:
    """The manifest is a file colsfm owns, so a stray field is an error, not silently dropped."""
    path = tmp_path / FRAME_SOURCE_FILE
    path.write_text('{"catalog_url": "x", "dataset": "d", "segment": "s", "timeline": "t", "streams": [], "extra": 1}')
    with pytest.raises(ValueError, match=FRAME_SOURCE_FILE):
        read_manifest(path)


def test_names_and_timestamps_must_pair_up() -> None:
    """A stream's two lists are parallel; a length mismatch would key frames to the wrong names."""
    with pytest.raises(ValueError, match="names for"):
        StreamKeyframes(entity="e", names=["a"], t_ns=[1, 2])


def test_an_input_directory_with_a_manifest_reads_the_video_streams(tmp_path: Path) -> None:
    """The backend follows the directory: a manifest means the catalog's streams, none means the files."""
    assert isinstance(open_frame_source(tmp_path), ImageFolderFrames)
    write_manifest(tmp_path / FRAME_SOURCE_FILE, _manifest())
    assert isinstance(open_frame_source(tmp_path), VideoStreamFrames)


@pytest.mark.integration
def test_the_image_folder_decodes_jpeg_and_png_on_the_gpu_like_opencv(tmp_path: Path) -> None:
    """nvJPEG and the OpenCV fallback give RGB `[3, h, w]` frames matching OpenCV's own decode."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (nvJPEG)")
    rng = np.random.default_rng(0)
    bgr = cv2.GaussianBlur(rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8), (5, 5), 0)
    (tmp_path / "cam").mkdir()
    # 4:4:4 so the comparison isolates the IDCT: decoders upsample 4:2:0 chroma differently (libjpeg's "fancy"
    # upsampling vs nvJPEG's), which moves single noisy pixels by tens of levels and is no fault of either.
    cv2.imwrite(str(tmp_path / "cam" / "a.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    cv2.imwrite(str(tmp_path / "cam" / "b.png"), bgr)
    frames = dict(ImageFolderFrames(tmp_path).frames(["cam/a.jpg", "cam/b.png"]))
    for name in ("cam/a.jpg", "cam/b.png"):
        rgb = frames[name]
        assert rgb.is_cuda and rgb.dtype == torch.uint8 and tuple(rgb.shape) == (3, 48, 64)
        decoded = cv2.imread(str(tmp_path / name), cv2.IMREAD_COLOR)
        assert decoded is not None
        reference = decoded[:, :, ::-1].transpose(2, 0, 1).astype(np.int16)
        difference = np.abs(rgb.cpu().numpy().astype(np.int16) - reference)
        # PNG is lossless; two JPEG decoders may round the IDCT differently by a level or two.
        assert difference.max() <= (0 if name.endswith(".png") else 3)
