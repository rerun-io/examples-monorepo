"""The extracted JPEGs are logged to Rerun as their own file bytes, never decoded and re-encoded."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from pysfm.apis.pycolmap_vid_recon import VidReconResult, log_features
from pysfm.apis.video_to_image import VideoToImageConfig, VideoToImageNode

JPEG_QUALITY: int = 60
"""Below the quality ``rr.Image.compress()`` uses, so a decode + re-encode changes the bytes."""


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    """Capture every ``rr.log`` call as ``(entity_path, archetype)``."""
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **_kwargs: calls.append((str(entity_path), archetype)))
    return calls


def _assert_logs_file_bytes(archetype: object, image_path: Path) -> None:
    assert isinstance(archetype, rr.EncodedImage), f"expected rr.EncodedImage, got {type(archetype).__name__}"
    assert archetype.blob is not None
    expected: rr.EncodedImage = rr.EncodedImage(contents=image_path.read_bytes(), media_type="image/jpeg")
    assert expected.blob is not None
    assert archetype.blob.as_arrow_array().equals(expected.blob.as_arrow_array())


def _write_synthetic_video(path: Path, num_frames: int) -> None:
    writer: cv2.VideoWriter = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 48))
    rng: np.random.Generator = np.random.default_rng(0)
    for _ in range(num_frames):
        frame: UInt8[ndarray, "48 64 3"] = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_to_image_verbose_logs_written_jpeg_bytes(tmp_path: Path, logged: list[tuple[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    video_path: Path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=10)
    # Write the frames below Rerun's compress() quality, so a re-encode cannot reproduce the file bytes.
    imwrite = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda path, img: imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))

    node: VideoToImageNode = VideoToImageNode(config=VideoToImageConfig(num_frames=3, verbose=True), parent_log_path=Path("world"))
    result = node(video_path=video_path, output_dir=tmp_path / "images")

    image_logs: list[tuple[str, object]] = [call for call in logged if call[0] == "world/extracted/image"]
    assert len(image_logs) == 3
    for (_path, archetype), image_path in zip(image_logs, result.image_paths, strict=True):
        _assert_logs_file_bytes(archetype, image_path)


def test_log_features_logs_jpeg_file_bytes(tmp_path: Path, logged: list[tuple[str, object]]) -> None:
    images_dir: Path = tmp_path / "images"
    images_dir.mkdir()
    image_path: Path = images_dir / "image1.jpg"
    noise: UInt8[ndarray, "48 64 3"] = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), noise, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    database_path: Path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as db:
        camera_id: int = db.write_camera(pycolmap.Camera.create_from_model_name(1, "SIMPLE_RADIAL", 50.0, 64, 48))
        image_id: int = db.write_image(pycolmap.Image(name=image_path.name, camera_id=camera_id))
        db.write_keypoints(image_id, np.array([[10.0, 20.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32))

    result: VidReconResult = VidReconResult(
        output_dir=tmp_path,
        images_dir=images_dir,
        database_path=database_path,
        model_dir=tmp_path / "sparse" / "0",
        num_frames_extracted=1,
        mapping_method="incremental",
        overlap=10,
    )
    log_features(result, Path("world"))

    image_logs: list[tuple[str, object]] = [call for call in logged if call[0] == "world/features/image"]
    assert len(image_logs) == 1
    _assert_logs_file_bytes(image_logs[0][1], image_path)
