"""The shared Rerun logging core: the source camera of a cover-cropped raster, derived videos, the input video's viewer copy."""

import shutil
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader

from metrictrail.eval.vipe_output import InputRasterRecord
from metrictrail.rerun_log import FrameVideo, SourceRaster, viewer_mp4

FFMPEG: str | None = shutil.which("ffmpeg")


def project(k4: Float64[ndarray, "4"], points_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
    """Pinhole projection ``fx x / z + cx``, continuous pixel coordinates."""
    return np.column_stack([k4[0] * points_cam[:, 0] / points_cam[:, 2] + k4[2], k4[1] * points_cam[:, 1] / points_cam[:, 2] + k4[3]])


def test_cover_crop_maps_raster_pixels_and_intrinsics_onto_the_source_frame() -> None:
    # ViPE's plan for a 1080p clip on the default 280x504 raster: scale 0.2625, resized 284x504, crop (0, 2).
    raster = SourceRaster(source_hw=(1080, 1920), resized_hw=(284, 504), crop_left_top=(0, 2))
    k_raster: Float64[ndarray, "4"] = np.array([400.0, 400.0, 252.0, 140.0])
    points: Float64[ndarray, "n 3"] = np.random.default_rng(0).uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 4.0], size=(20, 3))
    on_source: Float64[ndarray, "n 2"] = raster.source_px(project(k_raster, points).astype(np.float32)).astype(np.float64)
    np.testing.assert_allclose(on_source, project(raster.source_intrinsics(k_raster), points), atol=1e-3)
    np.testing.assert_allclose(raster.source_px(np.array([[0.0, 0.0], [504.0, 280.0]], dtype=np.float32)), [[0.0, 2 * 1080 / 284], [1920.0, 282 * 1080 / 284]])


def test_source_raster_comes_from_the_run_records_raster_block() -> None:
    record = InputRasterRecord(policy="cover_crop", prepared_hw=[280, 504], source_hw=[1080, 1920], resized_hw=[284, 504], crop_left_top=[0, 2])
    assert SourceRaster.from_record(record) == SourceRaster(source_hw=(1080, 1920), resized_hw=(284, 504), crop_left_top=(0, 2))
    assert SourceRaster.from_record(InputRasterRecord(policy="unchanged")) is None


def test_frame_video_logs_one_decodable_h264_sample_per_frame(tmp_path: Path) -> None:
    rec = rr.RecordingStream("metrictrail-test", recording_id="frame-video")
    rec.save(tmp_path / "video.rrd")
    video = FrameVideo(rec, "cam/rgb", fps=10.0)
    frames: list[UInt8[ndarray, "h w 3"]] = [np.full((64, 96, 3), 20 * index, dtype=np.uint8) for index in range(8)]
    for index, frame in enumerate(frames):
        rec.set_time("frame", sequence=index)
        video.log(frame)
    video.close()
    rec.flush()
    rec.disconnect()
    samples: list[bytes] = []
    times: list[int] = []
    codecs: list[int] = []
    for chunk in RrdReader(tmp_path / "video.rrd").stream():
        batch = chunk.to_record_batch()
        if "VideoStream:sample" in batch.schema.names:
            samples += [bytes(row[0]) for row in batch.column("VideoStream:sample").to_pylist() if row]
            times += batch.column("frame").to_pylist()
        if "VideoStream:codec" in batch.schema.names:
            codecs += [row[0] for row in batch.column("VideoStream:codec").to_pylist() if row]
    assert codecs == [rr.VideoCodec.H264.value]
    assert times == list(range(8))  # one sample per frame, at the frame's own time
    decoder = av.CodecContext.create("h264", "r")
    decoded: list[UInt8[ndarray, "h w 3"]] = []
    for sample in samples:
        decoded += [frame.to_ndarray(format="rgb24") for frame in decoder.decode(av.Packet(sample))]
    decoded += [frame.to_ndarray(format="rgb24") for frame in decoder.decode(None)]
    assert len(decoded) == 8
    assert all(abs(float(got.mean()) - float(want.mean())) < 6.0 for got, want in zip(decoded, frames, strict=True))


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg comes from the metrictrail env")
def test_viewer_mp4_keeps_an_h264_mp4_and_reencodes_hevc_and_other_containers(tmp_path: Path) -> None:
    def make(name: str, codec: str) -> Path:
        path: Path = tmp_path / name
        source: list[str] = ["-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1"]
        subprocess.run([str(FFMPEG), "-v", "error", "-y", *source, "-c:v", codec, "-pix_fmt", "yuv420p", str(path)], check=True)
        return path

    h264: Path = make("clip.mp4", "libx264")
    assert viewer_mp4(h264, tmp_path / "out") == h264
    for name, codec in (("clip_hevc.mp4", "libx265"), ("clip.mkv", "libx264")):
        copy: Path = viewer_mp4(make(name, codec), tmp_path / "out")
        assert copy.suffix == ".mp4" and copy.parent == tmp_path / "out"
        with av.open(str(copy)) as container:
            assert container.streams.video[0].codec_context.name == "h264"


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg comes from the metrictrail env")
def test_viewer_mp4_reencodes_a_clip_once_whatever_path_it_arrives_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The app gets every upload (and every click on an example) at a new temporary path: the copy is keyed by content."""
    first: Path = tmp_path / "upload-1" / "clip.mkv"
    first.parent.mkdir()
    source: list[str] = ["-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1"]
    subprocess.run([str(FFMPEG), "-v", "error", "-y", *source, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(first)], check=True)
    second: Path = tmp_path / "upload-2" / "renamed.mkv"
    second.parent.mkdir()
    second.write_bytes(first.read_bytes())
    copy: Path = viewer_mp4(first, tmp_path / "cache")
    encodes: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda args, **_: encodes.append(args))
    assert viewer_mp4(second, tmp_path / "cache") == copy
    assert encodes == []
