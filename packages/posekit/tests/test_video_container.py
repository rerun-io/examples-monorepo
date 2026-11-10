"""Videos in any container: QuickTime as is, others remuxed to MP4; an iPhone-style rotated HEVC .mov comes out upright (CPU)."""

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("av", reason="PyAV and TorchCodec come with the hands env")
pytest.importorskip("torchcodec", reason="PyAV and TorchCodec come with the hands env")

import av
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import log_video, read_video_stream_from_rrd

from posekit.apis.video_hands import as_mp4


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def landscape_hevc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Half a second of 320x180 HEVC with a red block in the top-left corner of the coded picture."""
    path = tmp_path_factory.mktemp("videos") / "landscape.mov"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=gray:size=320x180:rate=30:duration=0.5", "-vf", "drawbox=x=0:y=0:w=60:h=40:color=red:t=fill",
        "-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", "log-level=error", str(path),
    )
    return path


def _red_corner(frame: UInt8[ndarray, "h w 3"]) -> str:
    h, w = frame.shape[:2]
    corners = {"top-left": (0, 0), "top-right": (0, w - 1), "bottom-left": (h - 1, 0), "bottom-right": (h - 1, w - 1)}
    red = [name for name, (y, x) in corners.items() if frame[y, x, 0] > 200 and frame[y, x, 1] < 60]
    assert len(red) == 1, red
    return red[0]


def _first_viewer_frame(video: Path, rrd: Path) -> UInt8[ndarray, "h w 3"]:
    """First frame of the stream the viewer gets from ``log_video``, decoded with PyAV."""
    recording = rr.RecordingStream("video-container-test")
    recording.save(rrd)
    with recording:
        log_video(video, Path("camera/video"))
    recording.flush()
    codec, _, samples = read_video_stream_from_rrd(str(rrd), "camera/video", "video_time")
    decoder = av.CodecContext.create({rr.VideoCodec.H264: "h264", rr.VideoCodec.H265: "hevc"}[codec], "r")
    for row in samples.combine_chunks().to_pylist():  # one VideoStream sample per row
        for frame in decoder.decode(av.Packet(bytes(row[0]))):
            return frame.to_ndarray(format="rgb24")
    raise AssertionError("no frame decoded")


def test_rotated_iphone_mov_is_upright_for_the_models_and_the_viewer(landscape_hevc: Path, tmp_path: Path) -> None:
    from torchcodec.decoders import VideoDecoder

    # An iPhone portrait clip: landscape HEVC plus a 90° display matrix, in a QuickTime file with an upper-case suffix.
    rotated = tmp_path / "IMG_0001.MOV"
    _ffmpeg("-display_rotation", "90", "-i", str(landscape_hevc), "-c", "copy", str(rotated))
    assert as_mp4(rotated, tmp_path) == rotated  # QuickTime is ISO BMFF: no remux

    decoder = VideoDecoder(rotated, dimension_order="NHWC")
    assert (decoder.metadata.width, decoder.metadata.height) == (180, 320)  # the size the pinhole gets
    model_frame = decoder.get_frames_in_range(0, 1).data[0].numpy()
    viewer_frame = _first_viewer_frame(rotated, tmp_path / "rotated.rrd")
    # 90° counter-clockwise: the coded top-left block ends up bottom-left, in both copies.
    assert model_frame.shape == viewer_frame.shape == (320, 180, 3)
    assert _red_corner(model_frame) == _red_corner(viewer_frame) == "bottom-left"


@pytest.mark.parametrize(("suffix", "codec_args", "codec"), [(".mkv", ["-c", "copy"], "hevc"), (".webm", ["-c:v", "libvpx-vp9"], "vp9")])
def test_other_containers_are_remuxed_to_mp4_without_reencoding(
    landscape_hevc: Path, tmp_path: Path, suffix: str, codec_args: list[str], codec: str
) -> None:
    source = tmp_path / f"clip{suffix}"
    _ffmpeg("-i", str(landscape_hevc), *codec_args, str(source))
    remuxed = as_mp4(source, tmp_path / "out")
    assert remuxed.suffix == ".mp4"
    with av.open(str(source)) as before, av.open(str(remuxed)) as after:
        assert after.streams.video[0].codec_context.name == codec
        assert [p.size for p in before.demux(video=0) if p.size] == [p.size for p in after.demux(video=0) if p.size]  # same packets
    rr.experimental.Mp4Reader(remuxed, mode="stream", entity_path="video")  # the viewer's reader opens it


def test_a_codec_mp4_cannot_carry_is_reencoded_to_h264(landscape_hevc: Path, tmp_path: Path) -> None:
    source = tmp_path / "clip.mkv"
    _ffmpeg("-i", str(landscape_hevc), "-c:v", "ffv1", str(source))
    with av.open(str(as_mp4(source, tmp_path / "out"))) as remuxed:
        assert remuxed.streams.video[0].codec_context.name == "h264"
