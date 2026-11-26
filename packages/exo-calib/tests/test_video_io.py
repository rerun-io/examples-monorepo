"""The exo cameras decode on NVDEC over one shared reader, from a catalog served over a synthetic H.264 recording."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration
av = pytest.importorskip("av", reason="PyAV encodes the synthetic video")
rr = pytest.importorskip("rerun", reason="the catalog server comes from rerun-sdk")
pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")
pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

FRAMES: int = 24
CAMERAS: tuple[str, ...] = ("rig_00", "rig_01")


def _h264(frames: int, gop: int, seed: int) -> list[tuple[bytes, bool]]:
    """Moving noise, closed GOPs, no B-frames: every frame distinct, one packet per frame."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 96, 64, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"g": str(gop), "keyint_min": str(gop), "bf": "0", "sc_threshold": "0"}
    base: np.ndarray = np.random.default_rng(seed).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    packets: list = []
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(np.roll(base, 3 * index, axis=1), format="rgb24")
        frame.pts = index
        packets += codec.encode(frame)
    packets += codec.encode(None)
    return [(bytes(packet), bool(packet.is_keyframe)) for packet in packets]


def test_exo_streams_decode_every_camera_on_cuda_from_one_reader(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    from simplecv.catalog_video_codec import wrap_mp4
    from torchcodec.decoders import VideoDecoder

    from exo_calib.layer_io import TIMELINE
    from exo_calib.video_io import ExoVideoStreams, open_exo_streams

    times = (33_333_333 * np.arange(FRAMES, dtype=np.int64)).astype("timedelta64[ns]")
    encoded: dict[str, list[tuple[bytes, bool]]] = {name: _h264(FRAMES, 6, seed) for seed, name in enumerate(CAMERAS)}
    path: Path = tmp_path / "segment.rrd"
    with rr.RecordingStream("test", recording_id="segment") as recording:
        recording.save(path)
        for name, samples in encoded.items():
            recording.log(f"world/{name}/pinhole/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            recording.send_columns(
                f"world/{name}/pinhole/video",
                indexes=[rr.TimeColumn(TIMELINE, duration=times)],
                columns=rr.VideoStream.columns(sample=[sample for sample, _ in samples], is_keyframe=[keyframe for _, keyframe in samples]),
            )

    with rr.server.Server(datasets={"test": [path]}) as server:
        streams: ExoVideoStreams = open_exo_streams(server.client().get_dataset("test"), "segment", CAMERAS)
        frames = [streams.decoders[cam].get_frames_at([0, 13, 23]) for cam in range(len(CAMERAS))]
        rgb = streams.frame_rgb(1, 5)

    assert streams.decoders[0].reader is streams.decoders[1].reader
    for cam, name in enumerate(CAMERAS):
        np.testing.assert_array_equal(streams.times_ns[cam], times)
        samples = encoded[name]
        mp4: bytes = wrap_mp4([sample for sample, _ in samples], [keyframe for _, keyframe in samples], 30, codec="h264")
        whole = VideoDecoder(mp4, device="cuda", seek_mode="exact").get_frames_at([0, 5, 13, 23]).data
        assert frames[cam].device.type == "cuda"
        assert torch.equal(frames[cam], whole[[0, 2, 3]])
        if cam == 1:
            np.testing.assert_array_equal(rgb, whole[1].permute(1, 2, 0).cpu().numpy())
