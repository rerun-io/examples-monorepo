"""The source video sits on the run clock: frame 0 at time 0 is the input frame at start_time (CPU, synthetic video)."""

from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
import rerun as rr
from av.video.codeccontext import VideoCodecContext
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import read_video_stream_from_rrd

from fdanyone.video import write_video
from fdanyone_rerun.outputs import InputRecord
from fdanyone_rerun.rerun_log import SOURCE, TIMELINE, log_input_clip

CELL: int = 32
"""Each frame shows its index in 9 bits: a 3x3 grid of black or white 32-px cells."""


def _numbered(index: int) -> UInt8[ndarray, "96 96 3"]:
    bits: UInt8[ndarray, "3 3"] = np.array([(index >> bit) & 1 for bit in range(9)], dtype=np.uint8).reshape(3, 3) * 255
    return np.repeat(np.repeat(bits, CELL, axis=0), CELL, axis=1)[:, :, None].repeat(3, axis=2)


def _index(frame: UInt8[ndarray, "96 96 3"]) -> int:
    centers: ndarray = frame[CELL // 2 :: CELL, CELL // 2 :: CELL, 0].reshape(9)
    return int(sum(1 << bit for bit, value in enumerate(centers) if value > 127))


def test_the_canonical_clip_starts_at_start_time_on_the_run_clock(tmp_path: Path) -> None:
    video: Path = write_video(tuple(_numbered(index) for index in range(300)), tmp_path / "input.mp4", Fraction(30), crf=0, preset="ultrafast")
    clip = InputRecord(filename="input.mp4", fps="30/1", start_time_seconds=3.0, num_frames=121, width=96, height=96)
    recording = rr.RecordingStream("fdanyone-rerun-source-clip")
    recording.save(tmp_path / "source.rrd")
    log_input_clip(video, clip, recording)
    recording.flush()
    recording.disconnect()

    codec, times, samples = read_video_stream_from_rrd(str(tmp_path / "source.rrd"), f"{SOURCE}/video", TIMELINE)
    assert codec == rr.VideoCodec.H264
    seconds: ndarray = times.to_numpy().astype("timedelta64[ns]").astype(np.int64) / 1e9
    np.testing.assert_allclose(seconds, np.arange(121) / 30.0, atol=1e-6)  # 121 frames from 0 s, nothing of the rest
    decoder: VideoCodecContext = cast(VideoCodecContext, av.CodecContext.create("h264", "r"))
    frames: list[UInt8[ndarray, "h w 3"]] = []
    for row in samples.to_pylist():
        for packet in decoder.parse(b"".join(bytes(blob) for blob in row)):
            frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
    for packet in decoder.parse(None):
        frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
    frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(None))
    assert [_index(frame) for frame in frames] == list(range(90, 211))  # input frames 3.0 s .. 7.0 s
