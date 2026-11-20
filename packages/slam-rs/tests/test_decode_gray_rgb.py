"""The RGB copy a replay colours its landmarks from comes from the same decode, and leaves the gray frames unchanged."""

from io import BytesIO

import av
import numpy as np
from av.container import OutputContainer
from av.video.stream import VideoStream

from slam_rs.catalog_feed import decode_gray, decode_gray_rgb


def red_video(frames: int) -> bytes:
    buffer: BytesIO = BytesIO()
    container: OutputContainer
    with av.open(buffer, mode="w", format="mp4") as container:
        stream: VideoStream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        picture: np.ndarray = np.zeros((64, 96, 3), dtype=np.uint8)
        picture[..., 0] = 200
        for _ in range(frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def test_each_gray_frame_comes_with_its_smaller_rgb_copy() -> None:
    video: bytes = red_video(3)
    pairs = list(decode_gray_rgb(video, 1, 1, 4))
    assert [gray.tobytes() for gray, _ in pairs] == [gray.tobytes() for gray in decode_gray(video)], "tracking sees the same pixels"
    for _, rgb in pairs:
        assert rgb.shape == (16, 24, 3) and rgb.dtype == np.uint8
        assert rgb[..., 0].mean() > 150 and rgb[..., 1].mean() < 60, "the copy keeps the colour"
