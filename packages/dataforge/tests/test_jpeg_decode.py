"""JPEG planar decoding preserves chroma layout, order and active plane extents."""

import numpy as np
import pytest

jpeg = pytest.importorskip("turbojpeg", reason="PyTurboJPEG is required for JPEG decode tests")

from dataforge.video_encoding import FrameKind, FrameSource, decode_jpeg_frames, jpeg_frame_source  # noqa: E402


@pytest.mark.parametrize(
    "sampling,kind,divisors", [(3, "gray8", []), (0, "yuv444p", [(1, 1)] * 2), (1, "yuv422p", [(2, 1)] * 2), (2, "yuv420p", [(2, 2)] * 2)]
)
def test_jpeg_planes_omit_padding_and_keep_order(sampling: int, kind: FrameKind, divisors: list[tuple[int, int]]) -> None:
    codec = jpeg.TurboJPEG()
    # Odd dimensions exercise TurboJPEG's padded luma and ceil-sized chroma.
    height, width = 13, 17
    frames = []
    expected = []
    for value in (15, 210, 60):
        pixels = np.full((height, width, 3), value, dtype=np.uint8)
        pixels[:, :, 0] = np.arange(width, dtype=np.uint8) * 10
        pixels[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None] * 15
        encoded = codec.encode(pixels, jpeg_subsample=sampling)
        frames.append(encoded)
        planes = codec.decode_to_yuv_planes(encoded, strides=(32, 32, 32))
        expected.append(
            b"".join(
                plane[: (height + dy - 1) // dy, : (width + dx - 1) // dx].tobytes()
                for plane, (dx, dy) in zip(planes, [(1, 1), *divisors], strict=True)
            )
        )
    source = jpeg_frame_source(frames[0])
    assert source == FrameSource(kind, width, height, full_range=sampling != 3)
    assert list(decode_jpeg_frames(iter(frames), source=source, workers=2)) == expected


def test_jpeg_geometry_change_refused() -> None:
    codec = jpeg.TurboJPEG()
    first = codec.encode(np.zeros((8, 12, 3), dtype=np.uint8))
    changed = codec.encode(np.zeros((12, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="JPEG layout changed"):
        list(decode_jpeg_frames([first, changed], source=jpeg_frame_source(first)))


def test_jpeg_gray_clockwise_rotation_cpu() -> None:
    import shutil
    import subprocess

    from dataforge.video_encoding import TRANSPOSE_FILTERS

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("CPU ffmpeg executable absent")
    pixels = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
    codec = jpeg.TurboJPEG()
    encoded = codec.encode(pixels, pixel_format=jpeg.TJPF_GRAY, jpeg_subsample=jpeg.TJSAMP_GRAY)
    source = jpeg_frame_source(encoded)
    frame = next(decode_jpeg_frames([encoded], source=source))
    rotated = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-filter_threads",
            "1",
            *source.input_args(fps=30),
            "-vf",
            TRANSPOSE_FILTERS[1][0],
            "-c:v",
            "rawvideo",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-",
        ],
        input=frame,
        capture_output=True,
        check=True,
    )
    expected = np.rot90(codec.decode(encoded, pixel_format=jpeg.TJPF_GRAY).reshape(8, 12), -1)
    np.testing.assert_array_equal(np.frombuffer(rotated.stdout, dtype=np.uint8).reshape(12, 8), expected)


def test_jpeg_decode_lookahead_is_bounded() -> None:
    codec = jpeg.TurboJPEG()
    encoded = codec.encode(np.zeros((8, 12, 3), dtype=np.uint8))
    consumed = []

    def images():
        for i in range(100):
            consumed.append(i)
            yield encoded

    decoded = decode_jpeg_frames(images(), source=jpeg_frame_source(encoded), workers=2)
    next(decoded)
    assert len(consumed) <= 4
    decoded.close()
