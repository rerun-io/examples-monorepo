import numpy as np
import pytest

from depth_video_viewer.encode import decode_av1, encode_av1, join_ivf, split_ivf


def ramp(n: int, h: int, w: int) -> np.ndarray:
    f, y, x = np.meshgrid(np.arange(n), np.arange(h), np.arange(w), indexing="ij")
    return ((x * 37 + y * 101 + f * 53) % 1024).astype(np.uint16)


def test_lossless_round_trip_keeps_every_10_bit_code() -> None:
    codes = ramp(12, 16, 32)
    samples = encode_av1(codes, gop=5)
    assert len(samples) == 12
    assert np.array_equal(decode_av1(samples, 16, 32), codes)


def test_ivf_split_inverts_join() -> None:
    samples = [b"\x12\x00", b"abc", b""]
    assert split_ivf(join_ivf(samples, 8, 8)) == samples


def test_codes_above_10_bits_are_rejected() -> None:
    with pytest.raises(ValueError, match="10 bits"):
        encode_av1(np.full((1, 8, 8), 1024, np.uint16))
