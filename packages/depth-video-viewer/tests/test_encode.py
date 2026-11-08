import numpy as np
import pytest

from depth_video_viewer.encode import MAX_GOP, decode_av1, encode_av1, join_ivf, split_ivf, to_codes


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


@pytest.mark.parametrize("gop", [0, MAX_GOP + 1])
def test_gops_the_viewer_cannot_seek_are_rejected(gop: int) -> None:
    with pytest.raises(ValueError, match="gop must be"):
        encode_av1(ramp(2, 8, 8), gop=gop)


def test_codes_are_linear_in_depth_with_zero_kept_as_no_depth() -> None:
    depth = np.array([[[0, 1023, 2046]]], np.uint16)
    codes, unit = to_codes(depth)
    assert unit == 2.0
    assert codes.tolist() == [[[0, 512, 1023]]]


def test_positive_depths_that_would_become_no_depth_are_rejected() -> None:
    # unit = 3000 / 1023 mm, so 1 mm rounds to code 0.
    with pytest.raises(ValueError, match="1 depths below"):
        to_codes(np.array([[[0, 1, 3000]]], np.uint16))
