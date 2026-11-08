"""asmk.hamming (packages/asmk-rs, the Rust port of asmk's Cython extension) against a NumPy reference.

MASt3R-SLAM's retrieval runs ASMK with binary signatures, so every keyframe query and insert goes through
`binarize_and_pack_2D` and `hamming_cdist_packed`. The port was checked bit-exact against the Cython module
(9 widths x 3 thresholds x 3 normalizations, NaN and zeros included) before the extension was deleted; these
tests pin the same semantics with an independent NumPy implementation.
"""

import numpy as np
import pytest

pytest.importorskip("asmk_rs", reason="asmk.hamming needs packages/asmk-rs (mast3r-slam feature)")

from asmk import hamming  # noqa: E402
from numpy import ndarray  # noqa: E402


def _pack(arr: ndarray, threshold: int) -> ndarray:
    """MSB-first uint32 words of arr > threshold; a partial last word is left-aligned."""
    bits = arr > np.float32(threshold)
    pad = (-bits.shape[-1]) % 32
    bits = np.concatenate([bits, np.zeros((*bits.shape[:-1], pad), dtype=bool)], axis=-1)
    return np.packbits(bits, axis=-1, bitorder="big").view(">u4").astype(np.uint32)


def _cdist(a: ndarray, b: ndarray, normalization: float) -> ndarray:
    counts = np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(-1)
    norm = np.float32(normalization if normalization else a.shape[1] * 32)
    return (counts.astype(np.float32) / norm).astype(np.float32)


def test_asmk_hamming_is_asmk_rs() -> None:
    import asmk_rs

    assert hamming.hamming_cdist_packed is asmk_rs.hamming_cdist_packed


@pytest.mark.parametrize("dim", [1, 5, 31, 32, 33, 100, 128, 1024])
@pytest.mark.parametrize("threshold", [0, 1, -1])
def test_binarize_and_pack_matches_numpy(dim: int, threshold: int) -> None:
    rng = np.random.default_rng(dim)
    arr = rng.standard_normal((20, dim)).astype(np.float32)
    arr[rng.random(arr.shape) < 0.02] = np.nan
    arr[rng.random(arr.shape) < 0.05] = threshold
    packed = hamming.binarize_and_pack_2D(arr, threshold)
    assert packed.dtype == np.uint32
    np.testing.assert_array_equal(packed, _pack(arr, threshold))
    np.testing.assert_array_equal(hamming.binarize_and_pack(arr[0], threshold), _pack(arr[0], threshold))


@pytest.mark.parametrize("dim", [32, 33, 1024])
@pytest.mark.parametrize("normalization", [0.0, 2.0, 7.5])
def test_hamming_distances_match_numpy(dim: int, normalization: float) -> None:
    rng = np.random.default_rng(dim + 1)
    a = hamming.binarize_and_pack_2D(rng.standard_normal((7, dim)).astype(np.float32))
    b = hamming.binarize_and_pack_2D(rng.standard_normal((11, dim)).astype(np.float32))
    dist = hamming.hamming_cdist_packed(a, b, normalization)
    assert dist.dtype == np.float32
    np.testing.assert_array_equal(dist, _cdist(a, b, normalization))
    assert hamming.hamming_dist_packed(a[0], b[0], normalization) == _cdist(a[:1], b[:1], normalization)[0, 0]


def test_wrong_dtype_and_layout_raise() -> None:
    with pytest.raises(TypeError):
        hamming.binarize_and_pack_2D(np.zeros((2, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="contiguous"):
        hamming.binarize_and_pack_2D(np.zeros((8, 2), dtype=np.float32).T)
    with pytest.raises(ValueError, match="same number of words"):
        hamming.hamming_cdist_packed(np.zeros((1, 1), dtype=np.uint32), np.zeros((1, 2), dtype=np.uint32))
