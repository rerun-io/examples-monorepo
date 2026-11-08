"""ASMK's binary signatures (jenicek/asmk `cython/hamming.pyx`) in Rust: packages/asmk-rs/crates/hamming.

The API of the Cython module `asmk.hamming`, which recipes/asmk-src replaces with this one. Arrays must be
C-contiguous float32 (rows to binarize) or uint32 (packed rows); a wrong dtype raises TypeError, a strided
array ValueError. Bits are packed MSB first; a row whose length is not a multiple of 32 has its last word
left-aligned. Distances are popcounts divided by `normalization` (0 = 32 bits per word), in float32.
"""

import numpy as np
from jaxtyping import Float32, UInt32

def binarize_and_pack(arr: Float32[np.ndarray, "d"], threshold: int = 0) -> UInt32[np.ndarray, "w"]: ...
def binarize_and_pack_2D(arr: Float32[np.ndarray, "n d"], threshold: int = 0) -> UInt32[np.ndarray, "n w"]: ...  # noqa: N802 (upstream's name)
def hamming_dist_packed(n1: UInt32[np.ndarray, "w"], n2: UInt32[np.ndarray, "w"], normalization: float = 0) -> float: ...
def hamming_cdist_packed(
    arr1: UInt32[np.ndarray, "n1 w"], arr2: UInt32[np.ndarray, "n2 w"], normalization: float = 0
) -> Float32[np.ndarray, "n1 n2"]: ...
