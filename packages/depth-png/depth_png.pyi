"""Lossless PNG encoding and decoding of depth images (Rust: packages/depth-png/crates/depth-png).

``encode`` writes an 8- or 16-bit greyscale PNG with the Up row filter on every row and zlib compression (pure-Rust
zlib-rs), the contract of arkitscenes-download's ``encode_depth_png``. ``decode`` reads any 8- or 16-bit greyscale
PNG, whichever encoder wrote it. Arrays are C-contiguous and row-major; a Fortran-ordered or strided array raises
ValueError. The GIL is released while a PNG is compressed or decompressed.
"""

import numpy as np
from jaxtyping import UInt8, UInt16

def encode(depth: UInt16[np.ndarray, "h w"] | UInt8[np.ndarray, "h w"], level: int = 4) -> bytes:
    """A depth image as PNG bytes; uint16 becomes a 16-bit PNG, uint8 an 8-bit one.

    ``level`` is the zlib level: 0 stores the rows uncompressed, 1 is fastest, 9 smallest. Level 4 is within a few
    percent of level 6 in size on depth maps at about a third of the time.

    Raises:
        TypeError: ``depth`` is not a 2D uint16 or uint8 array.
        ValueError: ``depth`` is not C-contiguous, is empty, or ``level`` is outside 0-9.
    """
    ...

def decode(png: bytes) -> UInt16[np.ndarray, "h w"] | UInt8[np.ndarray, "h w"]:
    """An 8- or 16-bit greyscale PNG as a ``(h, w)`` uint8 or uint16 array, native byte order.

    Raises:
        ValueError: The bytes are not a valid PNG, or the PNG has colour, alpha, a palette or fewer than 8 bits.
    """
    ...
