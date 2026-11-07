"""Lossless PNG encoding and decoding of depth images, in parallel row bands (Rust: packages/depth-png/crates/depth-png).

``encode`` writes an 8- or 16-bit greyscale PNG with the Up row filter on every row and zlib compression (pure-Rust
zlib-rs). Large images are cut into bands of rows (a 1920x1440 depth map: 12 bands of 120 rows) compressed in
parallel into one standard zlib stream, with a private ``dpBN`` chunk that lets ``decode`` inflate the bands in
parallel too; any PNG reader decodes the file as usual. ``decode`` reads any 8- or 16-bit greyscale PNG, whichever
encoder wrote it.

Arrays are C-contiguous and row-major; a Fortran-ordered or strided array raises ValueError. They are read and written
in place with the GIL released, as NumPy's own functions do, so no other thread may write them during a call.

``threads=None`` uses the available parallelism, never more than the image's bands. Callers that already encode or
decode several images at once (a thread pool, data-loader producers) pass ``threads=1`` or 2.
"""

import numpy as np
from jaxtyping import UInt8, UInt16

def encode(depth: UInt16[np.ndarray, "h w"] | UInt8[np.ndarray, "h w"], level: int = 4, threads: int | None = None) -> bytes:
    """A depth image as PNG bytes; uint16 becomes a 16-bit PNG, uint8 an 8-bit one.

    ``level`` is the zlib level: 0 stores the rows uncompressed, 1 is fastest, 9 smallest. Level 4 is within a few
    percent of level 6 in size on depth maps at about a third of the time; level 2 encodes about as fast as libdeflate's
    level 1 at about 10 % more bytes.

    Raises:
        TypeError: ``depth`` is not a 2D uint16 or uint8 array.
        ValueError: ``depth`` is not C-contiguous, is empty, or ``level`` is outside 0-9.
    """
    ...

def decode(png: bytes | UInt8[np.ndarray, "n"], threads: int | None = None) -> UInt16[np.ndarray, "h w"] | UInt8[np.ndarray, "h w"]:
    """An 8- or 16-bit greyscale PNG as a new ``(h, w)`` uint16 or uint8 array, native byte order.

    Raises:
        TypeError: ``png`` is neither bytes nor a 1D uint8 array.
        ValueError: The bytes are not a valid PNG, or the PNG has colour, alpha, a palette or fewer than 8 bits.
    """
    ...

def decode_into(
    png: bytes | UInt8[np.ndarray, "n"], out: UInt16[np.ndarray, "h w"] | UInt8[np.ndarray, "h w"], threads: int | None = None
) -> None:
    """Decode into ``out``, a writable C-contiguous array of the PNG's shape and bit depth (e.g. ``pinned_tensor.numpy()``).

    Raises:
        TypeError: ``png`` or ``out`` has the wrong type.
        ValueError: The PNG is invalid or not greyscale, or ``out`` does not match its shape and bit depth.
    """
    ...

def header(png: bytes | UInt8[np.ndarray, "n"]) -> tuple[int, int, int]:
    """``(height, width, bit_depth)`` of an 8- or 16-bit greyscale PNG, from its header alone (to size a ``decode_into`` buffer).

    Raises:
        TypeError: ``png`` is neither bytes nor a 1D uint8 array.
        ValueError: The bytes are not a PNG header, or the PNG is not 8- or 16-bit greyscale.
    """
    ...
