"""Write the PNG fixtures of crates/depth-png/tests: depth images encoded by OpenCV and Pillow.

The pixels come from the formula in tests/third_party_pngs.rs (`sample`), so the Rust test checks the decoded values
against a source that is independent of either encoder. Noisy values keep the 16-bit images from compressing below
libpng's 8 KiB IDAT chunk size, so those files hold several IDAT chunks.

Run from the repository root: pixi run -e dataforge --frozen python -I packages/depth-png/scripts/make_fixtures.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "crates" / "depth-png" / "tests" / "fixtures"
WIDTH, HEIGHT = 97, 61


def sample(x: int, y: int, bits: int) -> int:
    """A 32-bit integer hash of the pixel position, cut to ``bits``; 0 (no depth) on every 11th diagonal."""
    if (x + 2 * y) % 11 == 0:
        return 0
    h = ((x * 73856093) ^ (y * 19349663)) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    return h & ((1 << bits) - 1)


def image(bits: int) -> np.ndarray:
    dtype = np.uint16 if bits == 16 else np.uint8
    return np.array([[sample(x, y, bits) for x in range(WIDTH)] for y in range(HEIGHT)], dtype=dtype)


OUT.mkdir(parents=True, exist_ok=True)
u16, u8 = image(16), image(8)
fixtures = {
    "opencv_u16_level4_up.png": cv2.imencode(".png", u16, [cv2.IMWRITE_PNG_COMPRESSION, 4, cv2.IMWRITE_PNG_FILTER, cv2.IMWRITE_PNG_FILTER_UP])[1].tobytes(),
    "opencv_u16_default.png": cv2.imencode(".png", u16)[1].tobytes(),
    "opencv_u8_default.png": cv2.imencode(".png", u8)[1].tobytes(),
}
for name, payload in fixtures.items():
    (OUT / name).write_bytes(payload)
Image.fromarray(u16).save(OUT / "pillow_u16_level6.png", compress_level=6)
Image.fromarray(u8).save(OUT / "pillow_u8_level6.png", compress_level=6)
Image.fromarray(np.stack([u8, u8, u8], axis=-1)).save(OUT / "pillow_rgb8.png")
for path in sorted(OUT.iterdir()):
    print(path.name, path.stat().st_size)
