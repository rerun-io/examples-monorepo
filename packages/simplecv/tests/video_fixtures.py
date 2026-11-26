"""Synthetic encoded video for catalog reader tests."""

from fractions import Fraction

import av
import numpy as np


def encode_h264(frames: int, gop: int, seed: int) -> list[tuple[bytes, bool]]:
    """H.264 samples of moving noise (every frame distinct), closed GOPs of ``gop`` frames, no B-frames."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 96, 64, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"g": str(gop), "keyint_min": str(gop), "bf": "0", "sc_threshold": "0"}
    rng: np.random.Generator = np.random.default_rng(seed)
    base: np.ndarray = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    samples: list[tuple[bytes, bool]] = []
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(np.roll(base, 3 * index, axis=1), format="rgb24")
        frame.pts = index
        samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(frame)]
    samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(None)]
    return samples
