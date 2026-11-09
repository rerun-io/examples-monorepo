"""Depth -> 10-bit codes <-> AV1 Main 10-bit monochrome lossless, one temporal unit per frame.

Codes are linear in depth with one unit per recording (`to_codes`); code 0 means no depth. Encoder settings: libaom `lossless=1`, so the codes the viewer decodes are the codes written here; a fast `realtime`
preset (`cpu-used 8`), since a lossless stream decodes to the same codes at any preset, only its size changes;
a fixed GOP (`-g` = `-keyint_min`), so a seek decodes at most `gop` samples; `gray10le` in full range, so no
limited-range scaling touches the codes.
"""

import subprocess

import numpy as np
from jaxtyping import UInt16
from numpy import ndarray

MAX_CODE = 1023
"""Largest code a 10-bit stream holds."""
MAX_GOP = 1000
"""The viewer's limit on samples a seek decodes (`decoder.rs` `MAX_GOP`)."""

_IVF_HEADER = 32
_IVF_FRAME_HEADER = 12


def to_codes(depth_mm: UInt16[ndarray, "n h w"]) -> tuple[UInt16[ndarray, "n h w"], float]:
    """Linear 10-bit codes and their unit in mm: code = round(mm / unit), unit = max depth / 1023 (at least 1 mm).

    The viewer's `meter` is then `1000 / unit`. Code 0 means no depth, so a positive depth below unit / 2 would
    vanish: such inputs are rejected.
    """
    unit = max(1.0, int(depth_mm.max()) / MAX_CODE)
    scaled = depth_mm / unit
    np.rint(scaled, out=scaled)
    np.clip(scaled, 0, MAX_CODE, out=scaled)
    codes = scaled.astype(np.uint16)
    lost = int(np.count_nonzero((depth_mm > 0) & (codes == 0)))
    if lost:
        raise ValueError(f"{lost} depths below {unit / 2:.3f} mm would quantise to code 0 (no depth)")
    return codes, unit


def encode_av1(codes: UInt16[ndarray, "n h w"], gop: int = 10) -> list[bytes]:
    """Encode 10-bit codes as AV1 Main 10-bit monochrome lossless; returns one sample (temporal unit) per frame.

    Every `gop`-th frame is a keyframe, so a viewer decodes at most `gop` samples for a seek.
    """
    if int(codes.max(initial=0)) > MAX_CODE:
        raise ValueError(f"codes must fit 10 bits (max {MAX_CODE}), got {int(codes.max())}")
    if not 1 <= gop <= MAX_GOP:
        raise ValueError(f"gop must be in [1, {MAX_GOP}], got {gop}")
    n, h, w = codes.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray10le", "-color_range", "pc", "-s", f"{w}x{h}",
           "-r", "30", "-i", "pipe:0", "-c:v", "libaom-av1", "-usage", "realtime", "-cpu-used", "8", "-row-mt", "1", "-threads", "8",
           "-aom-params", "lossless=1", "-g", str(gop), "-keyint_min", str(gop), "-color_range", "pc", "-f", "ivf", "pipe:1"]  # fmt: skip
    # stderr stays on the console: with `-loglevel error` it holds the only diagnostic of a failed encode.
    ivf = subprocess.run(cmd, input=codes.astype("<u2").tobytes(), stdout=subprocess.PIPE, check=True).stdout
    samples = split_ivf(ivf)
    if len(samples) != n:
        raise RuntimeError(f"libaom returned {len(samples)} temporal units for {n} frames")
    return samples


def split_ivf(ivf: bytes) -> list[bytes]:
    """The frames of an IVF container, in order."""
    if ivf[:4] != b"DKIF":
        raise ValueError("not an IVF stream")
    pos = int.from_bytes(ivf[6:8], "little")
    samples = []
    while pos < len(ivf):
        size = int.from_bytes(ivf[pos : pos + 4], "little")
        samples.append(ivf[pos + _IVF_FRAME_HEADER : pos + _IVF_FRAME_HEADER + size])
        pos += _IVF_FRAME_HEADER + size
    return samples


def join_ivf(samples: list[bytes], height: int, width: int) -> bytes:
    """Wrap AV1 temporal units in an IVF container (the inverse of `split_ivf`)."""
    header = b"".join([b"DKIF", (0).to_bytes(2, "little"), _IVF_HEADER.to_bytes(2, "little"), b"AV01", width.to_bytes(2, "little"),
                       height.to_bytes(2, "little"), (30).to_bytes(4, "little"), (1).to_bytes(4, "little"), len(samples).to_bytes(4, "little"),
                       bytes(4)])  # fmt: skip
    frames = [len(s).to_bytes(4, "little") + i.to_bytes(8, "little") + s for i, s in enumerate(samples)]
    return header + b"".join(frames)


def decode_av1(samples: list[bytes], height: int, width: int) -> UInt16[ndarray, "n h w"]:
    """Reference decode through ffmpeg (gray10le, no range conversion), for checking an encode offline."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "ivf", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "gray10le", "pipe:1"]
    raw = subprocess.run(cmd, input=join_ivf(samples, height, width), stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(len(samples), height, width).astype(np.uint16)
