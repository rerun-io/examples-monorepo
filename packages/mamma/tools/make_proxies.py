"""Transcode a sequence's 4K source videos to 720p H.264 proxies, once.

The streaming pipeline's wall time on a 4K-source clip is dominated by NVDEC
decoding the full-resolution HEVC before resizing (torchcodec has no
scaled-decode path); the dig measured a ~23 s irreducible decode floor on
running_jumping. A 720p H.264 proxy decodes ~9x cheaper, so a one-time offline
transcode lets the resident loop hit its realtime budget without touching
fidelity meaningfully (proxies feed track + 720p MammaNet crops, which the
fast preset already validated at 720p).

Proxies land in ``<data_dir>/proxies_<h>p/<cam>.mp4`` next to ``videos_light``;
pass that dir to ``--proxy-dir`` on the demo/dump/benchmark tools.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence


@dataclass
class MakeProxiesConfig:
    data_dir: Path = Path("data/inputs/outdoors/running_jumping")
    """Capture inputs (MAMMA NPZ layout) whose videos to proxy."""
    height: int = 720
    """Proxy height; width is derived to preserve aspect (16:9 -> 1280)."""
    cq: int = 23
    """NVENC constant-quality (lower = higher quality/larger). 23 is visually
    lossless at 720p for this content."""
    encoder: str = "h264_nvenc"
    """Proxy codec. H.264 (not the repo's AV1 default) so the streaming reader
    decodes proxies as cheaply as possible. Falls back to libx264 if NVENC is
    unavailable."""
    overwrite: bool = False
    """Re-transcode even if a proxy already exists."""


def _encoder_available(name: str) -> bool:
    import subprocess

    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return name in out.stdout


def main(config: MakeProxiesConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    width: int = int(round(config.height * sequence.cameras[0].width / sequence.cameras[0].height))
    width -= width % 2  # H.264 needs even dims
    proxy_dir: Path = config.data_dir / f"proxies_{config.height}p"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    encoder: str = config.encoder if _encoder_available(config.encoder) else "libx264"
    print(f"{sequence.name}: {len(sequence.video_paths)} cams -> {width}x{config.height} via {encoder} (cq {config.cq})")

    for cam_name, src in zip(sequence.camera_names, sequence.video_paths, strict=True):
        dst: Path = proxy_dir / f"{cam_name}.mp4"
        if dst.exists() and not config.overwrite:
            print(f"  {cam_name}: exists, skip")
            continue
        enc_args: list[str] = ["-c:v", encoder, "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0"]
        if "nvenc" in encoder:
            enc_args += ["-cq", str(config.cq)]
        else:
            enc_args += ["-crf", str(config.cq)]
        cmd: list[str] = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", f"scale={width}:{config.height}:flags=area",
            *enc_args,
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed for {cam_name}: {result.stderr[-300:]}")
        mb: float = dst.stat().st_size / 1e6
        print(f"  {cam_name}: {dst.name} ({mb:.1f} MB)")
    print(f"proxies in {proxy_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(MakeProxiesConfig)))
