"""Reproducible SHOW3D video comparison; run on the host with NVENC access."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json

from dataforge import archives, paths, writing
from dataforge.datasets.show3d_layers import VIDEO_GOP
from dataforge.datasets.show3d_source import CAMERAS, RecordingInfo, read_json, validate_component
from dataforge.logging_toolkit import log_video_stream
from dataforge.video_encoding import transcode_mp4_gray


@dataclass
class Config:
    """Compare CQ 28/32/36 on complete scenes."""

    root: Path = field(default_factory=lambda: paths.raw_root() / "show3d")
    """Raw Hub layout containing both scenes."""
    sequences: tuple[str, ...] = ("SPI102/keyboard_toss-away_83ef", "LYA722/birdhousetoy_shaking_8eca")
    """Subject/scene keys to measure."""
    output: Path = Path("data/show3d-video-measurements.json")
    """JSON report, published after all measurements succeed."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Measurement:
    """One complete scene/mode, with 20 grayscale PSNR samples per camera."""

    scene: str
    """Subject/scene key."""
    mode: str
    """cq28/cq32/cq36."""
    source_bytes: int
    """Total source MP4 size."""
    rrd_bytes: int
    """Total video-only recording size."""
    wall_seconds: float
    """Encoding plus RRD writing, excluding the PSNR decode pass."""
    median_psnr_db: float
    """Median over all sampled camera frames; 20 evenly spaced frames per camera."""


def sample_gray(path: Path, selected: set[int]) -> dict[int, UInt8[ndarray, "h w"]]:
    """Decode the source once and retain only selected presentation frames."""
    with av.open(str(path)) as container:
        return {index: frame.to_ndarray(format="gray") for index, frame in enumerate(container.decode(video=0)) if index in selected}


def rrd_psnr(path: Path, references: dict[str, dict[int, UInt8[ndarray, "h w"]]], expected_count: int) -> list[float]:
    """Stream the recording once, dispatching AV1 packets by camera entity."""
    decoders: dict[str, av.video.codeccontext.VideoCodecContext] = {}
    counts: dict[str, int] = dict.fromkeys(references, 0)
    for entity in references:
        decoder: av.CodecContext = av.CodecContext.create("libdav1d", "r")
        if not isinstance(decoder, av.video.codeccontext.VideoCodecContext):
            raise TypeError("AV1 decoder did not create a video codec context")
        decoders[entity] = decoder

    def decoded_packets() -> Iterator[tuple[str, av.VideoFrame]]:
        for chunk in rr.experimental.RrdReader(path).stream():
            entity: str = str(chunk.entity_path)
            batch: pa.RecordBatch = chunk.to_record_batch()
            if entity not in decoders or "VideoStream:sample" not in batch.schema.names:
                continue
            for payload in batch.column("VideoStream:sample").to_pylist():
                for frame in decoders[entity].decode(av.Packet(bytes(payload[0]))):
                    yield entity, frame

    def drained_frames() -> Iterator[tuple[str, av.VideoFrame]]:
        for entity, decoder in decoders.items():
            for frame in decoder.decode(None):
                yield entity, frame

    scores: list[float] = []
    for entity, frame in chain(decoded_packets(), drained_frames()):
        count: int = counts[entity]
        reference: UInt8[ndarray, "h w"] | None = references[entity].get(count)
        if reference is not None:
            decoded: UInt8[ndarray, "h w"] = frame.to_ndarray(format="gray")
            mse: float = float(np.mean((decoded.astype(np.float64) - reference.astype(np.float64)) ** 2))
            scores.append(float(10.0 * np.log10(255.0**2 / max(mse, 1e-12))))
        counts[entity] += 1
    for entity, count in counts.items():
        if count != expected_count:
            raise ValueError(f"{entity}: decoded {count} RRD frames, expected {expected_count}")
    if len(scores) != sum(len(reference) for reference in references.values()):
        raise ValueError("PSNR sample count mismatch")
    return scores


def main(config: Config) -> None:
    """Measure each mode; leave no temporary encoded clips after each scene."""
    measurements: list[Measurement] = []
    for key in config.sequences:
        parts: tuple[str, ...] = tuple(key.split("/"))
        if len(parts) != 2:
            raise ValueError(f"invalid subject/scene key: {key}")
        for part in parts:
            validate_component(part)
        scene: Path = config.root / "scenes" / key
        info: RecordingInfo = read_json(scene / "metadata/recording_info.json", RecordingInfo)
        videos: dict[str, Path] = {f"/video/{name}": scene / f"{name}.mp4" for camera in CAMERAS if (name := camera.source_name) in info.resolution}
        selected: set[int] = set(np.linspace(0, info.num_frames - 1, min(20, info.num_frames), dtype=np.int64).tolist())
        references: dict[str, dict[int, UInt8[ndarray, "h w"]]] = {entity: sample_gray(source, selected) for entity, source in videos.items()}
        work: Path = config.root / "work" / "measure" / key
        work.mkdir(parents=True, exist_ok=True)
        try:
            for cq in (28, 32, 36):
                mode: str = f"cq{cq}"
                target: Path = work / f"{mode}.rrd"
                start: float = time.perf_counter()
                with writing.atomic_recording(target, recording_id=f"measure-{mode}") as recording:
                    for entity, source in videos.items():
                        clip: Path = work / f"{source.stem}-{mode}.mp4"
                        try:
                            transcode_mp4_gray(source, clip, fps=int(info.fps), gop=VIDEO_GOP, cq=cq, frames=info.num_frames)
                            log_video_stream(recording, clip, entity)
                        finally:
                            clip.unlink(missing_ok=True)
                elapsed: float = time.perf_counter() - start
                scores: list[float] = rrd_psnr(target, references, info.num_frames)
                measured: Measurement = Measurement(
                    key, mode, sum(path.stat().st_size for path in videos.values()), target.stat().st_size, elapsed, float(np.median(scores))
                )
                measurements.append(measured)
                print(
                    f"{measured.scene} {measured.mode}: source={measured.source_bytes} B, RRD={measured.rrd_bytes} B, "
                    f"time={measured.wall_seconds:.3f} s, median PSNR={measured.median_psnr_db:.3f} dB"
                )
                target.unlink()
        finally:
            archives.remove_tree(work)
    with writing.atomic_write(config.output) as staged:
        staged.write_text(to_json(measurements))
