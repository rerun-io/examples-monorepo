"""Write the stitched panorama of an Insta360 recording as a derived, viewing-only layer ``.rrd``.

Every frameset's two lens images go through ``slam_rs._core.PanoramaStitch``, the dual-fisheye lane's own tables and
host remap, into the 1920x960 panorama the lane tracks on (lens 0's hemisphere on the left). The frames stream into
one AV1 NVENC encode (no B-frames, as ``VideoStream`` needs) and become a ``VideoStream`` at ``/world/rig_00/pano`` on
the recording's own ``video_time`` frame times, in a separate layer file whose application and recording ids are
the base recording's, so a viewer that opens both overlays them. The clip is never held in memory: the feed decodes
one window of lens samples at a time and ffmpeg takes each panorama as it is made.

The two lenses expose separately, so a brightness step shows at the seams. This tool does not correct it: the
tracker sees the same step, and the layer shows what the tracker saw.

The layer is for viewing only; nothing reads it back as input. AV1 plays in the native viewer and in the web viewer,
which the lens tracks' H.265 does not reach everywhere.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.chunk as rrc
from dataforge.logging_toolkit import log_video_stream
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, work_dir
from jaxtyping import Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import DEFAULT_WINDOW_S, LocalSegment, RigProfile, open_segment
from slam_rs.vio_log import PANORAMA_HEIGHT, PANORAMA_WIDTH

PANO_ENTITY: str = "/world/rig_00/pano"
"""Where the layer puts the panorama video."""


@dataclass(slots=True)
class Config:
    """Stitch every frameset of an Insta360 recording and write the panorama as a derived AV1 layer."""

    rrd: Path
    """The Insta360 base recording (``slam-rs-insv`` output)."""
    output: Path | None = None
    """Destination layer; defaults to ``<recording stem>.pano.rrd`` beside the recording."""
    margin_px: int = 64
    """The lane's tile overlap (``config.panorama_tile_margin_px`` of the insta360 config); recorded, pixels do not depend on it."""
    cq: int = 34
    """NVENC constant-quality target: a viewing layer, not an analysis input."""
    gop: int = 30
    """Keyframe interval in frames, about one second: the viewer seeks to any time within a second of decode."""
    decode_threads: int = 4
    """PyAV threads per lens decoder."""
    max_frames: int | None = None
    """Stitch only the first framesets; None stitches the whole recording."""
    window_s: float = DEFAULT_WINDOW_S
    """Longest stretch of lens samples the feed reads at once."""


@dataclass(frozen=True, slots=True)
class PanoLayer:
    """What one run wrote."""

    path: Path
    """The layer file."""
    frames: int
    """Panorama frames encoded."""
    duration_s: float
    """First to last frame time plus one frame period."""
    video_bytes: int
    """Size of the AV1 elementary stream (the encoded mp4)."""
    wall_s: float
    """Wall time of the whole run."""

    @property
    def bitrate_mbps(self) -> float:
        """Mean video bitrate, Mbit/s."""
        return self.video_bytes * 8 / self.duration_s / 1e6


def layer_path(rrd: Path) -> Path:
    """``<recording stem>.pano.rrd`` beside the recording."""
    return rrd.with_name(f"{rrd.stem}.pano.rrd")


def layer_duration_s(times_ns: Int64[ndarray, " n"]) -> float:
    """First to last frame time plus one median frame period, seconds; one frame counts as one 30 fps period."""
    period_ns: float = float(np.median(np.diff(times_ns))) if times_ns.size > 1 else 1e9 / 30
    return (float(times_ns[-1] - times_ns[0]) + period_ns) * 1e-9


def write_layer(config: Config) -> PanoLayer:
    """Stitch, encode and write the layer.

    Raises:
        FileNotFoundError: If the recording does not exist.
        ValueError: If it is not an Insta360 recording, or holds other than one recording.
    """
    if not config.rrd.is_file():
        raise FileNotFoundError(f"{config.rrd}: no such recording")
    start: float = time.perf_counter()
    stores: list[rrc.StoreEntry] = rrc.RrdReader(config.rrd).recordings()
    if len(stores) != 1:
        raise ValueError(f"{config.rrd}: expected one recording, found {len(stores)}")
    output: Path = config.output if config.output is not None else layer_path(config.rrd)
    times: list[int] = []
    with open_segment(
        LocalSegment(config.rrd), profile=RigProfile(decode_threads=config.decode_threads), window_s=config.window_s, include_ground_truth=False
    ) as feed:
        factory = feed.insta360
        if factory is None:
            raise ValueError(f"{config.rrd}: not an Insta360 recording (no insta360 property)")
        lens = feed.cameras[0]
        stitch = _core.PanoramaStitch(factory.lens_string, factory.window_crop, (lens.width, lens.height), margin_px=config.margin_px)
        offset_ns: int = feed.imu.cam_time_offset_ns

        def panoramas() -> Iterator[bytes]:
            for frameset in feed.framesets(feed.stop_ns_after(config.max_frames)):
                if config.max_frames is not None and len(times) == config.max_frames:
                    return
                times.append(frameset.t_ns - offset_ns)  # back on the recording's video_time
                yield stitch.panorama(frameset.images[0], frameset.images[1]).tobytes()

        with work_dir("insv-pano-") as work:
            clip: Path = work / "pano.mp4"
            # fps is the container's nominal rate only; log_video_stream puts every sample on its real frame time.
            encode_frames_to_mp4(
                panoramas(), clip, source=FrameSource("gray8", width=PANORAMA_WIDTH, height=PANORAMA_HEIGHT), fps=30, gop=config.gop, cq=config.cq
            )
            times_ns: Int64[ndarray, " n"] = np.asarray(times, dtype=np.int64)
            with rr.RecordingStream(stores[0].application_id, recording_id=stores[0].recording_id) as rec:
                rec.save(output)
                rec.send_property(
                    "pano_layer",
                    rr.AnyValues(
                        derived=True,
                        viewing_only=True,
                        note="Stitched for viewing only; nothing reads it as input. Lens brightness steps at the seams are not corrected.",
                        source_recording=config.rrd.name,
                        source_segment=feed.segment_id,
                        stitch="slam_rs._core.PanoramaStitch (dual-fisheye lane tables + host remap)",
                        lens_string_version="v6",
                        margin_px=config.margin_px,
                        codec="av1",
                        cq=config.cq,
                        gop=config.gop,
                    ),
                )
                log_video_stream(rec, clip, PANO_ENTITY, times_ns=times_ns)
            video_bytes: int = clip.stat().st_size
    return PanoLayer(output, len(times), layer_duration_s(times_ns), video_bytes, time.perf_counter() - start)


def main(config: Config) -> None:
    """Write the layer and report its size and bitrate."""
    layer: PanoLayer = write_layer(config)
    print(
        f"wrote {layer.frames} panorama frames ({layer.duration_s:.1f} s) to {layer.path} in {layer.wall_s:.1f} s: "
        f"AV1 {layer.video_bytes / 1e6:.1f} MB, {layer.bitrate_mbps:.2f} Mbit/s, layer {layer.path.stat().st_size / 1e6:.1f} MB"
    )
