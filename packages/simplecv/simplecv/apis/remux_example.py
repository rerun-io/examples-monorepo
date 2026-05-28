"""Query video stream from a recording and mux it to an mp4 video file."""

from dataclasses import dataclass
from pathlib import Path

from simplecv.rerun_log_utils import mux_h264_to_mp4, read_h264_samples_from_rrd


@dataclass
class RemuxConfig:
    input_rrd: Path
    output: Path = Path("output.mp4")
    entity: str = "video_stream"
    timeline: str = "time"


def main(cfg: RemuxConfig) -> None:
    # Load recording data
    print(f"Loading recording from: {cfg.input_rrd}")
    times, samples = read_h264_samples_from_rrd(cfg.input_rrd, cfg.entity, cfg.timeline)
    print(f"Creating video file: {cfg.output}")
    mux_h264_to_mp4(times, samples, cfg.output)
