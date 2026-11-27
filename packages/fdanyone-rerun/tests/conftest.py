"""Shared fixtures: one GPU run of the example clip per session (integration and golden lanes) and a decoder of logged videos."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
import rerun as rr
from av.video.codeccontext import VideoCodecContext


@pytest.fixture(scope="session")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the Space's policy (six views, one ring at 15 degrees, Turbo, seed 42) on the example clip, or reuse
    ``$FDANYONE_TEST_RUN`` (a finished run directory of the same policy)."""
    reuse: str | None = os.environ.get("FDANYONE_TEST_RUN")
    if reuse:
        return Path(reuse)
    # Imported here, not at module level: pytest-beartype hooks imports after conftest.py loads (AGENTS.md, Beartype).
    from fdanyone_rerun.assets import EXAMPLE_VIDEO, model_dir

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not EXAMPLE_VIDEO.is_file() or not (model_dir() / "4danyone" / "model.safetensors").is_file():
        pytest.skip("needs the weights and the example clip: pixi run -e fdanyone-rerun-dev fdanyone-download")
    from fdanyone_rerun.core import RunConfig, run_views

    output_dir: Path = tmp_path_factory.mktemp("fdanyone") / "example"
    run_views(RunConfig(video=EXAMPLE_VIDEO, output_dir=output_dir))
    return output_dir


VideoFrames = Callable[[Path, str, str], list[np.ndarray]]


@pytest.fixture
def rrd_video_frames() -> VideoFrames:
    """``decode(rrd, entity, timeline)``: every frame of the ``rr.VideoStream`` logged at ``entity``, as RGB uint8."""

    def decode(rrd: Path, entity: str, timeline: str) -> list[np.ndarray]:
        from simplecv.testing import read_video_stream_from_rrd

        codec, _, samples = read_video_stream_from_rrd(str(rrd), entity, timeline)
        names: dict[rr.VideoCodec, str] = {rr.VideoCodec.H264: "h264", rr.VideoCodec.H265: "hevc", rr.VideoCodec.AV1: "av1"}
        decoder: VideoCodecContext = cast(VideoCodecContext, av.CodecContext.create(names[codec], "r"))
        frames: list[np.ndarray] = []
        # One row per sample, each a list of blobs; then flush the parser (it holds the last access unit), then the decoder.
        for row in samples.to_pylist():
            for packet in decoder.parse(b"".join(bytes(blob) for blob in row)):
                frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
        for packet in decoder.parse(None):
            frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
        frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(None))
        return frames

    return decode
