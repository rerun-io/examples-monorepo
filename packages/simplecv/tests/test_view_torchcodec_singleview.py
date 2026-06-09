from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
from jaxtyping import Int, UInt8

import simplecv.apis.view_torchcodec_singleview as singleview
from simplecv.rerun_log_utils import RerunTyroConfig


def test_singleview_main_logs_video_stream_and_decoded_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Singleview logs the native video stream and matching TorchCodec decoded chunks."""
    video_path: Path = tmp_path / "sample.mp4"
    video_path.touch()
    sent_blueprints: list[Any] = []
    logged_videos: list[tuple[Path, Path, str]] = []
    decoded_calls: list[dict[str, Any]] = []

    class FakeTorchCodecVideoReader(singleview.TorchCodecVideoReader):
        instances: list["FakeTorchCodecVideoReader"] = []

        def __init__(self, source: Path, *, device: str, seek_mode: Literal["exact", "approximate"]) -> None:
            self.received_source: Path = source
            self.device: str = device
            self.seek_mode: Literal["exact", "approximate"] = seek_mode
            self.frame_count: int = 5
            self.ranges: list[tuple[int, int]] = []
            self.instances.append(self)

        def __len__(self) -> int:
            return self.frame_count

        def get_frames_in_range(self, start: int, stop: int) -> UInt8[torch.Tensor, "b 3 h w"]:
            self.ranges.append((start, stop))
            frame_count: int = stop - start
            video: UInt8[torch.Tensor, "b 3 h w"] = torch.zeros((frame_count, 3, 1, 1), dtype=torch.uint8)
            return video

    def fake_send_blueprint(blueprint: Any) -> None:
        sent_blueprints.append(blueprint)

    def fake_log_video(video_path_arg: Path, video_log_path: Path, timeline: str) -> Int[np.ndarray, "num_frames"]:
        logged_videos.append((video_path_arg, video_log_path, timeline))
        timestamps: Int[np.ndarray, "num_frames"] = np.arange(5, dtype=np.int64)
        return timestamps

    def fake_log_torchcodec_decoded_chunks(**kwargs: Any) -> None:
        chunks: list[list[UInt8[torch.Tensor, "b 3 h w"]]] = list(kwargs["chunked_videos"])
        kwargs["chunked_videos"] = chunks
        decoded_calls.append(kwargs)

    monkeypatch.setattr(singleview.rr, "send_blueprint", fake_send_blueprint)
    monkeypatch.setattr(singleview, "log_video", fake_log_video)
    monkeypatch.setattr(singleview, "TorchCodecVideoReader", FakeTorchCodecVideoReader)
    monkeypatch.setattr(singleview, "log_torchcodec_decoded_chunks", fake_log_torchcodec_decoded_chunks)

    rr_config: RerunTyroConfig = RerunTyroConfig(headless=True)
    config: singleview.TorchCodecSingleviewConfig = singleview.TorchCodecSingleviewConfig(
        rr_config=rr_config,
        video_path=video_path,
        max_frames=3,
        chunk_size=2,
        device="cpu",
    )

    singleview.main(config)

    assert len(sent_blueprints) == 1
    assert logged_videos == [(video_path, Path("/videos/sample/video"), "video_time")]
    assert FakeTorchCodecVideoReader.instances[0].received_source == video_path
    assert FakeTorchCodecVideoReader.instances[0].device == "cpu"
    assert FakeTorchCodecVideoReader.instances[0].seek_mode == "approximate"
    assert FakeTorchCodecVideoReader.instances[0].ranges == [(0, 2), (2, 3)]
    assert decoded_calls[0]["video_names"] == ["sample"]
    assert decoded_calls[0]["total_frames"] == 3
    assert decoded_calls[0]["chunk_size"] == 2
    assert [chunk[0].shape[0] for chunk in decoded_calls[0]["chunked_videos"]] == [2, 1]
