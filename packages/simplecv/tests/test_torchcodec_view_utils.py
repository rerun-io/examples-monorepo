from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from jaxtyping import Int, UInt8

import simplecv._torchcodec_view_utils as view_utils


def test_build_blueprint_creates_video_and_decoded_views() -> None:
    """The Rerun layout places native video and decoded images side by side."""
    blueprint: Any = view_utils.build_blueprint(["cam_a/output"])
    time_selection: Any = blueprint.time_panel.time_selection

    assert blueprint is not None
    assert blueprint.time_panel.timeline == "video_time"
    assert blueprint.time_panel.loop_mode == "selection"
    assert time_selection is not None
    assert time_selection.min.value == 0


def test_log_torchcodec_decoded_chunks_logs_bgr_images_with_video_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decoded TorchCodec chunks are logged beside the native video stream."""
    timeline_events: list[tuple[str, float | None, int | None]] = []
    log_paths: list[str] = []
    logged_images: list["FakeImage"] = []

    class FakeImage:
        def __init__(self, image: UInt8[np.ndarray, "h w 3"], color_model: object) -> None:
            self.image: UInt8[np.ndarray, "h w 3"] = image
            self.color_model: object = color_model
            self.jpeg_quality: int | None = None

        def compress(self, jpeg_quality: int) -> "FakeImage":
            self.jpeg_quality = jpeg_quality
            return self

    def fake_set_time(timeline: str, *, duration: float | None = None, sequence: int | None = None) -> None:
        timeline_events.append((timeline, duration, sequence))

    def fake_log(path: str, image: FakeImage) -> None:
        log_paths.append(path)
        logged_images.append(image)

    monkeypatch.setattr(view_utils.rr, "set_time", fake_set_time)
    monkeypatch.setattr(view_utils.rr, "log", fake_log)
    monkeypatch.setattr(view_utils.rr, "Image", FakeImage)
    monkeypatch.setattr(view_utils.rr, "ColorModel", SimpleNamespace(BGR="BGR"))

    first_chunk: UInt8[torch.Tensor, "b 3 h w"] = torch.tensor(
        [
            [[[1]], [[2]], [[3]]],
            [[[4]], [[5]], [[6]]],
        ],
        dtype=torch.uint8,
    )
    second_chunk: UInt8[torch.Tensor, "b 3 h w"] = torch.tensor(
        [
            [[[7]], [[8]], [[9]]],
        ],
        dtype=torch.uint8,
    )
    frame_timestamps: Int[np.ndarray, "num_frames"] = np.array([10, 20, 30], dtype=np.int64)

    view_utils.log_torchcodec_decoded_chunks(
        chunked_videos=[[first_chunk], [second_chunk]],
        video_names=["cam"],
        frame_timestamps_by_video=[frame_timestamps],
        total_frames=3,
        chunk_size=2,
    )

    assert log_paths == ["/videos/cam/decoded", "/videos/cam/decoded", "/videos/cam/decoded"]
    video_time_events: list[tuple[str, float | None, int | None]] = [
        event for event in timeline_events if event[0] == "video_time"
    ]
    assert [event[1] for event in video_time_events] == pytest.approx([10e-9, 20e-9, 30e-9])
    assert [event[2] for event in video_time_events] == [None, None, None]
    assert [event for event in timeline_events if event[0] == "frame"] == [
        ("frame", None, 0),
        ("frame", None, 1),
        ("frame", None, 2),
    ]
    assert logged_images[0].image[0, 0].tolist() == [3, 2, 1]
    assert logged_images[0].color_model == "BGR"
    assert logged_images[0].jpeg_quality == 80


def test_log_torchcodec_decoded_chunks_rejects_nonpositive_chunk_size() -> None:
    """Chunk logging fails clearly before deriving progress totals."""
    frame_timestamps: Int[np.ndarray, "num_frames"] = np.array([10], dtype=np.int64)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        view_utils.log_torchcodec_decoded_chunks(
            chunked_videos=[],
            video_names=["cam"],
            frame_timestamps_by_video=[frame_timestamps],
            total_frames=1,
            chunk_size=0,
        )
