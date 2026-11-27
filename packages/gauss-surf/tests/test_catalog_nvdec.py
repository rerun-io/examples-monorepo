"""The segment reader and frame selection decode on NVDEC, from a catalog served over a synthetic H.264 recording."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("av", reason="PyAV encodes the synthetic video")
rr = pytest.importorskip("rerun", reason="the catalog server comes from rerun-sdk")
pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")
pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

from arkitscenes_download.ingest.paths import TIMELINE, VIDEO_WIDE  # noqa: E402
from rerun.catalog import DatasetEntry  # noqa: E402
from simplecv.catalog_video_codec import wrap_mp4  # noqa: E402
from simplecv.testing import encode_h264  # noqa: E402
from torch import Tensor  # noqa: E402

FRAMES: int = 40
TIMES = (16_666_667 * np.arange(FRAMES, dtype=np.int64)).astype("timedelta64[ns]")


@pytest.fixture
def samples() -> list[tuple[bytes, bool]]:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    return encode_h264(FRAMES, 8)


@pytest.fixture
def dataset(tmp_path: Path, samples: list[tuple[bytes, bool]]) -> Iterator[DatasetEntry]:
    path: Path = tmp_path / "segment.rrd"
    with rr.RecordingStream("test", recording_id="segment") as recording:
        recording.save(path)
        recording.log(VIDEO_WIDE, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        recording.send_columns(
            VIDEO_WIDE,
            indexes=[rr.TimeColumn(TIMELINE, duration=TIMES)],
            columns=rr.VideoStream.columns(sample=[sample for sample, _ in samples], is_keyframe=[keyframe for _, keyframe in samples]),
        )
    with rr.server.Server(datasets={"test": [path]}) as server:
        yield server.client().get_dataset("test")


def _whole_stream(samples: list[tuple[bytes, bool]]) -> Tensor:
    from torchcodec.decoders import VideoDecoder

    mp4: bytes = wrap_mp4([sample for sample, _ in samples], [keyframe for _, keyframe in samples], 30, codec="h264")
    return VideoDecoder(mp4, device="cuda", seek_mode="exact").get_frames_at(list(range(FRAMES))).data


def test_decode_frames_yields_the_requested_frames_on_cuda(dataset: DatasetEntry, samples: list[tuple[bytes, bool]]) -> None:
    from gauss_surf.catalog import SegmentReader

    wanted: list[int] = [33, 0, 7, 8, 39]
    frames: list[Tensor] = list(SegmentReader(dataset, "segment").decode_frames(VIDEO_WIDE, TIMES[wanted], device=torch.device("cuda")))

    whole: Tensor = _whole_stream(samples)
    assert [frame.device.type for frame in frames] == ["cuda"] * len(wanted)
    for frame, index in zip(frames, wanted, strict=True):
        assert torch.equal(frame, whole[index])


def test_frame_selection_scores_every_packet_from_cuda_frames(dataset: DatasetEntry, monkeypatch: pytest.MonkeyPatch) -> None:
    import gauss_surf.apis.frame_selection as frame_selection

    devices: list[str] = []
    score = frame_selection.sharpness_scores

    def recording_scores(frames_bhw3: Tensor) -> Tensor:
        devices.append(frames_bhw3.device.type)
        return score(frames_bhw3)

    monkeypatch.setattr(frame_selection, "sharpness_scores", recording_scores)
    measured = frame_selection._score_camera(dataset, "segment", VIDEO_WIDE, torch.device("cuda"), batch_size=16)

    assert devices == ["cuda"] * 3
    np.testing.assert_array_equal(measured.timestamps, TIMES)
    assert measured.sharpness.shape == (FRAMES,) and np.isfinite(measured.sharpness).all()
