"""SelfCap's catalog video decodes chunk by chunk on NVDEC and gives the frames of the whole-stream decoder it replaces."""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
import torch

pytestmark = pytest.mark.integration

FIXTURE: Path = Path(__file__).parents[2] / "dataforge/tests/fixtures/av1_48f_192x160.mp4"
"""48 AV1 frames, keyframes every 12."""
ENTITY: str = "/world/rig_00/cam_00/pinhole/video"


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[rr.server.Server]:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: these callers decode on NVDEC")
    with av.open(str(FIXTURE)) as container:
        packets = [(bytes(packet), bool(packet.is_keyframe)) for packet in container.demux(video=0) if packet.size]
    path: Path = tmp_path_factory.mktemp("video") / "phone.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log(ENTITY, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
        rec.send_columns(
            ENTITY,
            indexes=[rr.TimeColumn("capture_time", duration=(41_666_667 * np.arange(len(packets), dtype=np.int64)).astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[sample for sample, _ in packets], is_keyframe=[key for _, key in packets]),
        )
    with rr.server.Server(datasets={"test": [path]}) as running:
        yield running


def _whole_stream(server: rr.server.Server) -> torch.Tensor:
    """The replaced path: every packet in one MP4, approximate seeks, on CUDA."""
    from simplecv.catalog_video import read_catalog_videos
    from simplecv.catalog_video_codec import wrap_mp4
    from torchcodec.decoders import VideoDecoder

    video = read_catalog_videos(server.client().get_dataset("test"), "segment", [ENTITY], "capture_time")[0]
    decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 24, codec=video.codec), device="cuda", seek_mode="approximate")
    return decoder.get_frames_at(list(range(len(video.samples)))).data


def _decoder(server: rr.server.Server):
    from simplecv.catalog_video_reader import open_decoders

    # 0.5 s chunks (12 frames): the requests below cross chunks.
    return open_decoders(server.client().get_dataset("test"), "segment", [ENTITY], "capture_time", torch.device("cuda"), chunk_s=0.5)[0]


def test_picked_hand_frames_decode_on_nvdec(server: rr.server.Server) -> None:
    from selfcap_curate.hands_gpu import decode_frames

    picked: np.ndarray = np.array([3, 11, 12, 30, 47], dtype=np.int64)
    frames: torch.Tensor = decode_frames(_decoder(server), picked)
    assert frames.device.type == "cuda"
    assert torch.equal(frames, _whole_stream(server)[picked].permute(0, 2, 3, 1))


def test_link_luma_decodes_on_nvdec(server: rr.server.Server) -> None:
    import torch.nn.functional as F

    from selfcap_curate.links import gray_frames

    grays: torch.Tensor = torch.cat(list(gray_frames(_decoder(server), 5, 40, 0.5)))
    assert grays.device.type == "cuda" and grays.shape == (36, 80, 96)
    rgb: torch.Tensor = _whole_stream(server)[5:41].float()
    luma = (rgb * torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)).sum(1, keepdim=True)
    expected = F.interpolate(luma, size=(80, 96), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255).to(torch.uint8)[:, 0]
    assert torch.equal(grays, expected)
