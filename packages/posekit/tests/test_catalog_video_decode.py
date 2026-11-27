"""posekit's catalog video decodes on the chunked reader: NVDEC whenever CUDA is available, the frames (and the body
calibration's pixel hashes) of the whole-stream decoder they replace."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

import rerun as rr  # noqa: E402
from simplecv.testing import encode_h264

pytestmark = pytest.mark.integration

ENTITY: str = "/cam/video"
FRAMES: int = 70
GOP: int = 9


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[rr.server.Server]:
    """H.264 of moving noise (every frame distinct), closed GOPs, no B-frames, at 30 fps on ``video_time``."""
    samples: list[tuple[bytes, bool]] = encode_h264(FRAMES, GOP)
    path: Path = tmp_path_factory.mktemp("video") / "camera.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log(ENTITY, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.send_columns(
            ENTITY,
            indexes=[rr.TimeColumn("video_time", duration=(33_333_333 * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[sample for sample, _ in samples], is_keyframe=[key for _, key in samples]),
        )
    with rr.server.Server(datasets={"test": [path]}) as running:
        yield running


def _whole_stream_nhwc(server: rr.server.Server, device: str) -> torch.Tensor:
    """The replaced path: every packet muxed into one MP4, decoded with approximate seeks (body_calibrate, hands_eval)."""
    from simplecv.catalog_video_codec import wrap_mp4
    from simplecv.testing import whole_column
    from torchcodec.decoders import VideoDecoder

    video = whole_column(server.client().get_dataset("test"), "segment", [ENTITY], "video_time")[0]
    decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec), device=device, dimension_order="NHWC", seek_mode="approximate")
    return decoder.get_frames_at(list(range(len(video.samples)))).data.contiguous()


def test_people_masks_decode_selected_frames_on_nvdec(server: rr.server.Server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    from simplecv.catalog_video_reader import CatalogVideoReader, read_video_index, select_decode_device

    from posekit.apis.people_masks import decode_selected

    dataset = server.client().get_dataset("test")
    device: str = select_decode_device()
    assert device == "cuda"
    # 0.5 s chunks (15 frames): the selection crosses chunk boundaries and GOPs.
    reader = CatalogVideoReader(dataset, "segment", read_video_index(dataset, "segment", [ENTITY], "video_time"), "video_time", chunk_s=0.5)
    decoded, frames = decode_selected(reader, 0, [44, 3, 17, 18, 3, 69], device)
    assert decoded == [3, 17, 18, 44, 69]
    assert frames.device.type == "cuda"
    assert torch.equal(frames, _whole_stream_nhwc(server, device)[decoded])


def test_body_calibration_hashes_survive_the_chunked_decode(server: rr.server.Server) -> None:
    """body_calibrate stores, and body_accuracy checks, SHA256s of NVDEC frames: the chunked decode must keep them."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the hashes are of NVDEC frames")
    from simplecv.catalog_video_reader import open_decoders

    from posekit.models.sam3d_body_quantization import frame_sha256

    decoder = open_decoders(server.client().get_dataset("test"), "segment", [ENTITY], "video_time", torch.device("cuda"), chunk_s=0.5)[0]
    indices: list[int] = np.linspace(0, len(decoder) - 1, 12, dtype=np.int64)[1:-1].tolist()
    frames: torch.Tensor = decoder.get_frames_at(indices).permute(0, 2, 3, 1).contiguous()
    assert frames.device.type == "cuda"
    whole: torch.Tensor = _whole_stream_nhwc(server, "cuda")
    assert [frame_sha256(frame) for frame in frames] == [frame_sha256(whole[index]) for index in indices]
