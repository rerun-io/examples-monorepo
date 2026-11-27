"""`simplecv.catalog_video_decode`: sequential decode on the reader gives the whole-stream decoder's frames, a chunk at a time."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

from video_fixtures import encode_h264  # noqa: E402

from simplecv import catalog_video_decode  # noqa: E402
from simplecv.catalog_video import read_catalog_videos  # noqa: E402
from simplecv.catalog_video_codec import wrap_mp4  # noqa: E402
from simplecv.catalog_video_decode import PrefetchedChunks, StreamDecoder  # noqa: E402
from simplecv.catalog_video_reader import CatalogVideoReader, VideoChunk, read_video_index, select_decode_device  # noqa: E402

pytestmark = pytest.mark.integration

FRAME_NS: int = 33_333_333
CAMERAS: dict[str, tuple[int, int, int]] = {"/cam_a/video": (95, 10, 0), "/cam_b/video": (87, 7, 4_000_000)}
"""entity: (frames, GOP, time offset in ns)"""


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[rr.server.Server]:
    path: Path = tmp_path_factory.mktemp("video") / "cameras.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for seed, (entity, (frames, gop, offset_ns)) in enumerate(CAMERAS.items()):
            samples: list[tuple[bytes, bool]] = encode_h264(frames, gop, seed)
            times = (offset_ns + FRAME_NS * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]")
            rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            rec.send_columns(
                entity,
                indexes=[rr.TimeColumn("video_time", duration=times)],
                columns=rr.VideoStream.columns(sample=[s for s, _ in samples], is_keyframe=[k for _, k in samples]),
            )
    with rr.server.Server(datasets={"test": [path]}) as running:
        yield running


def _reader(server: rr.server.Server) -> CatalogVideoReader:
    dataset = server.client().get_dataset("test")
    # 0.5 s chunks: 15 frames, so the batches below cross chunk boundaries.
    return CatalogVideoReader(dataset, "segment", read_video_index(dataset, "segment", list(CAMERAS), "video_time"), "video_time", chunk_s=0.5)


@pytest.fixture(scope="module")
def reference(server: rr.server.Server) -> list[torch.Tensor]:
    """Every frame NHWC from one decoder over the whole column (the path these classes replace), on the device the tests
    decode on: NVDEC's colour conversion differs from the CPU's by a level."""
    from torchcodec.decoders import VideoDecoder

    frames: list[torch.Tensor] = []
    for video in read_catalog_videos(server.client().get_dataset("test"), "segment", list(CAMERAS), "video_time"):
        decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec), device=select_decode_device(), seek_mode="exact")
        frames.append(decoder.get_frames_at(list(range(len(video.samples)))).data.permute(0, 2, 3, 1).cpu())
    return frames


def test_prefetched_batches_are_the_whole_stream_frames_on_the_selected_device(server: rr.server.Server, reference: list[torch.Tensor]) -> None:
    device: str = select_decode_device()
    wanted: list[int] = list(range(4, 86, 3))
    chunks: list[list[int]] = [wanted[begin : begin + 7] for begin in range(0, len(wanted), 7)]
    prefetched = PrefetchedChunks(_reader(server), chunks, 4, 87, device)
    batches: list[list[torch.Tensor]] = list(prefetched)
    assert len(batches) == len(chunks) and prefetched.wait_s >= 0.0
    for chunk, per_stream in zip(chunks, batches, strict=True):
        for frames, whole in zip(per_stream, reference, strict=True):
            assert frames.device.type == device
            assert torch.equal(frames.cpu(), whole[chunk])


def test_nvdec_decodes_whenever_cuda_is_available(server: rr.server.Server, reference: list[torch.Tensor]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    assert select_decode_device() == "cuda"
    wanted: list[int] = [5, 6, 14, 15, 16, 40, 59]
    frames: torch.Tensor = StreamDecoder(_reader(server), 1, 5, 60, "cuda").frames(wanted)
    assert frames.device.type == "cuda"
    assert torch.equal(frames.cpu(), reference[1][wanted])


def test_the_decoders_read_no_further_than_the_last_wanted_frame(server: rr.server.Server, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller passes the end of the clip while its batches stop early (a sample pack of a long clip): only the chunks up
    to the last wanted frame are fetched, and none is muxed past it."""
    muxed: list[tuple[int, int]] = []

    def counted_mux(samples: list[np.ndarray], keyframes: list[bool], fps: int, codec: str) -> bytes:
        muxed.append((len(samples), sum(keyframes)))
        return wrap_mp4(samples, keyframes, fps, codec=codec)

    monkeypatch.setattr(catalog_video_decode, "wrap_mp4", counted_mux)
    reader: CatalogVideoReader = _reader(server)
    fetched: list[int] = []
    fetch = reader.fetch

    def counted_fetch(chunk: int) -> VideoChunk:
        fetched.append(chunk)
        return fetch(chunk)

    monkeypatch.setattr(reader, "fetch", counted_fetch)
    assert reader.num_chunks >= 6
    batches = list(PrefetchedChunks(reader, [[10, 20], [30]], 10, 1000, select_decode_device()))
    assert [[len(frames) for frames in per_stream] for per_stream in batches] == [[2, 2], [1, 1]]
    assert fetched == list(range(max(reader.chunk_of(stream, 30) for stream in range(2)) + 1))
    # Each mux opens on the keyframe at or before its first frame and ends at the chunk's end or at frame 30.
    assert all(count <= 15 + 10 for count, _ in muxed)

    muxed.clear()
    assert len(StreamDecoder(reader, 0, 10, 13, "cpu").frames([10, 11, 12])) == 3
    assert muxed == [(3, 1)], "a caller's earlier end bounds the mux: keyframe 10 up to frame 12"
