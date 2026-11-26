"""Chunked catalog video reads give the frames of the whole-column path, and never hold the whole column."""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

from simplecv.catalog_video import CatalogVideo, read_catalog_videos  # noqa: E402
from simplecv.catalog_video_codec import wrap_mp4  # noqa: E402
from simplecv.catalog_video_reader import (  # noqa: E402
    CatalogVideoReader,
    ChunkedVideoDecoder,
    SharedChunks,
    StreamIndex,
    VideoChunk,
    iter_chunks,
    read_video_index,
)

pytestmark = pytest.mark.integration

FRAME_NS: int = 33_333_333
CAMERAS: dict[str, tuple[int, int, int, bool]] = {
    # entity: (frames, GOP, time offset in ns, keyframe flags logged on keyframes only)
    "/rig/cam_a/video": (95, 10, 0, True),
    "/rig/cam_b/video": (87, 7, 4_000_000, False),
}


def _encode(frames: int, gop: int, seed: int) -> list[tuple[bytes, bool]]:
    """H.264 samples of moving noise (every frame distinct), closed GOPs of ``gop`` frames, no B-frames."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 96, 64, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"g": str(gop), "keyint_min": str(gop), "bf": "0", "sc_threshold": "0"}
    rng: np.random.Generator = np.random.default_rng(seed)
    base: np.ndarray = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    samples: list[tuple[bytes, bool]] = []
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(np.roll(base, 3 * index, axis=1), format="rgb24")
        frame.pts = index
        samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(frame)]
    samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(None)]
    return samples


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path: Path = tmp_path_factory.mktemp("video") / "cameras.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for seed, (entity, (frames, gop, offset_ns, sparse)) in enumerate(CAMERAS.items()):
            samples: list[tuple[bytes, bool]] = _encode(frames, gop, seed)
            times = (offset_ns + FRAME_NS * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]")
            rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            rec.send_columns(
                entity, indexes=[rr.TimeColumn("video_time", duration=times)], columns=rr.VideoStream.columns(sample=[s for s, _ in samples])
            )
            flags: np.ndarray = np.asarray([k for _, k in samples])
            flagged: np.ndarray = flags if not sparse else flags[flags]
            rec.send_columns(
                entity,
                indexes=[rr.TimeColumn("video_time", duration=times if not sparse else times[flags])],
                columns=rr.VideoStream.columns(is_keyframe=flagged.tolist()),
            )
    return path


@pytest.fixture(scope="module")
def server(recording: Path) -> Iterator[rr.server.Server]:
    with rr.server.Server(datasets={"test": [recording]}) as running:
        yield running


@pytest.fixture(scope="module")
def whole(server: rr.server.Server) -> tuple[CatalogVideo, ...]:
    """The reference: today's whole-column read."""
    return read_catalog_videos(server.client().get_dataset("test"), "segment", list(CAMERAS), "video_time")


@pytest.fixture(scope="module")
def index(server: rr.server.Server) -> tuple[StreamIndex, ...]:
    # 0.7 s windows: several index queries per stream.
    return read_video_index(server.client().get_dataset("test"), "segment", list(CAMERAS), "video_time", window_s=0.7)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _whole_frames(video: CatalogVideo) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec), device=str(_device()), seek_mode="exact")
    return decoder.get_frames_at(list(range(len(video.samples)))).data


def test_index_lists_every_sample_without_reading_blobs(index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]) -> None:
    for stream, video in zip(index, whole, strict=True):
        np.testing.assert_array_equal(stream.times, video.times)
        np.testing.assert_array_equal(stream.keyframes, video.keyframes)
        assert stream.codec == video.codec == "h264"
        assert stream.times.dtype == np.dtype("timedelta64[ns]")
    assert sum(index[0].keyframes) == 10  # sparse flags: 95 frames, GOP 10


def test_chunks_partition_each_stream_into_whole_gops(
    server: rr.server.Server, index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]
) -> None:
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    assert reader.num_chunks >= 5
    chunks: list[VideoChunk] = [reader.fetch(chunk) for chunk in range(reader.num_chunks)]
    for position, video in enumerate(whole):
        joined: list[bytes] = [bytes(sample) for chunk in chunks for sample in chunk.streams[position].samples]
        assert joined == [bytes(sample) for sample in video.samples]
        for chunk in chunks:
            packets = chunk.streams[position]
            assert not packets.samples or packets.first == 0 or packets.keyframes[0]
    assert reader.fetch_count == reader.num_chunks


def test_random_access_gives_the_whole_stream_decoders_frames(
    server: rr.server.Server, index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]
) -> None:
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    for position, video in enumerate(whole):
        reference: torch.Tensor = _whole_frames(video)
        count: int = len(video.samples)
        boundaries: list[int] = [int(b) + d for b in reader.starts[1:-1, position] for d in (-1, 0, 1)]
        # First, last, both sides of every chunk boundary, a middle window, out of order and repeated.
        wanted: list[int] = sorted({0, count - 1, *[b for b in boundaries if 0 <= b < count]}) + list(range(40, 52)) + [count - 1, 3, 3, 61]
        decoder = ChunkedVideoDecoder(reader, position, _device())
        frames: torch.Tensor = decoder.get_frames_at(wanted)
        assert frames.shape[0] == len(wanted)
        assert torch.equal(frames, reference[wanted])


def test_sequential_read_ahead_decodes_every_frame_once(
    server: rr.server.Server, index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]
) -> None:
    from torchcodec.decoders import VideoDecoder

    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    decoded: list[list[torch.Tensor]] = [[] for _ in whole]
    for chunk in iter_chunks(reader, range(reader.num_chunks)):
        for position, packets in enumerate(chunk.streams):
            if packets.samples:
                decoder = VideoDecoder(packets.mp4("h264"), device=str(_device()), seek_mode="exact")
                decoded[position].append(decoder.get_frames_at(list(range(len(packets.samples)))).data)
    for position, video in enumerate(whole):
        assert torch.equal(torch.cat(decoded[position]), _whole_frames(video))


def test_reader_never_holds_the_whole_column(server: rr.server.Server, index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]) -> None:
    total: int = sum(len(sample) for video in whole for sample in video.samples)
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5, cache_bytes=total // 8)
    baseline: int = pa.total_allocated_bytes()
    peak: int = 0
    decoders: list[ChunkedVideoDecoder] = [ChunkedVideoDecoder(reader, position, _device()) for position in range(len(whole))]
    for position, video in enumerate(whole):
        for first in range(0, len(video.samples), 8):
            decoders[position].get_frames_at(list(range(first, min(first + 8, len(video.samples)))))
            peak = max(peak, pa.total_allocated_bytes() - baseline)
    largest: int = max(reader.fetch(chunk).nbytes for chunk in range(reader.num_chunks))
    assert largest < total / 3
    # The cache, the open decoders' chunks and one query in flight: a few chunks, never the column.
    assert peak < total / 2, f"Arrow held {peak} of {total} encoded bytes"
    assert reader._cached_bytes <= total // 8 + largest


def test_streams_walk_one_shared_read_ahead_in_their_own_pace(
    server: rr.server.Server, index: tuple[StreamIndex, ...], whole: tuple[CatalogVideo, ...]
) -> None:
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    chunks: range = range(1, reader.num_chunks - 1)
    shared = SharedChunks(reader, chunks)
    walks = [shared.stream(position) for position in range(len(whole))]
    # One stream runs three chunks ahead of the other; each still sees its own chunks in order, each fetched once.
    seen: list[list[bytes]] = [[], []]
    for _ in range(3):
        seen[0] += [bytes(sample) for sample in next(walks[0]).samples]
    for packets in walks[1]:
        seen[1] += [bytes(sample) for sample in packets.samples]
    for packets in walks[0]:
        seen[0] += [bytes(sample) for sample in packets.samples]
    for position, video in enumerate(whole):
        first, stop = int(reader.starts[chunks[0], position]), int(reader.starts[chunks[-1] + 1, position])
        assert seen[position] == [bytes(sample) for sample in video.samples[first:stop]]
    assert reader.fetch_count == len(chunks)
    assert shared.held == 0


def test_a_stream_without_samples_or_codec_is_named(tmp_path: Path) -> None:
    path: Path = tmp_path / "empty.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.set_time("video_time", duration=np.timedelta64(5, "ns"))
        rec.log("/other", rr.Scalars([1.0]))
        rec.log("/bare/video", rr.VideoStream.from_fields(sample=b"x", is_keyframe=True))
    with rr.server.Server(datasets={"test": [path]}) as running:
        dataset = running.client().get_dataset("test")
        with pytest.raises(ValueError, match="segment /cam/video: expected samples"):
            read_video_index(dataset, "segment", ["cam/video"], "video_time")
        with pytest.raises(ValueError, match="segment /bare/video: the video stream carries no codec"):
            read_video_index(dataset, "segment", ["/bare/video"], "video_time")


def test_a_stream_that_runs_too_far_ahead_fails_instead_of_holding_the_session(server: rr.server.Server, index: tuple[StreamIndex, ...]) -> None:
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    shared = SharedChunks(reader, range(reader.num_chunks), max_held=2)
    lead = shared.stream(0)
    next(lead)
    next(lead)
    with pytest.raises(ValueError, match="ahead"):
        next(lead)


def test_a_sequence_timeline_is_refused(tmp_path: Path) -> None:
    path: Path = tmp_path / "frames.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.set_time("frame", sequence=0)
        rec.log("/cam/video", rr.VideoStream.from_fields(sample=b"x", is_keyframe=True))
    with rr.server.Server(datasets={"test": [path]}) as running, pytest.raises(ValueError, match="seconds"):
        read_video_index(running.client().get_dataset("test"), "segment", ["/cam/video"], "frame")
