"""Chunked catalog video reads give the frames of the whole-column path, and never hold the whole column."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

from simplecv.catalog_video import CatalogVideo  # noqa: E402
from simplecv.catalog_video_codec import wrap_mp4  # noqa: E402
from simplecv.catalog_video_reader import (  # noqa: E402
    CatalogVideoReader,
    ChunkedVideoDecoder,
    SharedChunks,
    StreamIndex,
    VideoChunk,
    chunk_starts,
    iter_chunks,
    open_decoders,
    read_video_index,
    relay_video,
    select_decode_device,
)
from simplecv.testing import encode_h264, whole_column  # noqa: E402

pytestmark = pytest.mark.integration

FRAME_NS: int = 33_333_333
CAMERAS: dict[str, tuple[int, int, int, bool]] = {
    # entity: (frames, GOP, time offset in ns, keyframe flags logged on keyframes only)
    "/rig/cam_a/video": (95, 10, 0, True),
    "/rig/cam_b/video": (87, 7, 4_000_000, False),
}


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path: Path = tmp_path_factory.mktemp("video") / "cameras.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for seed, (entity, (frames, gop, offset_ns, sparse)) in enumerate(CAMERAS.items()):
            samples: list[tuple[bytes, bool]] = encode_h264(frames, gop, seed)
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
    return whole_column(server.client().get_dataset("test"), "segment", list(CAMERAS), "video_time")


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


def test_a_stream_without_keyframes_is_refused_rather_than_read_whole() -> None:
    times = np.arange(5, dtype=np.int64).astype("timedelta64[ns]")
    flagless = StreamIndex(entity="/cam/video", times=times, keyframes=np.zeros(5, dtype=bool), codec="h264")
    with pytest.raises(ValueError, match="/cam/video: the stream has no keyframe"):
        chunk_starts([flagless], chunk_ns=2)


def test_decoders_land_frames_on_cuda_whenever_it_is_available(server: rr.server.Server, whole: tuple[CatalogVideo, ...]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    assert select_decode_device() == "cuda"
    decoders = open_decoders(server.client().get_dataset("test"), "segment", list(CAMERAS), "video_time", torch.device(select_decode_device()))
    for decoder, video in zip(decoders, whole, strict=True):
        frame: torch.Tensor = decoder.get_frame_at(len(decoder) - 1)
        assert frame.device.type == "cuda"
        assert torch.equal(frame, _whole_frames(video)[-1])
        np.testing.assert_array_equal(decoder.index.times, video.times)


@pytest.fixture
def fresh_decode_policy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forget the cached NVDEC discovery and the printed notice for one test."""
    import simplecv.catalog_video_reader as reader_module

    discovery = reader_module.nvdec_unavailable_reason  # a test may replace it; clear the real cache
    discovery.cache_clear()
    monkeypatch.setattr(reader_module, "_CPU_NOTICE_PRINTED", [])
    yield
    discovery.cache_clear()


@pytest.mark.usefixtures("fresh_decode_policy")
@pytest.mark.parametrize(
    ("torchcodec_present", "cuda", "reason"),
    [(False, False, "no torchcodec"), (False, True, "no torchcodec"), (True, False, "torch sees no CUDA device"), (True, True, None)],
)
def test_nvdec_discovery_covers_every_combination(monkeypatch: pytest.MonkeyPatch, torchcodec_present: bool, cuda: bool, reason: str | None) -> None:
    import simplecv.catalog_video_reader as reader_module

    monkeypatch.setattr(reader_module, "find_spec", lambda name: object() if name != "torchcodec" or torchcodec_present else None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    found: str | None = reader_module.nvdec_unavailable_reason()
    assert (found is None) == (reason is None) and (reason is None or reason in str(found))
    if reason is None:
        assert select_decode_device("auto") == select_decode_device("cuda") == "cuda"
    else:
        assert select_decode_device("auto") == "cpu"
        with pytest.raises(ValueError, match="NVDEC decoding is unavailable"):
            select_decode_device("cuda")
    assert select_decode_device("cpu") == "cpu"


@pytest.mark.usefixtures("fresh_decode_policy")
def test_the_cpu_decoding_notice_is_printed_once_per_process(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import simplecv.catalog_video_reader as reader_module

    monkeypatch.setattr(reader_module, "nvdec_unavailable_reason", lambda: "no CUDA here")
    assert select_decode_device() == select_decode_device("auto") == "cpu"
    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(lambda _: select_decode_device(), range(32))) == {"cpu"}
    assert capsys.readouterr().out.count("decoding on the CPU") == 1


def test_a_sequence_timeline_reads_in_frame_counted_chunks(tmp_path: Path) -> None:
    path: Path = tmp_path / "frames.rrd"
    samples: list[tuple[bytes, bool]] = encode_h264(70, 9, seed=3)
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        frames = 2 * np.arange(len(samples), dtype=np.int64)  # every other frame number: counts need not be contiguous
        rec.send_columns(
            "/cam/video",
            indexes=[rr.TimeColumn("frame_index", sequence=frames)],
            columns=rr.VideoStream.columns(sample=[s for s, _ in samples], is_keyframe=[k for _, k in samples]),
        )
    with rr.server.Server(datasets={"test": [path]}) as running:
        dataset = running.client().get_dataset("test")
        video: CatalogVideo = whole_column(dataset, "segment", ["/cam/video"], "frame_index")[0]
        index: tuple[StreamIndex, ...] = read_video_index(dataset, "segment", ["/cam/video"], "frame_index", window_s=0.5)
        np.testing.assert_array_equal(index[0].times, video.times)
        reader = CatalogVideoReader(dataset, "segment", index, "frame_index", chunk_s=1.0)  # 30 frame numbers a chunk
        assert reader.num_chunks >= 4
        decoder = ChunkedVideoDecoder(reader, 0, _device())
        assert torch.equal(decoder.get_frames_at(list(range(len(samples)))), _whole_frames(video))


def test_relay_logs_the_window_from_the_keyframe_at_or_before_it(server: rr.server.Server, index: tuple[StreamIndex, ...], tmp_path: Path) -> None:
    dataset = server.client().get_dataset("test")
    reader = CatalogVideoReader(dataset, "segment", index, "video_time", chunk_s=0.5)
    first_ns, last_ns = 25 * FRAME_NS + 1, 71 * FRAME_NS
    targets: list[str] = [f"/relayed{entity}" for entity in CAMERAS]
    relayed_path: Path = tmp_path / "relayed.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(relayed_path)
        logged: int = relay_video(reader, targets, first_ns, last_ns, recording=rec)
    with rr.server.Server(datasets={"relayed": [relayed_path]}) as relayed_server:
        relayed = whole_column(relayed_server.client().get_dataset("relayed"), "segment", targets, "video_time")
    whole_videos = whole_column(dataset, "segment", list(CAMERAS), "video_time")
    expected_total: int = 0
    for got, video, (_, gop, offset_ns, _) in zip(relayed, whole_videos, CAMERAS.values(), strict=True):
        # The keyframes are every gop-th sample: the window opens on the last one at or before first_ns.
        first: int = ((first_ns - offset_ns) // FRAME_NS) // gop * gop
        stop: int = int(np.searchsorted(video.t_ns, last_ns, side="right"))
        assert [bytes(sample) for sample in got.samples] == [bytes(sample) for sample in video.samples[first:stop]]
        np.testing.assert_array_equal(got.times, video.times[first:stop])
        assert got.keyframes == video.keyframes[first:stop]
        expected_total += stop - first
    assert logged == expected_total


def test_concurrent_misses_of_one_chunk_share_one_fetch(
    server: rr.server.Server, index: tuple[StreamIndex, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    plain_fetch = CatalogVideoReader.fetch
    calls: list[int] = []
    lock = threading.Lock()

    def slow_fetch(self: CatalogVideoReader, chunk: int) -> VideoChunk:
        with lock:
            calls.append(chunk)
        time.sleep(0.2)  # long enough that every thread misses before the first fetch lands
        return plain_fetch(self, chunk)

    monkeypatch.setattr(CatalogVideoReader, "fetch", slow_fetch)
    with ThreadPoolExecutor(4) as pool:
        chunks: list[VideoChunk] = list(pool.map(lambda _: reader.packets(2), range(4)))
    assert calls == [2]
    assert all(chunk is chunks[0] for chunk in chunks)


def test_a_cuda_decoder_that_falls_back_to_the_cpu_says_so_once(
    server: rr.server.Server, index: tuple[StreamIndex, ...], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import torchcodec.decoders

    class _FallingBack:
        cpu_fallback = "NVDEC cannot decode this stream"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            return SimpleNamespace(data=torch.zeros((len(indices), 3, 2, 2), dtype=torch.uint8))

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", _FallingBack)
    reader = CatalogVideoReader(server.client().get_dataset("test"), "segment", index, "video_time", chunk_s=0.5)
    decoder = ChunkedVideoDecoder(reader, 0, torch.device("cuda"))
    decoder.get_frames_at([0, 1])
    decoder.get_frames_at([40])
    assert capsys.readouterr().out.count("decodes on the CPU") == 1


def test_a_stream_that_starts_mid_gop_decodes_from_its_first_keyframe(tmp_path: Path) -> None:
    samples: list[tuple[bytes, bool]] = encode_h264(40, 10, seed=7)[3:]  # the first 7 samples precede the first keyframe
    times = (33_333_333 * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]")
    path: Path = tmp_path / "mid_gop.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.send_columns(
            "/cam/video",
            indexes=[rr.TimeColumn("video_time", duration=times)],
            columns=rr.VideoStream.columns(sample=[s for s, _ in samples], is_keyframe=[k for _, k in samples]),
        )
    from torchcodec.decoders import VideoDecoder

    first_keyframe: int = next(position for position, (_, key) in enumerate(samples) if key)
    tail = samples[first_keyframe:]
    reference: torch.Tensor = VideoDecoder(wrap_mp4([s for s, _ in tail], [k for _, k in tail], 30, codec="h264"), device=str(_device())).get_frames_at(
        list(range(len(tail)))
    ).data
    with rr.server.Server(datasets={"test": [path]}) as running:
        decoder: ChunkedVideoDecoder = open_decoders(running.client().get_dataset("test"), "segment", ["/cam/video"], "video_time", _device(), chunk_s=0.5)[0]
        assert torch.equal(decoder.get_frames_at(list(range(first_keyframe, len(samples)))), reference)
        with pytest.raises(ValueError, match="before the stream's first keyframe"):
            decoder.get_frame_at(first_keyframe - 1)
