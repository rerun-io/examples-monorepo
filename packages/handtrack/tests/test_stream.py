import dataclasses

import numpy as np
import pyarrow as pa
import pytest
import torch
from simplecv.catalog_video import CatalogVideo

from handtrack.data.batches import CropKind
from handtrack.data.catalog import UMETRACK, HandTimeline, SegmentInfo
from handtrack.data.stream import (
    CatalogStream,
    DecoderOpener,
    DetNetSamples,
    ImageHands,
    KeyNetAugment,
    SamplePool,
    SegmentData,
    StreamConfig,
    bounding_circles,
    detnet_samples,
    empty_detnet_samples,
    evaluation_augment,
    is_fatal,
    keynet_samples,
    perspective_keynet_samples,
    sample_count,
    select_samples,
)
from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import HandPose, generic_hand_model
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import CROP_SIZE
from handtrack.labels.validity import HandLabel


def _tagged(ids: list[int]) -> DetNetSamples:
    samples: DetNetSamples = empty_detnet_samples(len(ids), torch.device("cpu"))
    return dataclasses.replace(samples, dataset=torch.tensor(ids, dtype=torch.int64))


def test_pool_draws_without_replacement_and_rejects_overflow() -> None:
    pool: SamplePool[DetNetSamples] = SamplePool(empty_detnet_samples(10, torch.device("cpu")), torch.Generator().manual_seed(0))
    pool.add(_tagged(list(range(6))))
    first: set[int] = set(pool.draw(4).dataset.tolist())
    assert len(first) == 4 and pool.count == 2
    assert set(pool.storage.dataset[: pool.count].tolist()) == set(range(6)) - first
    with pytest.raises(ValueError, match="full"):
        pool.add(_tagged(list(range(100, 112))))
    pool.add(_tagged(list(range(100, 108))))
    assert pool.count == 10 and pool.overwritten == 0
    everything: list[int] = pool.draw(10).dataset.tolist()
    assert len(set(everything)) == 10 and pool.count == 0
    with pytest.raises(ValueError, match="cannot draw"):
        pool.draw(1)


def test_detnet_samples_targets_and_masks() -> None:
    net: torch.Tensor = torch.full((2, 480, 640), 255, dtype=torch.uint8)
    net[1] = 0
    label: torch.Tensor = torch.tensor([[HandLabel.PRESENT, HandLabel.PARTIAL], [HandLabel.ABSENT, HandLabel.PRESENT]])
    circles: torch.Tensor = torch.tensor([[[320.0, 240.0, 64.0], [10.0, 10.0, 5.0]], [[float("nan")] * 3, [100.0, 120.0, 32.0]]])
    truth: ImageHands = ImageHands(
        net_xy=torch.zeros(2, 2, 21, 2), points_cam=torch.ones(2, 2, 21, 3), in_front=torch.ones(2, 2, 21, dtype=torch.bool), valid=torch.ones(2, 2, dtype=torch.bool)
    )
    samples: DetNetSamples = detnet_samples(net, label, circles, truth, dataset=1, camera=5)
    assert samples.pooled.dtype == torch.uint8 and samples.pooled[0].eq(255).all() and samples.pooled[1].eq(0).all()
    assert samples.presence.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert samples.circle_mask.tolist() == [[True, False], [False, True]]
    assert samples.presence_mask.tolist() == [[True, False], [True, True]]
    torch.testing.assert_close(samples.circle[0, 0], torch.tensor([0.5, 0.5, 0.1]))
    assert samples.circle[0, 1].eq(0).all() and samples.circle[1, 0].eq(0).all()
    assert not samples.in_front[1, 0].any() and samples.in_front[0, 1].all()
    assert samples.camera.tolist() == [5, 5] and samples.dataset.tolist() == [1, 1]


def _hand_points(centre: tuple[float, float], spread: float, seed: int) -> torch.Tensor:
    rng: np.random.Generator = np.random.default_rng(seed)
    return torch.from_numpy((np.asarray(centre) + rng.uniform(-spread, spread, (21, 2))).astype(np.float32))


def test_keynet_samples_positive_and_negatives() -> None:
    left: torch.Tensor = _hand_points((320.0, 240.0), 30.0, 0)
    net: torch.Tensor = torch.zeros((1, 480, 640), dtype=torch.uint8)
    net[0, 200:280, 280:360] = 200
    nan_hand: torch.Tensor = torch.full((21, 2), torch.nan)
    net_xy: torch.Tensor = torch.stack([left, nan_hand])[None]
    points_cam: torch.Tensor = torch.cat([net_xy / 1000.0, torch.full((1, 2, 21, 1), 0.4)], dim=-1)
    truth: ImageHands = ImageHands(net_xy=net_xy, points_cam=points_cam, in_front=torch.tensor([[[True] * 21, [False] * 21]]), valid=torch.tensor([[True, False]]))
    circles: torch.Tensor = torch.from_numpy(enclosing_circles(net_xy.numpy(), truth.in_front.numpy()))
    label: torch.Tensor = torch.tensor([[HandLabel.PRESENT, HandLabel.ABSENT]])
    augment: KeyNetAugment = dataclasses.replace(
        KeyNetAugment(), drift_probability=1.0, other_hand_probability=0.0, edge_probability=1.0, background_probability=1.0, zero_input_probability=0.0, stale_input_probability=0.0
    )
    samples = keynet_samples(net, truth, label, circles, truth, truth, 1.0, 0, augment, torch.Generator().manual_seed(3))
    kinds: list[int] = samples.kind.tolist()
    assert kinds.count(int(CropKind.POSITIVE)) == 1
    positive: int = kinds.index(int(CropKind.POSITIVE))
    assert samples.presence[positive] == 1.0 and (samples.presence[[i for i in range(len(kinds)) if i != positive]] == 0).all()
    inside: torch.Tensor = ((samples.points_crop[positive] >= 0) & (samples.points_crop[positive] < CROP_SIZE)).all(dim=-1)
    assert int(inside.sum()) >= 17
    assert samples.crops[positive].float().mean() > 50.0
    # Missing negative histories borrow a usable positive prior from this batch.
    assert samples.keypoints[positive].abs().sum() > 0
    right_negatives: list[int] = [i for i, k in enumerate(kinds) if k == int(CropKind.BACKGROUND)]
    assert right_negatives and all(samples.keypoints[i].abs().sum() > 0 for i in right_negatives)
    assert samples.points_crop[right_negatives].eq(0).all() and samples.crop_from_net.shape == (len(kinds), 3, 3)
    zero_inputs = keynet_samples(net, truth, label, circles, truth, truth, 1.0, 0, dataclasses.replace(augment, zero_input_probability=1.0), torch.Generator().manual_seed(3))
    assert zero_inputs.keypoints.eq(0).all()


def test_perspective_keynet_samples_positive_and_negatives() -> None:
    rig: CameraRig = CameraRig(names=("/cam",), image_size=torch.tensor([[640.0, 480.0]]), cam_from_rig=torch.eye(4)[None],
                               focal=torch.tensor([[300.0, 300.0]]), principal=torch.tensor([[319.5, 239.5]]), fisheye62=None)
    rng: np.random.Generator = np.random.default_rng(0)
    left_cam: torch.Tensor = torch.from_numpy(np.array([0.0, 0.0, 0.4]) + rng.uniform(-0.04, 0.04, (21, 3))).float()
    points_cam: torch.Tensor = torch.stack([left_cam, torch.full((21, 3), torch.nan)])[None]
    net_xy: torch.Tensor = points_cam[..., :2] / points_cam[..., 2:] * 300.0 + torch.tensor([319.5, 239.5])
    frames: torch.Tensor = torch.zeros((1, 480, 640), dtype=torch.uint8)
    frames[0, 200:280, 280:360] = 200  # the hand's image region: +-0.04 m at 0.4 m is +-30 px about the centre
    truth: ImageHands = ImageHands(net_xy=net_xy, points_cam=points_cam, in_front=torch.tensor([[[True] * 21, [False] * 21]]), valid=torch.tensor([[True, False]]))
    label: torch.Tensor = torch.tensor([[HandLabel.PRESENT, HandLabel.ABSENT]])
    augment: KeyNetAugment = dataclasses.replace(
        KeyNetAugment(), drift_probability=1.0, other_hand_probability=0.0, edge_probability=1.0, background_probability=1.0, zero_input_probability=0.0, stale_input_probability=0.0
    )
    samples = perspective_keynet_samples(frames, rig, letterbox_for(640, 480), 0.0, truth, label, truth, truth, 1.0, 0, augment, torch.Generator().manual_seed(3))
    kinds: list[int] = samples.kind.tolist()
    assert kinds.count(int(CropKind.POSITIVE)) == 1 and len(kinds) >= 2
    positive: int = kinds.index(int(CropKind.POSITIVE))
    assert samples.presence[positive] == 1.0 and (samples.presence[[i for i in range(len(kinds)) if i != positive]] == 0).all()
    inside: torch.Tensor = ((samples.points_crop[positive] >= 0) & (samples.points_crop[positive] < CROP_SIZE)).all(dim=-1)
    assert int(inside.sum()) >= 17
    assert samples.crops[positive].float().mean() > 50.0  # the crop sits on the bright hand region
    assert samples.keypoints[positive].abs().sum() > 0 and bool(torch.isfinite(samples.crop_from_net).all())
    negatives: list[int] = [i for i in range(len(kinds)) if i != positive]
    assert samples.points_crop[negatives].eq(0).all() and samples.d_rel_mm[negatives].eq(0).all()
    # The local affine maps the hand's net-frame keypoints close to their exact crop positions.
    mapped = torch.einsum('ij,kj->ki', samples.crop_from_net[positive, :2, :2], net_xy[0, 0]) + samples.crop_from_net[positive, :2, 2]
    assert float((mapped - samples.points_crop[positive]).norm(dim=-1).median()) < 2.0


def _two_hand_scene() -> tuple[CameraRig, torch.Tensor, ImageHands, torch.Tensor]:
    rig: CameraRig = CameraRig(names=("/cam",), image_size=torch.tensor([[640.0, 480.0]]), cam_from_rig=torch.eye(4)[None],
                               focal=torch.tensor([[300.0, 300.0]]), principal=torch.tensor([[319.5, 239.5]]), fisheye62=None)
    rng: np.random.Generator = np.random.default_rng(0)
    left: torch.Tensor = torch.from_numpy(np.array([-0.12, 0.0, 0.45]) + rng.uniform(-0.04, 0.04, (21, 3))).float()
    right: torch.Tensor = torch.from_numpy(np.array([0.12, 0.0, 0.45]) + rng.uniform(-0.04, 0.04, (21, 3))).float()
    points_cam: torch.Tensor = torch.stack([left, right])[None].repeat(8, 1, 1, 1)
    net_xy: torch.Tensor = points_cam[..., :2] / points_cam[..., 2:] * 300.0 + torch.tensor([319.5, 239.5])
    frames: torch.Tensor = torch.full((8, 480, 640), 90, dtype=torch.uint8)
    truth: ImageHands = ImageHands(net_xy=net_xy, points_cam=points_cam, in_front=torch.ones(8, 2, 21, dtype=torch.bool), valid=torch.ones(8, 2, dtype=torch.bool))
    return rig, frames, truth, torch.full((8, 2), int(HandLabel.PRESENT))


def test_other_hand_crops_are_flagged_and_the_margin_drops_near_negatives() -> None:
    rig, frames, truth, label = _two_hand_scene()
    augment: KeyNetAugment = dataclasses.replace(KeyNetAugment(), drift_probability=1.0, other_hand_probability=1.0, edge_probability=0.0, background_probability=0.0)
    samples = perspective_keynet_samples(frames, rig, letterbox_for(640, 480), 0.0, truth, label, truth, truth, 1.0, 0, augment, torch.Generator().manual_seed(5))
    kind = samples.kind
    other = kind == int(CropKind.OTHER_HAND)
    assert bool(other.any()) and bool(samples.other_inside[other].all())  # an OTHER_HAND crop always shows the other hand
    assert not bool(samples.other_inside[kind == int(CropKind.POSITIVE)].any())
    drifts = [int((perspective_keynet_samples(frames, rig, letterbox_for(640, 480), 0.0, truth, label, truth, truth, 1.0, 0,
                                             dataclasses.replace(augment, negative_margin=margin), torch.Generator().manual_seed(5)).kind
                   == int(CropKind.DRIFT)).sum()) for margin in (0.0, 2.0)]
    # DRIFT moves the crop 1.0-1.6 sides off the hand: a 2-side margin always finds the hand and drops them all.
    assert drifts[0] > 0 and drifts[1] == 0


def test_bounding_circles_and_evaluation_recipe() -> None:
    points: torch.Tensor = torch.stack([_hand_points((100.0, 50.0), 20.0, 1), _hand_points((0.0, 0.0), 5.0, 2)])
    valid: torch.Tensor = torch.ones(2, 21, dtype=torch.bool)
    valid[1] = False
    circles: torch.Tensor = bounding_circles(points, valid)
    assert ((points[0] - circles[0, :2]).norm(dim=-1) <= circles[0, 2] + 1e-4).all()
    assert torch.isnan(circles[1]).all()
    recipe: KeyNetAugment = evaluation_augment(KeyNetAugment())
    assert recipe.max_rotation == 0.0 and recipe.scale_range == (1.0, 1.0) and recipe.zero_input_probability == 0.0 and recipe.uv_noise_std == 0.0
    assert recipe.edge_probability == KeyNetAugment().edge_probability


def test_select_samples_keeps_every_field() -> None:
    samples: DetNetSamples = _tagged([7, 8, 9])
    picked: DetNetSamples = select_samples(samples, torch.tensor([2, 0]))
    assert picked.dataset.tolist() == [9, 7] and sample_count(picked) == 2 and picked.points.shape == (2, 2, 21, 2)


def _fake_segment(segment_id: str, marker: int) -> tuple[SegmentInfo, SegmentData]:
    """A 12-frame, one-camera segment with a left hand in view; ``marker`` tags its packets for the fake decoder."""
    frames: int = 12
    translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.4]).expand(frames, 3).clone()
    left: HandPose = HandPose(rotation=torch.eye(3).expand(frames, 3, 3).clone(), translation=translation, joint_angles=torch.zeros(frames, 22))
    right: HandPose = HandPose(rotation=torch.full((frames, 3, 3), torch.nan), translation=torch.full((frames, 3), torch.nan), joint_angles=torch.full((frames, 22), torch.nan))
    times: np.ndarray = np.arange(frames, dtype=np.int64) * 33_333_333
    timeline: HandTimeline = HandTimeline(
        video_time_ns=times,
        world_from_rig=torch.eye(4).expand(frames, 4, 4).clone(),
        headset_valid=torch.ones(frames, dtype=torch.bool),
        poses=(left, right),
        confidence=torch.tensor([[1.0, 0.0]]).expand(frames, 2).clone(),
        has_pose=torch.tensor([[True, False]]).expand(frames, 2).clone(),
        hand_model=generic_hand_model(),
        hand_scale=1.0,
    )
    rig: CameraRig = CameraRig(
        names=("/world/rig_00/cam_00",),
        image_size=torch.tensor([[640.0, 480.0]]),
        cam_from_rig=torch.eye(4)[None],
        focal=torch.tensor([[300.0, 300.0]]),
        principal=torch.tensor([[320.0, 240.0]]),
        fisheye62=None,
    )
    video: CatalogVideo = CatalogVideo(times.astype("timedelta64[ns]"), [np.full(4, marker, dtype=np.uint8)] * frames, [True] * frames, "av1")
    info: SegmentInfo = SegmentInfo(UMETRACK, segment_id, "real", "hand_hand", "training", "user_00", frames, 30)
    return info, SegmentData(rig=rig, letterboxes=(letterbox_for(640, 480),), timeline=timeline, videos=(video,))


class _Frames:
    def __init__(self, data: torch.Tensor) -> None:
        self.data: torch.Tensor = data


class _GrayDecoder:
    def get_frames_at(self, indices: list[int]) -> _Frames:
        return _Frames(torch.full((len(indices), 3, 480, 640), 128, dtype=torch.uint8))


def _stream(segments: dict[str, tuple[SegmentInfo, SegmentData]], opener: DecoderOpener) -> CatalogStream:
    return CatalogStream(
        StreamConfig(datasets=(UMETRACK,), nets="detnet", producers=1, fetchers=1, detnet_buffer=64, detnet_batch_size=2, min_fill=0.0, device="cpu"),
        segments=tuple(info for info, _ in segments.values()),
        read_segment=lambda info: segments[info.segment_id][1],
        open_decoder=opener,
    )


def test_a_segment_that_fails_twice_is_skipped_for_the_rest_of_the_run() -> None:
    segments: dict[str, tuple[SegmentInfo, SegmentData]] = {name: _fake_segment(name, marker) for name, marker in (("good-segment", 1), ("bad-segment", 0))}
    opened: list[int] = []

    def opener(video: CatalogVideo, fps: int, device: torch.device) -> _GrayDecoder:
        opened.append(int(video.samples[0][0]))
        if video.samples[0][0] == 0:
            raise RuntimeError("Failed to open input buffer: Invalid data found when processing input")
        return _GrayDecoder()

    with _stream(segments, opener) as stream:
        for epoch in (0, 1):
            stream.start_epoch(epoch)
            images: int = 0
            while (batch := stream.next_detnet_batch()) is not None:
                images += batch.pooled.shape[0]
            assert images == 2, "the good segment's two kept frames still arrive"
        assert stream.stats.skipped_segments == 1 and stream.stats.retried_segments == 1 and stream.stats.segments == 2
        assert len(stream.stats.failures) == 2
        assert all("bad-segment decode camera /world/rig_00/cam_00" in line and "Invalid data" in line for line in stream.stats.failures)
        assert opened.count(0) == 2, "one retry, then never again"


def test_machinery_failures_stay_fatal() -> None:
    segments: dict[str, tuple[SegmentInfo, SegmentData]] = {"good-segment": _fake_segment("good-segment", 1)}

    def opener(video: CatalogVideo, fps: int, device: torch.device) -> _GrayDecoder:
        raise torch.OutOfMemoryError("CUDA out of memory")

    with _stream(segments, opener) as stream:
        stream.start_epoch(0)
        with pytest.raises(RuntimeError, match=r"a catalog producer failed \(dataforge-umetrack good-segment\)"):
            stream.next_detnet_batch()
    assert is_fatal(torch.OutOfMemoryError("x")) and is_fatal(RuntimeError("Invalid data")) and is_fatal(ValueError("bad labels"))


@pytest.mark.parametrize('evaluation', [False, True])
def test_cancel_interrupts_producer_wait(evaluation: bool) -> None:
    import threading

    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    info, data = _fake_segment('waiting', 1)

    def reader(info: SegmentInfo) -> SegmentData:
        entered.set()
        release.wait(5.0)
        return data

    stream = CatalogStream(StreamConfig(device='cpu', validation=evaluation, producers=1, fetchers=1, detnet_buffer=4),
                           segments=(info,), read_segment=reader, open_decoder=lambda *_args: _GrayDecoder())
    batches = []

    def consume() -> None:
        stream.start_epoch(0)
        batches.append(stream.next_detnet_batch())
        returned.set()

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert entered.wait(2.0)
        stream.cancel()
        assert returned.wait(1.0), 'cancel must return before the catalog read completes'
        assert batches == [None]
    finally:
        release.set()
        worker.join(3.0)
        stream.close()


def test_evaluation_retains_native_hand_eligibility() -> None:
    info, data = _fake_segment('native', 1)
    with CatalogStream(StreamConfig(device='cpu', validation=True, producers=1, fetchers=1),
                       segments=(info,), read_segment=lambda info: data, open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        batch = stream.next_detnet_batch()
        assert batch is not None
        metadata = stream.detnet_validation()
        assert metadata.eligible.tolist() == [[True, False], [True, False]]


def test_epoch_publication_cannot_discard_claimed_work() -> None:
    import threading
    import time

    info, data = _fake_segment('race', 1)
    claimed = threading.Event()

    class InterleavedStream(CatalogStream):
        def _discard(self) -> None:
            if self._generation >= 0 and not self._stop.is_set():
                assert claimed.wait(2.0)
                deadline = time.monotonic() + 2.0
                while self._work.empty() and self._queue.empty() and time.monotonic() < deadline:
                    time.sleep(0.001)
            super()._discard()

        def _produce(self, worker: int) -> None:
            # Hold the decoder until epoch setup has discarded the claimed work.
            release.wait(3.0)
            super()._produce(worker)

    def reader(info: SegmentInfo) -> SegmentData:
        claimed.set()
        return data

    release = threading.Event()
    stream = InterleavedStream(StreamConfig(device='cpu', nets='detnet', producers=1, fetchers=1, detnet_buffer=4, detnet_batch_size=2, min_fill=0.0),
                               segments=(info,), read_segment=reader, open_decoder=lambda *_args: _GrayDecoder())
    returned = threading.Event()
    batches = []

    def consume() -> None:
        batches.append(stream.next_detnet_batch())
        returned.set()

    try:
        stream.start_epoch(0)
        release.set()
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        assert returned.wait(3.0), 'claimed segment must reach completion'
        assert batches[0] is not None
        assert stream.next_detnet_batch() is None
    finally:
        stream.cancel()
        release.set()
        stream.close()


@pytest.mark.parametrize('failures', [1, 2])
def test_camera_retry_does_not_duplicate_published_prefix(failures: int) -> None:
    info, data = _fake_segment('retry', 1)
    _, other = _fake_segment('other', 2)
    rig = dataclasses.replace(data.rig, names=('a', 'b'), image_size=data.rig.image_size.repeat(2, 1),
                              cam_from_rig=data.rig.cam_from_rig.repeat(2, 1, 1), focal=data.rig.focal.repeat(2, 1), principal=data.rig.principal.repeat(2, 1))
    data = dataclasses.replace(data, rig=rig, videos=(data.videos[0], other.videos[0]), letterboxes=data.letterboxes * 2)
    attempts = []
    first_camera_opens = []

    def opener(video: CatalogVideo, fps: int, device: torch.device) -> _GrayDecoder:
        if video.samples[0][0] == 1:
            first_camera_opens.append(1)
        if video.samples[0][0] == 2:
            attempts.append(1)
            if len(attempts) <= failures:
                raise RuntimeError('FFmpeg invalid data')
        return _GrayDecoder()

    with _stream({'retry': (info, data)}, opener) as stream:
        stream.start_epoch(0)
        cameras = []
        while stream.next_detnet_batch() is not None:
            cameras.extend(stream.detnet_validation().camera.tolist())
        assert sorted(cameras) == ([0, 0, 1, 1] if failures == 1 else [0, 0])
        assert len(first_camera_opens) == 1
        assert stream.stats.detnet_samples == len(cameras)
        assert stream.stats.segments == 1
        assert stream.stats.skipped_segments == (1 if failures == 2 else 0)


def test_slow_consumer_receives_all_samples_including_tail() -> None:
    import time

    segments = {str(i): _fake_segment(str(i), i) for i in range(5)}
    with CatalogStream(StreamConfig(device='cpu', nets='detnet', producers=1, fetchers=1, detnet_buffer=3,
                                    detnet_batch_size=2, queue_chunks=2, min_fill=1.0),
                       segments=tuple(info for info, _ in segments.values()), read_segment=lambda info: segments[info.segment_id][1],
                       open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        time.sleep(0.3)
        count = 0
        while (batch := stream.next_detnet_batch()) is not None:
            count += len(batch.dataset)
            time.sleep(0.01)
        assert count == 10
        assert stream.overwritten() == (0, 0)


def test_evaluation_selection_is_bounded_and_completion_order_independent() -> None:
    from handtrack.data.stream import _Chunk

    class BoundedStream(CatalogStream):
        def _insert(self, chunk: _Chunk) -> None:
            super()._insert(chunk)
            assert sum(sample_count(samples) for samples in (self._evaluation or ()) if samples is not None) <= self.config.validation_samples

    info, data = _fake_segment('evaluation', 1)
    results = []
    for order in (range(10), reversed(range(10))):
        with BoundedStream(StreamConfig(device='cpu', nets='detnet', validation=True, validation_samples=3),
                           segments=(info,), read_segment=lambda info: data, open_decoder=lambda *_args: _GrayDecoder()) as stream:
            # Feed the same decoded sample identities in opposite completion orders.
            stream._epoch_segments = [dataclasses.replace(info, segment_id=str(i)) for i in range(10)]
            for i in order:
                stream._insert(_Chunk(0, (i, 0, 0), _tagged([i]), None))
            results.append(sorted(stream._evaluation[0].dataset.tolist()) if stream._evaluation is not None else [])
    assert len(results[0]) == 3 and results[0] == results[1]


@pytest.mark.parametrize('error', [AssertionError('bug'), TypeError('bug'), IndexError('bug'), KeyError('bug'), AttributeError('bug'), NameError('bug'), RuntimeError('bug'), ValueError('bug')])
def test_programming_errors_cancel_workers_without_retry(error: Exception) -> None:
    info, data = _fake_segment('fatal', 1)
    calls = []

    def reader(info: SegmentInfo) -> SegmentData:
        calls.append(info)
        raise error

    with CatalogStream(StreamConfig(device='cpu', nets='detnet', producers=1, fetchers=1), segments=(info,),
                       read_segment=reader, open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        with pytest.raises(RuntimeError, match='catalog producer failed') as caught:
            stream.next_detnet_batch()
        assert caught.value.__cause__ is error
        assert len(calls) == 1 and stream.stats.retried_segments == 0
        assert stream._stop.is_set()


@pytest.mark.parametrize('evaluation', [False, True])
def test_cancel_is_terminal_for_readiness(evaluation: bool) -> None:
    info, data = _fake_segment('terminal', 1)
    with CatalogStream(StreamConfig(device='cpu', validation=evaluation), segments=(info,), read_segment=lambda info: data,
                       open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        stream.cancel()
        assert not stream.detnet_ready() and not stream.keynet_ready()
        assert stream.next_detnet_batch() is None and stream.next_keynet_batch() is None


def test_close_reports_workers_alive_at_deadline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import handtrack.data.stream as module

    info, data = _fake_segment('timeout', 1)
    stream = _stream({'timeout': (info, data)}, lambda *_args: _GrayDecoder())
    stream.close()

    class StuckThread:
        name = 'stuck-test-worker'

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            pass

    stream._threads = [StuckThread()]
    ticks = iter([0.0, 31.0])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    stream.close()
    assert 'stuck-test-worker' in capsys.readouterr().err


@pytest.mark.parametrize('stale_probability', [0.0, 1.0])
def test_negative_inputs_use_usable_selected_history(stale_probability: float) -> None:
    left = _hand_points((320.0, 240.0), 30.0, 7)
    xy = torch.stack([left, left + 900.0])[None]
    cam = torch.zeros(1, 2, 21, 3)
    cam[:, 0, :, 2] = 0.4
    cam[:, 1, :, 2] = -0.4
    prior = ImageHands(xy, cam, torch.tensor([[[True] * 21, [False] * 21]]), torch.ones(1, 2, dtype=torch.bool))
    truth = dataclasses.replace(prior, valid=torch.tensor([[True, False]]))
    stale = dataclasses.replace(prior, net_xy=xy + 2000.0)
    circles = bounding_circles(xy.reshape(2, 21, 2), prior.in_front.reshape(2, 21)).reshape(1, 2, 3)
    augment = dataclasses.replace(evaluation_augment(KeyNetAugment()), other_hand_probability=1.0, background_probability=1.0,
                                  stale_input_probability=stale_probability)
    samples = keynet_samples(torch.zeros(1, 480, 640, dtype=torch.uint8), truth,
                             torch.tensor([[HandLabel.PRESENT, HandLabel.ABSENT]]), circles, prior, stale,
                             1.0, 0, augment, torch.Generator().manual_seed(4))
    negatives = samples.keypoints[samples.presence == 0]
    assert len(negatives) > 0 and torch.isfinite(negatives).all()
    assert negatives.abs().max() < 2.0
    assert (negatives.abs().sum(dim=1) > 0).all()


def test_edge_negatives_cross_an_image_edge() -> None:
    left = _hand_points((320.0, 240.0), 30.0, 2)
    xy = torch.stack([left, left])[None]
    prior = ImageHands(xy, torch.ones(1, 2, 21, 3), torch.ones(1, 2, 21, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    truth = dataclasses.replace(prior, in_front=torch.zeros_like(prior.in_front), valid=torch.zeros_like(prior.valid))
    circles = bounding_circles(xy.reshape(2, 21, 2), prior.in_front.reshape(2, 21)).reshape(1, 2, 3)
    augment = dataclasses.replace(KeyNetAugment(), edge_probability=1.0, background_probability=0.0, other_hand_probability=0.0)
    for seed in range(8):
        samples = keynet_samples(torch.zeros(1, 480, 640, dtype=torch.uint8), truth,
                                 torch.zeros(1, 2, dtype=torch.int64), circles, prior, prior,
                                 1.0, 0, augment, torch.Generator().manual_seed(seed))
        edges = samples.crop_from_net[samples.kind == int(CropKind.EDGE)]
        assert len(edges) == 2
        corners = torch.tensor([[-0.5, -0.5, 1.0], [95.5, -0.5, 1.0], [-0.5, 95.5, 1.0], [95.5, 95.5, 1.0]])
        net_corners = corners[None] @ torch.linalg.inv(edges).transpose(-1, -2)
        low = net_corners[..., :2].amin(dim=1)
        high = net_corners[..., :2].amax(dim=1)
        crosses = ((low[:, 0] <= -0.5) & (high[:, 0] >= -0.5) | (low[:, 0] <= 639.5) & (high[:, 0] >= 639.5)
                   | (low[:, 1] <= -0.5) & (high[:, 1] >= -0.5) | (low[:, 1] <= 479.5) & (high[:, 1] >= 479.5))
        assert crosses.all()


def test_joint_readiness_allows_consumer_ratio_to_adapt() -> None:
    import time

    segments = {str(i): _fake_segment(str(i), i) for i in range(5)}
    with CatalogStream(StreamConfig(device='cpu', nets='both', producers=1, fetchers=1, detnet_buffer=1, keynet_buffer=1,
                                    detnet_batch_size=4, keynet_batch_size=4, queue_chunks=1, min_fill=1.0),
                       segments=tuple(info for info, _ in segments.values()), read_segment=lambda info: segments[info.segment_id][1],
                       open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        detnet = keynet = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if stream.detnet_ready():
                batch = stream.next_detnet_batch()
                assert batch is not None
                detnet += len(batch.dataset)
            elif stream.keynet_ready():
                keys = stream.next_keynet_batch()
                assert keys is not None
                keynet += len(keys.dataset)
            elif stream._epoch_produced() and stream._queue.empty() and stream._pending is None:
                break
            else:
                time.sleep(0.001)
        assert detnet == stream.stats.detnet_samples == 10
        assert keynet == stream.stats.keynet_samples > 0
        assert stream.overwritten() == (0, 0)
        assert not stream.detnet_ready() and not stream.keynet_ready()


@pytest.mark.parametrize('error', [OSError('transport'), pa.ArrowInvalid('bad arrow')])
def test_classified_transport_failures_retry(error: Exception) -> None:
    info, data = _fake_segment('transport', 1)
    calls = []

    def reader(info: SegmentInfo) -> SegmentData:
        calls.append(info)
        if len(calls) == 1:
            raise error
        return data

    with CatalogStream(StreamConfig(device='cpu', nets='detnet', producers=1, fetchers=1), segments=(info,),
                       read_segment=reader, open_decoder=lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        assert stream.next_detnet_batch() is not None
        assert stream.next_detnet_batch() is None
        assert len(calls) == 2 and stream.stats.retried_segments == 1


def test_missing_confidence_preserves_show3d_geometric_absence() -> None:
    from handtrack.data.segment_labels import segment_labels

    _, data = _fake_segment('missing', 1)
    timeline = dataclasses.replace(data.timeline, confidence=torch.full_like(data.timeline.confidence, torch.nan),
                                   has_pose=torch.ones_like(data.timeline.has_pose), poses=(data.timeline.poses[0], data.timeline.poses[0]))
    visible = segment_labels(timeline, data.rig, data.letterboxes, np.array([0], dtype=np.int64), True)
    assert not visible.image_valid.any()
    outside_pose = dataclasses.replace(timeline.poses[0], translation=timeline.poses[0].translation + torch.tensor([10.0, 0.0, 0.0]))
    outside = segment_labels(dataclasses.replace(timeline, poses=(outside_pose, outside_pose)), data.rig, data.letterboxes,
                             np.array([0], dtype=np.int64), True)
    assert outside.image_valid.all() and (outside.hand_label == HandLabel.ABSENT).all()


def test_invalid_catalog_confidence_is_a_classified_data_failure() -> None:
    info, data = _fake_segment('invalid-confidence', 1)
    data = dataclasses.replace(data, timeline=dataclasses.replace(data.timeline, confidence=torch.full_like(data.timeline.confidence, 0.5)))
    with _stream({'invalid-confidence': (info, data)}, lambda *_args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        assert stream.next_detnet_batch() is None
        assert stream.stats.retried_segments == stream.stats.skipped_segments == 1
