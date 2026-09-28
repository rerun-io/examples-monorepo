import dataclasses

import numpy as np
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


def test_pool_draws_without_replacement_and_overwrites_when_full() -> None:
    pool: SamplePool[DetNetSamples] = SamplePool(empty_detnet_samples(10, torch.device("cpu")), torch.Generator().manual_seed(0))
    pool.add(_tagged(list(range(6))))
    first: set[int] = set(pool.draw(4).dataset.tolist())
    assert len(first) == 4 and pool.count == 2
    assert set(pool.storage.dataset[: pool.count].tolist()) == set(range(6)) - first
    pool.add(_tagged(list(range(100, 112))))
    assert pool.count == 10 and pool.overwritten == 4
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
    # The positive's keypoint input is its (noisy) prior; the right hand has no prior, so its negatives get zeros.
    assert samples.keypoints[positive].abs().sum() > 0
    right_negatives: list[int] = [i for i, k in enumerate(kinds) if k == int(CropKind.BACKGROUND)]
    assert right_negatives and all(samples.keypoints[i].abs().sum() == 0 for i in right_negatives)
    assert samples.points_crop[right_negatives].eq(0).all() and samples.crop_from_net.shape == (len(kinds), 3, 3)
    zero_inputs = keynet_samples(net, truth, label, circles, truth, truth, 1.0, 0, dataclasses.replace(augment, zero_input_probability=1.0), torch.Generator().manual_seed(3))
    assert zero_inputs.keypoints.eq(0).all()


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
    assert is_fatal(torch.OutOfMemoryError("x")) and not is_fatal(RuntimeError("Invalid data")) and not is_fatal(ValueError("bad labels"))


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
                           segments=(info,), read_segment=reader, open_decoder=lambda *args: _GrayDecoder())
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
                       segments=(info,), read_segment=lambda info: data, open_decoder=lambda *args: _GrayDecoder()) as stream:
        stream.start_epoch(0)
        batch = stream.next_detnet_batch()
        assert batch is not None
        metadata = stream.detnet_validation()
        assert metadata.eligible.tolist() == [[True, False], [True, False]]
