"""Exercise the trainer and blocking consumers against real bounded producer pools."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import Event, Timer

import pytest
import torch
from test_stream import _fake_segment, _Frames
from test_train_loop import DETNET_SGD, KEYNET_SGD, FakeSource

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.data.stream import CatalogStream, KeyNetAugment, StreamConfig, evaluation_augment
from handtrack.train.loop import LoopSettings, Trainer


def joint_stream() -> CatalogStream:
    segments = [_fake_segment(str(i), i * 12) for i in range(6)]

    class Decoder:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def get_frames_at(self, indices: list[int]) -> _Frames:
            pixels = torch.tensor([self.marker + i + 1 for i in indices], dtype=torch.uint8)
            return _Frames(pixels[:, None, None, None].expand(-1, 3, 480, 640).clone())

    augment = replace(evaluation_augment(KeyNetAugment()), drift_probability=0.0, edge_probability=0.0,
                      other_hand_probability=0.0, background_probability=1.0)
    return CatalogStream(
        StreamConfig(nets='both', device='cpu', producers=1, fetchers=1, prefetch_segments=1,
                     detnet_buffer=4, keynet_buffer=4, detnet_batch_size=2, keynet_batch_size=2,
                     queue_chunks=1, min_fill=1.0, intensity_range=(1.0, 1.0), keynet=augment),
        segments=tuple(info for info, _ in segments),
        read_segment=lambda info: segments[int(info.segment_id)][1],
        open_decoder=lambda video, fps, device: Decoder(int(video.samples[0][0])),
    )


@pytest.mark.parametrize('minimum_ratio', [1, 3])
def test_actual_joint_trainer_delivers_every_sample_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimum_ratio: int) -> None:
    monkeypatch.setattr('handtrack.data.stream._BACKPRESSURE_TIMEOUT_S', 0.01)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    stream = joint_stream()
    timed_out = Event()

    def abort() -> None:
        timed_out.set()
        stream.cancel()

    timer = Timer(15.0, abort)
    trainer = Trainer('both', DETNET_SGD, KEYNET_SGD,
                      LoopSettings(epochs=1, validate_every=0, keynet_steps_per_detnet_step=minimum_ratio), tmp_path, 'cpu')
    det_ids: list[int] = []
    key_ids: list[tuple[int, int]] = []
    train_batch = trainer.train_batch

    def record(batch: DetNetBatch | KeyNetBatch) -> None:
        if isinstance(batch, DetNetBatch):
            det_ids.extend((batch.pooled[:, 0, 0, 0] * 255).round().int().tolist())
        else:
            ids = (batch.crops.flatten(1).amax(1) * 255).round().int().tolist()
            key_ids.extend(zip(ids, batch.kind.tolist(), strict=True))
        train_batch(batch)

    trainer.train_batch = record
    timer.start()
    try:
        state = trainer.run(stream, FakeSource(0, 0))
        assert not timed_out.is_set(), "trainer failed to reach epoch exhaustion"
        assert state.epoch == 1 and state.step == 18
        assert Counter(det_ids) == Counter(range(1, 73, 6))
        assert Counter(key_ids) == Counter((i, kind) for i in det_ids for kind in (0, 4))
        assert stream.overwritten() == (0, 0)
        assert stream.stats.segments == 6
    finally:
        timer.cancel()
        stream.close()
        torch.set_num_threads(previous)


@pytest.mark.parametrize('network', ['detnet', 'keynet'])
def test_one_sided_consumer_in_joint_mode_uses_valve(network: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr('handtrack.data.stream._BACKPRESSURE_TIMEOUT_S', 0.05)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    stream = joint_stream()
    timed_out = Event()

    def abort() -> None:
        timed_out.set()
        stream.cancel()

    timer = Timer(5.0, abort)
    timer.start()
    try:
        stream.start_epoch(0)
        counts = [0, 0]
        side = 0 if network == 'detnet' else 1
        next_batch = stream.next_detnet_batch if side == 0 else stream.next_keynet_batch
        while (batch := next_batch()) is not None:
            counts[side] += len(batch.dataset)
        assert not timed_out.is_set(), 'safety valve failed to reach exhaustion'
        assert counts[side] == (12, 24)[side]
        assert stream.overwritten()[side] == 0
        assert stream.overwritten()[1 - side] > 0
        assert capsys.readouterr().err.count('backpressure safety valve') == 1
        next_other = stream.next_keynet_batch if side == 0 else stream.next_detnet_batch
        while (batch := next_other()) is not None:
            counts[1 - side] += len(batch.dataset)
        assert tuple(n + lost for n, lost in zip(counts, stream.overwritten(), strict=True)) == (12, 24)
        assert not stream.detnet_ready() and not stream.keynet_ready()
        assert not stream.wait_for_batch()
    finally:
        timer.cancel()
        stream.close()
        torch.set_num_threads(previous)


def test_valve_discards_oldest_survivors_after_shuffle_draw() -> None:
    from test_stream import _tagged

    from handtrack.data.stream import SamplePool, empty_detnet_samples

    pool = SamplePool(empty_detnet_samples(4, torch.device('cpu')), torch.Generator().manual_seed(7))
    pool.add(_tagged([0, 1, 2, 3]))
    drawn = pool.draw(2).dataset.tolist()
    survivors = sorted(set(range(4)) - set(drawn))
    pool.add(_tagged([4, 5]))
    pool.discard_oldest(1)
    assert set(pool.draw(3).dataset.tolist()) == {survivors[1], 4, 5}
    assert pool.overwritten == 1
