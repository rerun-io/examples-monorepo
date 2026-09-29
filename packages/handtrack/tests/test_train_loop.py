"""Train real networks with one CPU source; checkpoints are the observable seam."""
import signal
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('torch', reason='requires handtrack environment')
import torch

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.models.detnet import DetNetF
from handtrack.models.keynet import KeyNetLoss, KeyNetOutput
from handtrack.train.loop import LoopSettings, Nets, OptimiserSettings, Trainer
from handtrack.train.source import DetNetValidation, KeyNetValidation

DETNET_SGD: OptimiserSettings = OptimiserSettings(0.001)
KEYNET_SGD: OptimiserSettings = OptimiserSettings(0.025)
"""The CLI defaults (``handtrack.apis.train.Config``)."""


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FakeSource:
    """Fixed tiny batches; independent pool lengths expose premature joint exits."""
    def __init__(self, det_steps: int = 2, key_steps: int = 3) -> None:
        self.lengths = (det_steps, key_steps)
        self.remaining = [det_steps, key_steps]
        self.closed = False
        self.det = DetNetBatch(torch.rand(2, 1, 120, 160), torch.full((2, 2, 3), 0.1), torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.int64))
        self.key = KeyNetBatch(torch.rand(2, 1, 96, 96), torch.zeros(2, 63), torch.zeros(2, 21, 18, 18), torch.zeros(2, 21, 18), torch.ones(2), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))

    def start_epoch(self, epoch: int) -> None:
        self.remaining = list(self.lengths)

    def next_detnet_batch(self) -> DetNetBatch | None:
        if self.remaining[0] == 0:
            return None
        self.remaining[0] -= 1
        return self.det

    def next_keynet_batch(self) -> KeyNetBatch | None:
        if self.remaining[1] == 0:
            return None
        self.remaining[1] -= 1
        return self.key

    def detnet_validation(self) -> DetNetValidation:
        return DetNetValidation(torch.full((2, 2, 21, 2), 100.0), torch.ones(2, 2, 21, dtype=torch.bool), torch.zeros(2, dtype=torch.int64), torch.ones(2, 2, dtype=torch.bool))

    def keynet_validation(self) -> KeyNetValidation:
        return KeyNetValidation(torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 21, 2), torch.zeros(2, 21))

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class InterruptSource(FakeSource):
    """Requests a stop on every KeyNet draw, as SIGTERM would mid-epoch."""
    trainer: Trainer

    def next_keynet_batch(self) -> KeyNetBatch | None:
        batch = super().next_keynet_batch()
        self.trainer.handle_sigterm(signal.SIGTERM, None)
        return batch


@pytest.mark.parametrize('nets,steps', [('detnet', 2), ('keynet', 3), ('both', 5)])
def test_training_and_resume(tmp_path: Path, nets: Nets, steps: int) -> None:
    torch.manual_seed(0)
    source = FakeSource()
    validation = FakeSource(1, 1)
    trainer = Trainer(nets, DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1, log_every=1, validate_every=2), tmp_path, 'cpu', max_val_batches=1)
    state = trainer.run(source, validation)
    assert state.step == steps and state.epoch == 1
    assert source.closed and validation.closed
    assert state.history
    assert all(torch.isfinite(p).all() for model in trainer.models.values() for p in model.parameters())
    resumed = Trainer(nets, DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=2, log_every=1), tmp_path, 'cpu', resume=True)
    assert resumed.run(FakeSource(), FakeSource(1, 1)).step == 2 * steps
    assert (tmp_path / 'train.jsonl').exists()


def test_sigterm_handler(tmp_path: Path) -> None:
    trainer = Trainer('detnet', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1), tmp_path, 'cpu')
    trainer.handle_sigterm(signal.SIGTERM, None)
    state = trainer.run(FakeSource(), FakeSource())
    assert state.step == 0
    assert (tmp_path / 'last.pt.sha256').exists()


@pytest.mark.parametrize('nets', ['detnet', 'keynet', 'both'])
def test_fixed_batch_overfit(tmp_path: Path, nets: Nets) -> None:
    torch.manual_seed(10)
    trainer = Trainer(nets, DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1), tmp_path, 'cpu')
    source = FakeSource()
    first: dict[str, float] = {}
    for iteration in range(10):
        if nets in ('detnet', 'both'):
            trainer.train_batch(source.det)
        if nets in ('keynet', 'both'):
            trainer.train_batch(source.key)
        if iteration == 0:
            first = {name: terms['total'] for name, terms in trainer.losses.items()}
    for name, initial in first.items():
        assert trainer.losses[name]['total'] < initial * 0.9


def test_heatmap_reduction_reaches_the_keynet_objective(tmp_path: Path) -> None:
    source = FakeSource()
    output = KeyNetOutput(torch.full((2, 21, 18, 18), 0.5), torch.full((2, 21, 18), 0.5), torch.zeros(2))
    losses: dict[str, KeyNetLoss] = {}
    for reduction in ('mean', 'pixel_sum'):
        trainer = Trainer('keynet', OptimiserSettings(0.001), OptimiserSettings(0.025), LoopSettings(epochs=1, heatmap_reduction=reduction), tmp_path, 'cpu')
        losses[reduction] = trainer.keynet_objective(source.key, output)
    assert LoopSettings().heatmap_reduction == 'mean'
    assert losses['mean'].heatmap.item() == pytest.approx(0.25)
    assert losses['pixel_sum'].heatmap.item() == pytest.approx(0.25 * 18 * 18)
    assert losses['pixel_sum'].distance.item() == pytest.approx(0.25 * 18)


def test_heatmap_warmup_uses_mean_for_its_epochs(tmp_path: Path) -> None:
    source = FakeSource()
    output = KeyNetOutput(torch.full((2, 21, 18, 18), 0.5), torch.full((2, 21, 18), 0.5), torch.zeros(2))
    cadence = LoopSettings(epochs=3, heatmap_reduction='pixel_sum', heatmap_warmup_epochs=1)
    trainer = Trainer('keynet', OptimiserSettings(0.001), OptimiserSettings(0.025), cadence, tmp_path, 'cpu')
    assert trainer.keynet_objective(source.key, output).heatmap.item() == pytest.approx(0.25)
    trainer.state = replace(trainer.state, epoch=1)
    assert trainer.keynet_objective(source.key, output).heatmap.item() == pytest.approx(0.25 * 18 * 18)
    with pytest.raises(ValueError, match='heatmap_warmup_epochs'):
        LoopSettings(heatmap_warmup_epochs=-1)
    by_steps = Trainer('keynet', OptimiserSettings(0.001), OptimiserSettings(0.025),
                       LoopSettings(heatmap_reduction='pixel_sum', heatmap_warmup_steps=5), tmp_path / 'steps', 'cpu')
    by_steps.state = replace(by_steps.state, step=4)
    assert by_steps.keynet_objective(source.key, output).heatmap.item() == pytest.approx(0.25)
    by_steps.state = replace(by_steps.state, step=5)
    assert by_steps.keynet_objective(source.key, output).heatmap.item() == pytest.approx(0.25 * 18 * 18)
    with pytest.raises(ValueError, match='heatmap_warmup_steps'):
        LoopSettings(heatmap_warmup_steps=-1)


def test_heatmap_ramp_grows_the_pixel_sum_from_the_mean_scale(tmp_path: Path) -> None:
    source = FakeSource()
    output = KeyNetOutput(torch.full((2, 21, 18, 18), 0.5), torch.full((2, 21, 18), 0.5), torch.zeros(2))
    trainer = Trainer('keynet', OptimiserSettings(0.001), OptimiserSettings(0.025),
                      LoopSettings(heatmap_reduction='pixel_sum', heatmap_warmup_steps=10, heatmap_ramp_steps=100, presence_weight=0.0), tmp_path, 'cpu')
    totals = {}
    for step in (9, 10, 60, 110):
        trainer.state = replace(trainer.state, step=step)
        loss = trainer.keynet_objective(source.key, output)
        assert loss.heatmap.item() == pytest.approx(0.25 if step < 10 else 0.25 * 18 * 18)  # logged terms stay unscaled
        totals[step] = loss.total.item()
    pixel_sum_total = 0.25 * 18 * 18 + 0.05 * 0.25 * 18
    assert totals[10] == pytest.approx(pixel_sum_total / 324) and totals[60] == pytest.approx(pixel_sum_total / 18)
    assert totals[110] == pytest.approx(pixel_sum_total)
    with pytest.raises(ValueError, match='pixel_sum'):
        LoopSettings(heatmap_ramp_steps=5)


def test_mid_epoch_joint_resume_preserves_updates(tmp_path: Path) -> None:
    torch.manual_seed(9)
    source = InterruptSource(2, 3)
    cadence = LoopSettings(epochs=1, checkpoint_every=1, keynet_steps_per_detnet_step=2, validate_every=0, resume_next_epoch=False)
    torch.manual_seed(42)
    interrupted = Trainer('both', DETNET_SGD, KEYNET_SGD, cadence, tmp_path / 'resume', 'cpu')
    source.trainer = interrupted
    state = interrupted.run(source, FakeSource(1, 1))
    assert state.step == 2
    assert state.epoch_steps == {'detnet': 1, 'keynet': 1}
    replay = FakeSource(2, 3)
    replay.det, replay.key = source.det, source.key
    resumed = Trainer('both', DETNET_SGD, KEYNET_SGD, cadence, tmp_path / 'resume', 'cpu', resume=True)
    assert resumed.run(replay, FakeSource(1, 1)).step == 5
    torch.manual_seed(42)
    uninterrupted = Trainer('both', DETNET_SGD, KEYNET_SGD, cadence, tmp_path / 'reference', 'cpu')
    uninterrupted.run(replay, FakeSource(1, 1))
    for name, model in resumed.models.items():
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, uninterrupted.models[name].state_dict()[key], rtol=0, atol=0)
    assert (tmp_path / 'resume' / 'epoch_001.pt').exists()
    assert (tmp_path / 'resume' / 'keynet' / 'best.weights.pt').exists()


def test_validation_without_metadata_has_no_geometric_score(tmp_path: Path) -> None:
    class TrainingOnlySource:
        def __init__(self) -> None:
            self.source = FakeSource(1, 1)

        def start_epoch(self, epoch: int) -> None:
            self.source.start_epoch(epoch)

        def next_detnet_batch(self) -> DetNetBatch | None:
            return self.source.next_detnet_batch()

        def next_keynet_batch(self) -> KeyNetBatch | None:
            return self.source.next_keynet_batch()

        def cancel(self) -> None:
            self.source.cancel()

        def close(self) -> None:
            self.source.close()

    trainer = Trainer('both', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1), tmp_path, 'cpu')
    trainer.run(TrainingOnlySource(), TrainingOnlySource())
    assert trainer.state.best == {}
    assert trainer.state.history[-1].values['keynet/error_px'] is None
    assert trainer.state.history[-1].values['detnet/recall'] is None


def test_best_checkpoint_uses_detection_score_and_exports_weights(tmp_path: Path) -> None:
    class ExactSource(FakeSource):
        def detnet_validation(self) -> DetNetValidation:
            points = torch.full((2, 2, 21, 2), 100.0)
            points[:, :, 0, 0] = 90.0
            points[:, :, 1, 0] = 110.0
            return DetNetValidation(points, torch.ones(2, 2, 21, dtype=torch.bool), torch.zeros(2, dtype=torch.int64), torch.ones(2, 2, dtype=torch.bool))

    trainer = Trainer('detnet', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1), tmp_path, 'cpu', max_val_batches=1)
    model = trainer.models['detnet']
    assert isinstance(model, DetNetF)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.center_head[1].bias.copy_(torch.tensor([100 / 640, 100 / 480] * 2))
        model.radius_head[1].bias.fill_(10 / 640)
        model.presence_head[1].bias.fill_(2.0)
    result = trainer.validate(ExactSource(1, 0))
    assert result.values['detnet/precision'] == result.values['detnet/recall'] == 1.0
    assert trainer.state.best == {'detnet': 1.0}
    checksum = (tmp_path / 'best.pt.sha256').read_text()
    exported = torch.load(tmp_path / 'best.weights.pt', weights_only=True)
    torch.testing.assert_close(exported['radius_head.1.bias'], model.radius_head[1].bias)
    with torch.no_grad():
        model.radius_head[1].bias.fill_(12.5 / 640)
    result = trainer.validate(ExactSource(1, 0))
    assert result.values['detnet/precision'] == 0.0
    assert (tmp_path / 'best.pt.sha256').read_text() == checksum
    assert model.training


def test_resume_next_epoch_skips_the_rest_of_the_interrupted_epoch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cadence = LoopSettings(epochs=2, checkpoint_every=1, validate_every=0)
    interrupted = Trainer('both', DETNET_SGD, KEYNET_SGD, cadence, tmp_path, 'cpu')
    source = InterruptSource(2, 3)
    source.trainer = interrupted
    state = interrupted.run(source, FakeSource(1, 1))
    assert state.epoch == 0 and state.step == 2 and state.epoch_steps == {'detnet': 1, 'keynet': 1}
    resumed = Trainer('both', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=2, checkpoint_every=1, validate_every=0, resume_next_epoch=True),
                      tmp_path, 'cpu', resume=True)
    assert resumed.state.epoch == 1 and resumed.state.step == 2 and resumed.state.epoch_steps == {}
    assert 'resume-next-epoch: skipping the rest of epoch 0' in capsys.readouterr().out
    # Epoch 1 runs in full from the source's first batch: 2 DetNet + 3 KeyNet steps, no replayed cursor.
    final = resumed.run(FakeSource(2, 3), FakeSource(1, 1))
    assert final.epoch == 2 and final.step == 2 + 5


def test_resume_records_recipe_changes_and_applies_new_sgd(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from serde.json import to_json

    from handtrack.apis.train import Config, StreamSettings
    from handtrack.train.checkpoint import TrainingState, load_checkpoint, save_checkpoint

    old = Config(nets='keynet', keynet=OptimiserSettings(0.004, 0.8), loop=LoopSettings(heatmap_reduction='pixel_sum'))
    new = replace(old, keynet=OptimiserSettings(0.025, 0.7, 'cosine'),
                  loop=LoopSettings(heatmap_reduction='mean', heatmap_warmup_epochs=1, presence_weight=2.0), stream=StreamSettings(seed=4))
    trainer = Trainer('keynet', old.detnet, old.keynet, old.loop, tmp_path, 'cpu')
    save_checkpoint(tmp_path / 'last.pt', trainer.models, trainer.optimisers, TrainingState(step=9, config_json=to_json(old)))
    resumed = Trainer('keynet', new.detnet, new.keynet, new.loop, tmp_path, 'cpu', resume=True, config_json=to_json(new))
    output = capsys.readouterr().out
    for field in ('keynet.lr', 'keynet.momentum', 'keynet.schedule', 'loop.heatmap_reduction', 'loop.heatmap_warmup_epochs', 'loop.presence_weight', 'stream.seed'):
        assert f'RECIPE CHANGE on resume at step 9: {field}:' in output
    assert resumed.optimisers['keynet'].param_groups[0]['lr'] == 0.025
    assert resumed.optimisers['keynet'].param_groups[0]['momentum'] == 0.7
    assert resumed.state.history[-1].recipe_changes['keynet.lr'] == ('0.004', '0.025')
    save_checkpoint(tmp_path / 'last.pt', resumed.models, resumed.optimisers, resumed.state)
    assert load_checkpoint(tmp_path / 'last.pt', resumed.models, resumed.optimisers).history == resumed.state.history


@pytest.mark.parametrize('during_validation', [False, True])
def test_cancel_unblocks_wait_without_completing_epoch(tmp_path: Path, during_validation: bool) -> None:
    import threading

    class WaitingSource(FakeSource):
        def __init__(self) -> None:
            super().__init__(1, 0)
            self.cancelled = threading.Event()

        def cancel(self) -> None:
            self.cancelled.set()

        def wait(self) -> None:
            trainer.handle_sigterm(signal.SIGTERM, None)
            assert self.cancelled.wait(0.5), 'SIGTERM must cancel the blocked source'

        def start_epoch(self, epoch: int) -> None:
            super().start_epoch(epoch)
            if during_validation:
                self.wait()

        def next_detnet_batch(self) -> DetNetBatch | None:
            self.wait()
            return None

    trainer = Trainer('detnet', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1, validate_every=1), tmp_path, 'cpu')
    waiting = WaitingSource()
    other = WaitingSource()
    source = FakeSource(1, 0) if during_validation else waiting
    validation = waiting if during_validation else other
    state = trainer.run(source, validation)
    assert waiting.cancelled.is_set()
    if not during_validation:
        assert other.cancelled.is_set()
    assert state.epoch == 0
    assert state.step == int(during_validation)
    assert source.closed and validation.closed
    assert (tmp_path / 'last.pt.sha256').exists()
