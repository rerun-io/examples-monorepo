"""Checkpoint integrity and training-state round trips."""
import errno
import os
from pathlib import Path

import pytest

pytest.importorskip('torch', reason='requires handtrack environment')
import torch

from handtrack.train.checkpoint import TrainingState, load_checkpoint, read_disk, save_checkpoint


def test_checkpoint_roundtrip_and_corruption(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 2)).sum().backward()
    optimiser.step()
    state = TrainingState(epoch=2, step=9, config_json='{"test":true}', history=[], epoch_steps={'detnet': 3})
    path = tmp_path / 'last.pt'
    save_checkpoint(path, {'detnet': model}, {'detnet': optimiser}, state)
    replacement = torch.nn.Linear(2, 1)
    replacement_optimiser = torch.optim.SGD(replacement.parameters(), lr=0.5)
    restored = load_checkpoint(path, {'detnet': replacement}, {'detnet': replacement_optimiser})
    assert restored.step == 9 and restored.epoch == 2
    assert restored.epoch_steps == {'detnet': 3}
    torch.testing.assert_close(model.weight, replacement.weight)
    assert replacement_optimiser.param_groups[0]['lr'] == 0.1
    with path.open('r+b') as stream:
        stream.seek(100)
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256'):
        load_checkpoint(path, {'detnet': replacement}, {'detnet': replacement_optimiser})


def test_direct_io_falls_back_only_when_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'bytes'
    path.write_bytes(b'checked bytes')

    def unsupported(*args: str, **kwargs: int) -> int:
        del args, kwargs
        raise OSError(errno.EINVAL, 'direct I/O not supported')

    with monkeypatch.context() as patch:
        patch.setattr(os, 'open', unsupported)
        assert read_disk(path) == b'checked bytes'

    def failed(*args: str, **kwargs: int) -> int:
        del args, kwargs
        raise OSError(errno.EIO, 'disk failed')

    with monkeypatch.context() as patch:
        patch.setattr(os, 'open', failed)
        with pytest.raises(OSError, match='disk failed'):
            read_disk(path)


@pytest.mark.parametrize('sidecar', [False, True])
def test_save_verifies_both_staged_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar: bool) -> None:
    import handtrack.train.checkpoint as checkpoint

    path = tmp_path / 'last.pt'
    checkpoint.save_checkpoint(path, {}, {}, TrainingState(step=7))
    original = checkpoint.read_disk

    def corrupt(staged: Path) -> bytes:
        data = original(staged)
        if staged.name.startswith('.') and staged.name.endswith('.sha256') == sidecar:
            return b'corrupt'
        return data

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, 'read_disk', corrupt)
        with pytest.raises(ValueError, match='SHA-256'):
            checkpoint.save_checkpoint(path, {}, {}, TrainingState(step=8))
    assert checkpoint.load_checkpoint(path, {}, {}).step == 7


def test_interrupted_publication_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'last.pt'
    save_checkpoint(path, {}, {}, TrainingState(step=7))
    original = os.replace

    def crash(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == Path(f'{path}.sha256'):
            raise OSError('simulated crash')
        original(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', crash)
        with pytest.raises(OSError, match='simulated crash'):
            save_checkpoint(path, {}, {}, TrainingState(step=8))
    assert load_checkpoint(path, {}, {}).step == 7
    path.unlink()
    assert load_checkpoint(path, {}, {}).step == 7


def test_loads_legacy_archive_and_hex_sidecar(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / 'last.pt'
    # Pre-fix metadata: MetricRecord has no recipe_changes field.
    metadata = ('{"epoch":2,"step":9,"config_json":"{}",'
                '"history":[{"step":8,"epoch":2,"values":{"loss":1.5}}],'
                '"epoch_steps":{"detnet":3},"best":{}}')
    torch.save({'models': {}, 'optimisers': {}, 'metadata': metadata,
                'rng': torch.get_rng_state(), 'cuda_rng': []}, path)
    Path(f'{path}.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '\n')
    state = load_checkpoint(path, {}, {})
    assert state.step == 9 and state.epoch_steps == {'detnet': 3}
    assert state.history[0].values == {'loss': 1.5}
    assert state.history[0].recipe_changes == {}
