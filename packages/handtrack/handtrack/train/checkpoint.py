"""Atomic checkpoints with disk-bypassing integrity verification when supported."""
import errno
import hashlib
import io
import mmap
import os
import tempfile
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path

import torch
from serde import SerdeError, serde
from serde.json import from_json, to_json
from torch import nn
from torch.optim import Optimizer


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MetricRecord:
    """A validation observation retained in checkpoints."""
    step: int
    """Global gradient-step count."""
    epoch: int
    """Zero-based epoch."""
    values: dict[str, float | None]
    """Named scores; missing scores are null."""
    recipe_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    """Resume overrides as field -> (old, new); absent in legacy records."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrainingState:
    """Portable metadata; model and optimiser tensors live in the torch archive."""
    epoch: int = 0
    """Epoch to start or resume (zero-based)."""
    step: int = 0
    """Total completed optimiser steps across both networks."""
    config_json: str = '{}'
    """Resolved CLI configuration as JSON."""
    history: list[MetricRecord] = field(default_factory=list)
    """All validation observations so far."""
    epoch_steps: dict[str, int] = field(default_factory=dict)
    """Consumed batches per network in the current epoch; replay skips these."""
    best: dict[str, float] = field(default_factory=dict)
    """Best validation score for each network."""


def atomic_write(path: Path, payload: bytes) -> None:
    """Fsync bytes in the destination directory, replace, then fsync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created: tuple[int, str] = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    descriptor: int = created[0]
    temporary: str = created[1]
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory: int = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_disk(path: Path) -> bytes:
    """Read through aligned O_DIRECT buffers; fallback only for unsupported I/O.

    Checkpoint loading deserializes these verified bytes, never a second cached
    read. Anonymous mmap provides page alignment; padded read sizes also meet
    sector alignment. Real I/O errors propagate rather than hiding disk faults.
    """
    if not hasattr(os, 'O_DIRECT'):
        return path.read_bytes()
    try:
        descriptor: int = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            size: int = os.fstat(descriptor).st_size
            parts: list[bytes] = []
            offset: int = 0
            with mmap.mmap(-1, 1024 * 1024) as buffer:
                while offset < size:
                    count: int = os.preadv(descriptor, [buffer], offset)
                    if count == 0:
                        raise OSError('Unexpected end of checkpoint')
                    parts.append(buffer[:min(count, size - offset)])
                    offset += count
            return b''.join(parts)
        finally:
            os.close(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS):
            raise
        return path.read_bytes()


def verified_payload(path: Path) -> bytes:
    """Read both persisted files and reject a mismatched digest."""
    payload: bytes = read_disk(path)
    expected: bytes = read_disk(Path(f'{path}.sha256')).strip()
    if hashlib.sha256(payload).hexdigest().encode('ascii') != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')
    return payload


def save_archive(path: Path, payload: bytes) -> None:
    """Verify a staged pair, retain current as .prev, then publish the new pair.

    Readers must not run concurrently with a writer. The archive/hex-sidecar
    format is unchanged. A partial publication falls back to the previous pair.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{path.name}.', dir=path.parent) as directory:
        staged: Path = Path(directory) / '.archive'
        sidecar: Path = Path(f'{staged}.sha256')
        digest: bytes = hashlib.sha256(payload).hexdigest().encode('ascii')
        for attempt in (1, 2):
            atomic_write(staged, payload)
            atomic_write(sidecar, digest + b'\n')
            try:
                persisted: bytes = verified_payload(staged)
                if hashlib.sha256(persisted).hexdigest().encode('ascii') != digest:
                    raise ValueError(f'SHA-256 mismatch for {staged}')
                break
            except ValueError:
                if attempt == 2:
                    raise
                print(f'SHA-256 mismatch after writing {path} (attempt {attempt})', flush=True)
        try:
            previous: bytes = verified_payload(path)
        except (OSError, ValueError):
            pass  # Preserve an existing fallback after interrupted publication.
        else:
            # Keep current intact until both previous files are durable.
            backup: Path = Path(f'{path}.prev')
            atomic_write(backup, previous)
            atomic_write(Path(f'{backup}.sha256'), hashlib.sha256(previous).hexdigest().encode('ascii') + b'\n')
            verified_payload(backup)
        os.replace(staged, path)
        os.replace(sidecar, Path(f'{path}.sha256'))
        descriptor: int = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def save_checkpoint(path: Path, models: dict[str, nn.Module], optimisers: dict[str, Optimizer], state: TrainingState) -> None:
    """Save tensors plus strict JSON metadata in one checksummed torch archive."""
    buffer: io.BytesIO = io.BytesIO()
    torch.save({'models': {name: model.state_dict() for name, model in models.items()},
                'optimisers': {name: optimiser.state_dict() for name, optimiser in optimisers.items()},
                'metadata': to_json(state), 'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}, buffer)
    save_archive(path, buffer.getvalue())


def load_checkpoint(path: Path, models: dict[str, nn.Module], optimisers: dict[str, Optimizer]) -> TrainingState:
    """Verify disk bytes before safe deserialization and restore RNG and SGD state."""
    try:
        payload: bytes = verified_payload(path)
    except (OSError, ValueError) as error:
        previous: Path = Path(f'{path}.prev')
        print(f'Checkpoint {path} failed verification: {error}; trying {previous}', flush=True)
        try:
            payload = verified_payload(previous)
        except (OSError, ValueError):
            raise error from None
        print(f'Loading checkpoint {previous}', flush=True)
    else:
        print(f'Loading checkpoint {path}', flush=True)
    archive = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    try:
        state: TrainingState = from_json(TrainingState, archive['metadata'])
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f'Invalid checkpoint metadata in {path}: {error}') from error
    if set(models) != set(archive['models']) or set(optimisers) != set(archive['optimisers']):
        raise ValueError('Checkpoint network selection differs from this run')
    for name, model in models.items():
        model.load_state_dict(archive['models'][name])
        optimisers[name].load_state_dict(archive['optimisers'][name])
    torch.set_rng_state(archive['rng'])
    if archive['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(archive['cuda_rng'])
    return state


def export_weights(path: Path, model: nn.Module) -> None:
    """Export only state_dict, with the same checksum guarantees as a checkpoint."""
    buffer: io.BytesIO = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    save_archive(path, buffer.getvalue())
