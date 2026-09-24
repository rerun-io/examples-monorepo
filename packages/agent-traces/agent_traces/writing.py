"""Atomic publication of catalog-readable files."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(target: Path) -> Iterator[Path]:
    """Yield a temporary path beside the target, publishing only on clean exit."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: tuple[int, str] = tempfile.mkstemp(dir=target.parent, suffix=f"{target.suffix}.tmp")
    os.close(temporary[0])
    path: Path = Path(temporary[1])
    try:
        yield path
        path.chmod(0o644)  # Catalog servers running as another user must be able to read the file.
        os.replace(path, target)
    finally:
        path.unlink(missing_ok=True)
