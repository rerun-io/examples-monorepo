"""Build machine-local artifacts once across processes."""

import fcntl
from collections.abc import Callable
from pathlib import Path


def build_once(artifact: Path, build: Callable[[], None]) -> Path:
    """Run ``build`` unless ``artifact`` exists; processes that find it missing take turns, so the first one's build serves them all.

    ``build`` must publish ``artifact`` last (stage it, then rename), because its presence means it is complete.
    The lock file ``<name>.lock`` sits beside the artifact.

    Returns:
        ``artifact``.
    """
    if artifact.exists():
        return artifact
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.with_name(f"{artifact.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not artifact.exists():
            build()
    return artifact
