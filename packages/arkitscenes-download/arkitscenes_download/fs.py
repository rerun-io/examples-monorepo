"""Shared filesystem helpers."""

from pathlib import Path


def directory_size(path: Path) -> int:
    """Return recursive file bytes, or zero for an absent directory."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
