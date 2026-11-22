"""Held-out camera selection: a fixed seed picks the cameras that never enter training or init."""

import random


def select_holdout(camera_names: tuple[str, ...], num_holdout: int, seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cameras into training and held-out sets.

    Args:
        camera_names: Every camera of the clip, in any order.
        num_holdout: How many cameras to hold out.
        seed: Seed of ``random.Random``; the same names and seed always give the same split.

    Returns:
        ``(train_names, holdout_names)``: training cameras in sorted order, held-out cameras in sample order
        (``random.Random(seed).sample(sorted(camera_names), num_holdout)``).

    Raises:
        ValueError: When the names repeat or ``num_holdout`` leaves no training camera.
    """
    names: list[str] = sorted(camera_names)
    if len(set(names)) != len(names):
        raise ValueError(f"camera names repeat: {names}")
    if not 0 <= num_holdout < len(names):
        raise ValueError(f"num_holdout={num_holdout} must be in [0, {len(names)}) for {len(names)} cameras")
    holdout: list[str] = random.Random(seed).sample(names, num_holdout)
    train: tuple[str, ...] = tuple(name for name in names if name not in holdout)
    return train, tuple(holdout)
