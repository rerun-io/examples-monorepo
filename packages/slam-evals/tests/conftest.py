"""pytest fixtures for slam-evals.

The synthetic-sequence builder lives in ``slam_evals.data.synthetic`` so
both the test suite and the ``slam-evals-smoke`` CLI tool can build the
same per-modality fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_evals.data.synthetic import Fixture, build_fixture
from slam_evals.data.types import Modality


@pytest.fixture
def fixture_factory(tmp_path: Path):
    """Factory fixture: ``fixture_factory(modality, n_frames=5)`` → ``Fixture``."""

    def _make(modality: Modality, *, n_frames: int = 5) -> Fixture:
        return build_fixture(tmp_path / "bench", modality=modality, n_frames=n_frames)

    return _make
