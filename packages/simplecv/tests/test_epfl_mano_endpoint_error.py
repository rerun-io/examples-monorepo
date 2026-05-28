"""Endpoint-error guard: MANO-derived joints must match CSV ground truth.

The full validator is in ``tools/validate_epfl_mano.py``. This guard exercises
the same metric on the first available train session of the AV1 mirror under
``/mnt/8tb/data/epfl-smart-kitchen-av1``. When that data is absent (e.g. on CI
runners), the test is skipped rather than failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on sys.path so we can import the CLI module.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simplecv.data.exoego.epfl_smart_kitchen import (  # noqa: E402
    EpflSmartKitchenConfig,
    EpflSmartKitchenSequence,
    hand_pose_path,
)
from tools.validate_epfl_mano import (  # noqa: E402
    DEFAULT_EPFL_ROOT,
    compute_endpoint_error,
    pick_first_train_sessions,
)


def _first_available_train_session() -> tuple[str, str] | None:
    if not DEFAULT_EPFL_ROOT.exists():
        return None
    picks = pick_first_train_sessions(DEFAULT_EPFL_ROOT, 1)
    return picks[0] if picks else None


def _have_mano_pkls() -> bool:
    mano_root: Path = _REPO_ROOT / "simplecv" / "data"
    return (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()


@pytest.mark.slow
def test_epfl_mano_endpoint_error_against_csv_ground_truth() -> None:
    pick = _first_available_train_session()
    if pick is None:
        pytest.skip("EPFL Smart Kitchen AV1 mirror not present; nothing to validate")
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")
    participant, session = pick

    cfg = EpflSmartKitchenConfig(
        root_directory=DEFAULT_EPFL_ROOT,
        split="train",
        participant_id=participant,
        session_name=session,
        load_labels=True,
    )
    # Sanity: the pose CSV must exist where the loader expects it.
    assert hand_pose_path(cfg).exists(), f"Missing pose3d_mano.csv at {hand_pose_path(cfg)}"

    sequence = EpflSmartKitchenSequence(cfg)
    labels = sequence.exoego_labels
    assert labels is not None and labels.mano_stack is not None

    per_frame_mean, per_frame_max, sampled = compute_endpoint_error(labels.mano_stack, labels.xyzc_stack, n_frames=50)
    assert sampled.size > 0
    mean_mm: float = float(per_frame_mean.mean())
    max_mm: float = float(per_frame_max.max())

    # Tolerances match the /goal pass condition.
    assert mean_mm < 20.0, f"mean endpoint error {mean_mm:.2f} mm exceeds 20 mm budget"
    assert max_mm < 50.0, f"max endpoint error {max_mm:.2f} mm exceeds 50 mm budget"

    # Per-hand betas must round-trip through the loader.
    assert labels.mano_stack.betas.shape == (2, 10)
    assert not np.allclose(labels.mano_stack.betas_for(0), labels.mano_stack.betas_for(1))
