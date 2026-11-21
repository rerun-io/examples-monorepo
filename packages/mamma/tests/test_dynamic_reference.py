"""The dynamic gate's frozen reference round-trips bit-exactly through its .rrd, and a missing file is an error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun", reason="the reference is an .rrd")


def test_reference_round_trips_bit_exactly(tmp_path: Path) -> None:
    from mamma.eval.dynamic_reference import load_reference, write_reference

    generator: np.random.Generator = np.random.default_rng(0)
    vertices = generator.standard_normal((6, 7, 3)).astype(np.float32)
    frames = np.arange(6, dtype=np.int64) + 3
    path: Path = tmp_path / "reference.rrd"
    write_reference(path, frames, vertices, 30.0, "unit test")

    loaded_frames, loaded_vertices, source = load_reference(path)
    assert loaded_frames.tolist() == frames.tolist()
    assert loaded_vertices.dtype == np.float32 and np.array_equal(loaded_vertices, vertices)
    assert source == "unit test"


def test_missing_reference_is_an_error(tmp_path: Path) -> None:
    from mamma.eval.dynamic_reference import load_reference

    with pytest.raises(FileNotFoundError, match="dynamic reference"):
        load_reference(tmp_path / "absent.rrd")


def test_prediction_rows_must_agree() -> None:
    from mamma.apis.freeze_dynamic_reference import MammaPredParams

    with pytest.raises(ValueError, match="5 pose rows for 4 translation rows"):
        MammaPredParams(betas=np.zeros(16, np.float32), poses=np.zeros((5, 165), np.float32), trans=np.zeros((4, 3), np.float32))
