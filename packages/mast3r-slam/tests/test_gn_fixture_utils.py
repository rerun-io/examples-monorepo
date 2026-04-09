from __future__ import annotations

from pathlib import Path

import torch

from mast3r_slam.gn_fixture_utils import (
    GN_CAPTURE_DIR_ENV,
    GN_CAPTURE_LIMIT_ENV,
    load_gn_fixture,
    maybe_capture_gn_fixture,
)


def test_capture_and_load_fixture_roundtrip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(GN_CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(GN_CAPTURE_LIMIT_ENV, "2")

    out_path = maybe_capture_gn_fixture(
        "rays",
        inputs={
            "Twc": torch.randn(3, 8),
            "ii": torch.tensor([0, 1], dtype=torch.long),
            "sigma_ray": 0.003,
        },
        metadata={"name": "unit-fixture"},
    )

    assert out_path is not None
    payload = load_gn_fixture(out_path)
    assert payload["kind"] == "rays"
    assert payload["metadata"]["name"] == "unit-fixture"
    assert payload["inputs"]["Twc"].device.type == "cpu"
    assert payload["inputs"]["ii"].dtype == torch.long


def test_capture_limit_is_respected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(GN_CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(GN_CAPTURE_LIMIT_ENV, "1")

    first = maybe_capture_gn_fixture("rays", inputs={"Twc": torch.zeros(1, 8)})
    second = maybe_capture_gn_fixture("rays", inputs={"Twc": torch.ones(1, 8)})

    assert first is not None
    assert second is None
    assert len(list(tmp_path.glob("rays-*.pt"))) == 1
