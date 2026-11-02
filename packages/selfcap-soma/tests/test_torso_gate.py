"""The calibration-frame gate of the two single-segment tools, on literal view counts around the threshold."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray

from selfcap_soma import evaluate, shoulders
from selfcap_soma.catalog import MIN_SHOULDER_VIEWS, Observations, well_observed_torso

TORSO: list[int] = [5, 6, 11, 12]


class FitReached(Exception):
    """Ends a tool at the fit that receives the selected frames, before any GPU work."""


def observations(stride: int) -> Observations:
    """Blocks of ``stride`` frames, one block per case, so a tool's every-Nth stride keeps the first frame of each block that passes."""
    cases: list[tuple[float, int, bool]] = [
        # block start time, shoulder views, shoulder finite
        (0.0, MIN_SHOULDER_VIEWS, True),  # at the threshold: kept
        (1.0, MIN_SHOULDER_VIEWS - 1, True),  # just below: dropped
        (2.0, 4, True),  # above: kept
        (3.0, 4, False),  # many views but a missing shoulder: dropped
        (30.0, 4, True),  # after the 25 s calibration window: dropped
    ]
    times: Float64[ndarray, "n"] = np.concatenate([start + 0.01 * np.arange(stride) for start, _, _ in cases])
    frames: int = len(times)
    xyz: Float64[ndarray, "n 133 3"] = np.zeros((frames, 133, 3))
    support: Int64[ndarray, "n 133"] = np.full((frames, 133), 4, dtype=np.int64)
    for index, (_, views, shoulder_finite) in enumerate(cases):
        block: slice = slice(index * stride, (index + 1) * stride)
        support[block, TORSO[0]] = views  # one weak shoulder decides the frame: the gate takes the shoulders' minimum
        support[block, TORSO[2:]] = 2  # hips at two views (the common case) never block a frame
        if not shoulder_finite:
            xyz[block, TORSO[0]] = np.nan
    return Observations(times, xyz, np.zeros((frames, 133)), xyz.copy(), np.zeros((frames, 3)), np.tile(np.eye(3), (frames, 1, 1)), 0.0, support)


def test_a_layer_without_view_counts_cannot_be_gated() -> None:
    data: Observations = observations(1)
    without: Observations = replace(data, support=None)
    with pytest.raises(ValueError, match="support_view_count"):
        well_observed_torso(without)


def test_shoulders_calibrates_on_frames_with_enough_torso_views(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stride: int = 4
    seen: list[Observations] = []

    def fake_fit(data: Observations, *_args: object, **_kwargs: object) -> Observations:
        seen.append(data)
        if len(seen) == 2:
            raise FitReached
        return data

    monkeypatch.setattr(shoulders, "Path", lambda _path: tmp_path)
    monkeypatch.setattr(shoulders, "read_window", lambda *_args, **_kwargs: observations(stride))
    monkeypatch.setattr(shoulders, "fit", fake_fit)
    monkeypatch.setattr(shoulders, "describe", lambda *_args: None)
    monkeypatch.setattr(shoulders, "save", lambda *_args: None)
    with pytest.raises(FitReached):
        shoulders.main()
    assert seen[1].times.tolist() == [0.0, 2.0]


def test_evaluate_calibrates_on_frames_with_enough_torso_views(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stride: int = 8
    seen: list[Observations] = []

    def fake_fit(data: Observations, *_args: object, **_kwargs: object) -> None:
        seen.append(data)
        raise FitReached

    monkeypatch.setattr(evaluate, "read_window", lambda *_args, **_kwargs: observations(stride))
    monkeypatch.setattr(evaluate, "read_views", lambda *_args, **_kwargs: ([], None))
    monkeypatch.setattr(evaluate, "CatalogClient", lambda _url: type("Client", (), {"get_dataset": lambda _self, _name: None})())
    monkeypatch.setattr(evaluate, "fit", fake_fit)
    with pytest.raises(FitReached):
        evaluate.main(evaluate.Config(output=tmp_path, calibrated=True))
    assert seen[0].times.tolist() == [0.0, 2.0]
