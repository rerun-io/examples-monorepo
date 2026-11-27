"""The engine warm-up builds the engines ``auto`` will load, and nothing where ``auto`` runs ONNX Runtime."""

from typing import get_args

import pytest

from posekit.apis.warm_engines import WarmedConfig, WarmEnginesConfig, main, warm_configs
from posekit.models.rtmpose import RtmPoseConfig, RtmPoseVariant
from posekit.models.yolox import YoloxDetectorConfig, YoloxVariant


def test_every_variant_is_warmed_with_its_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default config is what later runs set up, so its resolved ``auto`` is the engine they load."""
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: True)
    expected: list[WarmedConfig] = [YoloxDetectorConfig(variant=variant) for variant in get_args(YoloxVariant)]
    expected += [RtmPoseConfig(variant=variant) for variant in get_args(RtmPoseVariant)]
    assert warm_configs(WarmEnginesConfig()) == expected


def test_without_tensorrt_there_is_nothing_to_warm(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: False)
    main(WarmEnginesConfig())
    assert "nothing to build" in capsys.readouterr().out
