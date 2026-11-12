"""``compile_unless_dev`` compiles in production unless the development environment or ``POSEKIT_COMPILE=0`` says eager."""

import functools

import pytest
import torch

from posekit.ops import compile as compile_module


@pytest.mark.parametrize(("environment", "compiled"), [({}, True), ({"POSEKIT_COMPILE": "0"}, False), ({"PIXI_DEV_MODE": "1"}, False), ({"POSEKIT_COMPILE": "1"}, True)])
def test_compile_is_off_in_the_dev_env_or_on_request(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], compiled: bool) -> None:
    def double(values: torch.Tensor) -> torch.Tensor:
        return values * 2

    monkeypatch.delenv("PIXI_DEV_MODE", raising=False)
    monkeypatch.delenv("POSEKIT_COMPILE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch, "compile", lambda function, **options: functools.partial(function, **options))
    result = compile_module.compile_unless_dev(double)
    assert isinstance(result, functools.partial) == compiled and (compiled or result is double)
