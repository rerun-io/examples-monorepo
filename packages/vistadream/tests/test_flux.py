"""FLUX.1 Fill role (unit lane, no weights): told to use only cached files, loading never downloads."""

import diffusers
import pytest

from vistadream.models.flux import FluxConfig, FluxInpainting


def test_flux_loads_only_cached_files_when_told(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def from_pretrained(name: str, **kwargs: object) -> None:
        calls.append(kwargs)
        raise OSError(f"{name} is not cached")

    monkeypatch.setattr(diffusers.FluxFillPipeline, "from_pretrained", from_pretrained)
    with pytest.raises(OSError):
        FluxInpainting(FluxConfig(local_files_only=True))
    assert calls[0]["local_files_only"] is True
