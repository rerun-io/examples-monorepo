"""The run-scoped model prefetch: a failed or stopped run keeps no model alive (CPU, fake loaders)."""

import gc
import threading
import weakref

import pytest

from fdanyone.config import RANK64_DELTA4
from fdanyone.model import loader
from fdanyone.model.loader import GenerationPrefetch


class _Model:
    """Stands in for a loaded DiT or VAE."""


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[weakref.ref]:
    """Fake loaders that record a weak reference to every model they build."""
    built: list[weakref.ref] = []

    def load(*_args: object, **_kwargs: object) -> _Model:
        model = _Model()
        built.append(weakref.ref(model))
        return model

    monkeypatch.setattr(loader, "load_denoiser", load)
    monkeypatch.setattr(loader, "load_vae", load)
    return built


def _prefetch() -> GenerationPrefetch:
    return GenerationPrefetch(checkpoint_path="dit", turbo_lora_path=None, profile=RANK64_DELTA4, attention_backend="sdpa", vae_path="vae")


@pytest.mark.parametrize("taken", [(), ("denoiser",), ("denoiser", "vae")])
def test_close_drops_every_model_the_run_did_not_keep(loads: list[weakref.ref], taken: tuple[str, ...]) -> None:
    prefetch = _prefetch()
    kept = [prefetch.take_denoiser() if name == "denoiser" else prefetch.take_vae() for name in taken]
    prefetch.close()
    del prefetch
    gc.collect()
    # close() cancels a load that had not started yet, so fewer than two models may have been built at all.
    assert sum(ref() is not None for ref in loads) == len(kept)


def test_close_waits_for_a_load_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_load(*_args: object, **_kwargs: object) -> _Model:
        started.set()
        release.wait(5)
        finished.set()
        return _Model()

    monkeypatch.setattr(loader, "load_denoiser", slow_load)
    monkeypatch.setattr(loader, "load_vae", lambda *_a, **_k: _Model())
    prefetch = _prefetch()
    assert started.wait(5)
    threading.Timer(0.2, release.set).start()
    prefetch.close()
    assert finished.is_set()


def test_a_failed_load_raises_where_it_is_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: object, **_kwargs: object) -> _Model:
        raise OSError("checkpoint unreadable")

    monkeypatch.setattr(loader, "load_denoiser", broken)
    monkeypatch.setattr(loader, "load_vae", lambda *_a, **_k: _Model())
    prefetch = _prefetch()
    with pytest.raises(OSError, match="unreadable"):
        prefetch.take_denoiser()
    with pytest.raises(RuntimeError, match="already taken"):
        prefetch.take_denoiser()
    prefetch.close()
