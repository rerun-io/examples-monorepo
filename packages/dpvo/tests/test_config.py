"""DPVOConfig: the backend knobs default to plain DPVO and reject bad values."""

from collections.abc import Callable

import pytest
from beartype.roar import BeartypeException

from dpvo.config import DPVOConfig

PRESETS: list[Callable[[], DPVOConfig]] = [
    DPVOConfig,
    DPVOConfig.accurate,
    DPVOConfig.fast,
    DPVOConfig.slam,
    DPVOConfig.slam_fast,
    DPVOConfig.slam_classic,
    DPVOConfig.slam_classic_fast,
]


@pytest.mark.parametrize("preset", PRESETS, ids=lambda preset: preset.__name__)
def test_presets_keep_plain_dpvo_backend(preset: Callable[[], DPVOConfig]) -> None:
    cfg: DPVOConfig = preset()
    assert cfg.fmap_memory == 36
    assert cfg.store_inactive is True
    assert cfg.normalize_global_ba is True
    assert cfg.global_ba_t0 == "active_min"
    assert cfg.gba_schedule_k == 0
    assert cfg.raise_on_ba_failure is False
    assert cfg.motion_scale_by_tstamps is False
    # update() autocast follows mixed_precision; every preset keeps the former hard-coded True
    assert cfg.mixed_precision is True


def test_accepts_metric_backend_values() -> None:
    cfg: DPVOConfig = DPVOConfig(
        fmap_memory=512,
        store_inactive=False,
        normalize_global_ba=False,
        global_ba_t0="first",
        gba_schedule_k=8,
        raise_on_ba_failure=True,
        motion_scale_by_tstamps=True,
    )
    assert cfg.fmap_memory == 512
    assert cfg.global_ba_t0 == "first"


@pytest.mark.parametrize("fmap_memory", [0, 35])
def test_rejects_short_fmap_ring(fmap_memory: int) -> None:
    with pytest.raises(ValueError, match="fmap_memory"):
        DPVOConfig(fmap_memory=fmap_memory)


def test_rejects_negative_schedule() -> None:
    with pytest.raises(ValueError, match="gba_schedule_k"):
        DPVOConfig(gba_schedule_k=-1)


def test_rejects_unknown_global_ba_t0() -> None:
    # beartype (dev env) rejects the Literal at __init__; __post_init__ rejects it in prod
    with pytest.raises((ValueError, BeartypeException)):
        DPVOConfig(global_ba_t0="all")  # pyrefly: ignore[bad-argument-type]
