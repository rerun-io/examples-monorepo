"""MetricDPVOConfig and its adapter for ViPE's uppercase yacs keys."""

from collections.abc import Mapping
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from dpvo.metric.config import YACS_DEFAULTS, MetricDPVOConfig, YacsValue  # noqa: E402

COMMON: dict[str, YacsValue] = {
    "BUFFER_SIZE": 8192,
    "PATCHES_PER_FRAME": 96,
    "REMOVAL_WINDOW": 22,
    "OPTIMIZATION_WINDOW": 10,
    "PATCH_LIFETIME": 13,
    "KEYFRAME_THRESH": 15.0,
    "METRIC_DEPTH": True,
}
GBA: dict[str, YacsValue] = {**COMMON, "LOOP_CLOSURE": True, "METRIC_GLOBAL_BA": True, "METRIC_GLOBAL_BA_T0": "all"}
GBA_REFINE: dict[str, YacsValue] = {
    **GBA,
    "GBA_SCHEDULE_K": 8,
    "CULLED_INTERP": True,
    "GBA_REMEASURE_FINAL": "two",
    "REMEASURE_BURNIN": 3,
    "REMEASURE_WARM_ACTIVE": True,
    "REMEASURE_CARRY_STATE": True,
    "GBA_SCALE_LATENT": True,
    "LOOP_REVERSE_EDGES": True,
    "KEYFRAME_MAX_CULL": 10,
}
PRESETS: dict[str, dict[str, YacsValue]] = {"local": COMMON, "gba": GBA, "gba_refine": GBA_REFINE}
"""The non-default keys of ViPE's three MetricTrail presets (vipe/_configs/slam/metrictrail_*.yaml)."""


def block(overrides: Mapping[str, YacsValue]) -> dict[str, YacsValue]:
    return {**YACS_DEFAULTS, **overrides}


def test_defaults_have_the_43_keys() -> None:
    assert len(YACS_DEFAULTS) == 43


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_preset_round_trip(preset: str) -> None:
    yacs: dict[str, YacsValue] = block(PRESETS[preset])
    cfg: MetricDPVOConfig = MetricDPVOConfig.from_yacs(yacs)
    assert cfg.to_yacs() == yacs
    assert list(cfg.to_yacs()) == list(YACS_DEFAULTS)
    assert MetricDPVOConfig.from_yacs(cfg.to_yacs()) == cfg


def test_preset_fields() -> None:
    local: MetricDPVOConfig = MetricDPVOConfig.from_yacs(block(COMMON))
    assert (local.store_inactive, local.normalize_global_ba, local.global_ba_t0, local.fmap_memory) == (False, True, "active_min", 36)
    assert (local.buffer_size, local.patches_per_frame, local.gradient_bias) == (8192, 96, False)
    assert local.motion_scale_by_tstamps and local.raise_on_ba_failure

    gba: MetricDPVOConfig = MetricDPVOConfig.from_yacs(block(GBA))
    assert (gba.store_inactive, gba.normalize_global_ba, gba.global_ba_t0, gba.fmap_memory) == (True, False, "first", 36)

    refine: MetricDPVOConfig = MetricDPVOConfig.from_yacs(block(GBA_REFINE))
    assert (refine.fmap_memory, refine.gba_schedule_k, refine.remeasure_final, refine.remeasure_burnin) == (512, 8, "two", 3)
    assert refine.scale_latent and refine.loop_reverse_edges and refine.culled_interp
    assert (refine.keyframe_max_cull, refine.keyframe_max_cull_rot_deg) == (10, 10.0)


def test_typed_defaults_equal_yacs_defaults_with_metric_depth() -> None:
    assert MetricDPVOConfig.from_yacs(block({"METRIC_DEPTH": True})) == MetricDPVOConfig()


def test_int_widens_to_float_but_not_the_reverse() -> None:
    assert MetricDPVOConfig.from_yacs(block({**COMMON, "KEYFRAME_THRESH": 15})).keyframe_thresh == 15.0
    with pytest.raises(ValueError, match="PATCHES_PER_FRAME must be int"):
        MetricDPVOConfig.from_yacs(block({**COMMON, "PATCHES_PER_FRAME": 96.0}))
    with pytest.raises(ValueError, match="MIXED_PRECISION must be bool"):
        MetricDPVOConfig.from_yacs(block({**COMMON, "MIXED_PRECISION": 1}))


def test_unknown_or_absent_keys_raise() -> None:
    with pytest.raises(ValueError, match="unknown=\\['EXTRA'\\]"):
        MetricDPVOConfig.from_yacs(block({**COMMON, "EXTRA": 1}))
    absent: dict[str, YacsValue] = block(COMMON)
    del absent["FMAP_STORE"]
    with pytest.raises(ValueError, match="absent=\\['FMAP_STORE'\\]"):
        MetricDPVOConfig.from_yacs(absent)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({**COMMON, "METRIC_DEPTH": False}, "METRIC_DEPTH"),
        ({**COMMON, "CENTROID_SEL_STRAT": "GRADIENT_BIAS"}, "CENTROID_SEL_STRAT"),
        ({**COMMON, "MOTION_MODEL": "CONSTANT"}, "MOTION_MODEL"),
        ({**COMMON, "CLASSIC_LOOP_CLOSURE": True}, "CLASSIC_LOOP_CLOSURE"),
        ({**GBA_REFINE, "REMEASURE_MIN_AGE": 3}, "REMEASURE_MIN_AGE"),
        ({**GBA_REFINE, "GBA_SCALE_LOCAL_USE": "frozen"}, "GBA_SCALE_LOCAL_USE"),
        ({**GBA, "GBA_REMEASURE_FINAL": "one"}, "GBA_REMEASURE_FINAL"),
        ({**GBA, "METRIC_GLOBAL_BA_T0": "active_min"}, "METRIC_GLOBAL_BA_T0"),
        ({**COMMON, "METRIC_GLOBAL_BA_T0": "all"}, "METRIC_GLOBAL_BA_T0"),
        ({**GBA, "FMAP_STORE": 256}, "FMAP_STORE"),
    ],
)
def test_dead_keys_at_other_values_raise(overrides: dict[str, YacsValue], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MetricDPVOConfig.from_yacs(block(overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {**COMMON, "LOOP_CLOSURE": True},  # proximity edges need the metric global BA
        {**COMMON, "METRIC_GLOBAL_BA": True, "METRIC_GLOBAL_BA_T0": "all"},  # and the reverse
        {**COMMON, "GBA_SCHEDULE_K": 8},
        {**COMMON, "GBA_SCALE_LATENT": True},
        {**GBA, "LOOP_REVERSE_EDGES": True},  # needs the long fmap ring of the re-measurement
        {**GBA_REFINE, "FMAP_STORE": 20},
        {**COMMON, "KEYFRAME_MAX_CULL": -1},
        {**COMMON, "METRIC_DEPTH_MIN": 600.0},
    ],
)
def test_cross_key_guards(overrides: dict[str, YacsValue]) -> None:
    with pytest.raises(ValueError):
        MetricDPVOConfig.from_yacs(block(overrides))


def test_inconsistent_derived_fields_raise() -> None:
    with pytest.raises(ValueError, match="store_inactive"):
        MetricDPVOConfig(store_inactive=True)
    with pytest.raises(ValueError, match="fmap_memory"):
        MetricDPVOConfig(fmap_memory=512)


def test_to_yacs_rejects_values_yacs_cannot_express() -> None:
    with pytest.raises(ValueError, match="cannot express"):
        MetricDPVOConfig(motion_scale_by_tstamps=False).to_yacs()
    with pytest.raises(ValueError, match="cannot express"):
        MetricDPVOConfig(gradient_bias=True).to_yacs()


# ── against ViPE (metrictrail env only) ────────────────────────────────


def vipe_slam_configs() -> Path:
    vipe = pytest.importorskip("vipe", reason="ViPE is installed only in the metrictrail env")
    return Path(vipe.__file__).parent / "_configs" / "slam"


def test_yacs_defaults_equal_vipe_schema() -> None:
    """ViPE's Hydra schema (``vipe.metrictrail.config.DPVOConfig``) still lists DPVO's defaults: they must equal ours."""
    vipe_slam_configs()
    from vipe.metrictrail.config import DPVOConfig as VipeDPVOSchema
    from vipe.metrictrail.dpvo.config import cfg

    schema: dict[str, YacsValue] = VipeDPVOSchema().model_dump()
    assert schema == dict(YACS_DEFAULTS)
    assert {key: type(value) for key, value in schema.items()} == {key: type(value) for key, value in YACS_DEFAULTS.items()}
    assert dict(cfg) == dict(YACS_DEFAULTS)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_vipe_preset_yaml_loads(preset: str) -> None:
    yaml = pytest.importorskip("yaml")
    path: Path = vipe_slam_configs() / f"metrictrail_{preset}.yaml"
    dpvo_block: dict[str, YacsValue] = yaml.safe_load(path.read_text())["dpvo"]
    merged: dict[str, YacsValue] = block(dpvo_block)
    cfg: MetricDPVOConfig = MetricDPVOConfig.from_yacs(merged)
    assert cfg == MetricDPVOConfig.from_yacs(block(PRESETS[preset]))
    assert cfg.to_yacs() == merged
