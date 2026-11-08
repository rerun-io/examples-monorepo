"""dpvo.metric's yacs defaults against ViPE's DPVO schema; it lives here because only this env installs ViPE."""

from dpvo.metric.config import YACS_DEFAULTS, YacsValue
from vipe.metrictrail.config import DPVOConfig as VipeDPVOSchema
from vipe.metrictrail.dpvo.config import cfg


def test_yacs_defaults_equal_vipe_schema() -> None:
    """ViPE's Hydra schema (``vipe.metrictrail.config.DPVOConfig``) still lists DPVO's defaults: they must equal ours."""
    schema: dict[str, YacsValue] = VipeDPVOSchema().model_dump()
    assert schema == dict(YACS_DEFAULTS)
    assert {key: type(value) for key, value in schema.items()} == {key: type(value) for key, value in YACS_DEFAULTS.items()}
    assert dict(cfg) == dict(YACS_DEFAULTS)
