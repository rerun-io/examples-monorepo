"""Reports retain backend choices and preserve legacy input selections."""

from dataclasses import replace
from pathlib import Path

from serde.json import from_json, to_json

from posekit.apis.body_accuracy import BodyAccuracyReport
from posekit.apis.body_bench import BodyBenchResult
from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
from posekit.models.sam3d_body_quantization import BodyPrecisionPlan, CalibrationFrame, load_body_calibration, write_body_calibration


def test_report_identity_distinguishes_precision_plans() -> None:
    from posekit.body_run import body_run_identity

    config = Sam3dBodyFastConfig(backbone_precision="nvfp4")
    first = body_run_identity(config)
    second = body_run_identity(replace(config, backbone_plan=BodyPrecisionPlan(attention={22: "bf16"})))
    assert first != second
    assert first.backbone_artifacts != second.backbone_artifacts
    report = BodyAccuracyReport([CalibrationFrame("input", 0, "a" * 64)], 0, {}, False, [], {}, run=first)
    assert from_json(BodyAccuracyReport, to_json(report)) == report
    bench = BodyBenchResult("full", "test gpu", 1, 1.0, {}, run=second)
    assert from_json(BodyBenchResult, to_json(bench)) == bench


def test_legacy_reports_remain_loadable() -> None:
    accuracy = '{"frames":[],"people":0,"bands":{},"passed":false,"per_frame":[],"eager_repeat_bands":{}}'
    benchmark = '{"mode":"full","gpu":"test","frames":1,"people_per_frame":1.0,"stages":{}}'
    assert from_json(BodyAccuracyReport, accuracy).run is None
    assert from_json(BodyBenchResult, benchmark).run is None


def test_identity_records_the_effective_calibration_override(tmp_path: Path) -> None:
    from posekit.body_run import body_run_identity

    original = Sam3dBodyFastConfig()
    calibration = load_body_calibration()
    ranges = dict(calibration.input_amax)
    ranges[next(iter(ranges))] *= 1.01
    path = tmp_path / "override.rrd"
    write_body_calibration(path, replace(calibration, input_amax=ranges))
    effective = replace(original, calibration=path)
    identity = body_run_identity(effective)
    assert identity.config == effective
    assert identity.calibration_sha256 != body_run_identity(original).calibration_sha256
    assert identity.backbone_artifacts != body_run_identity(original).backbone_artifacts
