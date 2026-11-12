"""The CLI and app resolve the same patch budget, with explicit overrides retained."""

from pathlib import Path

import pytest

pytest.importorskip("metrictrail.apis.run", reason="run settings require the MetricTrail environment")

from metrictrail.apis.run import PreparedInput, RunSettings, hydra_overrides  # noqa: E402
from metrictrail.benchmarks import BenchmarkName  # noqa: E402


@pytest.mark.parametrize("benchmark", ["video", "tum"])
def test_default_patch_budget_reaches_the_video_and_accuracy_presets(benchmark: BenchmarkName) -> None:
    source: Path = Path("clip.mp4") if benchmark == "video" else Path("frames/desk")
    settings: RunSettings = RunSettings(input=source, output=Path("out"), benchmark=benchmark)
    prepared: PreparedInput = PreparedInput(stream_path=source, name=source.stem, gt_sequence=None, raster_overrides=[])
    overrides: list[str] = hydra_overrides(settings, prepared, settings.output)
    assert "pipeline.slam.dpvo.PATCHES_PER_FRAME=64" in overrides


def test_patch_budget_can_restore_96_and_extra_overrides_still_win() -> None:
    settings: RunSettings = RunSettings(input=Path("clip.mp4"), output=Path("out"), patches_per_frame=96)
    prepared: PreparedInput = PreparedInput(stream_path=settings.input, name="clip", gt_sequence=None, raster_overrides=[])
    assert "pipeline.slam.dpvo.PATCHES_PER_FRAME=96" in hydra_overrides(settings, prepared, settings.output)
    settings.extra_overrides = ["pipeline.slam.dpvo.PATCHES_PER_FRAME=80"]
    overrides: list[str] = hydra_overrides(settings, prepared, settings.output)
    assert overrides.index("pipeline.slam.dpvo.PATCHES_PER_FRAME=96") < overrides.index("pipeline.slam.dpvo.PATCHES_PER_FRAME=80")


def test_upstream_label_keeps_its_own_patch_budget() -> None:
    settings: RunSettings = RunSettings(input=Path("clip.mp4"), output=Path("out"), label="upstream")
    prepared: PreparedInput = PreparedInput(stream_path=settings.input, name="clip", gt_sequence=None, raster_overrides=[])
    assert all("PATCHES_PER_FRAME" not in value for value in hydra_overrides(settings, prepared, settings.output))


def test_the_cli_runs_the_shared_default_clip_without_an_input() -> None:
    import tyro

    from metrictrail.apis.run import Config
    from metrictrail.examples import DEFAULT_CLIP, DEFAULT_INPUT

    config: Config = tyro.cli(Config, args=["--output", "out", "--rr-config.headless"])
    assert config.input == DEFAULT_INPUT and DEFAULT_INPUT.name == DEFAULT_CLIP == "IMG_0497.mp4"
