from __future__ import annotations

from pathlib import Path

from mv_api.gradio_ui import full_pipeline_rrd_ui as ui


def test_cleanup_temp_rrds_removes_existing_files(tmp_path: Path) -> None:
    output = tmp_path / "output.rrd"
    output.write_bytes(b"rrd")

    ui.cleanup_temp_rrds([str(output)])

    assert not output.exists()


def test_run_rrd_pipeline_uses_sanitized_config(monkeypatch: object, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyDatasetConfig:
        def __init__(self, *, rrd_path: Path) -> None:
            captured["rrd_path"] = rrd_path

    class DummyCalibratorConfig:
        def __init__(self, *, segment_people: bool) -> None:
            captured["segment_people"] = segment_people

    class DummyPipelineConfig:
        def __init__(self, *, rr_config: object, calib_confg: object, dataset: object, max_frames: int | None) -> None:
            self.rr_config = rr_config
            self.calib_confg = calib_confg
            self.dataset = dataset
            self.max_frames = max_frames

    def fake_run_full_exoego_pipeline(*, config: object) -> None:
        captured["config"] = config

    monkeypatch.setattr(ui, "RRDExoEgoConfig", DummyDatasetConfig)
    monkeypatch.setattr(ui, "MultiViewCalibratorConfig", DummyCalibratorConfig)
    monkeypatch.setattr(ui, "RRDPipelineConfig", DummyPipelineConfig)
    monkeypatch.setattr(ui, "run_full_exoego_pipeline", fake_run_full_exoego_pipeline)

    input_rrd = tmp_path / "input.rrd"
    input_rrd.write_bytes(b"rrd")
    pending_cleanup: list[str] = []

    viewer_rrd, download_rrd = ui.run_rrd_pipeline(str(input_rrd), 0, pending_cleanup)

    assert viewer_rrd == download_rrd
    assert Path(viewer_rrd).exists()
    assert pending_cleanup == [viewer_rrd]
    assert captured["rrd_path"] == input_rrd
    assert captured["segment_people"] is False

    config = captured["config"]
    assert config is not None
    assert config.max_frames is None
