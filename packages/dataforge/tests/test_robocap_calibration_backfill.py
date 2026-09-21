"""Backfill augments existing catalog recordings without replacing their base."""

import hashlib
from pathlib import Path

import pytest
import rerun as rr

pytest.importorskip("rerun.catalog", reason="calibration backfill tests need the Rerun catalog dependencies")

from rerun.catalog import CatalogClient, DatasetEntry  # noqa: E402
from rerun.chunk import RrdReader  # noqa: E402

from dataforge.apis.register import Config as RegisterConfig  # noqa: E402
from dataforge.apis.register import main as register  # noqa: E402
from dataforge.apis.robocap_calibration import Config, main  # noqa: E402

pytestmark = pytest.mark.integration


def test_backfill_adds_only_static_metadata_and_preserves_base(tmp_path: Path) -> None:
    device: str = "f408193e6447b3b0"
    segment: str = f"robocap__{device}__s00000015"
    factory: Path = tmp_path / f"0factory-calibration-{device}/imus_intrinsic"
    factory.mkdir(parents=True)
    (factory / "imu_mid_0.yaml").write_text("update_rate: 200.0\ngyroscope_noise_density: 0.0007\n")
    base: Path = tmp_path / "base.rrd"
    with rr.RecordingStream("dataforge", recording_id=segment) as recording:
        recording.save(base)
        recording.send_property("capture", rr.AnyValues(schema="dataforge:v1"))
        recording.log("/world/rig_00/cam_04", rr.AnyValues(name="left"), static=True)
        recording.set_time("video_time", duration=1.0)
        recording.log("/world/rig_00/imu_00/gyro", rr.Scalars([1.0, 2.0, 3.0]))
    before: str = hashlib.sha256(base.read_bytes()).hexdigest()
    with rr.server.Server(datasets={"robocap": [base]}) as server:
        client: CatalogClient = server.client()
        dataset: DatasetEntry = client.get_dataset("robocap")
        for _ in range(2):
            main(Config(catalog_url=server.url(), root=tmp_path, output_dir=tmp_path / "calibration", segments=(segment,)))
        rows = dataset.segment_table().select("rerun_layer_names").to_arrow_table().to_pylist()
        assert [sorted(row["rerun_layer_names"]) for row in rows] == [["base", "sensor_metadata"]]
        table = dataset.filter_contents(["/world/rig_00/imu_00"]).reader(index=None).to_arrow_table()
        assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[-14_902_432]]
    assert hashlib.sha256(base.read_bytes()).hexdigest() == before
    # A layer must not restate the segment's recording properties (rerun-data-model gotcha 7).
    layer_entities: set[str] = {str(chunk.entity_path) for chunk in RrdReader(tmp_path / "calibration" / f"{segment}.rrd").stream()}
    assert "/__properties" not in layer_entities
    assert "/world/rig_00/imu_00" in layer_entities


def test_registration_restores_saved_sensor_metadata_after_a_catalog_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segment: str = "robocap__f408193e6447b3b0__s00000015"
    for layer in ("base", "sensor_metadata"):
        folder: Path = tmp_path / layer
        folder.mkdir()
        with rr.RecordingStream("dataforge", recording_id=segment) as recording:
            recording.save(folder / f"{segment}.rrd")
            metadata: rr.AnyValues = rr.AnyValues(kind="imu") if layer == "base" else rr.AnyValues(applied_time_shift_ns=-14_902_432)
            recording.log("/world/rig_00/imu_00", metadata, static=True)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    with rr.server.Server() as server:
        register(RegisterConfig(catalog_url=server.url()))
        client: CatalogClient = server.client()
        entry: DatasetEntry = client.get_dataset("robocap")
        table = entry.filter_contents(["/world/rig_00/imu_00"]).reader(index=None).to_arrow_table()
        assert table["/world/rig_00/imu_00:applied_time_shift_ns"].to_pylist() == [[-14_902_432]]
