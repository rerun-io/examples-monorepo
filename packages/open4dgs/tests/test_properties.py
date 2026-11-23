"""The property records round-trip through the cells a recording or a catalog row gives back, keep the key names and
encodings already on the catalog, and read older rows with defaults."""

from pathlib import Path

import pytest

pytest.importorskip("trigs", reason="RunProperties holds the trainer config (open4dgs envs)")

from open4dgs.prior import PriorConfig  # noqa: E402
from open4dgs.properties import (  # noqa: E402
    InitCloudProperties,
    MatteLayerProperties,
    RoiSignature,
    any_values,
    encode_roi_cameras,
    parse_roi_cameras,
    row_cells,
    rrd_cells,
)
from open4dgs.run_log import RunProperties  # noqa: E402
from open4dgs.trigs_adapter import TrainConfig  # noqa: E402


def cells_of(properties: dict[str, object]) -> dict[str, list[object]]:
    """What a reader returns for these values: one list per field."""
    return {k: list(v) if isinstance(v, list) else [v] for k, v in properties.items()}


def full_run() -> RunProperties:
    return RunProperties(
        dataset="fdanyone-views",
        segment_id="seg",
        catalog_url="rerun+http://host:1",
        frame_start=10,
        frame_end=40,
        num_frames=31,
        cameras=("a", "b", "c"),
        train_cameras=("a", "c"),
        holdout_cameras=("b",),
        max_side=1280,
        roi_pad=0.2,
        mask_dataset="open4dgs-mamma",
        selfcap_time_start_s=3.5,
        prior_mode="auto",
        prior_source="layer:x/y",
        prior_config=PriorConfig(lambda_depth=0.1),
        train=TrainConfig(steps=99, rs_readout_ms=(10.0, 20.0, 0.0), rs_sign=(1.0, -1.0, 1.0), time_offset_clamp_ms=4.0),
        train_time_offset_clamp_ms_resolved=4.0,
        rs_signs_per_camera=(0.5, -1.0, 0.0),
        holdout_fit_steps=300,
    )


def test_run_properties_round_trip_through_cells() -> None:
    run: RunProperties = full_run()
    props = run.to_properties()

    assert RunProperties.from_cells(cells_of(props)) == run
    assert props["roi"] == "people:0.2" and props["mask_source"] == "layer:open4dgs-mamma/people_matte"
    assert props["train_rs_readout_ms"] == "10.0,20.0,0.0" and props["prior"] is True and props["prior_lambda_depth"] == 0.1


def test_an_old_run_row_reads_with_defaults() -> None:
    """A run from before the ROI, prior-mode and compensation fields (and the ``masks`` layer name)."""
    old = {
        "dataset": ["fdanyone-views"],
        "segment_id": ["seg"],
        "catalog_url": ["rerun+http://host:1"],
        "frame_start": [0],
        "frame_end": [120],
        "num_frames": [121],
        "cameras": ["a", "b"],
        "train_cameras": ["a"],
        "holdout_cameras": ["b"],
        "max_side": [0],
        "mask_source": ["layer:fdanyone-views/masks"],
        "train_steps": [25000],
    }
    run: RunProperties = RunProperties.from_cells(old)

    assert run.max_side is None and run.roi_pad is None and run.mask_dataset == "fdanyone-views"
    assert run.prior_config is None and run.train == TrainConfig(steps=25000) and run.selfcap_time_start_s is None


def test_roi_signatures_survive_their_text_form() -> None:
    sig: RoiSignature = RoiSignature.of(0.15, None, 64, 48, (500.004, 499.996, 21.5, 54.5), [1.0, 0.0, 0.0, 0.0, 0.70711, -0.70711, 0.0, 0.70711, 0.70711])
    cameras = {"a": sig, "b": None}
    text: str = encode_roi_cameras(cameras)

    assert text.startswith("a@pad=0.15;max_side=None;64x48;K=500.00,500.00,21.50,54.50;R=1.0000,0.0000") and text.endswith("|b@none")
    assert parse_roi_cameras(text) == cameras
    assert parse_roi_cameras(None) == {}


def test_catalog_row_cells_unwrap_the_nesting() -> None:
    row = {"rerun_segment_id": "s", "property:init_roma:holdout": [["a", "b"]], "property:init_roma:max_side": [640], "property:other:x": [1]}
    assert row_cells(row, "init_roma") == {"holdout": ["a", "b"], "max_side": [640]}


def test_layer_records_write_what_the_reader_reads(tmp_path: Path) -> None:
    import rerun as rr
    from rerun.chunk import RrdReader

    cloud = InitCloudProperties("/world/init_cloud", 0, 29, 3, 30.0, 100, "d", "s", "body_surface", 1234)
    matte = MatteLayerProperties("birefnet", 0, 30, "rerun+http://h:1", "d", "s", "smplx box pad 0.1 + 8 px")
    recording = rr.RecordingStream("properties-test", recording_id="s")
    recording.save(tmp_path / "p.rrd")
    recording.send_property("init_cloud", any_values(cloud.to_properties()))
    recording.send_property("people_matte", any_values(matte.to_properties()))
    recording.flush()
    recording.disconnect()
    reader = RrdReader(tmp_path / "p.rrd")

    assert InitCloudProperties.from_cells(rrd_cells(reader, "init_cloud")) == cloud
    assert "time_start_s" not in rrd_cells(reader, "people_matte")  # unset optional fields are not written
