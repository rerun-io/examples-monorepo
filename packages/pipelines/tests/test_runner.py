"""The runner over fake stages on a real in-process catalog: every step publishes, a failure stops the flow, a killed
step leaves the last good layer, and the layer files restore a restarted catalog."""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")

# A catalog server and stage subprocesses: the integration lane (AGENTS.md, Testing).
pytestmark = pytest.mark.integration

SEGMENT: str = "insta360__clip"


class Catalog(NamedTuple):
    url: str
    """The private catalog's URL."""
    base: Path
    """The base layer file of SEGMENT."""


def _write_base(path: Path) -> Path:
    from simplecv.catalog_layers import atomic_recording

    with atomic_recording(path, application_id="fake-base", recording_id=SEGMENT) as recording:
        recording.log("/world/rig_00", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    return path


def _serve(base: Path, layers: dict[str, Path]) -> Iterator[str]:
    from simplecv.catalog_layers import ensure_dataset, register_layer

    with rr.server.Server() as server:
        entry = ensure_dataset(server.client(), "insta360")
        register_layer(entry, [base], None)
        for layer, path in layers.items():
            register_layer(entry, [path], layer)
        yield server.url()


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[Catalog]:
    base: Path = _write_base(tmp_path / "base" / f"{SEGMENT}.rrd")
    for url in _serve(base, {}):
        yield Catalog(url, base)


def _launch(tmp_path: Path):
    from pipelines.runner import Launch

    pixi: Path = tmp_path / "pixi"
    pixi.write_text(f"#!/bin/sh\nexec {sys.executable} {Path(__file__).with_name('fake_pixi.py')} \"$@\"\n")
    pixi.chmod(0o755)
    return Launch(manifest=tmp_path / "pixi.toml", pixi=str(pixi), gpu_lock=tmp_path / "gpu.lock", memory_max=None)


def _run(tmp_path: Path, url: str) -> int:
    from rerun.catalog import CatalogClient

    from pipelines.capture_to_splat import flow_steps
    from pipelines.runner import run_flow

    steps = flow_steps("insta360", segment=SEGMENT, catalog_url=url, output_root=tmp_path / "layers")
    return run_flow(steps, _launch(tmp_path), entry=CatalogClient(url).get_dataset(name="insta360"), segment=SEGMENT, logs=tmp_path / "logs")


def _records(url: str) -> dict[str, object]:
    from rerun.catalog import CatalogClient
    from simplecv.run_record import read_run_record

    entry = CatalogClient(url).get_dataset(name="insta360")
    found: dict[str, object] = {}
    for layer in ("lens_mask", "slam_rs"):
        try:
            found[layer] = read_run_record(entry, SEGMENT, layer)
        except ValueError:
            found[layer] = None
    return found


def test_a_run_publishes_every_step_with_the_arguments_the_flow_gave_it(tmp_path: Path, catalog: Catalog) -> None:
    assert _run(tmp_path, catalog.url) == 0
    records = _records(catalog.url)
    assert all(record is not None for record in records.values())
    command: str = records["slam_rs"].command  # pyrefly: ignore[missing-attribute]
    assert "-e slam-rs-cuda slam-rs-catalog-layer --segment insta360__clip" in command and command.endswith("--mask-entity people_mask")
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == ["00-lens_masks.log", "01-slam.log"]


def test_a_failed_step_stops_the_flow_with_its_exit_code(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STAGE_FAIL", "posekit-lens-masks")
    assert _run(tmp_path, catalog.url) == 3
    assert _records(catalog.url) == {"lens_mask": None, "slam_rs": None}, "SLAM never ran on a missing mask layer"


def test_a_step_that_exits_0_without_publishing_stops_the_flow(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STAGE_NOOP", "posekit-lens-masks")
    assert _run(tmp_path, catalog.url) == 1
    assert _records(catalog.url)["slam_rs"] is None


def test_a_killed_step_leaves_the_last_good_layer_and_stops_the_flow(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage dies with its layer's temp file open: the published file, its registration and the later layer stay."""
    assert _run(tmp_path, catalog.url) == 0
    before = _records(catalog.url)
    lens_file: Path = tmp_path / "layers" / "lens_mask" / f"{SEGMENT}.rrd"
    lens_bytes: bytes = lens_file.read_bytes()
    monkeypatch.setenv("FAKE_STAGE_KILL", "posekit-lens-masks")
    assert _run(tmp_path, catalog.url) != 0
    assert _records(catalog.url) == before and lens_file.read_bytes() == lens_bytes


def test_the_layer_files_restore_a_restarted_catalog(tmp_path: Path, catalog: Catalog) -> None:
    """The catalog is in memory; the layer files are the durable copy and carry their run records."""
    assert _run(tmp_path, catalog.url) == 0
    before = _records(catalog.url)
    layers: dict[str, Path] = {layer: tmp_path / "layers" / layer / f"{SEGMENT}.rrd" for layer in ("lens_mask", "slam_rs")}
    for url in _serve(catalog.base, layers):
        assert _records(url) == before
