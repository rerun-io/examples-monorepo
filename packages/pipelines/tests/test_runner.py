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
LAYERS: tuple[str, ...] = ("lens_mask", "slam_rs", "masks", "colsfm")
"""What the insta360 flow publishes, in order."""


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
    return run_flow(steps, _launch(tmp_path), client=CatalogClient(url), logs=tmp_path / "logs")


def _records(url: str) -> dict[str, object]:
    from rerun.catalog import CatalogClient
    from simplecv.run_record import read_run_record

    entry = CatalogClient(url).get_dataset(name="insta360")
    found: dict[str, object] = {}
    for layer in LAYERS:
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
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == ["00-lens_masks.log", "01-slam.log", "02-face_masks.log", "03-colsfm.log"]


def test_a_failed_step_stops_the_flow_with_its_exit_code(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STAGE_FAIL", "posekit-lens-masks")
    assert _run(tmp_path, catalog.url) == 3
    assert set(_records(catalog.url).values()) == {None}, "no later step ran on a missing mask layer"


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
    layers: dict[str, Path] = {layer: tmp_path / "layers" / layer / f"{SEGMENT}.rrd" for layer in LAYERS}
    for url in _serve(catalog.base, layers):
        assert _records(url) == before


def test_each_step_is_checked_on_the_dataset_and_segment_it_publishes_to(tmp_path: Path, catalog: Catalog) -> None:
    """A flow can hand off across datasets (person_to_4d: the views' dataset, then the training runs'); the runner reads
    each step's run record where that step registered its layer."""
    from rerun.catalog import CatalogClient
    from simplecv.catalog_layers import ensure_dataset
    from simplecv.run_record import read_run_record

    from pipelines.runner import run_flow
    from pipelines.step import Step, Task

    client = CatalogClient(catalog.url)
    ensure_dataset(client, "runs")
    out: Path = tmp_path / "layers"
    steps = (
        Step("masks", "lens_mask", "insta360", SEGMENT, (Task("posekit", "posekit-lens-masks",
             ("--segment", SEGMENT, "--catalog-url", catalog.url, "--output-dir", str(out / "lens_mask"))),)),
        Step("train", "slam_rs", "runs", "runs__r1", (Task("slam-rs-cuda", "slam-rs-catalog-layer",
             ("--segment", "runs__r1", "--catalog", catalog.url, "--output-dir", str(out / "slam_rs"))),)),
    )
    assert run_flow(steps, _launch(tmp_path), client=client, logs=tmp_path / "logs") == 0
    assert read_run_record(client.get_dataset(name="runs"), "runs__r1", "slam_rs").command.endswith("--segment runs__r1 --catalog " + catalog.url + " --output-dir " + str(out / "slam_rs"))


def test_a_step_runs_its_tasks_in_order_and_only_the_last_publishes(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """A folded step (person_to_4d: generate, then register) runs each task as its own process, one log each; a failing
    first task stops the step before the publishing one runs."""
    from rerun.catalog import CatalogClient

    from pipelines.runner import run_flow
    from pipelines.step import Step, Task

    out: Path = tmp_path / "layers"
    scratch = Task("posekit", "fake-scratch", ("--write", str(tmp_path / "scratch.txt")))
    publish = Task("posekit", "posekit-lens-masks", ("--segment", SEGMENT, "--catalog-url", catalog.url, "--output-dir", str(out / "lens_mask")))
    steps = (Step("views", "lens_mask", "insta360", SEGMENT, (scratch, publish)),)
    assert run_flow(steps, _launch(tmp_path), client=CatalogClient(catalog.url), logs=tmp_path / "logs") == 0
    assert (tmp_path / "scratch.txt").exists() and _records(catalog.url)["lens_mask"] is not None
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == ["00-views-0.log", "00-views-1.log"]
    monkeypatch.setenv("FAKE_STAGE_FAIL", "fake-scratch")
    assert run_flow(steps, _launch(tmp_path), client=CatalogClient(catalog.url), logs=tmp_path / "logs2") == 3
    assert sorted(path.name for path in (tmp_path / "logs2").iterdir()) == ["00-views-0.log"]
