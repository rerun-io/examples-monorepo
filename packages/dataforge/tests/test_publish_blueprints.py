"""SelfCap catalog layouts follow the layers an in-process catalog holds; the publish main registers both defaults and the device assets."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlparse

import pytest
import tyro

pytest.importorskip("rerun.catalog", reason="catalog dependencies are optional outside the dataforge catalog environment")

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402
from rerun.catalog import CatalogClient, DatasetEntry  # noqa: E402
from rerun.chunk import RrdReader  # noqa: E402
from rerun.server import Server  # noqa: E402

from dataforge import schema  # noqa: E402
from dataforge.apis import publish_blueprints  # noqa: E402
from dataforge.datasets.selfcap import SelfcapConfig, SelfcapDataset  # noqa: E402
from dataforge.logging_toolkit import log_rig_node  # noqa: E402
from dataforge.writing import TableField, TableFields, blueprint_views, recording_to  # noqa: E402

FOUR: str = "selfcap__2025-12-20__25aeb66a__ep001"
THREE: str = "selfcap__2025-12-21__1dbe8c49__ep002"
LAYER_PROPERTIES: dict[str, dict[str, str | int]] = {
    "capture": {"collector": "a person", "num_frames": 3},
    "calib": {"session": "2025-12-20__25aeb66a", "exo_cameras": 4},
    "soma_quality": {"status": "ok"},
}
"""The property columns a real catalog holds at publish time, one group per layer of the test catalog."""


@pytest.fixture
def server() -> Iterator[Server]:
    with Server(port=0) as running:
        yield running


@pytest.fixture
def layer(tmp_path: Path) -> Callable[[str, str], str]:
    """Write one layer file of one segment with a property group named after it; returns its URI."""

    def write(name: str, segment: str) -> str:
        target: Path = tmp_path / name / f"{segment}.rrd"
        target.parent.mkdir(parents=True, exist_ok=True)
        with recording_to(target, recording_id=segment, send_properties=False) as recording:
            properties = rr.AnyValues()
            for key, value in LAYER_PROPERTIES.get(name, {"present": True}).items():
                properties.with_component_from_data(key, value)
            recording.send_property(name, properties)
        return target.as_uri()

    return write


def soma_mesh_shown(entry: DatasetEntry) -> bool:
    segment, _ = SelfcapDataset(SelfcapConfig()).catalog_blueprints(entry)
    scene: rrb.View = next(view for view in blueprint_views(segment) if str(view.name) == "Scene")
    return "/world/pred/soma_x/mesh" in scene.visualizer_overrides


def test_the_soma_mesh_shows_however_many_segments_have_a_soma_layer(server: Server, layer: Callable[[str, str], str]) -> None:
    entry: DatasetEntry = server.client().create_dataset("selfcap-test")
    entry.register([layer("base", FOUR), layer("base", THREE)], layer_name="base").wait()
    assert soma_mesh_shown(entry), "no segment has a soma layer: the override has nothing to act on"
    entry.register([layer("soma", FOUR)], layer_name="soma").wait()
    assert soma_mesh_shown(entry), "one failed segment must not hide the mesh of the other"
    entry.register([layer("soma", THREE)], layer_name="soma").wait()
    assert soma_mesh_shown(entry)


def test_the_table_is_the_preview_card_when_any_segment_has_a_preview_layer(server: Server, layer: Callable[[str, str], str]) -> None:
    entry: DatasetEntry = server.client().create_dataset("selfcap-test")
    entry.register([layer("base", FOUR), layer("base", THREE)], layer_name="base").wait()
    assert isinstance(SelfcapDataset(SelfcapConfig()).catalog_blueprints(entry)[1], rrb.Blueprint), "no preview to show"
    entry.register([layer("preview", FOUR)], layer_name="preview").wait()
    assert isinstance(SelfcapDataset(SelfcapConfig()).catalog_blueprints(entry)[1], rrb.table.TableBlueprint), "one failed segment must not turn off the cards"
    entry.register([layer("preview", THREE)], layer_name="preview").wait()

    table = SelfcapDataset(SelfcapConfig()).catalog_blueprints(entry)[1]
    assert isinstance(table, rrb.table.TableBlueprint)
    assert table.default_layout == "cards"
    assert table.card_layout is not None and table.card_layout.title == "property:RecordingInfo:name"
    assert table.card_layout.link == "recording link"
    assert table.table_layout is not None
    for columns in (table.card_layout.fields, table.table_layout.columns):
        cell = columns[0].column_type
        assert columns[0].source == "recording link" and isinstance(cell, rrb.table.PreviewCell)
        body, videos = cell.views
        assert (str(body.name), str(body.origin)) == ("Body", "/preview/center")
        assert (str(videos.name), str(videos.origin)) == ("Exo + ego", "/preview/exo_cam/pinhole")
        assert isinstance(body.contents, list) and isinstance(videos.contents, list)
        assert "+ /preview/body/**" in body.contents and "- /world/rig_05/cam_02/pinhole/**" in body.contents and "+ /world/rig_05/cam_02/pinhole" in body.contents
        assert set(videos.contents) == {"+ /preview/exo_cam/pinhole/**", "+ /preview/body/skeleton/**", "+ /preview/body/soma_xlo/**"}
        assert [(column.source, column.name) for column in columns[1:4]] == [
            ("property:calib:session", "Capture / session"),
            ("property:calib:exo_cameras", "Phones"),
            ("property:soma_quality:status", "SOMA quality"),
        ]
        assert all(column.visible is False for column in columns[4:])
        hidden: set[str] = {column.source for column in columns[4:]}
        assert "property:capture:collector" in hidden, "hidden although this catalog has no such column"
        assert {"rerun_segment_id", "property:preview:present"} <= hidden, "every column of the table at publish time"


def test_the_preview_table_is_built_from_the_table_fields() -> None:
    class Renamed(SelfcapDataset):
        def table_fields(self) -> TableFields:
            return TableFields(cards=(TableField("a", "A"),), table=(TableField("b", "B"),), hidden=("c",), sort_by="b")

    table = Renamed(SelfcapConfig()).preview_table_blueprint(["a", "b", "rerun_segment_id"])
    assert table.card_layout is not None and table.table_layout is not None
    assert [(column.source, column.name, column.visible) for column in table.card_layout.fields[1:]] == [
        ("a", "A", None), ("b", None, False), ("rerun_segment_id", None, False), ("c", None, False),
    ]
    assert [(column.source, column.name, column.visible) for column in table.table_layout.columns[1:]] == [
        ("b", "B", None), ("a", None, False), ("rerun_segment_id", None, False), ("c", None, False),
    ]
    assert table.sort_by == rrb.table.SortBy("b", direction="Ascending")
    assert Renamed(SelfcapConfig()).preview_table_blueprint(["a"]).sort_by is None, "the table has no such column yet"


def table_entities(rbl: Path) -> dict[tuple[str, str], object]:
    """The first value of every component of a table blueprint file, by entity and component name."""
    reader: RrdReader = RrdReader(rbl)
    return {
        (str(chunk.entity_path), name.rsplit(":", 1)[-1]): chunk.to_record_batch().column(name).to_pylist()[0]
        for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks()
        for name in chunk.to_record_batch().schema.names
        if ":" in name and not name.startswith(("log_", "blueprint"))
    }


def registered_blueprint(entry: DatasetEntry, name: str | None) -> Path:
    """The file behind one registered blueprint, as the blueprint dataset lists it."""
    blueprints: DatasetEntry | None = entry.blueprint_dataset()
    assert blueprints is not None and name is not None
    rows: list[dict] = blueprints.segment_table().select("rerun_segment_id", "rerun_storage_urls").to_arrow_table().to_pylist()
    return Path(urlparse(next(row["rerun_storage_urls"][0] for row in rows if row["rerun_segment_id"] == name)).path)


def test_the_main_registers_both_defaults_and_a_rerun_retires_only_its_own_older_blueprints(
    tmp_path: Path, server: Server, layer: Callable[[str, str], str]
) -> None:
    entry: DatasetEntry = server.client().create_dataset("selfcap-test")
    for name in ("base", "capture", "calib", "soma_quality", "preview"):
        entry.register([layer(name, FOUR), layer(name, THREE)], layer_name=name).wait()
    layers: list[Path] = sorted(tmp_path.glob("*/*.rrd"))
    output_root: Path = tmp_path / "output"
    config = publish_blueprints.Config(dataset=SelfcapConfig(), catalog_url=server.url(), catalog_name="selfcap-test", output_root=output_root)

    publish_blueprints.main(config)
    read_back: DatasetEntry = CatalogClient(server.url()).get_dataset("selfcap-test")
    first = (registered_blueprint(read_back, read_back.default_blueprint()), registered_blueprint(read_back, read_back.default_segment_table_blueprint()))
    assert all(path.is_file() and path.parent == (output_root / "blueprints").resolve() for path in first)
    logged = table_entities(first[1])
    assert logged[("/table", "timeline")] == ["preview_time"]
    assert logged[("/table", "sort_by")] == [{"column": "property:calib:session", "direction": 1}]
    field_order = logged[("/table/layouts/cards", "field_order")]
    assert isinstance(field_order, list)
    assert field_order[:4] == ["recording link", "property:calib:session", "property:calib:exo_cameras", "property:soma_quality:status"]
    assert logged[(f"/table/layouts/cards/fields/{rr.escape_entity_path_part('property:capture:collector')}", "visible")] == [False]
    assert logged[(f"/table/layouts/table/columns/{rr.escape_entity_path_part('property:capture:num_frames')}", "visible")] == [False]

    publish_blueprints.main(config)
    read_back = CatalogClient(server.url()).get_dataset("selfcap-test")
    second = (registered_blueprint(read_back, read_back.default_blueprint()), registered_blueprint(read_back, read_back.default_segment_table_blueprint()))
    assert not set(first) & set(second), "a run writes fresh files; a registered file is never overwritten"
    assert sorted((output_root / "blueprints").iterdir()) == sorted(second), "the older files are gone"
    assert sorted(tmp_path.glob("*/*.rrd")) == layers, "retiring deletes no layer"
    assert sorted(read_back.segment_ids()) == sorted((FOUR, THREE))


def test_the_main_publishes_the_current_table_before_the_preview_layer_exists(tmp_path: Path, server: Server, layer: Callable[[str, str], str]) -> None:
    entry: DatasetEntry = server.client().create_dataset("selfcap-test")
    for name in ("base", "capture"):
        entry.register([layer(name, FOUR)], layer_name=name).wait()
    publish_blueprints.main(publish_blueprints.Config(dataset=SelfcapConfig(), catalog_url=server.url(), catalog_name="selfcap-test", output_root=tmp_path / "output"))
    read_back: DatasetEntry = CatalogClient(server.url()).get_dataset("selfcap-test")
    logged = table_entities(registered_blueprint(read_back, read_back.default_segment_table_blueprint()))
    assert logged[("/table", "timeline")] == ["video_time"]
    assert ("/table", "sort_by") not in logged, "no calibration session to sort by yet"
    assert logged[(f"/table/layouts/cards/fields/{rr.escape_entity_path_part('property:capture:num_frames')}", "visible")] == [False]


def test_the_dataset_is_a_subcommand_like_dataforge_register() -> None:
    args: list[str] = ["--catalog-url", "rerun+http://127.0.0.1:1", "--catalog-name", "selfcap-test", "--output-root", "/tmp/output", "selfcap"]
    config: publish_blueprints.Config = tyro.cli(publish_blueprints.Config, args=args)
    assert isinstance(config.dataset, SelfcapConfig)
    assert (config.catalog_url, config.catalog_name, config.output_root) == ("rerun+http://127.0.0.1:1", "selfcap-test", Path("/tmp/output"))
    with pytest.raises(SystemExit):
        tyro.cli(publish_blueprints.Config, args=args[:-1], console_outputs=False)


def test_the_main_registers_the_device_assets_too(tmp_path: Path, server: Server) -> None:
    target: Path = tmp_path / "base" / f"{FOUR}.rrd"
    target.parent.mkdir(parents=True)
    with recording_to(target, recording_id=FOUR, send_properties=False) as recording:
        for rig, (name, kind) in enumerate([("a", "exo"), ("oak", "ego"), ("quest", "quest")]):
            log_rig_node(recording, rig, reference=schema.CAM0_REFERENCE, num_cameras=1, name=name, kind=kind)
    entry: DatasetEntry = server.client().create_dataset("selfcap-test")
    entry.register([target.as_uri()], layer_name="base").wait()
    publish_blueprints.main(publish_blueprints.Config(dataset=SelfcapConfig(), catalog_url=server.url(), catalog_name="selfcap-test", output_root=tmp_path / "output"))
    assert sorted(entry.assets_for_segment(FOUR)) == ["devices__iphone12__rig_00", "devices__oak_d_pro_w__rig_01", "devices__quest3s__rig_02"]
