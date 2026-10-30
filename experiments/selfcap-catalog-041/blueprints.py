"""SelfCap 0.41 saved-sort/filter layouts; shared-server use requires explicit opt-in."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import rerun as rr
import rerun.blueprint as rrb
import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from serde import from_dict, serde


@dataclass
class Config:
    """Publish layouts on a 0.41 catalog; preserve existing subset membership by default."""

    output_root: Path
    """Server-visible directory for new stamped blueprint files."""
    allow_shared: bool = False
    """Explicitly allow the shared port after its authorized upgrade."""
    populate_refined: bool = False
    """Create a filtered-view dataset sharing all files; otherwise preserve existing membership."""
    catalog_url: str = "rerun+http://127.0.0.1:51241"
    """Catalog URL; port 51235 requires allow_shared."""
    catalog_name: str = "selfcap-post-ep1-v2"
    """Existing SelfCap dataset populated by the snapshot restore tool."""
    refined_name: str = "selfcap-post-ep1-v2-refined"
    """Filtered catalog view sharing all source files, rather than copying data."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerRef:
    """One registered layer from the catalog manifest."""

    rerun_segment_id: str
    """Segment identity."""
    rerun_layer_name: str
    """Layer identity."""
    rerun_storage_url: str
    """Existing file visible to the server."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BlueprintRef:
    """One registered blueprint file."""

    rerun_segment_id: str
    """Blueprint recording identity."""
    rerun_storage_url: str
    """Existing blueprint URI."""


def table_blueprint(columns: list[str], *, filtered: bool) -> rrb.table.TableBlueprint:
    """Rank ascending, midpoint previews, and an optional refined-only SQL filter."""
    body: list[str] = ["/world/skeleton/**", "/world/pred/soma_x/mesh/**"]
    scene: rrb.Spatial3DView = rrb.Spatial3DView(name="Skeleton + SOMA", origin="/world", contents=body)
    pane: rrb.Spatial2DView = rrb.Spatial2DView(
        name="Exo rig_00", origin="/world/rig_00/cam_00/pinhole", contents=["/world/rig_00/cam_00/pinhole/video/**", *body],
    )
    preview: rrb.table.Column = rrb.table.Column("recording link", column_type=rrb.table.PreviewCell(scene, pane))
    fields: list[tuple[str, str]] = [
        ("property:summary:quality_rank", "Quality rank"),
        ("property:calib:session", "Capture / session"),
        ("property:calib:exo_cameras", "Phones"),
        ("property:calib:calib_exo_refined", "Refined"),
        ("property:soma_quality:status", "SOMA quality"),
        ("property:calib:oak_posed", "OAK posed"),
        ("property:calib:oak_rotation_source", "OAK rotation"),
        ("property:capture:num_frames", "Frames"),
        ("property:capture:session", "Session UUID"),
    ]
    shown: list[rrb.table.Column] = [rrb.table.Column(column, name=name) for column, name in fields]
    hidden: list[rrb.table.Column] = [
        rrb.table.Column(column, visible=False) for column in columns if column not in dict(fields) and column != "recording link"
    ]
    return rrb.table.TableBlueprint(
        table_layout=rrb.table.TableLayout(columns=[preview, *shown, *hidden]),
        card_layout=rrb.table.CardLayout(
            title="property:RecordingInfo:name", link="recording link",
            fields=[preview, *shown[:5], *[rrb.table.Column(column, visible=False) for column, _ in fields[5:]], *hidden],
        ),
        default_layout="cards",
        previews_config=rrb.table.PreviewsConfig(timeline="video_time", start_point=0.5, play_state="Playing"),
        sort_by=rrb.table.SortBy("property:summary:quality_rank", direction="Ascending"),
        filters=[rrb.table.Filter('"property:calib:calib_exo_refined"[1] = true')] if filtered else [],
    )


def main(config: Config) -> None:
    """Share layer references with a filtered view and register fresh table blueprints."""
    if rr.__version__ != "0.41.0" or (urlparse(config.catalog_url).port == 51235 and not config.allow_shared):
        raise ValueError("Use Rerun 0.41.0; shared port 51235 requires --allow-shared")
    client: CatalogClient = CatalogClient(config.catalog_url)
    source: DatasetEntry = client.get_dataset(config.catalog_name)
    stamp: str = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    config.output_root.mkdir(parents=True, exist_ok=True)
    for filtered in (False, True):
        target: DatasetEntry = (client.create_dataset(config.refined_name, exist_ok=True) if config.populate_refined else client.get_dataset(config.refined_name)) if filtered else source
        if filtered and config.populate_refined:
            existing: set[LayerRef] = {
                from_dict(LayerRef, row) for row in target.manifest().select("rerun_segment_id", "rerun_layer_name", "rerun_storage_url").to_arrow_table().to_pylist()
            }
            layers: list[LayerRef] = [
                from_dict(LayerRef, row) for row in source.manifest().select("rerun_segment_id", "rerun_layer_name", "rerun_storage_url").to_arrow_table().to_pylist()
            ]
            changed: defaultdict[str, list[str]] = defaultdict(list)
            for layer in layers:
                if layer not in existing:
                    changed[layer.rerun_layer_name].append(layer.rerun_storage_url)
            for layer_name, uris in changed.items():
                target.register(uris, layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
            blueprint_dataset: DatasetEntry | None = source.blueprint_dataset()
            if blueprint_dataset is not None:
                defaults: list[BlueprintRef] = [
                    from_dict(BlueprintRef, row) for row in blueprint_dataset.manifest().select("rerun_segment_id", "rerun_storage_url").to_arrow_table().to_pylist()
                ]
                for blueprint in defaults:
                    if blueprint.rerun_segment_id == source.default_blueprint():
                        if target.default_blueprint() == blueprint.rerun_segment_id:
                            break
                        target.register_blueprint(blueprint.rerun_storage_url, set_default=True)
        path: Path = (config.output_root / f"{target.name}-{stamp}.rbl").resolve()
        table_blueprint(source.segment_table().schema().names, filtered=filtered).save("dataforge", path)
        target.register_blueprint(path.as_uri(), segment_table=True, set_default=True)
        print(f"Published {target.name}: {target.id} ({path})", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
