"""Run a local post cycle and publish its generated layers to a dev dataset."""

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dataforge.datasets.selfcap import SelfcapConfig, SelfcapDataset
from datafusion import col
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from serde import from_dict, serde
from serde.json import from_json

from selfcap_curate.apis import post
from selfcap_curate.exo_landmarks import Landmarks, write_landmarks
from selfcap_curate.exo_refine import FocalSource
from selfcap_curate.records import EpisodeKey


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SourceSegment:
    rerun_segment_id: str
    """Selected source segment."""
    rerun_layer_names: list[str]
    """Registered layers in storage URL order."""
    rerun_storage_urls: list[str]
    """Immutable source files, interpreted by the catalog host."""


@dataclass
class Config:
    segment: str
    """Explicit segment id."""
    tag: str
    """Unique run directory name, supplied by the operator."""
    dataset: str = "selfcap-post-dev-exofit"
    """Destination dev dataset; production names are refused."""
    reference_dataset: str = "selfcap-post-dev"
    """Source of existing layers; independent EgoHand and SOMA layers are never refreshed."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog that owns the source and candidate datasets."""
    catalog_host: str = "spark"
    """SSH destination used for storing files and registering the blueprint."""
    remote_root: Path = Path("/home/pablo/dataforge-dev")
    """Parent of candidate datasets on the catalog host."""
    remote_base: Path = Path("/home/pablo/dataforge-dev/rrd/base")
    """Read-only as-cut base directory on the catalog host."""
    work_root: Path = Path("/tmp/selfcap-dev")
    """Local run directory parent."""
    blueprint_python: Path = Path("/home/pablo/dataforge-dev/code2/.pixi/envs/dataforge/bin/python")
    """Catalog-host interpreter used to register its local blueprint file."""
    landmarks: Path | None = None
    """Typed Landmarks JSON to ingest into the candidate before fitting."""
    exo_refine: bool = False
    """Enable optical refinement before keypoints."""
    focal_source: FocalSource = "fit"
    """Focal source for the joint fit."""
    focal_file: Path | None = None
    """Typed fixed-focal record when focal_source is fixed."""
    oak_egohand_assets: Path = Path("data/egohand")
    """EgoHand weights and TensorRT engine cache."""


def ship(host: str, local: Path, remote: Path) -> None:
    """Copy one file to the catalog host, creating its remote directory first."""
    subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(str(remote.parent))}"], check=True)
    subprocess.run(["scp", "-q", str(local), f"{host}:{remote}"], check=True)


def main(config: Config) -> None:
    """Publish generated layers only; preserve independent layers in the main dev dataset."""
    EpisodeKey.parse(config.segment)
    for name in (config.dataset, config.tag):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("dataset and tag must be simple path components")
    refresh_main: bool = config.dataset == "selfcap-post-dev"
    if config.dataset in {"selfcap-post-ep1", "dataforge-selfcap"} or (config.dataset == config.reference_dataset and not refresh_main):
        raise ValueError("dev-cycle destination must be main dev or a separate candidate dataset")
    work = config.work_root / config.tag
    work.mkdir(parents=True, exist_ok=False)
    remote = config.remote_root / config.dataset
    run = remote / "runs" / config.tag
    client = CatalogClient(config.catalog_url)
    source = client.get_dataset(name=config.reference_dataset)
    rows = source.segment_table().filter(col("rerun_segment_id") == config.segment).select(
        "rerun_segment_id", "rerun_layer_names", "rerun_storage_urls").to_arrow_table().to_pylist()
    if len(rows) != 1:
        raise ValueError("reference dataset must contain the selected segment")
    reference = from_dict(SourceSegment, rows[0])
    source_layers = dict(zip(reference.rerun_layer_names, reference.rerun_storage_urls, strict=True))
    if not {"base", "sapiens2", "calib"} <= source_layers.keys():
        raise ValueError("reference dataset needs base, sapiens2 and calib layers")
    subprocess.run(["ssh", config.catalog_host,
                    f"mkdir -p {shlex.quote(str(run.parent))} && mkdir {shlex.quote(str(run))}"], check=True)
    candidate = client.create_dataset(config.dataset, exist_ok=True)
    # Seed new candidates, but never re-register independent layers in main dev.
    if not refresh_main:
        for layer in ("base", "sapiens2", "calib", "egohand"):
            if layer in source_layers:
                candidate.register([source_layers[layer]], layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    if config.landmarks is not None:
        label_path = work / "exo_landmarks" / f"{config.segment}.rrd"
        write_landmarks(label_path, config.segment, from_json(Landmarks, config.landmarks.read_text()))
        ship(config.catalog_host, label_path, run / "exo_landmarks" / f"{config.segment}.rrd")
        candidate.register([f"file://{run}/exo_landmarks/{config.segment}.rrd"], layer_name="exo_landmarks",
                           on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    elif "exo_landmarks" in source_layers and not refresh_main:
        candidate.register([source_layers["exo_landmarks"]], layer_name="exo_landmarks", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    base = work / "base"
    base.mkdir()
    subprocess.run(["scp", "-q", f"{config.catalog_host}:{config.remote_base}/{config.segment}.rrd", str(base)], check=True)
    post.main(post.Config(base_dir=base, output_dir=work / "out", segment_ids=(config.segment,), catalog_url=config.catalog_url,
                          oak_egohand_assets=config.oak_egohand_assets, exo_refine=config.exo_refine,
                          exo_focal_source=config.focal_source, exo_focal_file=config.focal_file, exo_reference_dataset=config.dataset))
    layers = {"base": work / "out/align" / f"{config.segment}.rrd",
              "sapiens2": work / "out/align/sapiens2" / f"{config.segment}.rrd",
              "calib": work / "out/keypoints" / f"{config.segment}.rrd",
              "egohand_calib": work / "out/calib/egohand" / f"{config.segment}.rrd",
              "exo_landmarks": work / "out/align/exo_landmarks" / f"{config.segment}.rrd"}
    if refresh_main:
        del layers["sapiens2"]  # Keep the existing alignment detections; keypoints_sapiens2 is a separate run.
    for layer, path in layers.items():
        if not path.exists():
            continue
        remote_path = run / layer / path.name
        ship(config.catalog_host, path, remote_path)
        candidate.register([remote_path.as_uri()], layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    layout = work / f"{config.dataset}.rbl"
    SelfcapDataset(SelfcapConfig()).default_blueprint(show_soma_mesh=True).save(config.dataset, str(layout))
    subprocess.run(["ssh", config.catalog_host, f"mkdir -p {shlex.quote(str(run / 'reports'))}"], check=True)
    subprocess.run(["scp", "-q", str(layout), f"{config.catalog_host}:{run}/default.rbl"], check=True)
    for path in (work / "out/calib").glob("*.json"):
        subprocess.run(["scp", "-q", str(path), f"{config.catalog_host}:{run}/reports/{path.name}"], check=True)
    command = (f"from rerun.catalog import CatalogClient; CatalogClient({config.catalog_url!r}).get_dataset(name={config.dataset!r})"
               f".register_blueprint({(run / 'default.rbl').as_uri()!r}, set_default=True)")
    subprocess.run(["ssh", config.catalog_host, f"{shlex.quote(str(config.blueprint_python))} -c {shlex.quote(command)}"], check=True)
    print(f"VIEW {config.catalog_url}/dataset/{candidate.id}?segment_id={config.segment}", flush=True)
    print(f"FILES {run}", flush=True)
