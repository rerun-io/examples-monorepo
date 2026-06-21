"""Register the iPhone quality RRDs into a local Rerun catalog dataset.

Mirrors the trossen-oss pattern (rerun-io/trossen-oss): the catalog is the local
in-memory ``rerun server`` (start it separately:
``rerun server --port 51234``). Each scene becomes one segment (id == the RRD's
recording_id) with two layers:

  * ``base``  -- the ``quality.rrd`` (data + the embedded multi-view blueprint).
  * ``props`` -- a tiny RRD (same recording_id) that sets the recording NAME and
                 sortable PROPERTIES (num_people, PVE p95/p99, per-camera realtime,
                 pass flags) read from each scene's ``gate.json``.

The shared multi-view layout is also registered once as the dataset DEFAULT
blueprint, so every recording opens with the 3D-world + 4-camera-grid + metrics
layout instead of the heuristic auto-layout.

Properties surface as ``property:*`` columns in the catalog's segment table, so
the viewer can sort scenes by subject count or error. The .rbl and props layers
are app-scoped: they MUST use the recordings' application_id ('dump_artifacts',
the dump_artifacts.py default), or the viewer ignores them.

Nothing here rewrites the (large) quality.rrd files; the props layers are a few
KB each. Run from packages/mamma:  python tools/register_catalog.py
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun.catalog as cat
import tyro

from mamma.viz.blueprint import default_blueprint

# Scene categories (mirrors sweep_iphones_quality.py's CATEGORIES — keep in sync).
CATEGORIES: tuple[str, ...] = ("indoors", "outdoors")


@dataclass
class RegisterCatalogConfig:
    catalog_url: str = "rerun+http://127.0.0.1:51234"
    """gRPC URL of a locally-running ``rerun server`` (start it separately)."""
    dataset_name: str = "mamma_iphones"
    """Catalog dataset to (re)create."""
    rrd_root: Path = Path("/mnt/8tb/data/mamma_markerless_iphones_rrds")
    """Root holding <cat>/<scene>/quality.rrd + gate.json."""
    data_root: Path = Path("/mnt/8tb/data/mamma_markerless_iphones")
    """Dataset root (used only to count pred subjects when gate.json lacks n_people)."""
    application_id: str = "dump_artifacts"
    """MUST match the recordings' application_id (dump_artifacts.py default)."""
    camera_names: tuple[str, ...] = ("A001", "B001", "C001", "D001")
    """Shared 4-camera rig; the default blueprint is built from these."""
    recreate: bool = True
    """Delete an existing dataset of the same name before registering."""


def main(config: RegisterCatalogConfig) -> int:
    props_dir: Path = config.rrd_root / "_catalog_props"
    props_dir.mkdir(parents=True, exist_ok=True)
    rbl_path: Path = config.rrd_root / f"{config.dataset_name}.rbl"
    default_blueprint(list(config.camera_names), timing_doc=True).save(config.application_id, str(rbl_path))

    # Discover + validate inputs BEFORE any destructive catalog op: a mispointed
    # or unmounted rrd_root must NOT delete the existing dataset (that IS the
    # catalog's data) and then leave an empty/failed registration behind.
    scenes: list[str] = sorted(p for c in CATEGORIES for p in glob.glob(f"{config.rrd_root}/{c}/*/quality.rrd"))
    if not scenes:
        print(f"no quality.rrd under {config.rrd_root} — refusing to (re)create {config.dataset_name!r}")
        return 1

    client = cat.CatalogClient(config.catalog_url)
    if config.recreate and config.dataset_name in client.dataset_names():
        client.get_dataset(config.dataset_name).delete()
    ds = client.create_dataset(config.dataset_name, exist_ok=True)
    replace = cat.OnDuplicateSegmentLayer.REPLACE

    for p in scenes:
        cat_, scene = Path(p).relative_to(config.rrd_root).parts[:2]
        name: str = f"{cat_}/{scene}"
        gate_path: Path = Path(p).with_name("gate.json")
        gate: dict = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        if gate.get("error"):  # crashed/incomplete scene (sweep wrote a synthetic error gate) — don't register garbage
            print(f"  SKIP {name}: {gate['error']}")
            continue
        n_people: int = int(gate.get("n_people") or len(list((config.data_root / cat_ / scene / "pred").glob("params_*.npz"))) or 1)

        rid: str = ds.register([f"file://{p}"], layer_name="base", on_duplicate=replace).wait().segment_ids[0]

        rr.init(config.application_id, recording_id=rid)
        rr.send_recording_name(name)
        rr.send_property("category", rr.AnyValues(category=cat_))
        rr.send_property("num_people", rr.AnyValues(num_people=n_people))
        if gate.get("pve_p95_mm") is not None:
            rr.send_property("pve_p95_mm", rr.AnyValues(pve_p95_mm=float(gate["pve_p95_mm"])))
        if gate.get("pve_p99_mm") is not None:
            rr.send_property("pve_p99_mm", rr.AnyValues(pve_p99_mm=float(gate["pve_p99_mm"])))
        if gate.get("per_cam_realtime") is not None:
            rr.send_property("per_cam_realtime", rr.AnyValues(per_cam_realtime=float(gate["per_cam_realtime"])))
        rr.send_property("passes", rr.AnyValues(passes=bool(gate.get("pass", False))))
        rr.send_property("accuracy_pass", rr.AnyValues(accuracy_pass=bool(gate.get("accuracy_pass", False))))
        props_path: Path = props_dir / f"{cat_}__{scene}_props.rrd"
        rr.save(str(props_path))
        ds.register([f"file://{props_path}"], layer_name="props", on_duplicate=replace).wait()
        print(f"  {name:42s} ppl={n_people} p99={gate.get('pve_p99_mm')} pass={gate.get('pass')}")

    ds.register_blueprint(rbl_path.resolve().as_uri(), set_default=True)
    print(f"\nDONE: {len(ds.segment_ids())} segments on {config.catalog_url}, default_blueprint={ds.default_blueprint()}")
    print(f"Open a Rerun 0.33 viewer on {config.catalog_url} (ensure no stale viewer holds :9876).")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(RegisterCatalogConfig)))
