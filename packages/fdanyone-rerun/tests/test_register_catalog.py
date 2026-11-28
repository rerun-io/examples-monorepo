"""``fdanyone-register-catalog`` writes the views' segment with the run record of the run that wrote it, so a flow runner
can tell this run's ``base`` layer from an older one."""

import os
from pathlib import Path

import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")

# A downloaded run recording and a catalog server: the integration lane (AGENTS.md, Testing).
pytestmark = pytest.mark.integration

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
RUN_RRD: Path = Path(os.environ.get("FDANYONE_GOLDEN_RRD", PACKAGE_ROOT / "data" / "golden" / "example-fp8.rrd"))
"""A finished run's recording (the banked example run; set FDANYONE_GOLDEN_RRD to its copy)."""


def test_the_registered_views_segment_carries_its_run_record(tmp_path: Path) -> None:
    if not RUN_RRD.exists():
        pytest.skip(f"the run recording {RUN_RRD} is absent (set FDANYONE_GOLDEN_RRD)")
    from simplecv.run_record import RunRecord, read_run_record

    from fdanyone_rerun.apis.register_catalog import Config, main

    with rr.server.Server() as server:
        path: Path = main(Config(rrd=RUN_RRD, output_dir=tmp_path, seed=42, catalog_url=server.url()))
        record: RunRecord = read_run_record(server.client().get_dataset(name="fdanyone-views"), path.stem, "base")
    assert (record.layer, record.segment) == ("base", path.stem) and record.wall_s > 0.0
