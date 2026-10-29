"""The migration preserves the served calibration's exact hands clock."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from dataforge import schema
from frozen_v2 import RAW as RAW_ROOT
from rerun.chunk import RrdReader
from rrd_fixtures import entity_times

from selfcap_curate.patch_base import patch
from selfcap_curate.records import read_sync


@pytest.mark.golden
@pytest.mark.parametrize("key", ["2025-12-30__65d00748__ep001", "2025-11-24__027d8215__ep002"])
def test_patched_hands_clock_equals_served_calib(key: str, tmp_path: Path) -> None:
    root = Path("/mnt/nas/datasets/dataforge-selfcap-rrd")
    base = root / "base" / f"selfcap__{key}.rrd"
    calib = root / "calib-v2" / base.name
    sync = read_sync(Path(__file__).parent / "fixtures" / f"{key}.sync.json")
    raw = RAW_ROOT / sync.date / sync.session / "episodes" / f"episode-{sync.episode.removeprefix('ep')}"
    for asset in (base, calib, raw):
        if not asset.exists():
            pytest.skip(f"SelfCap golden asset absent: {asset}")
    patched = patch(base, raw, sync, output_path=tmp_path / "patched.rrd", mode="migrate-keep-clock")
    actual = entity_times(patched)[schema.coco133_xyz_path()]
    expected = np.sort(
        np.concatenate(
            [
                np.asarray(chunk.to_record_batch().column("video_time").cast(pa.int64()))
                for chunk in RrdReader(calib).stream().filter(content=schema.coco133_xyz_path())
                if not chunk.is_static
            ]
        )
    )
    np.testing.assert_array_equal(actual.video_ns, expected)
