"""The robocap-live display asset: the layout the Rust logger sends, written without a mesh (no assets needed)."""

from collections import Counter
from pathlib import Path

import pyarrow as pa
from rerun.chunk import RrdReader

from handtrack.apis.robocap_live_display import Config, main


def view_classes(asset: Path) -> Counter[str]:
    """Count the blueprint's views by class."""
    reader = RrdReader(asset)
    classes: Counter[str] = Counter()
    for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks():
        if not str(chunk.entity_path).startswith("/view/"):
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        for name in batch.schema.names:
            if name.endswith("class_identifier"):
                classes.update(str(value) for row in batch.column(name).to_pylist() for value in row)
    return classes


def test_the_asset_holds_two_3d_views_six_camera_panes_and_three_plots(tmp_path: Path) -> None:
    asset: Path = tmp_path / "display.rrd"
    main(Config(output=asset, root=tmp_path))
    assert view_classes(asset) == Counter({"3D": 2, "2D": 6, "TimeSeries": 3})
