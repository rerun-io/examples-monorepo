"""LabelStore: in-memory labels, the hand_boxes layer file, and its read-back."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pyarrow as pa
import pytest
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader, StoreEntry

from posekit.labeler.paths import label_path
from posekit.labeler.schema import Box, FrameLabels, ViewLabel
from posekit.labeler.store import LabelRow, LabelStore, base_application_id, load_layer, load_layer_file, rows_from_table

SEGMENT: str = "robocap__test__s00000001"


def sample_store() -> LabelStore:
    store: LabelStore = LabelStore(SEGMENT, split="eval", application_id="dataforge")
    store.put(
        FrameLabels(
            SEGMENT,
            1_000_000_000,
            [
                ViewLabel(0, "left", "accepted", Box(10.0, 20.0, 110.0, 220.5), "egohand", "pablo"),
                ViewLabel(3, "right", "no_hand", None, "person", "astra"),
            ],
        )
    )
    store.put(FrameLabels(SEGMENT, 1_200_000_000, [ViewLabel(0, "left", "rejected", None, "person", "pablo")]))
    store.put(FrameLabels(SEGMENT, 1_400_000_000, [ViewLabel(5, "left", "corrected", Box(1.5, 2.5, 30.0, 40.0), "agent", "astra")]))
    return store


def test_put_get_and_overwrite() -> None:
    store: LabelStore = sample_store()
    assert len(store) == 4
    assert [(label.cam, label.side) for label in store.get(1_000_000_000)] == [(0, "left"), (3, "right")]
    assert store.get(999) == []
    replaced: ViewLabel = ViewLabel(0, "left", "corrected", Box(11.0, 21.0, 111.0, 221.0), "person", "pablo")
    assert store.put(FrameLabels(SEGMENT, 1_000_000_000, [replaced])) == 1
    assert store.get_view(1_000_000_000, 0, "left") == replaced
    assert store.get_view(1_000_000_000, 1, "left") is None
    assert len(store) == 4
    assert [row.t_ns for row in store.rows()] == [1_000_000_000, 1_000_000_000, 1_200_000_000, 1_400_000_000]


def test_put_rejects_other_segment() -> None:
    store: LabelStore = sample_store()
    with pytest.raises(ValueError, match="posted to store"):
        store.put(FrameLabels("other", 0, []))


def test_put_is_thread_safe() -> None:
    store: LabelStore = LabelStore(SEGMENT, split="train", application_id="dataforge")

    def worker(cam: int) -> None:
        for step in range(200):
            store.put(FrameLabels(SEGMENT, step * 200_000_000, [ViewLabel(cam, "left", "no_hand", None, "person", f"w{cam}")]))

    threads: list[threading.Thread] = [threading.Thread(target=worker, args=(cam,)) for cam in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store) == 6 * 200
    assert store.unsaved == 6 * 200


def test_save_layer_writes_atomic_rrd(tmp_path: Path) -> None:
    store: LabelStore = sample_store()
    target: Path = tmp_path / "layers" / f"{SEGMENT}.rrd"
    assert store.save_layer(target) == 4
    assert target.stat().st_mode & 0o777 == 0o644
    assert sorted(path.name for path in target.parent.iterdir()) == [f"{SEGMENT}.rrd"]
    assert store.unsaved == 0
    entry: StoreEntry = RrdReader(target).recordings()[0]
    assert (entry.application_id, entry.recording_id) == ("dataforge", SEGMENT)
    # A second save replaces the file in place.
    store.put(FrameLabels(SEGMENT, 1_600_000_000, [ViewLabel(1, "right", "no_hand", None, "person", "pablo")]))
    assert store.unsaved == 1
    assert store.save_layer(target) == 5
    assert store.unsaved == 0


def test_rows_from_table_decodes_reader_rows() -> None:
    entity: str = label_path(2, "right")
    other: str = label_path(0, "left")
    centers = pa.list_(pa.list_(pa.float32(), 2))
    strings = pa.list_(pa.string())
    table: pa.Table = pa.table(
        {
            "rerun_segment_id": [SEGMENT, SEGMENT, SEGMENT],
            "video_time": pa.array([1_000, 2_000, 3_000], type=pa.duration("ns")),
            f"{entity}:status": pa.array([["accepted"], ["no_hand"], None], type=strings),
            f"{entity}:source": pa.array([["egohand"], ["person"], None], type=strings),
            f"{entity}:reviewer": pa.array([["pablo"], ["astra"], None], type=strings),
            f"{entity}:Boxes2D:centers": pa.array([[[60.0, 120.0]], [], None], type=centers),
            f"{entity}:Boxes2D:half_sizes": pa.array([[[50.0, 100.0]], [], None], type=centers),
            f"{other}:status": pa.array([None, None, ["rejected"]], type=strings),
            f"{other}:source": pa.array([None, None, ["person"]], type=strings),
            f"{other}:reviewer": pa.array([None, None, ["pablo"]], type=strings),
        }
    )
    rows: list[LabelRow] = rows_from_table(table)
    assert rows == [
        LabelRow(1_000, ViewLabel(2, "right", "accepted", Box(10.0, 20.0, 110.0, 220.0), "egohand", "pablo")),
        LabelRow(2_000, ViewLabel(2, "right", "no_hand", None, "person", "astra")),
        LabelRow(3_000, ViewLabel(0, "left", "rejected", None, "person", "pablo")),
    ]


def test_rows_from_table_rejects_bad_status() -> None:
    entity: str = label_path(0, "left")
    strings = pa.list_(pa.string())
    table: pa.Table = pa.table(
        {
            "video_time": pa.array([1_000], type=pa.duration("ns")),
            f"{entity}:status": pa.array([["maybe"]], type=strings),
            f"{entity}:source": pa.array([["person"]], type=strings),
            f"{entity}:reviewer": pa.array([["pablo"]], type=strings),
        }
    )
    with pytest.raises(Exception, match="maybe|status"):
        rows_from_table(table)


@pytest.mark.integration
def test_layer_file_round_trip(tmp_path: Path) -> None:
    """Write the layer, read it back through an in-process catalog server (the same decoder as the real catalog)."""
    store: LabelStore = sample_store()
    target: Path = tmp_path / f"{SEGMENT}.rrd"
    store.save_layer(target)
    assert load_layer_file(target) == store.rows()


REAL_CATALOG: str = "rerun+http://127.0.0.1:9897"
BASE_RRD: Path = Path("/home/pablo/dataforge-dev/rrd/base/robocap__f408193e6447b3b0__s00000044.rrd")
"""Base layer of one small real segment; its recording id is the segment id."""
REAL_SEGMENT: str = "robocap__f408193e6447b3b0__s00000044"


@pytest.mark.integration
def test_real_catalog_register_and_read_back(tmp_path: Path) -> None:
    """Register a tiny hand_boxes layer on :9897 with REPLACE and read it back, in a throwaway dataset.

    The test creates its own dataset (``robocap-labeling-test-<uuid>``) holding one real base layer, writes its
    layer under ``tmp_path`` and deletes the whole dataset at the end. It never touches ``robocap-labeling``.
    """
    if not BASE_RRD.is_file():
        pytest.skip(f"base layer {BASE_RRD} is not on this host")
    try:
        client: CatalogClient = CatalogClient(REAL_CATALOG)
    except (ConnectionError, OSError, ValueError, LookupError) as error:
        pytest.skip(f"catalog {REAL_CATALOG} unavailable: {error}")
    dataset: DatasetEntry = client.create_dataset(f"robocap-labeling-test-{uuid.uuid4()}")
    try:
        dataset.register([BASE_RRD.resolve().as_uri()], layer_name="base").wait()
        table: pa.Table = dataset.segment_table().to_arrow_table()
        assert table.column("rerun_segment_id").to_pylist() == [REAL_SEGMENT]
        start_ns: int = table.column("video_time:start").cast(pa.int64()).to_pylist()[0]

        application_id: str = base_application_id(dataset, REAL_SEGMENT)
        assert application_id == "dataforge"
        store: LabelStore = LabelStore(REAL_SEGMENT, split="eval", application_id=application_id)
        t_ns: int = start_ns + 1_000_000_000
        store.put(
            FrameLabels(
                REAL_SEGMENT,
                t_ns,
                [
                    ViewLabel(0, "left", "accepted", Box(100.0, 200.0, 300.0, 400.0), "egohand", "integration-test"),
                    ViewLabel(1, "right", "no_hand", None, "person", "integration-test"),
                ],
            )
        )
        target: Path = tmp_path / "hand_boxes" / f"{REAL_SEGMENT}.rrd"
        store.save_layer(target)
        dataset.register([target.resolve().as_uri()], layer_name="hand_boxes", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        assert load_layer(dataset, REAL_SEGMENT) == store.rows()
        properties: pa.Table = dataset.segment_table().to_arrow_table()
        assert properties.column("property:hand_boxes:split").to_pylist() == [["eval"]]
    finally:
        dataset.delete()
