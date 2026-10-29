"""Public rig selection does not assume four phones."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import rerun as rr
import rerun.server
from dataforge.writing import atomic_recording
from rerun.catalog import CatalogClient, DatasetEntry
from serde.json import to_json
from simplecv.catalog_rig_layout import CatalogComponent, parse_rig_layout

import selfcap_curate.quest_fit as quest
from selfcap_curate import catalog_io
from selfcap_curate.catalog_io import Segment, rig_layout, select_layout
from selfcap_curate.records import CatalogHands, SelfcapLayout


@pytest.mark.parametrize("phones", [3, 4])
def test_discovers_episode_rigs(phones: int) -> None:
    components = [
        CatalogComponent(f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole/video", "VideoStream", "sample", False)
        for rig, count in [(i, 1) for i in range(phones)] + [(phones, 3), (phones + 1, 2)]
        for cam in range(count)
    ]
    kinds = {f"rig_{i:02d}": "exo" for i in range(phones)} | {f"rig_{phones:02d}": "ego", f"rig_{phones + 1:02d}": "quest"}
    layout = select_layout(parse_rig_layout(components, kinds), tuple(f"phone-{i}" for i in range(phones)))
    assert layout.oak_rig == phones
    assert layout.quest_rig == phones + 1
    assert layout.video_entities[0] == f"/world/rig_{phones:02d}/cam_01/pinhole/video"


@pytest.mark.integration
def test_mixed_sessions_only_discover_segment_cameras(tmp_path, private_server, synthetic_base, fixture_sync) -> None:

    fixture = Path(__file__).parents[2] / "dataforge/tests/fixtures/av1_48f_192x160.mp4"
    bases = []
    for phones in (3, 4):
        path = tmp_path / f"session-{phones}.rrd"
        sync = fixture_sync(offsets_ms={f"exo/phone-{i}": 0.0 for i in range(phones)})
        synthetic_base(
            path, segment=f"session-{phones}", phones=tuple(f"phone-{i}" for i in range(phones)), video=fixture, sync=sync, calibrated=True
        )
        bases.append(path)
    with private_server({"mixed": bases}) as server:
        dataset = CatalogClient(server.url()).get_dataset("mixed")
        for phones in (3, 4):
            layout = rig_layout(dataset, f"session-{phones}")
            assert len(layout.cameras) == phones + 5
            assert sum(camera.rig_kind == "exo" for camera in layout.cameras) == phones
            assert select_layout(layout, tuple(f"phone-{i}" for i in range(phones))).oak_rig == phones



@pytest.mark.integration
def test_catalog_time_sync_reads_only_the_named_base(tmp_path, private_server, fixture_sync) -> None:

    sync = fixture_sync()
    path = tmp_path / "base.rrd"
    with atomic_recording(path, application_id="dataforge", recording_id=f"selfcap__{sync.key}") as recording:
        recording.send_property("time_sync", rr.AnyValues(record_json=to_json(sync)))
    with private_server({catalog_io.DATASET_NAME: [path]}) as server:
        result = catalog_io.read_time_sync(server.client().get_dataset(catalog_io.DATASET_NAME), f"selfcap__{sync.key}")
    assert result == sync


@pytest.mark.integration
def test_calibration_layout_uses_the_segments_physical_phone_names(tmp_path, private_server, synthetic_base) -> None:

    path = tmp_path / "layout.rrd"
    segment = "selfcap__2025-11-24__027d8215__ep002"
    synthetic_base(path, segment=segment, phones=("A", "B", "C"))
    with private_server({catalog_io.DATASET_NAME: [path]}) as server:
        layout = catalog_io.open_segment(server.client().get_dataset(catalog_io.DATASET_NAME), segment)
    assert layout.layout == SelfcapLayout(3, 4, ("A", "B", "C"))


def test_session_inputs_discover_one_complete_layout(monkeypatch, fixture_sync) -> None:

    sync = fixture_sync()
    layout = SelfcapLayout(4, 5, tuple(sync.exo_fit))
    discoveries, supplied = [], []
    stamps = np.arange(3, dtype=np.float64) * 0.02
    hands = CatalogHands(stamps, np.ones((3, 133, 3)), stamps, np.zeros((3, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)))

    dataset = Mock(spec=DatasetEntry)

    def discover(_dataset, segment_id):
        discoveries.append(segment_id)
        return Segment(_dataset, segment_id, layout)

    def gyro(segment):
        supplied.append(segment)
        return np.empty(0), np.empty((0, 3))

    def raw_hands(segment):
        supplied.append(segment)
        return hands

    monkeypatch.setattr(catalog_io, "open_segment", discover)
    monkeypatch.setattr(quest, "read_gyro", gyro)
    monkeypatch.setattr(quest, "read_catalog_hands", raw_hands)
    segment = catalog_io.open_segment(dataset, sync.episode_key.segment)
    episode = quest.read_session_episode(segment, sync)
    assert discoveries == [f"selfcap__{sync.key}"]
    assert episode.layout is layout
    assert supplied == [segment, segment]


@pytest.mark.integration
def test_catalog_time_sync_rejects_another_episodes_valid_record(tmp_path, private_server, fixture_sync):
    sync = fixture_sync()
    segment = sync.episode_key.segment.removesuffix("002") + "001"
    path = tmp_path / "wrong-sync.rrd"
    with atomic_recording(path, application_id="dataforge", recording_id=segment) as recording:
        recording.send_property("time_sync", rr.AnyValues(record_json=to_json(sync)))
    with private_server({catalog_io.DATASET_NAME: [path]}) as server, pytest.raises(ValueError, match="identity"):
        catalog_io.read_time_sync(server.client().get_dataset(catalog_io.DATASET_NAME), segment)
