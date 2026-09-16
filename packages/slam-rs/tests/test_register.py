"""Registering downloaded MSD recordings into a local catalog: one dataset per device, one layer per directory."""

from pathlib import Path

import pytest
import rerun as rr
import rerun.blueprint as rrb
from rerun.catalog import CatalogClient

from slam_rs.apis.register import register

INDEX: str = "msd-index__MIO_others__MIO10_short_2_panorama"
G2: str = "msd-g2__MGO_others__MGO09_short_1_updown"


def write_recording(path: Path, recording_id: str) -> None:
    """A minimal recording the catalog can register: one static scalar under the rig."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream: rr.RecordingStream = rr.RecordingStream("msd", recording_id=recording_id)
    stream.save(str(path))
    stream.log("/world/rig_00/imu_00/gyro", rr.Scalars([0.0, 0.0, 0.0]), static=True)
    stream.flush(blocking=True)
    stream.disconnect()


def write_layout(root: Path, layers: tuple[str, ...] = ("base", "gt", "sensor_metadata")) -> None:
    for recording_id in (INDEX, G2):
        for layer in layers:
            write_recording(root / layer / f"{recording_id}.rrd", recording_id)


def test_one_dataset_per_device_and_one_layer_per_directory(tmp_path: Path) -> None:
    write_layout(tmp_path)
    with rr.server.Server() as server:
        client: CatalogClient = CatalogClient(server.url)
        counts: dict[str, dict[str, int]] = register(tmp_path, client)
        assert counts == {
            "msd-index": {"base": 1, "gt": 1, "sensor_metadata": 1},
            "msd-g2": {"base": 1, "gt": 1, "sensor_metadata": 1},
        }
        assert {"msd-index", "msd-g2"} <= set(client.dataset_names())
        assert list(client.get_dataset("msd-index").segment_ids()) == [INDEX]
        assert list(client.get_dataset("msd-g2").segment_ids()) == [G2]


def test_registering_twice_is_idempotent(tmp_path: Path) -> None:
    write_layout(tmp_path)
    with rr.server.Server() as server:
        client: CatalogClient = CatalogClient(server.url)
        register(tmp_path, client)
        counts: dict[str, dict[str, int]] = register(tmp_path, client)
        assert counts["msd-index"]["base"] == 1
        assert list(client.get_dataset("msd-index").segment_ids()) == [INDEX]


def test_a_derived_layer_nobody_downloaded_is_simply_absent(tmp_path: Path) -> None:
    write_layout(tmp_path, layers=("base",))
    with rr.server.Server() as server:
        client: CatalogClient = CatalogClient(server.url)
        counts: dict[str, dict[str, int]] = register(tmp_path, client)
        assert counts == {"msd-index": {"base": 1}, "msd-g2": {"base": 1}}


def test_a_root_without_base_recordings_is_refused(tmp_path: Path) -> None:
    (tmp_path / "gt").mkdir()
    with rr.server.Server() as server, pytest.raises(FileNotFoundError, match="base"):
        register(tmp_path, CatalogClient(server.url))


def test_the_device_blueprint_becomes_the_default_once(tmp_path: Path) -> None:
    write_layout(tmp_path, layers=("base",))
    (tmp_path / "blueprints").mkdir()
    rrb.Blueprint(rrb.Spatial3DView(origin="/world")).save("msd-index", str(tmp_path / "blueprints" / "msd-index.rbl"))
    with rr.server.Server() as server:
        client: CatalogClient = CatalogClient(server.url)
        register(tmp_path, client)
        assert client.get_dataset("msd-index").default_blueprint() is not None
        assert client.get_dataset("msd-g2").default_blueprint() is None
        before: int = len(list(client.get_dataset("msd-index").blueprints()))
        register(tmp_path, client)
        assert len(list(client.get_dataset("msd-index").blueprints())) == before
