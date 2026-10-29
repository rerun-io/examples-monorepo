"""Bounded layer replacement against a fake catalog; no server access."""

from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import rerun as rr
from rerun.catalog import OnDuplicateSegmentLayer, RegistrationResult

from dataforge.apis import swap_layer as swap_api
from dataforge.catalog_swap import swap_layer
from dataforge.writing import atomic_recording


class Task:
    def __init__(self, segments: list[str] | None = None) -> None:
        self.segments = segments

    def wait(self) -> RegistrationResult | None:
        return None if self.segments is None else RegistrationResult(segment_ids=self.segments)


class Entry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> Task:
        assert segments_to_drop, "empty means ALL segments"
        assert layers_to_drop == ["calib"]
        self.calls.append(("drop", segments_to_drop))
        return Task()

    def register(self, uris: list[str], *, layer_name: str, on_duplicate: OnDuplicateSegmentLayer) -> Task:
        assert layer_name == "calib"
        assert on_duplicate == OnDuplicateSegmentLayer.SKIP
        self.calls.append(("add", uris))
        return Task([Path(unquote(urlparse(uri).path)).stem for uri in uris])


def write_recording(path: Path, recording_id: str | None = None) -> None:
    with atomic_recording(path, recording_id=recording_id or path.stem) as recording:
        recording.log("/test", rr.Scalars(1.0), static=True)


def test_swap_waits_and_drops_only_exact_ids_in_batches(tmp_path: Path) -> None:
    files: list[Path] = [tmp_path / f"segment-{i}.rrd" for i in range(1001)]
    for path in files:
        write_recording(path)
    entry: Entry = Entry()
    answers = iter([False, False, True, True, True])
    sleeps: list[float] = []
    swap_layer(entry, "calib", files, idle_probe=lambda: next(answers), sleep=sleeps.append)
    assert sleeps == [30.0, 30.0]
    assert [len(ids) for operation, ids in entry.calls if operation == "drop"] == [500, 500, 1]
    assert [item for operation, ids in entry.calls if operation == "drop" for item in ids] == [path.stem for path in files]
    assert [item for operation, ids in entry.calls if operation == "add" for item in ids] == [path.as_uri() for path in files]
    assert [operation for operation, _ in entry.calls] == ["drop", "add"] * 3


@pytest.mark.parametrize("batch_size", [0, -1])
def test_swap_rejects_invalid_batches_before_mutation(tmp_path: Path, batch_size: int) -> None:
    entry: Entry = Entry()
    with pytest.raises(ValueError):
        swap_layer(entry, "calib", [], batch_size=batch_size, idle_probe=lambda: True)
    assert not entry.calls


def test_empty_swap_never_unregisters() -> None:
    entry: Entry = Entry()
    with pytest.raises(ValueError, match="requires files"):
        swap_layer(entry, "calib", [], idle_probe=lambda: pytest.fail("empty swap must not probe"))
    assert not entry.calls


@pytest.mark.parametrize("backlog,listing,ops,milliseconds,expected", [
    (False, True, 10, 100.0, False),
    (True, False, 10, 100.0, False),
    (True, True, 10, 999.0, True),
    (True, True, 10, 1000.0, False),
    (True, True, 0, 0.0, True),
])
def test_idle_gate_checks_all_required_probes(backlog: bool, listing: bool, ops: int, milliseconds: float, expected: bool) -> None:
    from dataforge.catalog_swap import catalog_idle

    readings = iter([(20, 100.0), (20 + ops, 100.0 + milliseconds)])
    sleeps: list[float] = []
    assert catalog_idle(
        "rerun+http://catalog.example:9988", Path("/test/nfs"),
        backlog_probe=lambda _: backlog, listing_probe=lambda _: listing,
        read_probe=lambda _: next(readings), sleep=sleeps.append,
    ) == expected
    assert sleeps == ([20.0] if backlog else [])


def test_swap_cli_uses_only_get_dataset_and_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry: Entry = Entry()
    source: Path = tmp_path / "episode.rrd"
    write_recording(source)

    class Client:
        def __init__(self, url: str) -> None:
            assert url == "rerun+http://test.invalid:9999"

        def get_dataset(self, name: str) -> Entry:
            assert name == "test-dataset"
            return entry

    monkeypatch.setattr(swap_api, "CatalogClient", Client)
    monkeypatch.setattr(swap_api, "catalog_idle", lambda *_, **_kwargs: True)
    swap_api.main(swap_api.SwapConfig(catalog_url="rerun+http://test.invalid:9999", catalog_name="test-dataset", layer="calib", files=(source,)))
    assert entry.calls == [("drop", ["episode"]), ("add", [source.as_uri()])]


@pytest.mark.parametrize("rows,expected", [("LISTEN 0 4096 127.0.0.1:9999 0.0.0.0:*\n", True), ("LISTEN 2 4096 *:9999 *:*\n", False), ("", False)])
def test_local_backlog_probe_uses_configured_port(rows: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from dataforge import catalog_swap

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["ss", "-H", "-ltn", "( sport = :9999 )"]
        assert kwargs["timeout"] == 5.0
        return subprocess.CompletedProcess(command, 0, stdout=rows)

    monkeypatch.setattr(catalog_swap.subprocess, "run", run)
    assert catalog_swap.listen_backlog_idle(9999) == expected


def test_nfs_probe_reads_execute_time_from_exact_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataforge.catalog_swap import nfs_read_stats

    def read_text(path: Path) -> str:
        assert path == Path("/proc/self/mountstats")
        return (
            "device other:/export mounted on /other with fstype nfs4\n READ: 1 2 3 4 5 6 7 800\n"
            "device server:/export mounted on /test/nfs with fstype nfs4\n READ: 20 2 3 4 5 6 7 900\n"
            "device last:/export mounted on /last with fstype nfs4\n READ: 3 2 3 4 5 6 7 1000\n"
        )

    monkeypatch.setattr(Path, "read_text", read_text)
    assert nfs_read_stats(Path("/test/nfs")) == (20, 900.0)
    with pytest.raises(ValueError, match="no READ stats"):
        nfs_read_stats(Path("/missing"))


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_listing_probe_is_bounded_and_fails_closed(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from dataforge import catalog_swap

    def run(command: list[str], **kwargs: object) -> None:
        assert kwargs["timeout"] == 10.0
        assert command[-1] == "rerun+http://test.invalid:9999"
        assert "segment_table" not in command[2]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10.0)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(catalog_swap.subprocess, "run", run)
    assert not catalog_swap.listing_idle("rerun+http://test.invalid:9999")



def test_swap_checks_all_file_identities_before_any_mutation(tmp_path: Path) -> None:
    files: list[Path] = [tmp_path / "correct.rrd", tmp_path / "wrong.rrd"]
    write_recording(files[0])
    write_recording(files[1], "other-recording")
    entry: Entry = Entry()
    with pytest.raises(ValueError, match="wrong.rrd.*other-recording"):
        swap_layer(entry, "calib", files, batch_size=1, idle_probe=lambda: True)
    assert not entry.calls


@pytest.mark.parametrize("layer", [" calib ", "calib layer", "calib/other", "", " "])
def test_swap_rejects_invalid_layer_names(layer: str) -> None:
    entry: Entry = Entry()
    with pytest.raises(ValueError, match="layer"):
        swap_layer(entry, layer, [], idle_probe=lambda: True)
    assert not entry.calls


@pytest.mark.parametrize("returned", [["wrong"], [], ["correct", "correct"], None])
def test_swap_refuses_unexpected_registration_result(tmp_path: Path, returned: list[str] | None) -> None:
    source: Path = tmp_path / "correct.rrd"
    write_recording(source)

    class WrongEntry(Entry):
        def register(self, uris: list[str], *, layer_name: str, on_duplicate: OnDuplicateSegmentLayer) -> Task:
            super().register(uris, layer_name=layer_name, on_duplicate=on_duplicate)
            return Task(returned)

    with pytest.raises(RuntimeError, match="registration.*correct"):
        swap_layer(WrongEntry(), "calib", [source], idle_probe=lambda: True)


def test_gate_timeout_logs_progress_without_mutating(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source: Path = tmp_path / "episode.rrd"
    write_recording(source)
    entry: Entry = Entry()
    now: list[float] = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(TimeoutError, match="idle gate"):
        swap_layer(entry, "calib", [source], idle_probe=lambda: False,
                            gate_timeout_s=45.0, sleep=sleep, monotonic=lambda: now[0])
    assert now[0] == 45.0
    assert not entry.calls
    assert capsys.readouterr().out.count("waiting") == 2


def test_swap_logs_segment_ids_before_unregister(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source: Path = tmp_path / "episode.rrd"
    write_recording(source)

    class LoggedEntry(Entry):
        def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> Task:
            assert "episode" in capsys.readouterr().out
            return super().unregister(segments_to_drop=segments_to_drop, layers_to_drop=layers_to_drop)

    swap_layer(LoggedEntry(), "calib", [source], idle_probe=lambda: True)




@pytest.mark.integration
def test_swap_verifies_real_registration_on_private_server(tmp_path: Path) -> None:
    from rerun.catalog import DatasetEntry

    source: Path = tmp_path / "episode.rrd"
    write_recording(source)
    with rr.server.Server(port=0, datasets={"swap-test": [source]}) as server:
        entry: DatasetEntry = server.client().get_dataset("swap-test")
        # Repeat to exercise both absent and already registered target layers.
        for _ in range(2):
            swap_layer(entry, "calib", [source], idle_probe=lambda: True)


@pytest.mark.parametrize("kind", ["blueprint", "multiple"])
def test_swap_requires_exactly_one_recording(tmp_path: Path, kind: str) -> None:
    import subprocess
    import sys

    import rerun.blueprint as rrb

    source: Path = tmp_path / "episode.rrd"
    if kind == "blueprint":
        rrb.Blueprint(rrb.Spatial3DView()).save("test", source)
    else:
        first: Path = tmp_path / "first.rrd"
        write_recording(first, "episode")
        other: Path = tmp_path / "other.rrd"
        write_recording(other)
        subprocess.run([str(Path(sys.executable).with_name("rerun")), "rrd", "merge", str(first), str(other), "-o", str(source)], check=True)
    entry: Entry = Entry()
    with pytest.raises(ValueError, match="exactly one recording"):
        swap_layer(entry, "calib", [source], idle_probe=lambda: True)
    assert not entry.calls


def test_swap_cli_requires_its_own_inputs() -> None:
    import tyro

    parsed = tyro.cli(swap_api.SwapConfig, args=[ "--layer", "calib", "--files", "/tmp/episode.rrd", "--catalog-name", "test"])
    assert isinstance(parsed, swap_api.SwapConfig)
    assert parsed.files == (Path("/tmp/episode.rrd"),)
    assert parsed.catalog_name == "test"
    assert not hasattr(parsed, "dataset")
    with pytest.raises(SystemExit):
        tyro.cli(swap_api.SwapConfig, args=[ "--layer", "calib"])


def test_register_cli_keeps_original_dataset_spelling() -> None:
    import tyro

    from dataforge.apis.register import Config
    from dataforge.datasets.selfcap import SelfcapConfig

    parsed = tyro.cli(Config, args=["selfcap"])
    assert isinstance(parsed, Config)
    assert isinstance(parsed.dataset, SelfcapConfig)
    assert not any(name.startswith("swap_") for name in vars(parsed))


def test_remote_catalog_skips_local_backlog_probe() -> None:
    from dataforge.catalog_swap import catalog_idle

    assert catalog_idle(
        "rerun+http://remote.invalid:9988", local=False,
        backlog_probe=lambda _: pytest.fail("remote catalog must not probe ss"),
        listing_probe=lambda _: True,
    )
