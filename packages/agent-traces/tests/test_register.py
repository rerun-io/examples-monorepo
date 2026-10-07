"""Catalog registration through the CLI API and a fake SDK boundary."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pytest
from rerun.catalog import OnDuplicateSegmentLayer, SegmentRegistrationResult
from rerun.chunk import RrdReader

from agent_traces.apis import convert_all, register
from tests.conftest import SessionBuilder


@dataclass
class FakeRegistration:
    """Completed registration results with an explicit wait boundary."""
    results: list[SegmentRegistrationResult]
    """Results returned by the catalog."""
    waited: bool = False
    """Whether the caller waited before reading results."""
    def wait(self) -> None:
        self.waited = True
        if any(result.error for result in self.results):
            raise ValueError("registration failed")

    def iter_results(self) -> Iterator[SegmentRegistrationResult]:
        assert self.waited
        return iter(self.results)


@dataclass
class FakeBlueprintTable:
    """Blueprint storage rows at the SDK Arrow boundary."""

    uris: list[str]
    """Registered blueprint files, used as segment identities by the fake."""

    def select(self, *columns: str) -> "FakeBlueprintTable":
        assert columns == ("rerun_segment_id", "rerun_storage_urls")
        return self

    def to_arrow_table(self) -> pa.Table:
        return pa.table({"rerun_segment_id": pa.array(self.uris, type=pa.string()),
                         "rerun_storage_urls": pa.array([[uri] for uri in self.uris], type=pa.list_(pa.string()))})


@dataclass
class FakeEntry:
    """Catalog state and calls visible at the SDK boundary."""
    calls: list[tuple[list[str], str, OnDuplicateSegmentLayer]] = field(default_factory=list)
    """Batches submitted for registration."""
    segments: set[str] = field(default_factory=set)
    """Segments available in the catalog."""
    blueprints: list[str] = field(default_factory=list)
    """Registered default blueprints."""
    omit: str | None = None
    """Successful segment hidden from listing to simulate a catalog fault."""
    fail: str | None = None
    """Segment whose registration fails."""
    def register(self, uris: list[str], *, layer_name: str, on_duplicate: OnDuplicateSegmentLayer) -> FakeRegistration:
        self.calls.append((uris, layer_name, on_duplicate))
        results: list[SegmentRegistrationResult] = []
        for uri in uris:
            session_id: str = Path(uri).stem
            error: str | None = "bad recording" if session_id == self.fail else None
            if error is None and session_id != self.omit:
                self.segments.add(session_id)
            results.append(SegmentRegistrationResult(uri=uri, segment_id=session_id, error=error))
        return FakeRegistration(results)

    def segment_ids(self) -> list[str]:
        return sorted(self.segments)

    def blueprint_dataset(self):
        owner = self

        class Blueprints:
            def segment_table(self) -> FakeBlueprintTable:
                return FakeBlueprintTable(owner.blueprints)

            def segment_ids(self) -> list[str]:
                return list(owner.blueprints)

            def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> FakeRegistration:
                return owner.unregister(segments_to_drop=segments_to_drop, layers_to_drop=layers_to_drop)

        return Blueprints()

    def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> FakeRegistration:
        assert not layers_to_drop
        self.blueprints[:] = [uri for uri in self.blueprints if uri not in segments_to_drop]
        return FakeRegistration([])

    def segment_table(self):
        class Table:
            def schema(self) -> pa.Schema:
                return pa.schema([("property:skipped:unknown", pa.int64())])

        return Table()

    def default_blueprint(self) -> str | None:
        return self.blueprints[0] if self.blueprints else None

    def register_blueprint(self, uri: str, *, set_default: bool, segment_table: bool = False) -> None:
        assert set_default
        assert segment_table == ("table-" in uri)
        self.blueprints.append(uri)


@dataclass
class FakeClient:
    """Share datasets across repeated CLI calls."""
    url: str
    """Requested catalog URL."""
    entries: ClassVar[dict[str, FakeEntry]] = {}
    """Datasets created by the test."""
    def dataset_names(self) -> list[str]:
        return list(self.entries)

    def get_dataset(self, name: str) -> FakeEntry:
        return self.entries[name]

    def create_dataset(self, name: str) -> FakeEntry:
        assert name not in self.entries
        assert self.url == "catalog"
        return self.entries.setdefault(name, FakeEntry())


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeEntry]:
    """Patch SDK types as well as construction for beartype local checks."""
    monkeypatch.setattr(FakeClient, "entries", {})
    monkeypatch.setattr(register, "CatalogClient", FakeClient)
    monkeypatch.setattr(register, "DatasetEntry", FakeEntry)
    monkeypatch.setattr(register, "RegistrationHandle", FakeRegistration)
    return FakeClient.entries


@pytest.fixture
def converted(tmp_path: Path) -> Path:
    """Convert two real synthetic sessions in each of two profiles."""
    out: Path = tmp_path / "out"
    for profile in ("x", "y"):
        home: Path = tmp_path / profile
        for session_id in ("a", "b"):
            SessionBuilder(home / "projects/project" / f"{session_id}.jsonl").add("user", message={"content": session_id})
        convert_all.main(convert_all.Config(home=home, out=out))
    (out / "unconverted").mkdir()
    return out


@pytest.mark.parametrize("replace", [False, True])
def test_profiles_are_batched(converted: Path, catalog: dict[str, FakeEntry], replace: bool) -> None:
    """Each profile becomes one dataset with one base-layer batch."""
    register.main(register.Config(catalog_url="catalog", out=(converted,), replace=replace))
    assert set(catalog) == {"agent-traces-x", "agent-traces-y"}
    for profile in ("x", "y"):
        assert catalog[f"agent-traces-{profile}"].calls == [
            (
                [(converted / profile / name).resolve().as_uri() for name in ("a.rrd", "b.rrd")],
                "base",
                OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.SKIP,
            )
        ]


def test_selection_blueprint_and_duplicate_summary(converted: Path, catalog: dict[str, FakeEntry], capsys: pytest.CaptureFixture[str]) -> None:
    """Selection reuses the dataset and republishes both layouts without overwriting files."""
    config: register.Config = register.Config(catalog_url="catalog", out=(converted,), profile="x", dataset="custom")
    register.main(config)
    assert set(catalog) == {"custom"}
    entry: FakeEntry = catalog["custom"]
    assert len(entry.blueprints) == 2
    originals = {uri: Path(uri.removeprefix("file://")).read_bytes() for uri in entry.blueprints}
    assert all(originals.values())
    table_uri = next(uri for uri in entry.blueprints if "table-" in uri)
    reader = RrdReader(Path(table_uri.removeprefix("file://")))
    entities = {str(chunk.entity_path): chunk.to_record_batch() for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks()}
    assert entities["/table/layouts/table"]["TableLayout:column_order"].to_pylist() == [[
        "property:RecordingInfo:name", "wall:start", "property:session:n_turns",
        "property:session:host", "property:session:agent", "property:session:profile",
    ]]
    hidden = next(batch for path, batch in entities.items() if path.startswith("/table/layouts/table/columns/") and "skipped" in path)
    assert hidden["TableColumn:visible"].to_pylist() == [[False]]
    assert "registered=2 skipped_duplicates=0 errors=0 missing=0 dataset=custom segments=2 url=catalog" in capsys.readouterr().out
    register.main(config)
    assert len(entry.blueprints) == 2
    assert not set(entry.blueprints) & originals.keys()
    assert all(not Path(uri.removeprefix("file://")).exists() for uri in originals)
    assert "registered=0 skipped_duplicates=2 errors=0 missing=0" in capsys.readouterr().out
    register.main(register.Config(catalog_url=config.catalog_url, out=(converted,), profile="x", dataset="custom", replace=True))
    assert "registered=2 skipped_duplicates=0 errors=0 missing=0" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["a", "both"])
def test_missing_rrds_are_nonfatal(converted: Path, catalog: dict[str, FakeEntry], capsys: pytest.CaptureFixture[str], missing: str) -> None:
    """Missing files are reported and omitted from the batch and verification."""
    (converted / "x/a.rrd").unlink()
    if missing == "both":
        (converted / "x/b.rrd").unlink()
    register.main(register.Config(catalog_url="catalog", out=(converted,), profile="x"))
    output: str = capsys.readouterr().out
    assert f"MISSING {converted / 'x/a.rrd'}" in output
    assert (
        "registered=0 skipped_duplicates=0 errors=0 missing=2" if missing == "both" else "registered=1 skipped_duplicates=0 errors=0 missing=1"
    ) in output
    assert catalog["agent-traces-x"].segments == (set() if missing == "both" else {"b"})


@pytest.mark.parametrize(("directory", "name", "scheme", "authority", "deleted"), [
    ("second root/x", "agent-traces-0123456789abcdef0123456789abcdef.rbl", "file", "", True),
    ("second root/x", "agent-traces-table-0123456789abcdef0123456789abcdef.rbl", "file", "localhost", True),
    ("out/x", "agent-traces.rbl", "file", "", False),
    ("out/x", "agent-traces-custom.rbl", "file", "", False),
    ("out/x", "agent-traces-0123456789abcdef0123456789abcdeg.rbl", "file", "", False),
    ("out/x", "agent-traces-0123456789abcdef0123456789abcdef0.rbl", "file", "", False),
    ("out/y", "agent-traces-0123456789abcdef0123456789abcdef.rbl", "file", "", False),
    ("out/x", "agent-traces-0123456789abcdef0123456789abcdef.rbl", "https", "localhost", False),
    ("out/x", "agent-traces-0123456789abcdef0123456789abcdef.rbl", "file", "remote", False),
])
def test_retired_blueprint_ownership(converted: Path, catalog: dict[str, FakeEntry], directory: str,
                                   name: str, scheme: str, authority: str, deleted: bool) -> None:
    """Only stamped local files in matching profile directories are removed, across output roots."""
    retired: Path = converted.parent / directory / name
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_bytes(b"retired blueprint")
    uri: str = f"{scheme}://{authority}{retired.as_uri().removeprefix('file://')}"
    catalog["agent-traces-x"] = FakeEntry(blueprints=[uri])
    register.main(register.Config(catalog_url="catalog", out=(converted,), profile="x"))
    assert retired.exists() is not deleted
    assert uri not in catalog["agent-traces-x"].blueprints


def test_missing_manifest_names_path(tmp_path: Path, catalog: dict[str, FakeEntry]) -> None:
    """An explicit unconverted profile is a user error before catalog access."""
    with pytest.raises(ValueError, match=str(tmp_path / "x/manifest.json")):
        register.main(register.Config(catalog_url="catalog", out=(tmp_path,), profile="x"))
    assert not catalog


def test_catalog_must_list_successful_session(converted: Path, catalog: dict[str, FakeEntry]) -> None:
    """A completed registration must be visible under the manifest session ID."""
    catalog["agent-traces-x"] = FakeEntry(omit="a")
    with pytest.raises(RuntimeError, match="a"):
        register.main(register.Config(catalog_url="catalog", out=(converted,), profile="x"))


def test_errors_are_reported_per_recording(converted: Path, catalog: dict[str, FakeEntry], capsys: pytest.CaptureFixture[str]) -> None:
    """A wait error still exposes individual failed and successful results."""
    catalog["agent-traces-x"] = FakeEntry(fail="a", blueprints=["existing"])
    register.main(register.Config(catalog_url="catalog", out=(converted,), profile="x"))
    output: str = capsys.readouterr().out
    assert f"ERROR {(converted / 'x/a.rrd').as_uri()}: bad recording" in output
    assert "registered=1 skipped_duplicates=0 errors=1 missing=0" in output
    assert len(catalog["agent-traces-x"].blueprints) == 2
    assert "existing" not in catalog["agent-traces-x"].blueprints
    assert not (converted / "x/agent-traces.rbl").exists()


def test_all_published_files_are_readable(converted: Path, catalog: dict[str, FakeEntry]) -> None:
    """The catalog's user can read recordings, manifests, and the default blueprint."""
    register.main(register.Config(catalog_url="catalog", out=(converted,)))
    for profile in ("x", "y"):
        for name in ("a.rrd", "manifest.json", *(path.name for path in (converted / profile).glob("*.rbl"))):
            assert (converted / profile / name).stat().st_mode & 0o777 == 0o644


def test_blueprint_failure_leaves_no_partial_file(converted: Path, catalog: dict[str, FakeEntry], monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed blueprint save neither publishes nor leaves its temporary file."""
    import rerun.blueprint as rrb

    before = set((converted / "x").iterdir())

    def fail_save(self: rrb.Blueprint, application_id: str, path: Path) -> None:
        assert application_id == "agent_traces"
        path.write_bytes(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(rrb.Blueprint, "save", fail_save)
    with pytest.raises(RuntimeError, match="save failed"):
        register.main(register.Config(catalog_url="catalog", out=(converted,), profile="x"))
    assert set((converted / "x").iterdir()) == before
    assert not catalog["agent-traces-x"].blueprints


@pytest.mark.parametrize("failure", ["catalog", "missing"])
def test_registration_cli_exits_one_on_partial_failure(converted: Path, catalog: dict[str, FakeEntry], monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """The actual CLI shim maps both failure summaries to process status one."""
    import runpy
    import sys

    if failure == "catalog":
        catalog["agent-traces-x"] = FakeEntry(fail="a")
    else:
        (converted / "x/a.rrd").unlink()
    config = register.Config(catalog_url="catalog", out=(converted,), profile="x")
    summary = register.main(config)
    assert summary.exit_code == 1
    assert (summary.errors, summary.missing) == ((1, 0) if failure == "catalog" else (0, 1))
    monkeypatch.setattr(sys, "argv", ["register", "--catalog-url", "catalog", "--out", str(converted), "--profile", "x"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(__file__).parents[1] / "tools/apps/register.py"), run_name="__main__")
    assert error.value.code == 1


@pytest.mark.parametrize("replace", [False, True])
def test_multiple_roots_share_one_batch_per_profile(converted: Path, catalog: dict[str, FakeEntry], replace: bool,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """CLI roots form a profile union; the first copy wins even when replacing layers."""
    import tyro

    second: Path = converted.parent / "second-out"
    for profile, identities in (("x", ("b", "c")), ("z", ("d",))):
        home: Path = converted.parent / "second-home" / profile
        for identity in identities:
            SessionBuilder(home / "projects/project" / f"{identity}.jsonl").add("user", message={"content": identity})
        convert_all.main(convert_all.Config(home=home, out=second))
    args: list[str] = ["--catalog-url", "catalog", "--out", str(converted), str(second)]
    config: register.Config = tyro.cli(register.Config, args=args + (["--replace"] if replace else []))
    assert config.out == (converted, second)
    assert register.main(config) == register.Summary(6, 1, 0, 0)
    assert set(catalog) == {"agent-traces-x", "agent-traces-y", "agent-traces-z"}
    assert catalog["agent-traces-x"].calls == [([
        (converted / "x/a.rrd").as_uri(), (converted / "x/b.rrd").as_uri(), (second / "x/c.rrd").as_uri(),
    ], "base", OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.SKIP)]
    for name in ("y", "z"):
        assert len(catalog[f"agent-traces-{name}"].calls) == 1
    assert "registered=3 skipped_duplicates=1 errors=0 missing=0 dataset=agent-traces-x" in capsys.readouterr().out
    assert len(list((converted / "x").glob("*.rbl"))) == 2
    assert not list((second / "x").glob("*.rbl"))
    assert len(list((second / "z").glob("*.rbl"))) == 2
    assert register.main(register.Config(catalog_url="catalog", out=(converted, second), profile="z")) == register.Summary(0, 1, 0, 0)
