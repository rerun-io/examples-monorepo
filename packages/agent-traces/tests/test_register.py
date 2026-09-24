"""Catalog registration through the CLI API and a fake SDK boundary."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest
from rerun.catalog import OnDuplicateSegmentLayer, SegmentRegistrationResult

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

    def default_blueprint(self) -> str | None:
        return self.blueprints[0] if self.blueprints else None

    def register_blueprint(self, uri: str, *, set_default: bool) -> None:
        assert set_default
        self.blueprints.append(uri)


@dataclass
class FakeClient:
    """Share datasets across repeated CLI calls."""

    url: str
    """Requested catalog URL."""
    entries: ClassVar[dict[str, FakeEntry]] = {}
    """Datasets created by the test."""

    def create_dataset(self, name: str, *, exist_ok: bool) -> FakeEntry:
        assert exist_ok
        assert self.url == "rerun+http://127.0.0.1:1"
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
    register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, replace=replace))
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
    """Selection and name override persist one blueprint across reruns."""
    config: register.Config = register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, profile="x", dataset="custom")
    register.main(config)
    assert set(catalog) == {"custom"}
    entry: FakeEntry = catalog["custom"]
    blueprint: Path = converted / "x/agent-traces.rbl"
    assert entry.blueprints == [blueprint.resolve().as_uri()]
    assert blueprint.stat().st_size > 0
    original: bytes = blueprint.read_bytes()
    assert "registered=2 skipped_duplicates=0 errors=0 missing=0 dataset=custom segments=2 url=rerun+http://127.0.0.1:1" in capsys.readouterr().out
    register.main(config)
    assert len(entry.blueprints) == 1
    assert blueprint.read_bytes() == original
    assert "registered=0 skipped_duplicates=2 errors=0 missing=0" in capsys.readouterr().out
    register.main(register.Config(catalog_url=config.catalog_url, out=converted, profile="x", dataset="custom", replace=True))
    assert "registered=2 skipped_duplicates=0 errors=0 missing=0" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["a", "both"])
def test_missing_rrds_are_nonfatal(converted: Path, catalog: dict[str, FakeEntry], capsys: pytest.CaptureFixture[str], missing: str) -> None:
    """Missing files are reported and omitted from the batch and verification."""
    (converted / "x/a.rrd").unlink()
    if missing == "both":
        (converted / "x/b.rrd").unlink()
    register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, profile="x"))
    output: str = capsys.readouterr().out
    assert f"MISSING {converted / 'x/a.rrd'}" in output
    assert (
        "registered=0 skipped_duplicates=0 errors=0 missing=2" if missing == "both" else "registered=1 skipped_duplicates=0 errors=0 missing=1"
    ) in output
    assert catalog["agent-traces-x"].segments == (set() if missing == "both" else {"b"})


def test_missing_manifest_names_path(tmp_path: Path, catalog: dict[str, FakeEntry]) -> None:
    """An explicit unconverted profile is a user error before catalog access."""
    with pytest.raises(ValueError, match=str(tmp_path / "x/manifest.json")):
        register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=tmp_path, profile="x"))
    assert not catalog


def test_catalog_must_list_successful_session(converted: Path, catalog: dict[str, FakeEntry]) -> None:
    """A completed registration must be visible under the manifest session ID."""
    catalog["agent-traces-x"] = FakeEntry(omit="a")
    with pytest.raises(RuntimeError, match="a"):
        register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, profile="x"))


def test_errors_are_reported_per_recording(converted: Path, catalog: dict[str, FakeEntry], capsys: pytest.CaptureFixture[str]) -> None:
    """A wait error still exposes individual failed and successful results."""
    catalog["agent-traces-x"] = FakeEntry(fail="a", blueprints=["existing"])
    register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, profile="x"))
    output: str = capsys.readouterr().out
    assert f"ERROR {(converted / 'x/a.rrd').as_uri()}: bad recording" in output
    assert "registered=1 skipped_duplicates=0 errors=1 missing=0" in output
    assert catalog["agent-traces-x"].blueprints == ["existing"]
    assert not (converted / "x/agent-traces.rbl").exists()


def test_all_published_files_are_readable(converted: Path, catalog: dict[str, FakeEntry]) -> None:
    """The catalog's user can read recordings, manifests, and the default blueprint."""
    register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted))
    for profile in ("x", "y"):
        for name in ("a.rrd", "manifest.json", "agent-traces.rbl"):
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
        register.main(register.Config(catalog_url="rerun+http://127.0.0.1:1", out=converted, profile="x"))
    assert set((converted / "x").iterdir()) == before
    assert not catalog["agent-traces-x"].blueprints
