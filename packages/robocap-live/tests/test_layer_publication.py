"""Layer publication failures and overlapping registrations, without a catalog server."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import pytest
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle, RegistrationResult

from robocap_live.apis.hands_layer import publish_layer


class FakeRegistration(RegistrationHandle):
    def __init__(self, complete: Callable[[], None]) -> None:
        self.complete: Callable[[], None] = complete

    def wait(self, timeout_secs: int | None = None) -> RegistrationResult:
        self.complete()
        return RegistrationResult(segment_ids=["segment"])


class FakeDataset(DatasetEntry):
    def __init__(self, old: Path, outcome: Literal["success", "failed", "ambiguous"] = "success") -> None:
        self.registered: Path = old
        self.outcome: Literal["success", "failed", "ambiguous"] = outcome
        self.candidates: list[Path] = []
        self.during_wait: Callable[[], None] | None = None

    def register(self, recording_uri: list[str], *, layer_name: str | Sequence[str] = "base",
                 on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.ERROR) -> RegistrationHandle:
        assert layer_name == "hands" and on_duplicate == OnDuplicateSegmentLayer.REPLACE
        assert len(recording_uri) == 1
        candidate: Path = Path(unquote(urlparse(recording_uri[0]).path))
        self.candidates.append(candidate)

        def complete() -> None:
            if self.during_wait is not None:
                nested: Callable[[], None] = self.during_wait
                self.during_wait = None
                nested()
            if self.outcome == "failed":
                raise RuntimeError("registration rejected")
            self.registered = candidate
            if self.outcome == "ambiguous":
                raise TimeoutError("reply lost after commit")

        return FakeRegistration(complete)


@pytest.mark.parametrize("outcome", ["failed", "ambiguous"])
def test_failed_registration_keeps_old_and_candidate_bytes(tmp_path: Path, outcome: Literal["failed", "ambiguous"]) -> None:
    old: Path = tmp_path / "segment.rrd"
    old.write_bytes(b"old registered layer")
    partial: Path = tmp_path / ".run.partial"
    partial.write_bytes(b"new completed layer")
    dataset: FakeDataset = FakeDataset(old, outcome)

    with pytest.raises((RuntimeError, TimeoutError)):
        publish_layer(partial, old, dataset)

    assert old.read_bytes() == b"old registered layer"
    assert dataset.candidates[0] != old
    assert dataset.candidates[0].suffix == ".rrd"
    assert dataset.candidates[0].read_bytes() == b"new completed layer"
    assert dataset.registered == (old if outcome == "failed" else dataset.candidates[0])
    assert not partial.exists()


def test_overlapping_registrations_each_name_their_own_bytes(tmp_path: Path) -> None:
    old: Path = tmp_path / "segment.rrd"
    old.write_bytes(b"old registered layer")
    first: Path = tmp_path / ".first.partial"
    second: Path = tmp_path / ".second.partial"
    first.write_bytes(b"first run")
    second.write_bytes(b"second run")
    dataset: FakeDataset = FakeDataset(old)

    def publish_second() -> None:
        assert dataset.registered == old
        assert old.read_bytes() == b"old registered layer"
        publish_layer(second, old, dataset)

    dataset.during_wait = publish_second
    output: Path = publish_layer(first, old, dataset)
    assert len(set(dataset.candidates)) == 2
    assert [path.read_bytes() for path in dataset.candidates] == [b"first run", b"second run"]
    assert dataset.registered == output == dataset.candidates[0]
    assert old.read_bytes() == b"old registered layer"


def test_unregistered_runs_also_keep_prior_artifacts(tmp_path: Path) -> None:
    old: Path = tmp_path / "segment.first30.rrd"
    old.write_bytes(b"previous smoke run")
    partial: Path = tmp_path / ".run.partial"
    partial.write_bytes(b"new smoke run")
    output: Path = publish_layer(partial, old, None)
    assert output != old and output.suffix == ".rrd"
    assert output.read_bytes() == b"new smoke run"
    assert old.read_bytes() == b"previous smoke run"
