"""The shared decode door: typed records, errors that name their source, and redaction."""

import traceback
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pytest
from serde import serde

from dataforge.records import decode, read_json, read_member


@serde
@dataclass(frozen=True, slots=True)
class Entry:
    url: str
    size: int

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("size must be non-negative")


def test_read_json_and_read_member_decode_typed_records(tmp_path: Path) -> None:
    (tmp_path / "entry.json").write_text('{"url": "https://a", "size": 3}')
    with ZipFile(tmp_path / "bundle.zip", "w") as archive:
        archive.writestr("meta/entry.yaml", "url: https://b\nsize: 4\n")
    assert read_json(tmp_path / "entry.json", Entry) == Entry("https://a", 3)
    with ZipFile(tmp_path / "bundle.zip") as archive:
        assert read_member(archive, "meta/entry.yaml", Entry) == Entry("https://b", 4)


@pytest.mark.parametrize(
    "text",
    ['{"url": "https://a"', '{"url": "https://a"}', '{"url": "https://a", "size": -1}'],
    ids=["parser", "schema", "post-init"],
)
def test_every_decode_failure_names_the_source(text: str) -> None:
    with pytest.raises(ValueError, match=r"^entries\.json: "):
        decode(Entry, text, source="entries.json")


def test_member_errors_name_archive_and_member(tmp_path: Path) -> None:
    with ZipFile(tmp_path / "bundle.zip", "w") as archive:
        archive.writestr("entry.yaml", "url: [unclosed\n")
    with ZipFile(tmp_path / "bundle.zip") as archive, pytest.raises(ValueError, match=r"bundle\.zip:entry\.yaml: "):
        read_member(archive, "entry.yaml", Entry)


def test_redacted_errors_never_quote_the_document() -> None:
    document = '{"url": ["https://cdn/x?sig=SECRET"], "size": 1}'
    with pytest.raises(ValueError, match=r"^urls\.json: not a URL file \(SerdeError\)$") as failure:
        decode(Entry, document, source="urls.json", redact="not a URL file")
    assert "SECRET" not in "".join(traceback.format_exception(failure.value))
