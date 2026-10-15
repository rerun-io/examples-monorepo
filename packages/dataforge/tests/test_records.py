"""The shared decode door: typed records, errors that name their source, and redaction."""

import traceback
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pytest
from serde import serde

from dataforge.records import decode, read_csv_columns, read_json, read_member


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


def test_csv_columns_read_typed_by_name(tmp_path: Path) -> None:
    path: Path = tmp_path / "poses.csv"
    path.write_text("label,stamp_us,x\na,10,0.5\nb,20,nan\nc,30,\n")
    table: pa.Table = read_csv_columns(path, {"x": pa.float64(), "stamp_us": pa.int64()})
    assert table.column_names == ["x", "stamp_us"]  # as asked; other columns are never parsed
    np.testing.assert_array_equal(table.column("stamp_us").to_numpy(), np.array([10, 20, 30], dtype=np.int64))
    np.testing.assert_array_equal(table.column("x").to_numpy(), [0.5, np.nan, np.nan])  # empty and nan float cells are NaN


@pytest.mark.parametrize(
    ("text", "match"),
    [("stamp_us\n0.5\n", "int64"), ("stamp_us,x\n,1\n10,2\n", "empty"), ("other\n1\n", "stamp_us")],
    ids=["fractional-int", "empty-int", "missing-column"],
)
def test_csv_column_errors_name_the_file(tmp_path: Path, text: str, match: str) -> None:
    path: Path = tmp_path / "stamps.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match) as error:
        read_csv_columns(path, {"stamp_us": pa.int64()})
    assert str(path) in str(error.value)
