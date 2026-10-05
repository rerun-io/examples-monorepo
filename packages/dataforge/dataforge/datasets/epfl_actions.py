"""Read the release's small inline-string XLSX and activity JSON sidecars."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from serde import coerce, from_dict, serde

from dataforge.actions import Segment
from dataforge.records import read_json


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FineRow:
    """One shipped worksheet row, including empty labels and confusion flags."""

    Start: float
    """Start seconds from first video frame."""
    End: float
    """End seconds from first video frame."""
    Verbs: str = ""
    """Shipped verb."""
    Nouns: str = ""
    """Shipped noun."""
    Confusion: str = ""
    """Shipped ambiguity flag, including empty values."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Activity:
    """One coarse activity segment."""

    start: float
    """Start seconds."""
    end: float
    """End seconds."""
    Activities: str
    """Shipped activity label."""


@serde
@dataclass(frozen=True, slots=True)
class ActivityDocument:
    """Partial third-party JSON envelope."""

    annotations: list[Activity]
    """Coarse segments."""


def read_xlsx(path: Path) -> list[FineRow]:
    """Read only the release's inlineStr single-sheet layout; fail loudly otherwise.

    Numeric cells and absent cell references are supported. Shared strings,
    formulas, other cell types and additional worksheets are rejected.
    """
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        sheets: list[str] = [name for name in archive.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml")]
        if sheets != ["xl/worksheets/sheet1.xml"] or "xl/sharedStrings.xml" in archive.namelist():
            raise ValueError(f"{path}: expected the release's inlineStr single-sheet layout")
        root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    result = []
    headers = {}
    for index, row in enumerate(root.findall("s:sheetData/s:row", ns)):
        cells = {}
        for cell in row.findall("s:c", ns):
            if cell.attrib.get("t", "n") not in ("n", "inlineStr") or cell.find("s:f", ns) is not None:
                raise ValueError(f"{path}: unsupported worksheet cell {cell.attrib['r']}")
            column = "".join(c for c in cell.attrib["r"] if c.isalpha())
            text = "".join(cell.itertext()) if cell.attrib.get("t") == "inlineStr" else cell.findtext("s:v", default="", namespaces=ns)
            cells[column] = text
        if index == 0:
            headers = cells
            if list(headers.values()) != ["Start", "End", "Verbs", "Nouns", "Confusion"]:
                raise ValueError(f"{path}: unexpected worksheet header")
        elif any(cells.values()):
            result.append(from_dict(FineRow, {name: cells.get(column, "") for column, name in headers.items()}))
    if not headers:
        raise ValueError(f"{path}: missing worksheet header")
    return result


def read_actions(root: Path) -> dict[str, list[Segment]]:
    """Round seconds to 30 Hz frame boundaries; no timestamp rebasing."""
    fine = read_xlsx(root / "actions_annotations.xlsx")
    coarse: ActivityDocument = read_json(root / "activity_annotations.json", ActivityDocument)
    return {
        "fine": [Segment(round(row.Start * 30), round(row.End * 30), f"{row.Verbs} {row.Nouns} [Confusion={row.Confusion}]".strip()) for row in fine],
        "coarse": [Segment(round(row.start * 30), round(row.end * 30), row.Activities) for row in coarse.annotations],
    }

