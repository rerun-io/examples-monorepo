"""What the package docs must keep true: nothing points at a build host, and every decision tag the code cites is explained."""

import re
from pathlib import Path

PACKAGE: Path = Path(__file__).resolve().parents[1]
DOCS: tuple[Path, ...] = (PACKAGE / "README.md", *sorted((PACKAGE / "docs").glob("*.md")))


def test_no_doc_links_into_a_build_host() -> None:
    """A path under /tmp or /mnt exists on one machine for one afternoon; a reader elsewhere gets nothing from it."""
    for doc in DOCS:
        hits: list[str] = [line for line in doc.read_text().splitlines() if re.search(r"(?<![\w.])/(tmp|mnt)/", line)]
        assert not hits, f"{doc.name}: {hits[:3]}"


def test_every_decision_tag_the_code_cites_is_in_the_index() -> None:
    """`Dnn` tags in comments are only useful if the design notes still say what each decided."""
    cited: set[str] = set()
    for source in [*(PACKAGE / "slam_rs").rglob("*.py"), *(PACKAGE / "crates").rglob("*.rs"), *(PACKAGE / "tools").rglob("*.py")]:
        if "target" in source.parts:
            continue
        cited.update(re.findall(r"\bD\d{2}\b", source.read_text()))
    notes: str = (PACKAGE / "docs" / "design-notes.md").read_text()
    index: str = notes[notes.index("## Decision references") :]
    missing: set[str] = {tag for tag in cited if not re.search(rf"\*\*{tag}\*\*", index)}
    assert not missing, sorted(missing)
