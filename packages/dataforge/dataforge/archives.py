"""Reading named members out of one dataset sequence's archive, whole or split.

An Info-ZIP multi-volume set (``<stem>.z01``, ``<stem>.z02``, …, ``<stem>.zip``)
and a plain ``.zip`` need completely different machinery — Python's ``zipfile``
cannot read a spanned archive at all — but a converter only ever asks two things
of either: give me this member, and give me these members in this order. That
pair is the whole seam, and ``open_member_reader`` picks the implementation.
Neither method knows what a member holds: the small one is a csv today and the
streamed ones are PNG frames, and a reader that named either would have to grow
a method per format.

Monado SLAM is the first dataset that needs this; nothing here is MSD-specific.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from types import TracebackType

PART_SUFFIX_RE: re.Pattern[str] = re.compile(r"^\.z\d+$")
"""Info-ZIP split-volume suffix (``.z01``, ``.z02``, …); the closing volume is the plain ``.zip``."""


def remove_tree(root: Path) -> None:
    """Delete a scratch directory tree, reporting what it could not delete.

    ``shutil.rmtree(..., ignore_errors=True)`` is the wrong default for a raw
    tree under a size budget: a directory that quietly fails to go away is
    exactly how the next sequence's fetch finds the disk full and blames its own
    leftovers. A tree that was never there is not a failure, so only real errors
    are printed — and they are printed rather than raised, because every caller
    is already cleaning up after something else.

    Args:
        root: Directory to remove, along with everything under it.
    """
    if not root.exists():
        return
    shutil.rmtree(root, onexc=lambda _function, failed, error: print(f"  warning: could not remove {failed}: {type(error).__name__}: {error}"))


def resolve_seven_zip() -> Path:
    """Locate the 7-Zip CLI that reads multi-volume archives.

    conda-forge's ``7zip`` package installs the modern ``7zz`` and keeps ``7z``
    as a compatibility name, so both are accepted.
    """
    for binary in ("7zz", "7z"):
        found: str | None = shutil.which(binary)
        if found is not None:
            return Path(found)
    raise FileNotFoundError("no 7zz/7z on PATH; the conda-forge '7zip' package provides it (see [feature.dataforge.dependencies])")


class MemberReader(AbstractContextManager["MemberReader"], ABC):
    """Reads named members out of one sequence archive; the seam both readers implement.

    A reader owns whatever scratch it needed: leaving the context removes it,
    whether the caller finished the members or abandoned them mid-iteration.
    """

    @abstractmethod
    def read_member(self, member: str) -> bytes:
        """Whole contents of one small member, e.g. ``<SEQ>/mav0/imu0/data.csv``."""

    @abstractmethod
    def iter_members(self, members: Sequence[str]) -> Iterator[bytes]:
        """Contents of ``members``, in the given order, one at a time.

        For a stream too big to hold at once — one camera's PNG frames — which
        is why the members must all live in **one** directory: a multi-volume
        archive is extracted a directory at a time, so mixing two cameras in one
        call would extract both and defeat the extraction budget.

        Raises:
            ValueError: The members do not share one parent directory.
        """


def one_parent_directory(members: Sequence[str]) -> str:
    """The single directory ``members`` live in, refusing a call that spans two.

    Checked in both readers rather than in the split one alone: the constraint is
    part of the seam, so the cheap in-process reader has to reject the same calls
    the expensive one cannot serve, or a converter would only find out on the one
    sequence that happens to ship a split archive.

    Args:
        members: Archive-relative member names, at least one.

    Returns:
        Their common parent, as a posix path string.

    Raises:
        ValueError: They do not all share one parent.
    """
    parents: set[str] = {str(PurePosixPath(member).parent) for member in members}
    if len(parents) != 1:
        raise ValueError(f"iter_members reads one directory at a time, but these members span {sorted(parents)}")
    return parents.pop()


class ZipMemberReader(MemberReader):
    """A plain single-file ``.zip``, read in-process with the stdlib."""

    def __init__(self, archive: Path) -> None:
        self.archive: zipfile.ZipFile = zipfile.ZipFile(archive)

    def read_member(self, member: str) -> bytes:
        return self.archive.read(member)

    def iter_members(self, members: Sequence[str]) -> Iterator[bytes]:
        if members:
            one_parent_directory(members)
        for member in members:
            yield self.archive.read(member)

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        self.archive.close()


class SevenZipMemberReader(MemberReader):
    """An Info-ZIP multi-volume set (``.z01``…``.zip``), read through the 7-Zip CLI.

    Python's ``zipfile`` opens the closing volume — its central directory is
    intact — and then fails on the first member whose data crosses a volume
    boundary, so it cannot be used at all here. 7-Zip cannot stream either, so
    ``iter_members`` extracts the members' directory into ``work_dir``, yields the
    files from there, and deletes them again: peak scratch is one directory's
    members rather than the whole sequence.
    """

    def __init__(self, closing_volume: Path, work_dir: Path) -> None:
        self.archive: Path = closing_volume
        self.work_dir: Path = work_dir
        self.binary: Path = resolve_seven_zip()

    def _run(self, arguments: Sequence[str]) -> bytes:
        """Run one 7-Zip command and return its stdout; ``-bso0 -bsp0`` keep progress off it."""
        completed: subprocess.CompletedProcess[bytes] = subprocess.run(
            [str(self.binary), *arguments], capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{self.binary.name} exited {completed.returncode} on {self.archive.name}:\n{completed.stderr.decode(errors='replace')}")
        return completed.stdout

    def read_member(self, member: str) -> bytes:
        return self._run(["x", "-so", "-bso0", "-bsp0", str(self.archive), member])

    def iter_members(self, members: Sequence[str]) -> Iterator[bytes]:
        if not members:
            return
        directory: str = one_parent_directory(members)
        extract_dir: Path = self.work_dir / "extract"
        remove_tree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        self._run(["x", f"-o{extract_dir}", "-y", "-bso0", "-bsp0", str(self.archive), f"{directory}/*"])
        try:
            for member in members:
                yield (extract_dir / member).read_bytes()
        finally:
            remove_tree(extract_dir)

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        remove_tree(self.work_dir / "extract")


def open_member_reader(archives: Sequence[Path], work_dir: Path) -> MemberReader:
    """Pick the reader one sequence's archive files need.

    Args:
        archives: Local volume paths, parts first and the closing ``.zip`` last.
        work_dir: Scratch directory the multi-volume reader extracts into.

    Returns:
        A stdlib reader for a single file, the 7-Zip-backed one for a volume set.
    """
    if len(archives) == 1:
        return ZipMemberReader(archives[0])
    return SevenZipMemberReader(archives[-1], work_dir)


def group_archives(entries: Sequence[tuple[str, int]]) -> dict[str, list[tuple[str, int]]]:
    """Group a collection listing into ``stem → volumes``, dropping non-archives.

    An MSD sequence is either one ``<stem>.zip`` or an Info-ZIP multi-volume set
    (``<stem>.z01``, ``<stem>.z02``, …, ``<stem>.zip``). Volumes come back parts
    first and ascending, with the closing ``.zip`` last, which is the order 7-Zip
    wants them named in.

    Args:
        entries: ``(repo-relative path, size)`` pairs from one collection directory.

    Returns:
        One entry per sequence stem; README files and anything else are dropped.
    """
    grouped: dict[str, list[tuple[str, int]]] = {}
    for path, size in entries:
        suffix: str = Path(path).suffix
        if suffix != ".zip" and PART_SUFFIX_RE.match(suffix) is None:
            continue
        grouped.setdefault(Path(path).stem, []).append((path, size))
    # ".zip" sorts after ".z01".."z99" lexicographically, so one sort gives both
    # the ascending part order and the closing volume's place at the end.
    return {stem: sorted(volumes) for stem, volumes in sorted(grouped.items())}
